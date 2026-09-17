"""Synthetic tests only. Never write simulated outcomes into production results."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import analyze
import protocol as p
import run


def synthetic_records(k):
    ids = list(dict.fromkeys(r["fixture_id"] for r in p.rows_for(k)))
    records = []
    for row in p.rows_for(k):
        index = ids.index(row["fixture_id"])
        correct = index < len(ids) // 2 or (row["cell"] == "NN" and index < len(ids) // 2 + 9)
        choice = row["ideal"] if correct else ("B" if row["ideal"] == "A" else "A")
        records.append(dict(fixture=row, text=choice, parsed=choice, strict_parse=choice,
                            generated_correct=correct, logit_correct=correct, logit_choice=choice,
                            raw_ab_margin=1.0 if choice == "A" else -1.0,
                            processed_ab_margin=1.0 if choice == "A" else -1.0,
                            generated_ids=[1 if choice == "A" else 2],
                            first_generated_id=1 if choice == "A" else 2, first_raw_argmax_id=1,
                            a_token_id=1, b_token_id=2, input_tokens=100, batch_padded_input_tokens=104,
                            batch_seconds=0.01, run_id="SYNTHETIC_UNIT_TEST", commit="SYNTHETIC_UNIT_TEST",
                            prompt_sha256="0" * 64))
    return records


def synthetic_payload(k=2, shard=0, n=8):
    return dict(protocol=p.VERSION, freeze_hash="test-freeze", model=p.MODEL, revision=p.REVISION,
                dtype="torch.bfloat16", backend="transformers-cpu", factor_count=k, shard=shard,
                status="COMPLETE" if n == 8 else "PARTIAL", attempts=[],
                records=synthetic_records(k)[shard * 8:shard * 8 + n])


def fake_inference_modules(input_length=100):
    """Minimal local protocol double, never a real model or production observation."""
    class Scalar:
        def __init__(self, value):
            self.value = value
        def item(self):
            return self.value
        def __sub__(self, other):
            return Scalar(self.value - other.value)

    class Raw:
        def float(self):
            return self
        def cpu(self):
            return self
        def __getitem__(self, key):
            return Scalar(1.0 if key[1] == 1 else 0.0) if isinstance(key, tuple) else SimpleNamespace(argmax=lambda: Scalar(1))

    generated = MagicMock()
    generated.__getitem__.return_value.cpu.return_value.tolist.return_value = [1]
    sequence = MagicMock()
    sequence.__getitem__.return_value = generated
    output = SimpleNamespace(sequences=sequence, logits=[Raw()], scores=[Raw()])
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: json.dumps(messages)
    tokenizer.encode.side_effect = lambda text, **kwargs: [1] if text == "A" else [2] if text == "B" else [3] * input_length
    tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(4, input_length))}
    tokenizer.batch_decode.return_value = ["A"] * 4
    model = MagicMock()
    model.generation_config.to_dict.return_value = {}
    model.generate.return_value = output
    transformers = SimpleNamespace(__version__="TEST_DOUBLE", AutoTokenizer=MagicMock(), AutoModelForCausalLM=MagicMock())
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    torch = SimpleNamespace(__version__="TEST_DOUBLE", bfloat16="TEST_DOUBLE",
                            manual_seed=lambda seed: None, set_num_threads=lambda threads: None,
                            get_num_threads=lambda: 4, inference_mode=nullcontext)
    numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None))
    return {"torch": torch, "numpy": numpy, "transformers": transformers}, model, output


class FixtureTests(unittest.TestCase):
    def test_sizes_budget_and_balance(self):
        audit = p.audit()
        self.assertEqual(sum(x["prompts"] for x in audit.values()), 640)
        self.assertEqual(640 * p.MAX_NEW, 2560)
        self.assertEqual(640 * p.MAX_INPUT, 327680)
        self.assertEqual(sum(x["slices"] for x in audit.values()), 20)
        for value in audit.values():
            exposure = value["active_factor_exposure"].values()
            self.assertLessEqual(max(exposure) - min(exposure), 2)

    def test_nested_oracle_and_order(self):
        baseline = {r["fixture_id"]: r for r in p.rows_for(5) if r["cell"] == "LL"}
        for k in p.COUNTS:
            for row in p.rows_for(k):
                permitted = sum(row["self_bits"][i] for i in row["active_order"]) >= k // 2 + 1
                self.assertEqual(permitted, row["permitted"])
                self.assertEqual(row["ideal"], row["allow_code"] if permitted else row["block_code"])
                self.assertEqual(row["ideal"], baseline[row["fixture_id"]]["ideal"])
                self.assertEqual(len(row["active_order"]), k)
                self.assertEqual(row["active_order"], [i for i in row["display_order"] if i in row["insertion_order"][:k]])

    def test_semantic_cells_are_matched_and_counterordered(self):
        for k in p.COUNTS:
            rows = p.rows_for(k)
            for i in range(0, len(rows), 4):
                batch = rows[i:i + 4]
                self.assertEqual({r["cell"] for r in batch}, set(p.CELLS))
                for key in ("fixture_id", "ideal", "active_order", "allow_code", "self_bits", "peer_bits", "persona"):
                    self.assertTrue(all(r[key] == batch[0][key] for r in batch))
                self.assertEqual(len({json.dumps(r["messages"]) for r in batch}), 4)
            for position in range(4):
                self.assertEqual(Counter(r["cell"] for r in rows[position::4]), {c: p.COUNTS[k] // 4 for c in p.CELLS})

    def test_reproducible_shards_cover_dataset(self):
        for k in p.COUNTS:
            merged = [r for s in range(p.COUNTS[k] // 2) for r in p.shard_rows(k, s)]
            self.assertEqual(merged, p.rows_for(k))
            self.assertEqual(p.digest(p.rows_for(k)), p.digest(p.rows_for(k)))
        for k, shard in ((1, 0), (2, -1), (2, 16), (5, 32)):
            with self.assertRaises(ValueError):
                p.shard_rows(k, shard)

    def test_parser_parity(self):
        for text, choice in (("A", "A"), (" b\n", "B"), ("Answer: A.", "A"),
                             ("AB", None), ("A or B", None), ("BLOCK", None), ("", None)):
            self.assertEqual(p.parse_choice(text), choice)

    def test_source_freeze_when_present(self):
        if (p.ROOT / "manifest.json").exists():
            self.assertEqual(len(p.check_freeze()), 64)


class StatisticsTests(unittest.TestCase):
    def test_exact_paired_known_values(self):
        self.assertEqual(analyze.mcnemar_exact(9, 0), .00390625)
        self.assertEqual(analyze.mcnemar_exact(0, 9), .00390625)
        self.assertEqual(analyze.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(analyze.mcnemar_exact(11, 5), .210113525390625)

    def test_exact_intervals(self):
        self.assertAlmostEqual(analyze.exact_ci(0, 32)[1], 1 - .025 ** (1 / 32))
        self.assertAlmostEqual(analyze.exact_ci(32, 32)[0], .025 ** (1 / 32))
        low, high = analyze.exact_ci(16, 32)
        self.assertAlmostEqual(low, 1 - high)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)

    def test_only_five_factor_primary(self):
        for k in p.COUNTS:
            report = analyze.analyze_stage(k, synthetic_records(k), fit_secondary=False)
            self.assertEqual(set(report["cells"]), set(p.CELLS))
            self.assertEqual(len(report["generated_pairs_descriptive"]), 6)
            if k == 5:
                self.assertEqual(report["primary"]["p_value"], .00390625)
                self.assertTrue(report["primary"]["neutral_benefit_replicated"])
            else:
                self.assertNotIn("p_value", report["primary"])

    def test_missing_duplicate_rejected(self):
        rows = synthetic_records(2)
        for invalid in (rows[:-1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                analyze.analyze_stage(2, invalid)

    def test_invalid_output_counts_wrong(self):
        rows = synthetic_records(2)[:4]
        rows[0].update(text="unclear", parsed=None, strict_parse=None, generated_correct=False)
        summary = analyze.cell_summary(rows)
        self.assertEqual(summary["parser_coverage"], .75)
        self.assertFalse(summary["descriptive_criterion_met"])

    @unittest.skipUnless(os.environ.get("RES_TEST_GLM") == "1", "Optional pinned analysis dependency integration test")
    def test_secondary_fit(self):
        result = analyze.random_intercept_model(synthetic_records(5))
        self.assertEqual(result["status"], "FIT_OK", result)
        self.assertEqual(len(result["effects"]), 4)
        self.assertNotIn("p_value", result)


class SafetyTests(unittest.TestCase):
    def test_inference_path_checkpoint_then_resume_after_interruption(self):
        modules, model, output = fake_inference_modules()
        with tempfile.TemporaryDirectory(prefix="res-ladder-inference-test-") as directory:
            with patch.object(p, "ROOT", Path(directory)), patch.dict("sys.modules", modules):
                model.generate.side_effect = [output, TimeoutError("synthetic interruption")]
                with self.assertRaises(TimeoutError):
                    run.run_shard(2, 0, "test-freeze")
                partial = run.load_checkpoint(2, 0, "test-freeze")
                self.assertEqual(len(partial["records"]), 4)
                model.generate.side_effect = None
                model.generate.reset_mock()
                self.assertTrue(run.run_shard(2, 0, "test-freeze"))
                complete = run.load_checkpoint(2, 0, "test-freeze")
                self.assertEqual(complete["records"][:4], partial["records"])
                self.assertEqual(len(complete["records"]), 8)
                self.assertEqual(model.generate.call_count, 1)

    def test_token_cap_blocks_before_model_load(self):
        modules, model, _ = fake_inference_modules(input_length=513)
        with tempfile.TemporaryDirectory(prefix="res-ladder-input-test-") as directory:
            with patch.object(p, "ROOT", Path(directory)), patch.dict("sys.modules", modules):
                with self.assertRaisesRegex(ValueError, "Input cap exceeded"):
                    run.run_shard(2, 0, "test-freeze")
                modules["transformers"].AutoModelForCausalLM.from_pretrained.assert_not_called()
                model.generate.assert_not_called()

    def test_checkpoint_prefix_and_corruption(self):
        for n in (0, 4, 8):
            self.assertEqual(len(run.validate(synthetic_payload(n=n), 2, 0, "test-freeze")), n)
        for key, value in (("freeze_hash", "wrong"), ("model", "wrong"), ("revision", "wrong"),
                           ("backend", "mock"), ("dtype", "float32"), ("status", "PARTIAL")):
            payload = synthetic_payload()
            payload[key] = value
            with self.assertRaises(ValueError):
                run.validate(payload, 2, 0, "test-freeze")
        for mutate in (lambda x: x["records"].pop(),
                       lambda x: x["records"][0].update(input_tokens=513),
                       lambda x: x["records"][0].update(generated_ids=[1] * 5),
                       lambda x: x["records"][0].update(generated_correct=False),
                       lambda x: x["records"][0].update(raw_ab_margin=float("nan")),
                       lambda x: x["records"].reverse()):
            payload = synthetic_payload()
            mutate(payload)
            with self.assertRaises(ValueError):
                run.validate(payload, 2, 0, "test-freeze")

    def test_complete_resume_does_not_import_or_run_model(self):
        with patch.object(run, "load_checkpoint", return_value=synthetic_payload()):
            self.assertTrue(run.run_shard(2, 0, "test-freeze"))

    def test_advancement_requires_completion_not_accuracy(self):
        control = dict(operation="run", factor_count=2, slice=0, attempt=2)
        with patch.object(run, "load_checkpoint", return_value=None):
            self.assertEqual(run.plan(control, "test")["shards"], [0, 1, 2, 3])
            for k, part in ((2, 1), (3, 0), (4, 0), (5, 0)):
                with self.assertRaises(ValueError):
                    run.plan(dict(control, factor_count=k, slice=part), "test")
        with patch.object(run, "load_checkpoint", return_value={"status": "COMPLETE"}):
            ready = run.plan(dict(control, factor_count=5, slice=7), "test")
            self.assertEqual(ready["operation"], "report")
            self.assertEqual(ready["shards"], [])
        with patch.object(run, "load_checkpoint", side_effect=lambda k, s, freeze: {"status": "COMPLETE"} if s == 0 else None):
            self.assertEqual(run.plan(control, "test")["shards"], [1, 2, 3])
        for control_update in ({"operation": "auto"}, {"slice": -1}, {"factor_count": True}, {"attempt": 0}):
            with self.assertRaises(ValueError):
                run.plan(dict(control, **control_update), "test")

    def test_collection_retains_partial_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory(prefix="res-ladder-test-") as directory:
            base = Path(directory)
            with patch.object(p, "ROOT", base):
                source = base / "incoming" / "artifact" / "k2-s0.json"
                p.atomic_json(source, synthetic_payload(n=4))
                self.assertFalse(run.collect(2, 0, base / "incoming", "test-freeze"))
                self.assertEqual(len(run.load_checkpoint(2, 0, "test-freeze")["records"]), 4)
                p.atomic_json(source, synthetic_payload())
                self.assertFalse(run.collect(2, 0, base / "incoming", "test-freeze"))
                original = p.shard_path(2, 0).read_bytes()
                changed = synthetic_payload()
                changed["records"][0]["run_id"] = "UNAUTHORIZED_REPLACEMENT"
                p.atomic_json(source, changed)
                with self.assertRaises(ValueError):
                    run.collect(2, 0, base / "incoming", "test-freeze")
                self.assertEqual(p.shard_path(2, 0).read_bytes(), original)

    def test_duplicate_artifacts_rejected(self):
        with tempfile.TemporaryDirectory(prefix="res-ladder-test-") as directory:
            base = Path(directory)
            with patch.object(p, "ROOT", base):
                for name in ("a", "b"):
                    p.atomic_json(base / "incoming" / name / "k2-s0.json", synthetic_payload())
                with self.assertRaises(ValueError):
                    run.collect(2, 0, base / "incoming", "test-freeze")

    def test_time_and_artifact_bounds_in_workflow(self):
        workflow = (p.REPO / ".github/workflows/res-complexity-ladder.yml").read_text()
        for expected in ("timeout-minutes: 150", "timeout-minutes: 110", "max-parallel: 2", "fail-fast: false",
                         "ref: ${{ github.sha }}", "-s${{ env.SHARD }}.json", "cancel-in-progress: false"):
            self.assertIn(expected, workflow)
        self.assertLessEqual(10 + 2 * 150 + 20, 360)


if __name__ == "__main__":
    unittest.main()
