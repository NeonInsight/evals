from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phase0
import representation_validator
import study
import analysis_secondary


class SurfaceEfficiencyStudyTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(study.parse_choice("A"), "A")
        self.assertEqual(study.parse_choice("answer: b"), "B")
        self.assertIsNone(study.parse_choice("A or B"))
        self.assertIsNone(study.parse_choice("none"))

    def test_fresh_pool_invariants(self):
        candidate = study.build_pool(study.SCIENTIFIC_N, "candidate")
        reserve = study.build_pool(
            study.RESERVE_N, "reserve", study.SCIENTIFIC_N // 32
        )
        profiles = []
        for fixture in candidate + reserve:
            study.validate_fixture_invariants(fixture)
            profiles.append(fixture["semantic_profile_sha256"])
        self.assertEqual(len(profiles), len(set(profiles)))
        self.assertFalse(set(profiles) & phase0.prior_profile_hashes())

    def test_conditions_change_only_declared_graph_field(self):
        fixture = study.build_pool(32, "candidate")[0]
        for surface in ("VERBOSE", "ABBREVIATED", "SYMBOLIC"):
            preserved = f"PRESERVED_{surface}"
            changed = f"LOCUS_CHANGED_{surface}"
            self.assertEqual(
                study.graph_signature(preserved)["edges"],
                study.graph_signature(changed)["edges"],
            )
            self.assertNotEqual(
                study.graph_signature(preserved)["decision_locus"],
                study.graph_signature(changed)["decision_locus"],
            )
            self.assertNotEqual(
                study.expected_for(fixture, preserved),
                study.expected_for(fixture, changed),
            )
        self.assertEqual(
            study.graph_signature("PRESERVED_SYMBOLIC"),
            study.graph_signature("NONREL_CONTROL_SYMBOLIC"),
        )
        self.assertEqual(
            study.expected_for(fixture, "PRESERVED_SYMBOLIC"),
            study.expected_for(fixture, "NONREL_CONTROL_SYMBOLIC"),
        )

    def test_condition_blind_representation_parser(self):
        fixture = study.build_pool(32, "candidate")[0]
        for condition in study.CONDITIONS:
            packet = {
                "packet_id": condition,
                "text": study.messages_for(fixture, condition)[1]["content"],
            }
            parsed = representation_validator.parse_packet(packet)
            self.assertTrue(parsed["valid"])
            self.assertEqual(parsed["surface_form"], study.SURFACE_FORM[condition])
            self.assertEqual(parsed["graph_signature"], study.graph_signature(condition))
            self.assertEqual(
                parsed["marker"],
                "Y" if condition == "NONREL_CONTROL_SYMBOLIC" else "X",
            )

    def test_zero_variance_is_reported_not_fitted(self):
        predictor_constant = analysis_secondary._spearman([0, 0, 0], [0, 1, 0])
        outcome_constant = analysis_secondary._spearman([0, 1, 2], [1, 1, 1])
        self.assertEqual(
            predictor_constant["status"], "NOT_ESTIMABLE_ZERO_VARIANCE_PREDICTOR"
        )
        self.assertEqual(
            outcome_constant["status"], "NOT_ESTIMABLE_ZERO_VARIANCE_OUTCOME"
        )

    def test_workflow_caps_each_inference_job(self):
        workflow = (study.REPO / ".github/workflows/res-surface-efficiency-assay.yml").read_text()
        self.assertIn("timeout-minutes: 270", workflow)
        self.assertIn("timeout-minutes: 250", workflow)
        self.assertIn("RES_SFE_SOFT_SECONDS: '14400'", workflow)
        self.assertEqual(study.SCIENTIFIC_N // study.SHARD_FIXTURES, 16)
        self.assertEqual(study.SHARD_FIXTURES * len(study.CONDITIONS), 56)

    @unittest.skipUnless((HERE / "frozen_manifest.json").exists(), "Phase 0 not built")
    def test_phase0_hashes_and_counts(self):
        phase0.verify()
        manifest = json.loads((HERE / "frozen_manifest.json").read_text())
        fixtures = study.read_jsonl(HERE / "fixtures.jsonl")
        rows = study.read_jsonl(HERE / "rendered_prompts.jsonl")
        self.assertEqual(len(fixtures), study.SCIENTIFIC_N)
        self.assertEqual(len(rows), len(fixtures) * len(study.CONDITIONS))
        self.assertEqual(manifest["expected_observations"], len(rows))
        self.assertEqual({row["condition"] for row in rows}, set(study.CONDITIONS))

    @unittest.skipUnless((HERE / "surface_validation.json").exists(), "Phase 0 not built")
    def test_surface_ladder_and_matched_symbolic_edits(self):
        rows = study.read_jsonl(HERE / "rendered_prompts.jsonl")
        by_fixture = {}
        for row in rows:
            by_fixture.setdefault(row["fixture_id"], {})[row["condition"]] = row
        for group in by_fixture.values():
            self.assertGreater(
                group["PRESERVED_VERBOSE"]["input_token_count"],
                group["PRESERVED_ABBREVIATED"]["input_token_count"],
            )
            self.assertGreater(
                group["PRESERVED_ABBREVIATED"]["input_token_count"],
                group["PRESERVED_SYMBOLIC"]["input_token_count"],
            )
            relational = group["LOCUS_CHANGED_SYMBOLIC"]
            control = group["NONREL_CONTROL_SYMBOLIC"]
            self.assertEqual(
                relational["edit_distance_from_preserved_same_form"],
                control["edit_distance_from_preserved_same_form"],
            )
            self.assertEqual(
                abs(relational["input_token_delta_from_preserved_same_form"]),
                abs(control["input_token_delta_from_preserved_same_form"]),
            )
            self.assertEqual(
                relational["character_delta_from_preserved_same_form"],
                control["character_delta_from_preserved_same_form"],
            )


if __name__ == "__main__":
    unittest.main()
