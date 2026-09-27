#!/usr/bin/env python3
"""Build and verify every pre-inference Phase 0 artifact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer

import fixture_generation
import perturbation_generation
import study

VALIDATION_QUESTION = (
    "Which version requires maintaining more relational state, including actors, roles, "
    "delegation relationships, nested dependencies, or locus-of-action distinctions, "
    "in order to make the same underlying decision?"
)
RESERVE_N = 64


def _tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(study.MODEL_ID, revision=study.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def write_configs(template_hashes: dict[str, str]) -> None:
    code_representation = {
        "identifier": study.CANONICAL_CONDITION,
        "source_study": study.CANONICAL_STUDY,
        "source_sequence": study.CANONICAL_SEQUENCE,
        "source_commit": study.CANONICAL_SOURCE_COMMIT,
        "source_path": study.CANONICAL_SOURCE_PATH,
        "source_sha256": study.CANONICAL_SOURCE_SHA256,
        "local_byte_identical_artifact": str(study.CANONICAL_LOCAL_COPY.relative_to(study.REPO)),
        "local_artifact_sha256": study.file_hash(study.CANONICAL_LOCAL_COPY),
        "factor_labels": list(study.FACTOR_LABELS_CODE_MAPPED),
        "tasks": [family[1] for family in study.FAMILIES],
        "system_text": study.SYSTEM_TEXT,
        "verified_historical_low_prompt_sha256": template_hashes["CODE_MAPPED"],
        "generated_result": "17/32",
    }
    neutral_representation = {
        "identifier": "five-count-neutral",
        "source_study": study.CANONICAL_STUDY,
        "source_sequence": study.CANONICAL_SEQUENCE,
        "source_commit": study.CANONICAL_SOURCE_COMMIT,
        "shared_source_path": study.CANONICAL_SOURCE_PATH,
        "shared_source_sha256": study.CANONICAL_SOURCE_SHA256,
        "factor_labels": list(study.FACTOR_LABELS_NEUTRAL),
        "task": study.NEUTRAL_TASK,
        "system_text": study.SYSTEM_TEXT,
        "verified_historical_low_prompt_sha256": template_hashes["NEUTRAL"],
        "generated_result": "26/32",
    }
    scorer = {
        "type": "deterministic",
        "parser": "exactly one isolated A or B, case-insensitive",
        "correctness": "parsed choice equals prespecified correct code",
        "condition_metadata_visible": False,
        "model_scorer": None,
        "retry_on_substantive_wrong_answer": False,
    }
    inference = {
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "backend": "transformers CPU",
        "dtype": study.MODEL_DTYPE,
        "do_sample": False,
        "max_new_tokens": study.MAX_NEW_TOKENS,
        "max_input_tokens": study.MAX_INPUT_TOKENS,
        "seed": study.SEED,
        "padding_side": "left",
        "first_scorable_token": "first generation step, admissible tokens A and B",
        "decision_margin": "logit(correct/scored decision token) - logit(other admissible token)",
        "technical_retries": study.TECHNICAL_RETRIES,
        "retry_identity": "same five-rendering fixture batch, prompt, model, revision, decoding, and seed",
    }
    rubric = {
        "question": VALIDATION_QUESTION,
        "validator": "deterministic blinded relational-state adjudicator v1",
        "input_fields": ["packet_id", "question", "version_a", "version_b"],
        "forbidden_fields": ["high_low_labels", "framing_labels", "hypotheses", "predicted_outcomes", "prior_performance"],
        "decision_dimensions": ["actors", "directed relations", "nesting", "locus distinctions", "dependencies"],
        "prompt_length_used": False,
        "initial_global_threshold": 0.85,
        "individual_final_threshold": "intended HIGH must be selected",
        "maximum_cycles": 2,
    }
    perturbation = {
        "generator": perturbation_generation.GENERATOR_ID,
        "maximum_attempts": perturbation_generation.MAX_ATTEMPTS,
        "seed": study.PERTURBATION_SEED,
        "source": "exact Neutral×High rendering",
        "transformation": "insert a fixture-seeded generic DOCUMENT MARKER selected from a fixed lexical bank",
        "outcome_visibility": False,
        "matching": "absolute token and character delta versus CodeMapped×High",
        "relative_tolerance": 0.20,
        "small_change_absolute_tolerance": {"tokens": 2, "characters": 20},
    }
    study.atomic_json(study.ROOT / "code_mapped_representation.json", code_representation)
    study.atomic_json(study.ROOT / "neutral_representation.json", neutral_representation)
    study.atomic_json(study.ROOT / "scorer_config.json", scorer)
    (study.ROOT / "scorer_prompt.txt").write_text(
        "NONE: scoring is deterministic and receives no condition metadata or model prompt.\n", encoding="utf-8"
    )
    study.atomic_json(study.ROOT / "inference_config.json", inference)
    study.atomic_json(study.ROOT / "complexity_validation_rubric.json", rubric)
    study.atomic_json(study.ROOT / "perturbation_generation_config.json", perturbation)


def freeze_pools(n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = fixture_generation.build_pool(n, "candidate", 0)
    reserve = fixture_generation.build_pool(RESERVE_N, "reserve", n // 64)
    for fixture in candidate + reserve:
        fixture_generation.validate_fixture_invariants(fixture)
    study.write_jsonl(study.ROOT / "candidate_pool.jsonl", candidate)
    study.write_jsonl(study.ROOT / "reserve_pool.jsonl", reserve)
    freeze = {
        "stage": "written and hashed before first blinded validation",
        "candidate_count": len(candidate),
        "reserve_count": len(reserve),
        "candidate_order_frozen": [fixture["fixture_id"] for fixture in candidate],
        "reserve_order_frozen": [fixture["fixture_id"] for fixture in reserve],
        "candidate_pool_sha256": study.file_hash(study.ROOT / "candidate_pool.jsonl"),
        "reserve_pool_sha256": study.file_hash(study.ROOT / "reserve_pool.jsonl"),
        "candidate_fixture_sha256": {fixture["fixture_id"]: fixture["fixture_sha256"] for fixture in candidate},
        "reserve_fixture_sha256": {fixture["fixture_id"]: fixture["fixture_sha256"] for fixture in reserve},
    }
    study.atomic_json(study.ROOT / "pool_freeze.json", freeze)
    return candidate, reserve


def packet_for(fixture: dict[str, Any], packet_id: str, rng: random.Random) -> tuple[dict[str, str], dict[str, Any]]:
    low = study.messages_for(fixture, "NEUTRAL", "LOW")
    high = study.messages_for(fixture, "NEUTRAL", "HIGH")
    low_text = "\n\n".join(message["content"] for message in low)
    high_text = "\n\n".join(message["content"] for message in high)
    high_side = rng.choice(("A", "B"))
    packet = {
        "packet_id": packet_id,
        "question": VALIDATION_QUESTION,
        "version_a": high_text if high_side == "A" else low_text,
        "version_b": high_text if high_side == "B" else low_text,
    }
    key = {
        "packet_id": packet_id,
        "fixture_id": fixture["fixture_id"],
        "intended_high_side": high_side,
        "packet_sha256": study.digest_json(packet),
    }
    return packet, key


def run_validator(packet_path: Path, output_path: Path) -> list[dict[str, Any]]:
    subprocess.run(
        [sys.executable, str(study.ROOT / "complexity_validator.py"), str(packet_path), str(output_path)],
        check=True,
        cwd=study.REPO,
    )
    return study.read_jsonl(output_path)


def validate_complexity(candidate: list[dict[str, Any]], reserve: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(study.VALIDATION_SEED)
    packets, keys = [], []
    for index, fixture in enumerate(candidate):
        packet, key = packet_for(fixture, f"INITIAL-{index:03d}", rng)
        packets.append(packet)
        keys.append(key)
    packet_path = study.ROOT / "complexity_validation_packets_initial.jsonl"
    output_path = study.ROOT / "complexity_validator_judgments_initial.jsonl"
    study.write_jsonl(packet_path, packets)
    study.write_jsonl(study.ROOT / "complexity_validation_key_initial.jsonl", keys)
    judgments = run_validator(packet_path, output_path)
    by_packet = {row["packet_id"]: row for row in judgments}
    initial_records = []
    failing_indices = []
    for index, key in enumerate(keys):
        judgment = by_packet[key["packet_id"]]
        concordant = judgment["choice"] == key["intended_high_side"]
        if not concordant:
            failing_indices.append(index)
        initial_records.append({**key, "validator_choice": judgment["choice"], "concordant": concordant})
    initial_rate = sum(record["concordant"] for record in initial_records) / len(initial_records)
    if initial_rate < 0.85:
        payload = {
            "status": "STOP_INITIAL_GLOBAL_CONCORDANCE_BELOW_85_PERCENT",
            "initial_concordance": initial_rate,
            "initial_records": initial_records,
            "cycles_used": 1,
        }
        study.atomic_json(study.ROOT / "complexity_validation.json", payload)
        raise RuntimeError("STOP: blinded initial complexity validation did not reach 85%")

    final = list(candidate)
    replacement_records = []
    if failing_indices:
        if len(failing_indices) > len(reserve):
            raise RuntimeError("STOP: frozen reserve is too small for the single replacement cycle")
        replacement_packets, replacement_keys = [], []
        for sequence, candidate_index in enumerate(failing_indices):
            replacement = reserve[sequence]
            packet, key = packet_for(replacement, f"REPLACEMENT-{sequence:03d}", rng)
            key["replaces_fixture_id"] = candidate[candidate_index]["fixture_id"]
            key["candidate_index"] = candidate_index
            replacement_packets.append(packet)
            replacement_keys.append(key)
        replacement_packet_path = study.ROOT / "complexity_validation_packets_replacement.jsonl"
        replacement_output_path = study.ROOT / "complexity_validator_judgments_replacement.jsonl"
        study.write_jsonl(replacement_packet_path, replacement_packets)
        study.write_jsonl(study.ROOT / "complexity_validation_key_replacement.jsonl", replacement_keys)
        replacement_judgments = run_validator(replacement_packet_path, replacement_output_path)
        replacement_index = {row["packet_id"]: row for row in replacement_judgments}
        for key in replacement_keys:
            judgment = replacement_index[key["packet_id"]]
            concordant = judgment["choice"] == key["intended_high_side"]
            replacement_records.append({**key, "validator_choice": judgment["choice"], "concordant": concordant})
            if not concordant:
                payload = {
                    "status": "STOP_REPLACEMENT_FAILED_SINGLE_ALLOWED_REVALIDATION",
                    "initial_concordance": initial_rate,
                    "initial_records": initial_records,
                    "replacement_records": replacement_records,
                    "cycles_used": 2,
                }
                study.atomic_json(study.ROOT / "complexity_validation.json", payload)
                raise RuntimeError("STOP: a replacement failed the only allowed revalidation")
            final[key["candidate_index"]] = next(row for row in reserve if row["fixture_id"] == key["fixture_id"])

    payload = {
        "status": "PASSED",
        "validator": "deterministic blinded relational-state adjudicator v1",
        "question": VALIDATION_QUESTION,
        "initial_n": len(candidate),
        "initial_concordant": sum(record["concordant"] for record in initial_records),
        "initial_concordance": initial_rate,
        "threshold": 0.85,
        "initial_global_threshold_passed": True,
        "initial_records": initial_records,
        "replacement_records": replacement_records,
        "cycles_used": 2 if failing_indices else 1,
        "maximum_cycles": 2,
        "final_n": len(final),
        "every_final_fixture_validated": True,
        "final_fixture_ids": [fixture["fixture_id"] for fixture in final],
    }
    study.atomic_json(study.ROOT / "complexity_validation.json", payload)
    lines = [
        "# Phase 0B complexity validation",
        "",
        "The complete candidate and reserve pools were ordered and SHA-256 hashed before the first validation pass. The validator received only randomized A/B versions and the frozen question; it received no HIGH/LOW labels, framing labels, hypothesis, expected direction, prior performance, or target-model output.",
        "",
        f"Initial concordance: **{payload['initial_concordant']}/{payload['initial_n']} ({initial_rate:.1%})**. Frozen threshold: ≥85%.",
        f"Validation cycles used: {payload['cycles_used']} of 2 maximum.",
        f"Replacements from the frozen reserve: {len(replacement_records)}.",
        "Every fixture retained in the final confirmatory set has a validated intended HIGH > LOW ordering.",
        "",
        "The adjudicator compares relational actors, directed relations, nesting, locus distinctions, and dependencies lexicographically. Prompt length is unavailable to its decision rule. These judgments are design-validation data and contain no scored target-model behavior.",
        "",
    ]
    (study.ROOT / "complexity_validation.md").write_text("\n".join(lines), encoding="utf-8")
    return final, payload


def _load_ladder_protocol():
    path = study.REPO / "experiments/res_complexity_ladder/protocol.py"
    specification = importlib.util.spec_from_file_location("historical_res_ladder_protocol", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader
    specification.loader.exec_module(module)
    return module


def summarize_historical_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list(study.STRUCTURAL_METRICS) + ["character_count", "input_token_count"]
    summary = {"n": len(rows)}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def historical_comparison(tokenizer: Any) -> dict[str, Any]:
    low_metrics = study.structural_metrics("LOW")
    original = {}
    for label, path in (("v1_code_mapped", study.CANONICAL_RESULT_PATH), ("v1_neutral", study.CANONICAL_NEUTRAL_RESULT_PATH)):
        data = json.loads(path.read_text(encoding="utf-8"))["fixtures"]
        original[label] = [
            {
                **low_metrics,
                "character_count": len(row["rendered_prompt"]),
                "input_token_count": len(tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)),
            }
            for row in data
        ]
    ladder = _load_ladder_protocol()
    replication = {}
    for cell, label in (("LL", "replication_loaded"), ("NN", "replication_neutral")):
        selected = [row for row in ladder.rows_for(5) if row["cell"] == cell]
        records = []
        for row in selected:
            prompt = study.chat_prompt(tokenizer, row["messages"])
            records.append(
                {
                    **low_metrics,
                    "character_count": len(prompt),
                    "input_token_count": len(tokenizer.encode(prompt, add_special_tokens=False)),
                }
            )
        replication[label] = records
    summary = {key: summarize_historical_rows(value) for key, value in {**original, **replication}.items()}
    payload = {
        "status": "RETROSPECTIVE_NON_CONFIRMATORY_MOTIVATION_ONLY",
        "prohibited_uses": [
            "redefine LOW/HIGH",
            "change fixture-generation rules",
            "alter the hypothesis",
            "select current fixtures",
            "determine exclusions",
            "count as confirmatory evidence",
        ],
        "metric_definitions": {
            "task_relevant_actor_entity_count": "distinct task actors whose role or record must be distinguished",
            "relational_binding_count": "actor-role, actor-action, record-source, and handoff bindings",
            "directed_role_delegation_count": "directed delegation or record-relay edges",
            "maximum_relational_nesting_depth": "longest actor-to-actor dependency path",
            "locus_of_action_transition_count": "required switches among action/request/record loci",
            "conjunction_dependency_count": "independent relational dependencies that must jointly hold",
            "character_count": "full chat-template prompt characters",
            "input_token_count": "pinned Qwen tokenizer tokens",
        },
        "summary": summary,
        "source_hashes": {
            str(study.CANONICAL_RESULT_PATH.relative_to(study.REPO)): study.file_hash(study.CANONICAL_RESULT_PATH),
            str(study.CANONICAL_NEUTRAL_RESULT_PATH.relative_to(study.REPO)): study.file_hash(study.CANONICAL_NEUTRAL_RESULT_PATH),
            "experiments/res_complexity_ladder/protocol.py": study.file_hash(study.REPO / "experiments/res_complexity_ladder/protocol.py"),
            "experiments/res_complexity_ladder/manifest.json": study.file_hash(study.REPO / "experiments/res_complexity_ladder/manifest.json"),
        },
    }
    study.atomic_json(study.ROOT / "historical_structural_comparison.json", payload)
    lines = [
        "# Historical structural comparison",
        "",
        "**RETROSPECTIVE AND NON-CONFIRMATORY — MOTIVATION ONLY.**",
        "",
        "The original v1 fixture set and the later independent replication were characterized with the same frozen metrics used for the present LOW/HIGH validation. This comparison cannot redefine complexity, modify fixtures, alter hypotheses, select exclusions, or count as confirmatory evidence.",
        "",
        "| Historical set | n | actors | bindings | directed relations | nesting | locus transitions | dependencies | mean characters | mean tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in summary.items():
        lines.append(
            f"| {label} | {values['n']} | {values['task_relevant_actor_entity_count']['mean']:.1f} | "
            f"{values['relational_binding_count']['mean']:.1f} | {values['directed_role_delegation_count']['mean']:.1f} | "
            f"{values['maximum_relational_nesting_depth']['mean']:.1f} | {values['locus_of_action_transition_count']['mean']:.1f} | "
            f"{values['conjunction_dependency_count']['mean']:.1f} | {values['character_count']['mean']:.1f} | "
            f"{values['input_token_count']['mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Both historical protocols use the same two-record actor topology at this metric resolution. Their principal structural differences are therefore not evidence for the prospective interaction hypothesis. Any length differences remain descriptive historical characteristics.",
            "",
        ]
    )
    (study.ROOT / "historical_structural_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def render_final(fixtures: list[dict[str, Any]], tokenizer: Any, validation: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_key = {record["fixture_id"]: record for record in validation["initial_records"] if record["concordant"]}
    replacement_key = {record["fixture_id"]: record for record in validation["replacement_records"]}
    final_fixtures, rows = [], []
    condition_base = list(study.CONDITIONS)
    position_counts = {condition: Counter() for condition in study.CONDITIONS}
    for fixture_index, raw_fixture in enumerate(fixtures):
        fixture = dict(raw_fixture)
        fixture_generation.validate_fixture_invariants(fixture)
        neutral_low = study.messages_for(fixture, "NEUTRAL", "LOW")
        code_low = study.messages_for(fixture, "CODE_MAPPED", "LOW")
        neutral_high = study.messages_for(fixture, "NEUTRAL", "HIGH")
        code_high = study.messages_for(fixture, "CODE_MAPPED", "HIGH")
        control, control_match = perturbation_generation.make_nonrelational_control(
            fixture, tokenizer, neutral_high, code_high
        )
        messages_by_condition = {
            "NEUTRAL_LOW": neutral_low,
            "CODE_MAPPED_LOW": code_low,
            "NEUTRAL_HIGH": neutral_high,
            "CODE_MAPPED_HIGH": code_high,
            "NONREL_CONTROL_HIGH": control,
        }
        neutral_prompts = {
            "LOW": study.chat_prompt(tokenizer, neutral_low),
            "HIGH": study.chat_prompt(tokenizer, neutral_high),
        }
        base = list(condition_base)
        block = fixture_index - fixture_index % 5
        random.Random(f"{study.SEED}:condition-latin:{block}").shuffle(base)
        offset = fixture_index % 5
        order = base[offset:] + base[:offset]
        for condition_position, condition in enumerate(order):
            messages = messages_by_condition[condition]
            prompt = study.chat_prompt(tokenizer, messages)
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 0 < input_tokens <= study.MAX_INPUT_TOKENS:
                raise RuntimeError(f"STOP: {fixture['fixture_id']} {condition} exceeds input limit")
            complexity = "LOW" if condition.endswith("LOW") else "HIGH"
            framing = (
                "NEUTRAL" if condition.startswith("NEUTRAL") else
                "CODE_MAPPED" if condition.startswith("CODE_MAPPED") else
                "NON_RELATIONAL_CONTROL"
            )
            baseline = neutral_prompts[complexity]
            metric = study.structural_metrics(complexity)
            record = {
                "observation_id": f"{fixture['fixture_id']}:{condition}",
                "fixture_id": fixture["fixture_id"],
                "condition": condition,
                "framing": framing,
                "complexity": complexity,
                "expected": fixture["correct_code"],
                "messages": messages,
                "messages_sha256": study.digest_json(messages),
                "chat_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "character_count": len(prompt),
                "input_token_count": input_tokens,
                "output_token_count": None,
                "edit_distance_from_neutral_baseline": Levenshtein.distance(baseline, prompt),
                **metric,
                "condition_position": condition_position,
                "evaluation_order": fixture_index * 5 + condition_position,
                "shard_index": fixture_index // study.SHARD_FIXTURES,
                "h2b_control_valid": control_match["valid"],
                "control_matching": control_match if condition == "NONREL_CONTROL_HIGH" else None,
            }
            record["rendering_hash"] = study.digest_json(record)
            rows.append(record)
            position_counts[condition][condition_position] += 1
        validation_record = initial_key.get(fixture["fixture_id"]) or replacement_key.get(fixture["fixture_id"])
        fixture["validation_record"] = validation_record
        fixture["h2b_control_valid"] = control_match["valid"]
        fixture["control_matching"] = control_match
        fixture["final_order"] = fixture_index
        fixture["shard_index"] = fixture_index // study.SHARD_FIXTURES
        fixture["canonical_fixture_hash"] = study.digest_json(
            {key: value for key, value in fixture.items() if key not in {"validation_record", "control_matching", "fixture_sha256", "canonical_fixture_hash"}}
        )
        final_fixtures.append(fixture)
    if any(max(counts.values()) - min(counts.values()) > 1 for counts in position_counts.values()):
        raise AssertionError("Condition positions are not balanced to within one")
    return final_fixtures, rows


def write_preregistration(n: int, power: dict[str, Any], validation: dict[str, Any], controls_valid: int) -> None:
    mde = power["mde_results"][str(n)]["mde_grid_estimate"]
    text = f"""# {study.STUDY_NAME} — preregistration

Frozen before confirmatory target-model inference. This is the next confirmatory continuation of the existing RES Study 1 sequence. It tests behavioral sensitivity to representations of locus of action and relational structure. It does not test consciousness, subjective experience, self-awareness, or literal human-like selfhood.

## Canonical prior representation

The canonical CODE-MAPPED manipulation is **{study.CANONICAL_STUDY}**, sequence `{study.CANONICAL_SEQUENCE}`, condition `{study.CANONICAL_CONDITION}`: generated accuracy 17/32 versus 26/32 for `five-count-neutral`.

- source commit: `{study.CANONICAL_SOURCE_COMMIT}`
- source artifact: `{study.CANONICAL_SOURCE_PATH}`
- source SHA-256: `{study.CANONICAL_SOURCE_SHA256}`
- current byte-identical artifact: `{study.CANONICAL_LOCAL_COPY.relative_to(study.REPO)}`
- exact template identifier: `{study.CANONICAL_CONDITION}`

The current artifact hash is required to equal the source hash at every preflight. The later LL/NN “loaded” wording and its approximately 35/64 versus 38/64 result are separate and are not merged into this framing manipulation.

## Role of prior data

The hypothesis is prospective and is not a post-hoc explanation for the later replication non-result. Earlier RES results are used only for Phase 0 power/covariance estimation and the explicitly retrospective, non-confirmatory structural comparison in `historical_structural_comparison.md`. That comparison cannot redefine complexity, change fixtures or hypotheses, select fixtures or exclusions, or count as confirmatory evidence.

## Hypotheses

For each complexity level, FramingEffect = P(Neutral pass) − P(CodeMapped pass). The primary probability interaction is FramingEffect_HIGH − FramingEffect_LOW. The directional expectation is FramingEffect_LOW ≥ 0, FramingEffect_HIGH > 0, and a **positive** interaction.

H2a predicts that HIGH-complexity Neutral→CodeMapped transition fixtures have smaller `|margin(Neutral_HIGH)|` than framing-stable fixtures. The frozen test is a one-sided Mann–Whitney U with transition margins predicted smaller; rank-biserial effect and a bootstrap interval are reported.

H2b predicts more transitions for Neutral_HIGH→CodeMapped_HIGH than for Neutral_HIGH→NonRelationalPerturbation_HIGH. Both use the identical Neutral_HIGH observation and margin. The frozen test is a one-sided paired exact McNemar test. Descriptive absolute-margin strata are `≤0.75`, `(0.75,1.50]`, `(1.50,3.00]`, and `>3.00`.

## Sample size and power

Scientific N is **{n} fixtures**. Alpha is .05 two-sided, target power .80, and probability-scale interaction SESOI is .15. The power simulation used earlier RES outcomes only, explicitly modeled within-fixture correlation, framing and complexity discordance, and empirical all-pass/all-fail prevalence. Required approximately-zero aggregate framing scenarios were included. The selected-N MDE grid estimate is {mde}. Full assumptions, inputs, seeds, achieved power, Monte Carlo intervals, and diagnostics are in `power_analysis.md` and `power_analysis.json`.

## Matched design and frozen complexity

Every fixture has Neutral×LOW, CodeMapped×LOW, Neutral×HIGH, CodeMapped×HIGH, and a NonRelationalPerturbation×HIGH control. The four primary versions preserve the same five factor values, response mapping, actor records, task decision, and correct code. LOW has CURRENT EXECUTOR and PEER. HIGH preserves that decision while adding REQUESTER→COORDINATOR→CURRENT EXECUTOR delegation and PEER→OBSERVER→REQUESTER record relay.

Frozen metrics are task-relevant actor/entity count, relational-binding count, directed role/delegation count, maximum relational nesting depth, locus-of-action transition count, conjunction/dependency count, characters, and pinned-tokenizer input tokens. Their operational definitions are in `historical_structural_comparison.json`. Longer text alone cannot validate HIGH.

The candidate pool ({n}) and ordered reserve pool ({RESERVE_N}) were written and hashed before blinded validation. A/B order was randomized. The validator saw only the frozen question and A/B texts. Initial global concordance was {validation['initial_concordance']:.1%} against a frozen ≥85% threshold. It used {validation['cycles_used']} of two allowed cycles, and every final fixture has validated HIGH > LOW ordering. No scored target-model output was used.

## Non-relational perturbation

All controls were generated before inference from the exact Neutral_HIGH rendering by a deterministic fixture-seeded document-marker algorithm with at most two attempts. It preserves actor identity, bindings, delegation, locus semantics, relational complexity, and the answer, while avoiding the v1 factor/task wording. Both absolute token-delta and character-delta magnitudes must match the CodeMapped_HIGH change within ±20%; small changes use absolute tolerances of ±2 tokens and ±20 characters. {controls_valid}/{n} controls are valid. Invalid controls remain in the primary study and are excluded only from H2b. No regeneration is allowed after outputs.

## Scoring, margin, and technical failures

Scoring is deterministic and blinded to condition metadata: exactly one isolated A or B is parsed, then compared with the prespecified correct code. The inference configuration is pinned in `inference_config.json`. At the first generation step, A and B are the finite admissible decision-token set. The signed margin is `logit(correct/scored token) − logit(other admissible token)`; this is the already-frozen mathematically equivalent RES decision-margin implementation. H2a/H2b use `|margin(Neutral_HIGH)|`.

Technical invalidity is limited to invocation failure, truncation preventing scoring, corrupted/missing data, parser failure after frozen retries, scorer failure after retries, or wrong frozen configuration. Each observation receives one initial call plus at most two identical retries for technical failure only. A valid wrong, surprising, ambiguous, ceiling/floor, large-margin, or small-margin result is never substantively excluded or retried. If a primary cell remains technically missing, available raw observations remain and that fixture is excluded only from analyses requiring all four cells. The fifth-cell failure affects only H2b.

## Primary analysis and fallback

The primary model is fixture-stratified conditional logistic regression: `pass ~ neutral + high + neutral×high + strata(fixture_id)`. Coding makes the predicted interaction positive. The two-sided α=.05 interaction coefficient is primary. GEE with binary outcome, logit link, fixture clustering, and exchangeable working correlation is used only when conditional logistic regression cannot produce a finite converged interaction estimate and interval. Model choice never depends on significance.

The transparent probability difference-in-differences and 95% fixture-bootstrap interval are always reported. Precommitted categories are: CONFIRMATORY SUPPORT for a positive estimate whose 95% CI excludes zero; PRACTICALLY NEGLIGIBLE only when the full probability CI lies strictly inside (−.15,+.15); PREDICTED-SIZE POSITIVE RULED OUT, CONTRARY EFFECT STILL POSSIBLE when upper<+.15 and lower≤−.15; INCONCLUSIVE when zero and +.15 remain included; and CONTRARY-DIRECTION when the CI excludes zero negatively.

## Runtime, sharding, and no-feedback boundary

There are {n // study.SHARD_FIXTURES} deterministic shards of {study.SHARD_FIXTURES} fixtures (40 evaluations). Each fixture checkpoints atomically. A shard approaching the approximately 4.5-hour workflow ceiling finishes its current fixture, flushes, and exits. Resume skips an observation only after rendering hash, model/revision, inference configuration, and scorer configuration all match the freeze; any mismatch stops execution.

After the Phase 0 commit, scored outputs cannot change fixtures, complexity, representation, perturbations, scorer, margin, N, models, alpha, SESOI, strata, exclusions, retries, or hypotheses. All-pass and all-fail fixtures remain in descriptive data even when they provide zero conditional-likelihood information.

## Interpretation boundary

The study can test whether sensitivity to the frozen v1 manipulation changes with relational complexity, whether transitions concentrate near the decision boundary, and whether relational transitions exceed matched generic surface perturbation. It cannot establish the internal mechanism or consciousness, subjective experience, self-awareness, or human-equivalent self-representation. Optimized-label search remains a later held-out exploratory phase.
"""
    (study.ROOT / "preregistration.md").write_text(text, encoding="utf-8")


def frozen_files() -> list[str]:
    root = "experiments/res_confirmatory_study1_continuation/"
    names = [
        "canonical/v1_code_mapped_source.py",
        "study.py",
        "fixture_generation.py",
        "perturbation_generation.py",
        "conditional_logit.py",
        "power_analysis.py",
        "complexity_validator.py",
        "analysis_common.py",
        "analysis_primary.py",
        "analysis_secondary.py",
        "analysis_controls.py",
        "analyze.py",
        "run.py",
        "phase0.py",
        "test_study.py",
        "README.md",
        "preregistration.md",
        "power_analysis.md",
        "power_analysis.json",
        "historical_structural_comparison.md",
        "historical_structural_comparison.json",
        "complexity_validation.md",
        "complexity_validation.json",
        "complexity_validation_rubric.json",
        "complexity_validation_packets_initial.jsonl",
        "complexity_validation_key_initial.jsonl",
        "complexity_validator_judgments_initial.jsonl",
        "candidate_pool.jsonl",
        "reserve_pool.jsonl",
        "pool_freeze.json",
        "fixtures.jsonl",
        "rendered_prompts.jsonl",
        "code_mapped_representation.json",
        "neutral_representation.json",
        "scorer_config.json",
        "scorer_prompt.txt",
        "inference_config.json",
        "perturbation_generation_config.json",
        "frozen_manifest.json",
    ]
    optional = [
        "complexity_validation_packets_replacement.jsonl",
        "complexity_validation_key_replacement.jsonl",
        "complexity_validator_judgments_replacement.jsonl",
    ]
    files = [root + name for name in names]
    files.extend(root + name for name in optional if (study.ROOT / name).exists())
    files.append(".github/workflows/res-confirmatory-study1-continuation.yml")
    return sorted(files)


def make_manifest(fixtures: list[dict[str, Any]], rows: list[dict[str, Any]], power: dict[str, Any]) -> dict[str, Any]:
    parent_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.REPO, text=True).strip()
    return {
        "protocol": study.VERSION,
        "state": "PHASE0_FROZEN_PRE_INFERENCE",
        "phase0_parent_commit_sha": parent_commit,
        "freeze_commit_note": "The Git commit containing this manifest is the non-self-referential freeze identifier.",
        "source_commit_sha": study.CANONICAL_SOURCE_COMMIT,
        "canonical_v1_source_study": study.CANONICAL_STUDY,
        "canonical_v1_template_identifier": study.CANONICAL_CONDITION,
        "canonical_v1_source_path": study.CANONICAL_SOURCE_PATH,
        "canonical_v1_source_hash": study.CANONICAL_SOURCE_SHA256,
        "current_code_mapped_artifact_path": str(study.CANONICAL_LOCAL_COPY.relative_to(study.REPO)),
        "current_code_mapped_artifact_hash": study.file_hash(study.CANONICAL_LOCAL_COPY),
        "code_mapped_representation_hash": study.file_hash(study.ROOT / "code_mapped_representation.json"),
        "neutral_artifact_hash": study.file_hash(study.ROOT / "neutral_representation.json"),
        "scorer_hash": study.file_hash(study.ROOT / "scorer_config.json"),
        "scorer_prompt_hash": study.file_hash(study.ROOT / "scorer_prompt.txt"),
        "complexity_validation_rubric_hash": study.file_hash(study.ROOT / "complexity_validation_rubric.json"),
        "inference_config_hash": study.file_hash(study.ROOT / "inference_config.json"),
        "preregistration_hash": study.file_hash(study.ROOT / "preregistration.md"),
        "power_analysis_hashes": {
            "script": study.file_hash(study.ROOT / "power_analysis.py"),
            "json": study.file_hash(study.ROOT / "power_analysis.json"),
            "markdown": study.file_hash(study.ROOT / "power_analysis.md"),
        },
        "analysis_code_hashes": {
            "conditional_logistic": study.file_hash(study.ROOT / "conditional_logit.py"),
            "primary": study.file_hash(study.ROOT / "analysis_primary.py"),
            "secondary": study.file_hash(study.ROOT / "analysis_secondary.py"),
            "controls": study.file_hash(study.ROOT / "analysis_controls.py"),
            "orchestrator": study.file_hash(study.ROOT / "analyze.py"),
        },
        "fixture_generation_code_hash": study.file_hash(study.ROOT / "fixture_generation.py"),
        "fixture_generation_implementation_hash": study.file_hash(study.ROOT / "study.py"),
        "perturbation_generation_code_hash": study.file_hash(study.ROOT / "perturbation_generation.py"),
        "perturbation_generation_implementation_hash": study.file_hash(study.ROOT / "study.py"),
        "runner_hash": study.file_hash(study.ROOT / "run.py"),
        "model_identifier": study.MODEL_ID,
        "model_revision": study.MODEL_REVISION,
        "inference_parameters": json.loads((study.ROOT / "inference_config.json").read_text(encoding="utf-8")),
        "random_seeds": {
            "study": study.SEED,
            "pool": study.POOL_SEED,
            "validation": study.VALIDATION_SEED,
            "perturbation": study.PERTURBATION_SEED,
            "bootstrap": study.BOOTSTRAP_SEED,
            **power["seeds"],
        },
        "scientific_n": len(fixtures),
        "primary_observations": len(fixtures) * 4,
        "specificity_observations": len(fixtures),
        "fixture_hashes": {fixture["fixture_id"]: fixture["canonical_fixture_hash"] for fixture in fixtures},
        "rendering_hashes": {row["observation_id"]: row["rendering_hash"] for row in rows},
        "control_rendering_hashes": {
            row["observation_id"]: row["rendering_hash"] for row in rows if row["condition"] == "NONREL_CONTROL_HIGH"
        },
        "fixture_file_hash": study.file_hash(study.ROOT / "fixtures.jsonl"),
        "rendering_file_hash": study.file_hash(study.ROOT / "rendered_prompts.jsonl"),
        "candidate_pool_hash": study.file_hash(study.ROOT / "candidate_pool.jsonl"),
        "reserve_pool_hash": study.file_hash(study.ROOT / "reserve_pool.jsonl"),
    }


def build() -> None:
    if (study.ROOT / "frozen_manifest.json").exists() or (study.ROOT / "results").exists():
        raise RuntimeError("Refusing to replace an existing freeze or scored-results directory")
    study.verify_canonical_copy()
    power = json.loads((study.ROOT / "power_analysis.json").read_text(encoding="utf-8"))
    n = int(power["selected_n"])
    tokenizer = _tokenizer()
    template_hashes = study.verify_low_templates_against_history(tokenizer)
    write_configs(template_hashes)
    candidate, reserve = freeze_pools(n)
    final, validation = validate_complexity(candidate, reserve)
    historical_comparison(tokenizer)
    fixtures, rows = render_final(final, tokenizer, validation)
    controls_valid = sum(fixture["h2b_control_valid"] for fixture in fixtures)
    study.write_jsonl(study.ROOT / "fixtures.jsonl", fixtures)
    study.write_jsonl(study.ROOT / "rendered_prompts.jsonl", rows)
    write_preregistration(n, power, validation, controls_valid)
    manifest = make_manifest(fixtures, rows, power)
    if manifest["current_code_mapped_artifact_hash"] != manifest["canonical_v1_source_hash"]:
        raise RuntimeError("STOP: current code-mapped artifact hash differs from canonical v1")
    study.atomic_json(study.ROOT / "frozen_manifest.json", manifest)
    lines = []
    for relative in frozen_files():
        path = study.REPO / relative
        if not path.exists():
            raise FileNotFoundError(f"Required frozen file missing: {relative}")
        lines.append(f"{study.file_hash(path)}  {relative}")
    (study.ROOT / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verify()
    print(
        json.dumps(
            {
                "status": "PHASE0_FROZEN_PRE_INFERENCE",
                "n": n,
                "renderings": len(rows),
                "valid_h2b_controls": controls_valid,
                "initial_validation_concordance": validation["initial_concordance"],
            },
            sort_keys=True,
        )
    )


def verify() -> None:
    study.verify_canonical_copy()
    manifest = json.loads((study.ROOT / "frozen_manifest.json").read_text(encoding="utf-8"))
    if manifest["current_code_mapped_artifact_hash"] != study.CANONICAL_SOURCE_SHA256:
        raise ValueError("Canonical/current artifact hash mismatch")
    for line in (study.ROOT / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if study.file_hash(study.REPO / relative) != expected:
            raise ValueError(f"Frozen hash mismatch: {relative}")
    fixtures = study.read_jsonl(study.ROOT / "fixtures.jsonl")
    rows = study.read_jsonl(study.ROOT / "rendered_prompts.jsonl")
    if len(fixtures) != manifest["scientific_n"] or len(rows) != 5 * len(fixtures):
        raise ValueError("Frozen sample count mismatch")
    if {fixture["fixture_id"]: fixture["canonical_fixture_hash"] for fixture in fixtures} != manifest["fixture_hashes"]:
        raise ValueError("Fixture hashes differ from manifest")
    if {row["observation_id"]: row["rendering_hash"] for row in rows} != manifest["rendering_hashes"]:
        raise ValueError("Rendering hashes differ from manifest")
    validation = json.loads((study.ROOT / "complexity_validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASSED" or validation["initial_concordance"] < 0.85 or not validation["every_final_fixture_validated"]:
        raise ValueError("Complexity validation freeze is invalid")
    if any(fixture["control_matching"]["attempts_used"] > 2 for fixture in fixtures):
        raise ValueError("Perturbation attempt limit exceeded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    command = parser.parse_args().command
    build() if command == "build" else verify()
    if command == "verify":
        print(json.dumps({"status": "PHASE0_HASHES_VERIFIED"}))


if __name__ == "__main__":
    main()

