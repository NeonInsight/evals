#!/usr/bin/env python3
"""Build and verify the pre-inference freeze for the surface-efficiency assay."""
from __future__ import annotations

import argparse
import hashlib
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
import study


def _tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(study.MODEL_ID, revision=study.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def prior_profile_hashes() -> set[str]:
    path = study.REPO / "experiments/res_confirmatory_study1_continuation/fixtures.jsonl"
    hashes = set()
    for row in study.read_jsonl(path):
        profile = {
            "task": row["loaded_task"],
            "persona": row["persona"],
            "allow_code": row["allow_code"],
            "permitted": row["permitted"],
            "difficulty": row["difficulty"],
            "executor_bits": row["executor_bits"],
            "reference_bits": row["peer_bits"],
            "factor_order": row["factor_order"],
        }
        hashes.add(study.digest_json(profile))
    return hashes


def freeze_pools() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = fixture_generation.build_pool(study.SCIENTIFIC_N, "candidate", 0)
    reserve = fixture_generation.build_pool(study.RESERVE_N, "reserve", study.SCIENTIFIC_N // 32)
    for fixture in candidate + reserve:
        fixture_generation.validate_fixture_invariants(fixture)
    hashes = [fixture["semantic_profile_sha256"] for fixture in candidate + reserve]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Candidate and reserve pools contain duplicate semantic profiles")
    overlap = set(hashes) & prior_profile_hashes()
    if overlap:
        raise RuntimeError("Fresh held-out pool overlaps a previously scored semantic profile")
    study.write_jsonl(study.ROOT / "candidate_pool.jsonl", candidate)
    study.write_jsonl(study.ROOT / "reserve_pool.jsonl", reserve)
    payload = {
        "status": "FROZEN_BEFORE_REPRESENTATION_VALIDATION",
        "candidate_count": len(candidate),
        "reserve_count": len(reserve),
        "candidate_order": [row["fixture_id"] for row in candidate],
        "reserve_order": [row["fixture_id"] for row in reserve],
        "candidate_sha256": study.file_hash(study.ROOT / "candidate_pool.jsonl"),
        "reserve_sha256": study.file_hash(study.ROOT / "reserve_pool.jsonl"),
        "prior_scored_profile_overlap": 0,
        "replacement_rule": "No replacement after target-model inference; Phase 0 stops if any candidate representation fails.",
    }
    study.atomic_json(study.ROOT / "pool_freeze.json", payload)
    return candidate, reserve


def _condition_order(fixture_index: int) -> list[str]:
    block = fixture_index - fixture_index % len(study.CONDITIONS)
    base = list(study.CONDITIONS)
    random.Random(f"{study.SEED}:latin:{block}").shuffle(base)
    offset = fixture_index % len(base)
    return base[offset:] + base[:offset]


def render_fixtures(
    fixtures: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frozen_fixtures, all_rows = [], []
    position_counts = {condition: Counter() for condition in study.CONDITIONS}
    for fixture_index, raw in enumerate(fixtures):
        fixture = dict(raw)
        messages = {condition: study.messages_for(fixture, condition) for condition in study.CONDITIONS}
        prompts = {condition: study.chat_prompt(tokenizer, value) for condition, value in messages.items()}
        token_counts = {
            condition: len(tokenizer.encode(prompt, add_special_tokens=False))
            for condition, prompt in prompts.items()
        }
        baseline_verbose = prompts["PRESERVED_VERBOSE"]
        preserved_same_form = {
            "VERBOSE": "PRESERVED_VERBOSE",
            "ABBREVIATED": "PRESERVED_ABBREVIATED",
            "SYMBOLIC": "PRESERVED_SYMBOLIC",
        }
        order = _condition_order(fixture_index)
        for condition_position, condition in enumerate(order):
            specification = study.condition_spec(condition)
            prompt = prompts[condition]
            input_tokens = token_counts[condition]
            if not 0 < input_tokens <= study.MAX_INPUT_TOKENS:
                raise RuntimeError(f"{fixture['fixture_id']} {condition} exceeds the input-token ceiling")
            same_form_condition = preserved_same_form[specification["surface_form"]]
            same_form_prompt = prompts[same_form_condition]
            same_form_tokens = token_counts[same_form_condition]
            expected = study.expected_for(fixture, condition)
            row = {
                "observation_id": f"{fixture['fixture_id']}:{condition}",
                "fixture_id": fixture["fixture_id"],
                "condition": condition,
                **specification,
                "graph_signature": study.graph_signature(condition),
                "graph_edit_distance_from_preserved": 1 if specification["graph_state"] == "LOCUS_CHANGED" else 0,
                "expected": expected,
                "messages": messages[condition],
                "messages_sha256": study.digest_json(messages[condition]),
                "chat_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "character_count": len(prompt),
                "input_token_count": input_tokens,
                "output_token_count": None,
                "character_savings_from_preserved_verbose": len(baseline_verbose) - len(prompt),
                "input_token_savings_from_preserved_verbose": token_counts["PRESERVED_VERBOSE"] - input_tokens,
                "levenshtein_from_preserved_verbose": Levenshtein.distance(baseline_verbose, prompt),
                "edit_distance_from_preserved_same_form": Levenshtein.distance(same_form_prompt, prompt),
                "input_token_delta_from_preserved_same_form": input_tokens - same_form_tokens,
                "character_delta_from_preserved_same_form": len(prompt) - len(same_form_prompt),
                "condition_position": condition_position,
                "evaluation_order": fixture_index * len(study.CONDITIONS) + condition_position,
                "shard_index": fixture_index // study.SHARD_FIXTURES,
            }
            row["rendering_hash"] = study.digest_json(row)
            all_rows.append(row)
            position_counts[condition][condition_position] += 1
        fixture["final_order"] = fixture_index
        fixture["shard_index"] = fixture_index // study.SHARD_FIXTURES
        fixture["canonical_fixture_hash"] = study.digest_json(fixture)
        frozen_fixtures.append(fixture)
    if any(max(counts.values()) - min(counts.values()) > 1 for counts in position_counts.values()):
        raise AssertionError("Condition positions are not balanced to within one")
    return frozen_fixtures, all_rows


def validate_representations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    packets, keys = [], []
    shuffled = list(rows)
    random.Random(study.VALIDATION_SEED).shuffle(shuffled)
    for index, row in enumerate(shuffled):
        packet_id = f"RVP-{index:04d}"
        packets.append({"packet_id": packet_id, "text": row["messages"][1]["content"]})
        keys.append(
            {
                "packet_id": packet_id,
                "observation_id": row["observation_id"],
                "expected_surface_form": row["surface_form"],
                "expected_graph_signature": row["graph_signature"],
                "expected_marker": "Y" if row["nonrelational_marker_changed"] else "X",
            }
        )
    packet_path = study.ROOT / "representation_validation_packets.jsonl"
    key_path = study.ROOT / "representation_validation_key.jsonl"
    judgment_path = study.ROOT / "representation_validator_judgments.jsonl"
    study.write_jsonl(packet_path, packets)
    study.write_jsonl(key_path, keys)
    subprocess.run(
        [sys.executable, str(study.ROOT / "representation_validator.py"), str(packet_path), str(judgment_path)],
        check=True,
        cwd=study.REPO,
    )
    judgments = {row["packet_id"]: row for row in study.read_jsonl(judgment_path)}
    records = []
    for key in keys:
        judgment = judgments[key["packet_id"]]
        concordant = (
            judgment["valid"]
            and judgment["surface_form"] == key["expected_surface_form"]
            and judgment["graph_signature"] == key["expected_graph_signature"]
            and judgment["marker"] == key["expected_marker"]
        )
        records.append({**key, "judgment": judgment, "concordant": concordant})
    concordant_n = sum(record["concordant"] for record in records)
    payload = {
        "status": "PASSED" if concordant_n == len(records) else "STOP_REPRESENTATION_VALIDATION_FAILED",
        "validator": "condition-blind deterministic relation-graph parser v1",
        "packet_count": len(records),
        "concordant": concordant_n,
        "concordance": concordant_n / len(records),
        "target_model_outputs_visible": False,
        "condition_labels_visible": False,
        "records": records,
    }
    study.atomic_json(study.ROOT / "representation_validation.json", payload)
    (study.ROOT / "representation_validation.md").write_text(
        "# Representation validation\n\n"
        "The validator received shuffled prompt texts and opaque packet IDs, without condition labels, hypotheses, or target-model outputs. It independently parsed role aliases, path edges, decision locus, and the irrelevant marker.\n\n"
        f"Concordance: **{concordant_n}/{len(records)} ({payload['concordance']:.1%})**.\n",
        encoding="utf-8",
    )
    if payload["status"] != "PASSED":
        raise RuntimeError("STOP: a frozen candidate representation failed semantic validation")
    return payload


def validate_surface_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["fixture_id"], {})[row["condition"]] = row
    fixture_records = []
    for fixture_id, group in sorted(groups.items()):
        verbose = group["PRESERVED_VERBOSE"]
        abbreviated = group["PRESERVED_ABBREVIATED"]
        symbolic = group["PRESERVED_SYMBOLIC"]
        relational = group["LOCUS_CHANGED_SYMBOLIC"]
        control = group["NONREL_CONTROL_SYMBOLIC"]
        record = {
            "fixture_id": fixture_id,
            "verbose_tokens": verbose["input_token_count"],
            "abbreviated_tokens": abbreviated["input_token_count"],
            "symbolic_tokens": symbolic["input_token_count"],
            "abbreviated_savings": abbreviated["input_token_savings_from_preserved_verbose"],
            "symbolic_savings": symbolic["input_token_savings_from_preserved_verbose"],
            "relational_symbolic_edit_distance": relational["edit_distance_from_preserved_same_form"],
            "control_symbolic_edit_distance": control["edit_distance_from_preserved_same_form"],
            "relational_symbolic_token_delta": relational["input_token_delta_from_preserved_same_form"],
            "control_symbolic_token_delta": control["input_token_delta_from_preserved_same_form"],
        }
        if not (verbose["input_token_count"] > abbreviated["input_token_count"] > symbolic["input_token_count"]):
            raise RuntimeError("Compression ladder is not strictly token-decreasing for every fixture")
        if relational["edit_distance_from_preserved_same_form"] != control["edit_distance_from_preserved_same_form"]:
            raise RuntimeError("Symbolic relational and non-relational controls are not edit-distance matched")
        if abs(relational["input_token_delta_from_preserved_same_form"]) != abs(control["input_token_delta_from_preserved_same_form"]):
            raise RuntimeError("Symbolic relational and non-relational controls are not token-delta matched")
        if relational["character_delta_from_preserved_same_form"] != control["character_delta_from_preserved_same_form"]:
            raise RuntimeError("Symbolic relational and non-relational controls are not character-delta matched")
        fixture_records.append(record)
    preserved = [row for row in rows if row["condition"].startswith("PRESERVED_")]
    unique_savings = sorted({row["input_token_savings_from_preserved_verbose"] for row in preserved})
    unique_edits = sorted({row["levenshtein_from_preserved_verbose"] for row in preserved})
    if len(unique_savings) < 3 or len(unique_edits) < 3:
        raise RuntimeError("Surface-distance design lacks the preregistered three-level variation")
    payload = {
        "status": "PASSED",
        "fixtures": len(groups),
        "strict_token_compression_every_fixture": True,
        "matched_symbolic_relational_nonrelational_edits": True,
        "unique_preserved_token_savings": unique_savings,
        "unique_preserved_levenshtein_distances": unique_edits,
        "zero_variance_guard": {
            "minimum_distinct_token_savings": 3,
            "minimum_distinct_levenshtein_distances": 3,
            "passed": True,
        },
        "fixture_records": fixture_records,
    }
    study.atomic_json(study.ROOT / "surface_validation.json", payload)
    (study.ROOT / "surface_validation.md").write_text(
        "# Surface-design validation\n\n"
        "Every fixture has a strictly token-decreasing VERBOSE → ABBREVIATED → SYMBOLIC ladder. The symbolic decision-locus edit and irrelevant-marker edit have matched character, token, and Levenshtein costs.\n\n"
        f"Distinct preserved token-savings values: {len(unique_savings)}. Distinct preserved Levenshtein distances: {len(unique_edits)}.\n",
        encoding="utf-8",
    )
    return payload


def write_configs() -> None:
    study.atomic_json(
        study.ROOT / "scorer_config.json",
        {
            "type": "deterministic",
            "parser": "exactly one isolated A or B, case-insensitive",
            "correctness": "parsed choice equals the precomputed code for the rendered decision locus",
            "retry_on_substantive_wrong_answer": False,
            "condition_metadata_visible": False,
        },
    )
    (study.ROOT / "scorer_prompt.txt").write_text(
        "NONE: scoring is deterministic and receives no condition metadata or model prompt.\n",
        encoding="utf-8",
    )
    study.atomic_json(
        study.ROOT / "inference_config.json",
        {
            "model": study.MODEL_ID,
            "revision": study.MODEL_REVISION,
            "backend": "transformers CPU",
            "dtype": study.MODEL_DTYPE,
            "do_sample": False,
            "max_new_tokens": study.MAX_NEW_TOKENS,
            "max_input_tokens": study.MAX_INPUT_TOKENS,
            "seed": study.SEED,
            "technical_retries": study.TECHNICAL_RETRIES,
            "first_scorable_token": "first generation step; admissible tokens A and B",
            "decision_margin": "raw logit(expected token) minus raw logit(other admissible token)",
        },
    )
    study.atomic_json(
        study.ROOT / "representation_config.json",
        {
            "surface_forms": ["VERBOSE", "ABBREVIATED", "SYMBOLIC"],
            "canonical_edges": [list(edge) for edge in study.CANONICAL_EDGES],
            "preserved_locus": "EXECUTION_ACTOR",
            "changed_locus": "REFERENCE_ACTOR",
            "symbolic_relational_edit": "L:X to L:Y",
            "symbolic_nonrelational_control": "M:X to M:Y while L:X is unchanged",
            "factor_record_wording": "identical across all conditions",
            "adaptive_search": False,
        },
    )


def write_preregistration(power: dict[str, Any], validation: dict[str, Any], surface: dict[str, Any]) -> None:
    text = f"""# {study.STUDY_NAME} — preregistration

Frozen before target-model inference. This is a new continuation, not a modification of the completed 128-fixture relational-complexity study. It tests behavioral efficiency of relation-preserving surface compression and specificity to a relational decision-locus edit. It does not test consciousness, subjective experience, self-awareness, or human-equivalent selfhood.

## Design

Scientific N is **{study.SCIENTIFIC_N} fresh held-out fixtures** plus an unused frozen reserve of {study.RESERVE_N}. No semantic profile overlaps the prior scored continuation. Every fixture has seven matched renderings: preserved VERBOSE, ABBREVIATED, and SYMBOLIC forms; decision-locus-changed versions of those three forms; and one SYMBOLIC non-relational marker control.

The relation graph edges are fixed as REQUESTER→COORDINATOR→EXECUTION_ACTOR and REFERENCE_ACTOR→OBSERVER→REQUESTER. Preserved conditions set the decision locus to EXECUTION_ACTOR. Relational counterfactuals set it to REFERENCE_ACTOR. Executor and reference records have opposite correct action codes in every fixture. Surface form changes role notation only; factor records, values, task, persona, response mapping, and graph semantics stay fixed.

At SYMBOLIC form, the relational edit is `L:X`→`L:Y`; the non-relational control is `M:X`→`M:Y` while `L:X` stays unchanged. Their character, token, and Levenshtein edit costs must match exactly. Levenshtein distance is a surface metric, never a semantic or graph metric.

## Prospective validation

The condition-blind parser validated {validation['concordant']}/{validation['packet_count']} shuffled renderings. It saw no condition labels, hypotheses, or model outputs. The compression ladder is strictly token-decreasing for every fixture. The zero-variance guard requires at least three distinct token-savings values and three distinct Levenshtein distances; observed pre-inference counts are {len(surface['unique_preserved_token_savings'])} and {len(surface['unique_preserved_levenshtein_distances'])}.

## Primary outcomes and decision rule

Generated accuracy is primary. The confirmatory endpoint is the paired risk difference P(pass|PRESERVED_SYMBOLIC) − P(pass|PRESERVED_VERBOSE). Non-inferiority margin is −{study.NONINFERIORITY_MARGIN:.2f}, tested by a frozen 10,000-repetition fixture bootstrap; the one-sided 95% lower bound must exceed the margin. Exact parsed-choice agreement is a co-primary fidelity criterion with a frozen floor of {study.CHOICE_FIDELITY_FLOOR:.2f}. Every fixture must also save at least one pinned-tokenizer input token. The joint surface-efficiency claim requires all three criteria; because it is an intersection-union claim, no multiplicity adjustment is used.

Power was calculated before inference using one-sided α=.05, target power .80, true design difference zero, and paired discordance .20, conservatively rounded above the prior LOW representation disagreement. The selected N is {power['selected_n']} and does not change after scoring. Higher-discordance scenarios are sensitivity analyses only.

## Secondary outcomes

Relational specificity compares exact choice transitions from PRESERVED_SYMBOLIC to LOCUS_CHANGED_SYMBOLIC against transitions to NONREL_CONTROL_SYMBOLIC using a one-sided paired exact McNemar test. Compression-dose choice change and absolute decision-margin change are descriptive Spearman analyses across abbreviated and symbolic levels, computed only when both variables vary. Generated accuracy remains primary; first-step raw A/B logits are secondary.

## Technical rules

Scoring parses exactly one isolated A or B and compares it with the precomputed answer for that rendering's locus. Valid wrong or surprising outputs are never retried or excluded. Technical retries are limited to invocation failure, truncation preventing scoring, or parser failure, with one initial call plus at most {study.TECHNICAL_RETRIES} identical retries. No adaptive paraphrase search, label search, fixture repair, exclusion, threshold change, or sample-size change is permitted after inference begins.

Execution uses {study.SCIENTIFIC_N // study.SHARD_FIXTURES} deterministic shards of {study.SHARD_FIXTURES} fixtures ({study.SHARD_FIXTURES * len(study.CONDITIONS)} evaluations each). Each fixture checkpoints atomically. A four-hour soft ceiling stops before the next fixture and preserves a resumable prefix; GitHub jobs have a 270-minute hard timeout. Old runs are never rerun in place: resume requires a fresh control commit and a larger attempt number.

## Interpretation boundary

The assay can show whether compact relation notation preserves generated behavior and whether a matched relational edit changes behavior more than a non-relational edit. It cannot by itself identify an internal mechanism. Any later optimized-symbol or natural-language paraphrase search must use different held-out fixtures and a separate preregistration.
"""
    (study.ROOT / "preregistration.md").write_text(text, encoding="utf-8")


def frozen_files() -> list[str]:
    root = "experiments/res_surface_efficiency_assay/"
    names = [
        "__init__.py",
        "README.md",
        "study.py",
        "fixture_generation.py",
        "representation_validator.py",
        "power_analysis.py",
        "analysis_common.py",
        "analysis_primary.py",
        "analysis_secondary.py",
        "analyze.py",
        "run.py",
        "phase0.py",
        "test_study.py",
        "power_analysis.json",
        "power_analysis.md",
        "candidate_pool.jsonl",
        "reserve_pool.jsonl",
        "pool_freeze.json",
        "fixtures.jsonl",
        "rendered_prompts.jsonl",
        "representation_validation_packets.jsonl",
        "representation_validation_key.jsonl",
        "representation_validator_judgments.jsonl",
        "representation_validation.json",
        "representation_validation.md",
        "surface_validation.json",
        "surface_validation.md",
        "scorer_config.json",
        "scorer_prompt.txt",
        "inference_config.json",
        "representation_config.json",
        "preregistration.md",
        "frozen_manifest.json",
    ]
    files = [root + name for name in names]
    files.append(".github/workflows/res-surface-efficiency-assay.yml")
    return sorted(files)


def make_manifest(fixtures: list[dict[str, Any]], rows: list[dict[str, Any]], power: dict[str, Any]) -> dict[str, Any]:
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=study.REPO, text=True).strip()
    prior_results = (
        study.REPO
        / "experiments/res_confirmatory_study1_continuation/results/posthoc_surface_efficiency_analysis.json"
    )
    return {
        "protocol": study.VERSION,
        "state": "PHASE0_FROZEN_PRE_INFERENCE",
        "phase0_parent_commit_sha": parent,
        "model_identifier": study.MODEL_ID,
        "model_revision": study.MODEL_REVISION,
        "scientific_n": len(fixtures),
        "reserve_n": study.RESERVE_N,
        "conditions": list(study.CONDITIONS),
        "expected_observations": len(fixtures) * len(study.CONDITIONS),
        "fixture_hashes": {row["fixture_id"]: row["canonical_fixture_hash"] for row in fixtures},
        "rendering_hashes": {row["observation_id"]: row["rendering_hash"] for row in rows},
        "fixture_file_hash": study.file_hash(study.ROOT / "fixtures.jsonl"),
        "rendering_file_hash": study.file_hash(study.ROOT / "rendered_prompts.jsonl"),
        "candidate_pool_hash": study.file_hash(study.ROOT / "candidate_pool.jsonl"),
        "reserve_pool_hash": study.file_hash(study.ROOT / "reserve_pool.jsonl"),
        "inference_config_hash": study.file_hash(study.ROOT / "inference_config.json"),
        "scorer_hash": study.file_hash(study.ROOT / "scorer_config.json"),
        "representation_config_hash": study.file_hash(study.ROOT / "representation_config.json"),
        "preregistration_hash": study.file_hash(study.ROOT / "preregistration.md"),
        "power_analysis_hashes": {
            "script": study.file_hash(study.ROOT / "power_analysis.py"),
            "json": study.file_hash(study.ROOT / "power_analysis.json"),
            "markdown": study.file_hash(study.ROOT / "power_analysis.md"),
        },
        "analysis_code_hashes": {
            "primary": study.file_hash(study.ROOT / "analysis_primary.py"),
            "secondary": study.file_hash(study.ROOT / "analysis_secondary.py"),
            "orchestrator": study.file_hash(study.ROOT / "analyze.py"),
        },
        "runner_hash": study.file_hash(study.ROOT / "run.py"),
        "prior_motivation_artifact": {
            "path": str(prior_results.relative_to(study.REPO)),
            "sha256": study.file_hash(prior_results),
            "role": "prospective power motivation only",
        },
        "power": power,
        "seeds": {
            "study": study.SEED,
            "pool": study.POOL_SEED,
            "validation": study.VALIDATION_SEED,
            "bootstrap": study.BOOTSTRAP_SEED,
        },
    }


def build() -> None:
    if (study.ROOT / "frozen_manifest.json").exists() or (study.ROOT / "results").exists():
        raise RuntimeError("Refusing to replace an existing freeze or scored-results directory")
    power_path = study.ROOT / "power_analysis.json"
    if not power_path.exists():
        raise RuntimeError("Run power_analysis.py before Phase 0 build")
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if power["selected_n"] != study.SCIENTIFIC_N:
        raise RuntimeError("Power-selected N differs from the study constant")
    write_configs()
    candidate, _ = freeze_pools()
    tokenizer = _tokenizer()
    fixtures, rows = render_fixtures(candidate, tokenizer)
    study.write_jsonl(study.ROOT / "fixtures.jsonl", fixtures)
    study.write_jsonl(study.ROOT / "rendered_prompts.jsonl", rows)
    validation = validate_representations(rows)
    surface = validate_surface_design(rows)
    write_preregistration(power, validation, surface)
    manifest = make_manifest(fixtures, rows, power)
    study.atomic_json(study.ROOT / "frozen_manifest.json", manifest)
    hash_lines = []
    for relative in frozen_files():
        path = study.REPO / relative
        if not path.exists():
            raise FileNotFoundError(f"Required frozen file missing: {relative}")
        hash_lines.append(f"{study.file_hash(path)}  {relative}")
    (study.ROOT / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    verify()
    print(
        json.dumps(
            {
                "status": "PHASE0_FROZEN_PRE_INFERENCE",
                "n": len(fixtures),
                "renderings": len(rows),
                "conditions": len(study.CONDITIONS),
            },
            sort_keys=True,
        )
    )


def verify() -> None:
    manifest = json.loads((study.ROOT / "frozen_manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != study.VERSION or manifest["state"] != "PHASE0_FROZEN_PRE_INFERENCE":
        raise ValueError("Frozen manifest protocol/state mismatch")
    for line in (study.ROOT / "hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if study.file_hash(study.REPO / relative) != expected:
            raise ValueError(f"Frozen hash mismatch: {relative}")
    fixtures = study.read_jsonl(study.ROOT / "fixtures.jsonl")
    rows = study.read_jsonl(study.ROOT / "rendered_prompts.jsonl")
    if len(fixtures) != study.SCIENTIFIC_N or len(rows) != study.SCIENTIFIC_N * len(study.CONDITIONS):
        raise ValueError("Frozen sample count mismatch")
    if {row["condition"] for row in rows} != set(study.CONDITIONS):
        raise ValueError("Frozen condition set mismatch")
    if {row["fixture_id"]: row["canonical_fixture_hash"] for row in fixtures} != manifest["fixture_hashes"]:
        raise ValueError("Fixture hashes differ from manifest")
    if {row["observation_id"]: row["rendering_hash"] for row in rows} != manifest["rendering_hashes"]:
        raise ValueError("Rendering hashes differ from manifest")
    if json.loads((study.ROOT / "representation_validation.json").read_text())["status"] != "PASSED":
        raise ValueError("Representation validation is not passed")
    if json.loads((study.ROOT / "surface_validation.json").read_text())["status"] != "PASSED":
        raise ValueError("Surface validation is not passed")
    prior = prior_profile_hashes()
    if any(row["semantic_profile_sha256"] in prior for row in fixtures):
        raise ValueError("A frozen fixture overlaps prior scored semantic content")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    command = parser.parse_args().command
    build() if command == "build" else verify()
    if command == "verify":
        print(json.dumps({"status": "PHASE0_HASHES_VERIFIED"}))


if __name__ == "__main__":
    main()
