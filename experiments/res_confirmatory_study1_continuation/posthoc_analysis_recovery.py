#!/usr/bin/env python3
"""Recover the frozen analysis and add a clearly post-hoc surface-efficiency summary.

This file is intentionally outside the Phase 0 hash manifest. It does not
change fixtures, observations, hypotheses, or the frozen primary/secondary
analysis code. Its only compatibility correction is to serialize non-finite
descriptive statistics as JSON null; it then writes separate post-hoc outputs.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import analysis_common as common
import analyze as frozen_analyze
import study

RESULTS = study.ROOT / "results"
SOURCE_RUN_ID = 36324399210
SOURCE_ARTIFACT_ID = 10936466534
SOURCE_ARTIFACT_NAME = "res-cs1c-collected-1"

COMPARISONS = (
    {
        "id": "code_mapped_low_vs_neutral_low",
        "baseline": "NEUTRAL_LOW",
        "variant": "CODE_MAPPED_LOW",
        "role": "same LOW relational structure and decision; alternate frozen factor/task representation",
    },
    {
        "id": "code_mapped_high_vs_neutral_high",
        "baseline": "NEUTRAL_HIGH",
        "variant": "CODE_MAPPED_HIGH",
        "role": "same HIGH relational structure and decision; alternate frozen factor/task representation",
    },
    {
        "id": "nonrel_control_high_vs_neutral_high",
        "baseline": "NEUTRAL_HIGH",
        "variant": "NONREL_CONTROL_HIGH",
        "role": "same HIGH relational structure and decision; matched generic surface perturbation",
    },
)

SURFACE_MEASURES = (
    "character_count",
    "input_token_count",
    "output_token_count",
    "edit_distance_from_neutral_baseline",
)


def finite_json_ready(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: finite_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [finite_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return finite_json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def describe(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def estimable_spearman(predictor: list[float], outcome: list[float]) -> dict[str, Any]:
    """Compute Spearman only when both variables vary; otherwise explain why."""
    if len(predictor) != len(outcome):
        raise ValueError("Predictor and outcome lengths differ")
    predictor_unique = sorted({float(value) for value in predictor})
    outcome_unique = sorted({float(value) for value in outcome})
    base = {
        "n": len(predictor),
        "predictor_unique_values": predictor_unique,
        "outcome_unique_values": outcome_unique,
    }
    if len(predictor) < 3:
        return {**base, "status": "NOT_ESTIMABLE_INSUFFICIENT_N", "spearman_rho": None, "two_sided_p_descriptive": None}
    if len(predictor_unique) < 2 and len(outcome_unique) < 2:
        return {
            **base,
            "status": "NOT_ESTIMABLE_ZERO_VARIANCE_BOTH",
            "spearman_rho": None,
            "two_sided_p_descriptive": None,
        }
    if len(predictor_unique) < 2:
        return {
            **base,
            "status": "NOT_ESTIMABLE_ZERO_VARIANCE_PREDICTOR",
            "spearman_rho": None,
            "two_sided_p_descriptive": None,
        }
    if len(outcome_unique) < 2:
        return {
            **base,
            "status": "NOT_ESTIMABLE_ZERO_VARIANCE_OUTCOME",
            "spearman_rho": None,
            "two_sided_p_descriptive": None,
        }
    statistic, p_value = spearmanr(predictor, outcome)
    if not math.isfinite(float(statistic)) or not math.isfinite(float(p_value)):
        return {
            **base,
            "status": "NOT_ESTIMABLE_NONFINITE_RESULT",
            "spearman_rho": None,
            "two_sided_p_descriptive": None,
        }
    return {
        **base,
        "status": "ESTIMABLE_DESCRIPTIVE",
        "spearman_rho": float(statistic),
        "two_sided_p_descriptive": float(p_value),
    }


def correlation_estimability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("technical_status") == "VALID"]
    scopes = {
        "all_primary": [row for row in valid if row["condition"] in study.PRIMARY_CONDITIONS],
        **{condition: [row for row in valid if row["condition"] == condition] for condition in study.CONDITIONS},
    }
    output = {}
    for scope, selected in scopes.items():
        outcome = [int(row["passed"]) for row in selected]
        output[scope] = {
            measure: estimable_spearman([float(row[measure]) for row in selected], outcome)
            for measure in SURFACE_MEASURES
        }
    return {
        "status": "COMPLETE_DESCRIPTIVE_ESTIMABILITY_AUDIT",
        "role": (
            "Companion metadata for the frozen length/structure analysis. "
            "It does not substitute a new inferential test."
        ),
        "source_analysis": "length_structural_control_analysis.json",
        "rule": (
            "A rank correlation is reported only when n >= 3 and both predictor and correctness vary. "
            "Otherwise the coefficient and p-value are null with an explicit reason."
        ),
        "scopes": output,
    }


def compactness_class(token_savings: float, same_choice: bool) -> str:
    if token_savings > 0:
        prefix = "VARIANT_SHORTER"
    elif token_savings < 0:
        prefix = "VARIANT_LONGER"
    else:
        prefix = "TOKEN_LENGTH_EQUAL"
    return f"{prefix}_{'CHOICE_PRESERVED' if same_choice else 'CHOICE_CHANGED'}"


def matched_surface_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("technical_status") == "VALID"]
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid:
        groups[row["fixture_id"]][row["condition"]] = row

    comparison_results = {}
    pair_records = []
    for specification in COMPARISONS:
        baseline_condition = specification["baseline"]
        variant_condition = specification["variant"]
        records = []
        for fixture_id in sorted(groups):
            group = groups[fixture_id]
            if baseline_condition not in group or variant_condition not in group:
                continue
            baseline = group[baseline_condition]
            variant = group[variant_condition]
            if baseline["expected"] != variant["expected"]:
                raise ValueError("Matched conditions disagree on the frozen expected decision")

            graph_signature_preserved = all(
                baseline[metric] == variant[metric] for metric in study.STRUCTURAL_METRICS
            )
            same_choice = baseline.get("parsed_choice") == variant.get("parsed_choice")
            same_pass = bool(baseline["passed"]) == bool(variant["passed"])
            character_savings = float(baseline["character_count"] - variant["character_count"])
            token_savings = float(baseline["input_token_count"] - variant["input_token_count"])
            margin_shift = float(variant["decision_margin"] - baseline["decision_margin"])
            record = {
                "comparison": specification["id"],
                "fixture_id": fixture_id,
                "baseline_condition": baseline_condition,
                "variant_condition": variant_condition,
                "baseline_observation_id": baseline["observation_id"],
                "variant_observation_id": variant["observation_id"],
                "frozen_structural_signature_preserved": graph_signature_preserved,
                "baseline_character_count": baseline["character_count"],
                "variant_character_count": variant["character_count"],
                "character_savings_variant_vs_baseline": character_savings,
                "baseline_input_token_count": baseline["input_token_count"],
                "variant_input_token_count": variant["input_token_count"],
                "input_token_savings_variant_vs_baseline": token_savings,
                "surface_edit_distance_characters": variant["edit_distance_from_neutral_baseline"],
                "baseline_parsed_choice": baseline.get("parsed_choice"),
                "variant_parsed_choice": variant.get("parsed_choice"),
                "parsed_choice_preserved": same_choice,
                "baseline_passed": bool(baseline["passed"]),
                "variant_passed": bool(variant["passed"]),
                "correctness_preserved": same_pass,
                "pass_transition": not same_pass,
                "correction": (not bool(baseline["passed"])) and bool(variant["passed"]),
                "regression": bool(baseline["passed"]) and (not bool(variant["passed"])),
                "baseline_decision_margin": baseline["decision_margin"],
                "variant_decision_margin": variant["decision_margin"],
                "decision_margin_shift_variant_minus_baseline": margin_shift,
                "absolute_decision_margin_shift": abs(margin_shift),
                "compactness_behavior_class": compactness_class(token_savings, same_choice),
            }
            records.append(record)
            pair_records.append(record)

        n = len(records)
        if not n:
            comparison_results[specification["id"]] = {
                "status": "NO_COMPLETE_VALID_PAIRS",
                "baseline": baseline_condition,
                "variant": variant_condition,
            }
            continue

        class_counts: dict[str, int] = defaultdict(int)
        for record in records:
            class_counts[record["compactness_behavior_class"]] += 1

        choice_change = [int(not record["parsed_choice_preserved"]) for record in records]
        margin_change = [record["absolute_decision_margin_shift"] for record in records]
        character_savings = [record["character_savings_variant_vs_baseline"] for record in records]
        token_savings = [record["input_token_savings_variant_vs_baseline"] for record in records]
        edit_distance = [float(record["surface_edit_distance_characters"]) for record in records]
        comparison_results[specification["id"]] = {
            "status": "COMPLETE_POSTHOC_DESCRIPTIVE",
            "role": specification["role"],
            "baseline": baseline_condition,
            "variant": variant_condition,
            "n": n,
            "frozen_structural_signature_preserved_n": sum(
                record["frozen_structural_signature_preserved"] for record in records
            ),
            "baseline_accuracy": sum(record["baseline_passed"] for record in records) / n,
            "variant_accuracy": sum(record["variant_passed"] for record in records) / n,
            "accuracy_change_variant_minus_baseline": (
                sum(record["variant_passed"] for record in records)
                - sum(record["baseline_passed"] for record in records)
            ) / n,
            "parsed_choice_agreement_rate": sum(record["parsed_choice_preserved"] for record in records) / n,
            "correctness_agreement_rate": sum(record["correctness_preserved"] for record in records) / n,
            "pass_transition_rate": sum(record["pass_transition"] for record in records) / n,
            "corrections": sum(record["correction"] for record in records),
            "regressions": sum(record["regression"] for record in records),
            "compactness_behavior_classes": dict(sorted(class_counts.items())),
            "character_savings_variant_vs_baseline": describe(character_savings),
            "input_token_savings_variant_vs_baseline": describe(token_savings),
            "surface_edit_distance_characters": describe(edit_distance),
            "absolute_decision_margin_shift": describe(margin_change),
            "descriptive_associations": {
                "character_savings_vs_choice_change": estimable_spearman(character_savings, choice_change),
                "input_token_savings_vs_choice_change": estimable_spearman(token_savings, choice_change),
                "surface_edit_distance_vs_choice_change": estimable_spearman(edit_distance, choice_change),
                "surface_edit_distance_vs_absolute_margin_shift": estimable_spearman(edit_distance, margin_change),
            },
        }

    study.write_jsonl(RESULTS / "posthoc_surface_efficiency_pairs.jsonl", pair_records)
    payload = {
        "status": "POSTHOC_DESCRIPTIVE_COMPLETE",
        "role": (
            "Surface-form and representation-efficiency summary added after outcomes were observed. "
            "It is not preregistered, confirmatory, or a replacement for the frozen primary/secondary analyses."
        ),
        "measurement_definition": {
            "relational_structure": (
                "Operationally checked here only as equality of the six frozen structural metrics within a matched fixture. "
                "This is not an independently validated graph-isomorphism or mechanistic measure."
            ),
            "surface_edit_distance_characters": (
                "Frozen character-level Levenshtein distance between a variant's full chat-formatted prompt "
                "and the same fixture/complexity Neutral prompt."
            ),
            "character_savings_variant_vs_baseline": (
                "baseline full-prompt character count minus variant count; positive means the variant is shorter."
            ),
            "input_token_savings_variant_vs_baseline": (
                "baseline pinned-tokenizer input count minus variant count; positive means the variant uses fewer model tokens."
            ),
            "behavioral_fidelity": (
                "Exact parsed-choice agreement, correctness transitions, and absolute change in the frozen decision margin."
            ),
            "efficiency_question": (
                "Whether a more compact representation preserves the same choice and similar decision margin. "
                "Compactness and behavioral fidelity are reported separately rather than collapsed into an unsupported scalar."
            ),
        },
        "interpretation_limits": [
            "The frozen variants were not generated as a paraphrase-distance experiment.",
            "Levenshtein distance measures surface edits, not relational or semantic distance.",
            "Token savings are model-specific because they use the pinned tokenizer.",
            "Choice preservation does not establish an internal relational mechanism.",
            "Post-hoc associations are descriptive and are omitted when either variable has no variation.",
        ],
        "comparisons": comparison_results,
    }
    study.atomic_json(RESULTS / "posthoc_surface_efficiency_analysis.json", finite_json_ready(payload))
    return payload


def write_posthoc_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Post-hoc surface-form efficiency analysis",
        "",
        "**Status: descriptive and post-hoc.** This does not alter or replace the frozen confirmatory analysis.",
        "",
        "Efficiency is split into two observable pieces: (1) compactness, measured by signed character and pinned-token savings, and (2) behavioral fidelity, measured by exact choice agreement, correctness transitions, and decision-margin shift. Levenshtein distance remains a surface-edit measure; equality of the six frozen structural metrics is only an operational check that the matched pair retained the same frozen structural signature.",
        "",
        "| Matched representation pair | n | mean token savings | choice agreement | pass transitions | accuracy change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for specification in COMPARISONS:
        result = payload["comparisons"][specification["id"]]
        if result.get("status") != "COMPLETE_POSTHOC_DESCRIPTIVE":
            lines.append(f"| {specification['id']} | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {specification['id']} | {result['n']} | "
            f"{result['input_token_savings_variant_vs_baseline']['mean']:+.2f} | "
            f"{result['parsed_choice_agreement_rate']:.1%} | "
            f"{result['corrections'] + result['regressions']} | "
            f"{result['accuracy_change_variant_minus_baseline']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "Positive savings means the variant is shorter than its Neutral baseline. A shorter prompt with an unchanged choice is evidence of surface efficiency for this pinned model and fixture, not evidence that the two strings are semantically interchangeable in general.",
            "",
            "The existing frozen prompts do not instantiate contrasts such as 'happy' versus an emoji or a controlled paraphrase family. Testing that idea directly requires a new preregistered fixture set with relation-preserving paraphrases, relation-changing counterfactuals, and matched non-relational controls.",
            "",
        ]
    )
    (RESULTS / "POSTHOC_SURFACE_EFFICIENCY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_manifest_path = RESULTS / "run_manifest.json"
    if not run_manifest_path.exists():
        raise FileNotFoundError("Collected run_manifest.json is required")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    observations = common.load_observations()
    rows = common.validate_and_join(observations)
    if run_manifest.get("status") != "COMPLETE_ATTEMPTED_ALL_FIXTURES":
        raise RuntimeError("Recovery requires all predetermined fixtures to have been attempted")
    if len(rows) != run_manifest.get("expected_observations"):
        raise RuntimeError("Collected observation count does not match the completed run manifest")

    original_json_ready = common.json_ready
    common.json_ready = finite_json_ready
    try:
        frozen_result = frozen_analyze.main()
    finally:
        common.json_ready = original_json_ready
    if frozen_result.get("status") != "COMPLETE":
        raise RuntimeError("Frozen analysis did not complete after strict-JSON recovery")

    estimability = correlation_estimability(rows)
    study.atomic_json(
        RESULTS / "length_structural_correlation_estimability.json",
        finite_json_ready(estimability),
    )
    posthoc = matched_surface_efficiency(rows)
    write_posthoc_report(posthoc)

    provenance = {
        "status": "ANALYSIS_RECOVERY_COMPLETE",
        "protocol": study.VERSION,
        "source_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_name": SOURCE_ARTIFACT_NAME,
        "source_observations": len(rows),
        "source_run_manifest_status": run_manifest["status"],
        "recovery_workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "recovery_commit": os.environ.get("GITHUB_SHA", "local"),
        "compatibility_correction": (
            "Non-finite descriptive statistics caused by constant-input Spearman correlations "
            "were serialized as JSON null instead of invalid NaN."
        ),
        "estimability_metadata": "length_structural_correlation_estimability.json",
        "posthoc_addition": "posthoc_surface_efficiency_analysis.json",
        "frozen_inputs_or_observations_changed": False,
        "frozen_primary_or_secondary_definition_changed": False,
        "target_model_calls_made_by_recovery": 0,
        "phase0_hash_verification": "required before collection and again before publication",
    }
    study.atomic_json(RESULTS / "analysis_recovery_provenance.json", provenance)
    print(
        json.dumps(
            {
                "status": "ANALYSIS_RECOVERY_COMPLETE",
                "observations": len(rows),
                "posthoc_comparisons": len(posthoc["comparisons"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
