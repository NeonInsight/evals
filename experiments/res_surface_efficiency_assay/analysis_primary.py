"""Frozen primary generated-accuracy and choice-fidelity analysis."""
from __future__ import annotations

from typing import Any

import analysis_common as common
import study


def summarize_pair(
    groups: dict[str, dict[str, dict[str, Any]]], baseline_condition: str, variant_condition: str
) -> dict[str, Any]:
    fixture_ids = sorted(groups)
    baseline = [groups[fid][baseline_condition] for fid in fixture_ids]
    variant = [groups[fid][variant_condition] for fid in fixture_ids]
    baseline_pass = [int(row["passed"]) for row in baseline]
    variant_pass = [int(row["passed"]) for row in variant]
    bootstrap = common.paired_accuracy_bootstrap(baseline_pass, variant_pass)
    agreements = sum(left["parsed_choice"] == right["parsed_choice"] for left, right in zip(baseline, variant))
    corrections = sum((not left["passed"]) and right["passed"] for left, right in zip(baseline, variant))
    regressions = sum(left["passed"] and (not right["passed"]) for left, right in zip(baseline, variant))
    n = len(fixture_ids)
    token_savings = [left["input_token_count"] - right["input_token_count"] for left, right in zip(baseline, variant)]
    character_savings = [left["character_count"] - right["character_count"] for left, right in zip(baseline, variant)]
    edit_distance = [right["levenshtein_from_preserved_verbose"] for right in variant]
    return {
        "baseline_condition": baseline_condition,
        "variant_condition": variant_condition,
        "n": n,
        "baseline_accuracy": sum(baseline_pass) / n,
        "variant_accuracy": sum(variant_pass) / n,
        **bootstrap,
        "choice_agreements": agreements,
        "choice_agreement_rate": agreements / n,
        "choice_agreement_exact_95_ci": common.exact_binomial_interval(agreements, n),
        "corrections": corrections,
        "regressions": regressions,
        "input_token_savings": common.describe(token_savings),
        "character_savings": common.describe(character_savings),
        "levenshtein_from_preserved_verbose": common.describe(edit_distance),
    }

def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = study.PRIMARY_PAIR + ("PRESERVED_ABBREVIATED",)
    groups, incomplete = common.complete_groups(rows, required)
    if not groups:
        return {
            "status": "INCOMPLETE_NO_PRIMARY_ANALYSIS",
            "complete_fixtures": 0,
            "incomplete_fixtures": incomplete,
        }
    compact = summarize_pair(groups, "PRESERVED_VERBOSE", "PRESERVED_SYMBOLIC")
    abbreviated = summarize_pair(groups, "PRESERVED_VERBOSE", "PRESERVED_ABBREVIATED")
    noninferior = compact["one_sided_95_lower_bound"] > -study.NONINFERIORITY_MARGIN
    fidelity = compact["choice_agreement_rate"] >= study.CHOICE_FIDELITY_FLOOR
    shorter = compact["input_token_savings"]["minimum"] > 0
    if noninferior and fidelity and shorter:
        category = "SURFACE_EFFICIENT_WITHIN_FROZEN_CRITERIA"
    elif not noninferior:
        category = "COMPACT_ACCURACY_NONINFERIORITY_NOT_ESTABLISHED"
    elif not fidelity:
        category = "CHOICE_FIDELITY_FLOOR_NOT_MET"
    else:
        category = "COMPACT_FORM_NOT_TOKEN_EFFICIENT"
    return {
        "status": "COMPLETE" if not incomplete else "COMPLETE_CASE_WITH_TECHNICAL_MISSINGNESS",
        "complete_fixtures": len(groups),
        "incomplete_fixtures": incomplete,
        "primary_endpoint": "paired generated-accuracy difference: PRESERVED_SYMBOLIC minus PRESERVED_VERBOSE",
        "noninferiority_margin": -study.NONINFERIORITY_MARGIN,
        "alpha": 0.05,
        "sidedness": "one-sided",
        "choice_fidelity_floor": study.CHOICE_FIDELITY_FLOOR,
        "joint_claim_rule": (
            "Compact form is surface-efficient only if its one-sided 95% paired-bootstrap lower bound "
            "exceeds -0.10, exact parsed-choice agreement is at least 0.75, and every fixture saves tokens."
        ),
        "interpretation_category": category,
        "criteria": {
            "generated_accuracy_noninferior": noninferior,
            "choice_fidelity_floor_met": fidelity,
            "positive_token_savings_every_fixture": shorter,
        },
        "compact_vs_verbose": compact,
        "abbreviated_vs_verbose_secondary": abbreviated,
        "outcome_priority": "generated accuracy primary; parsed-choice fidelity co-primary criterion; logits secondary",
    }
