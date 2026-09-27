"""Frozen graph-specificity, compression-dose, and margin analyses."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from scipy.stats import binomtest, spearmanr

import analysis_common as common
import study


def _spearman(predictor: list[float], outcome: list[float]) -> dict[str, Any]:
    predictor_unique = sorted(set(predictor))
    outcome_unique = sorted(set(outcome))
    base = {
        "n": len(predictor),
        "predictor_unique_values": predictor_unique,
        "outcome_unique_values": outcome_unique,
    }
    if len(predictor) < 3:
        return {**base, "status": "NOT_ESTIMABLE_INSUFFICIENT_N", "rho": None, "p": None}
    if len(predictor_unique) < 2:
        return {**base, "status": "NOT_ESTIMABLE_ZERO_VARIANCE_PREDICTOR", "rho": None, "p": None}
    if len(outcome_unique) < 2:
        return {**base, "status": "NOT_ESTIMABLE_ZERO_VARIANCE_OUTCOME", "rho": None, "p": None}
    rho, p_value = spearmanr(predictor, outcome)
    if not math.isfinite(float(rho)) or not math.isfinite(float(p_value)):
        return {**base, "status": "NOT_ESTIMABLE_NONFINITE", "rho": None, "p": None}
    return {**base, "status": "ESTIMABLE_DESCRIPTIVE", "rho": float(rho), "p": float(p_value)}


def graph_specificity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups, incomplete = common.complete_groups(rows, study.SPECIFICITY_TRIPLET)
    fixture_rows = []
    for fixture_id, group in sorted(groups.items()):
        baseline = group["PRESERVED_SYMBOLIC"]
        relational = group["LOCUS_CHANGED_SYMBOLIC"]
        control = group["NONREL_CONTROL_SYMBOLIC"]
        relational_transition = baseline["parsed_choice"] != relational["parsed_choice"]
        control_transition = baseline["parsed_choice"] != control["parsed_choice"]
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "baseline_choice": baseline["parsed_choice"],
                "relational_choice": relational["parsed_choice"],
                "control_choice": control["parsed_choice"],
                "baseline_expected": baseline["expected"],
                "relational_expected": relational["expected"],
                "control_expected": control["expected"],
                "expected_choice_flips_for_graph_change": baseline["expected"] != relational["expected"],
                "expected_choice_stable_for_control": baseline["expected"] == control["expected"],
                "relational_transition": relational_transition,
                "nonrelational_transition": control_transition,
                "baseline_margin": baseline["decision_margin"],
                "relational_surface_edit_distance": relational["edit_distance_from_preserved_same_form"],
                "control_surface_edit_distance": control["edit_distance_from_preserved_same_form"],
                "relational_input_token_delta": relational["input_token_delta_from_preserved_same_form"],
                "control_input_token_delta": control["input_token_delta_from_preserved_same_form"],
            }
        )
    relational_only = sum(row["relational_transition"] and not row["nonrelational_transition"] for row in fixture_rows)
    control_only = sum(row["nonrelational_transition"] and not row["relational_transition"] for row in fixture_rows)
    discordant = relational_only + control_only
    p_value = float(binomtest(relational_only, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    table = Counter((row["relational_transition"], row["nonrelational_transition"]) for row in fixture_rows)
    n = len(fixture_rows)
    return {
        "status": "COMPLETE" if n else "NO_COMPLETE_VALID_TRIPLETS",
        "n": n,
        "incomplete_fixtures": incomplete,
        "test": "one-sided paired exact McNemar",
        "alternative": "decision-locus edit changes choice more often than matched non-relational marker edit",
        "paired_table": {
            "neither": table[(False, False)],
            "relational_only": relational_only,
            "nonrelational_only": control_only,
            "both": table[(True, True)],
        },
        "one_sided_p": p_value,
        "relational_transition_rate": sum(row["relational_transition"] for row in fixture_rows) / n if n else None,
        "nonrelational_transition_rate": sum(row["nonrelational_transition"] for row in fixture_rows) / n if n else None,
        "all_expected_graph_answers_flip": all(row["expected_choice_flips_for_graph_change"] for row in fixture_rows),
        "all_control_answers_stable": all(row["expected_choice_stable_for_control"] for row in fixture_rows),
        "fixtures": fixture_rows,
    }

def compression_and_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups, incomplete = common.complete_groups(rows, study.FACTORIAL_CONDITIONS)
    conditions = {}
    graph_response = {}
    for condition in study.FACTORIAL_CONDITIONS:
        selected = [group[condition] for group in groups.values()]
        conditions[condition] = {
            "n": len(selected),
            "passes": sum(row["passed"] for row in selected),
            "accuracy": sum(row["passed"] for row in selected) / len(selected) if selected else None,
            "input_token_count": common.describe([row["input_token_count"] for row in selected]),
            "character_count": common.describe([row["character_count"] for row in selected]),
        }
    for form in ("VERBOSE", "ABBREVIATED", "SYMBOLIC"):
        preserved = f"PRESERVED_{form}"
        changed = f"LOCUS_CHANGED_{form}"
        pairs = [(group[preserved], group[changed]) for group in groups.values()]
        graph_response[form] = {
            "n": len(pairs),
            "choice_transition_rate": (
                sum(left["parsed_choice"] != right["parsed_choice"] for left, right in pairs) / len(pairs)
                if pairs else None
            ),
            "preserved_accuracy": conditions[preserved]["accuracy"],
            "changed_accuracy": conditions[changed]["accuracy"],
        }

    token_savings, choice_change, margin_change = [], [], []
    for group in groups.values():
        baseline = group["PRESERVED_VERBOSE"]
        for condition in ("PRESERVED_ABBREVIATED", "PRESERVED_SYMBOLIC"):
            variant = group[condition]
            token_savings.append(float(baseline["input_token_count"] - variant["input_token_count"]))
            choice_change.append(float(baseline["parsed_choice"] != variant["parsed_choice"]))
            margin_change.append(abs(float(variant["decision_margin"] - baseline["decision_margin"])))
    return {
        "status": "COMPLETE" if groups else "NO_COMPLETE_FACTORIAL_FIXTURES",
        "complete_fixtures": len(groups),
        "incomplete_fixtures": incomplete,
        "condition_summaries": conditions,
        "graph_response_by_surface_form": graph_response,
        "compression_dose_descriptive_associations": {
            "token_savings_vs_choice_change": _spearman(token_savings, choice_change),
            "token_savings_vs_absolute_margin_change": _spearman(token_savings, margin_change),
        },
        "interpretation_rule": (
            "Compression-dose associations are secondary and descriptive; Levenshtein/token distance is not semantic distance."
        ),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "graph_specificity": graph_specificity(rows),
        "compression_and_graph": compression_and_graph(rows),
        "margin_definition": (
            "first-step raw logit(expected decision token) minus raw logit(other admissible token); "
            "margin analyses are secondary"
        ),
    }
