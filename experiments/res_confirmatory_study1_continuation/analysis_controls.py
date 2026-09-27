"""Frozen descriptive length and structural-control analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import analysis_common as common
import study

MEASURES = (
    "character_count",
    "input_token_count",
    "output_token_count",
    "edit_distance_from_neutral_baseline",
    "task_relevant_actor_entity_count",
    "relational_binding_count",
    "directed_role_delegation_count",
    "maximum_relational_nesting_depth",
    "locus_of_action_transition_count",
    "conjunction_dependency_count",
)


def describe(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def paired_difference(groups: dict[str, dict[str, dict[str, Any]]], left: str, right: str, measure: str) -> dict[str, Any] | None:
    values = []
    for group in groups.values():
        if left in group and right in group:
            values.append(float(group[left][measure]) - float(group[right][measure]))
    return describe(values)


def diagnostic_length_gee(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["condition"] in study.PRIMARY_CONDITIONS and row.get("technical_status") == "VALID"]
    if not primary:
        return {"status": "NO_VALID_PRIMARY_ROWS"}
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial

        records = []
        token_mean = np.mean([row["input_token_count"] for row in primary])
        for row in primary:
            neutral = int(row["condition"].startswith("NEUTRAL"))
            high = int(row["condition"].endswith("HIGH"))
            records.append(
                {
                    "fixture_id": row["fixture_id"],
                    "passed": int(row["passed"]),
                    "neutral": neutral,
                    "high": high,
                    "neutral_by_high": neutral * high,
                    "centered_tokens": (row["input_token_count"] - token_mean) / 10,
                }
            )
        frame = pd.DataFrame(records)
        exog = sm.add_constant(frame[["neutral", "high", "neutral_by_high", "centered_tokens"]], has_constant="add")
        fit = sm.GEE(
            frame["passed"], exog, groups=frame["fixture_id"], family=Binomial(), cov_struct=Exchangeable()
        ).fit(maxiter=200)
        return {
            "status": "DESCRIPTIVE_FIT",
            "role": "length diagnostic only; not a replacement primary model",
            "coefficients": {name: float(fit.params[name]) for name in exog.columns},
            "standard_errors": {name: float(fit.bse[name]) for name in exog.columns},
        }
    except Exception as error:
        return {"status": "DIAGNOSTIC_FIT_FAILED", "reason": str(error)}


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("technical_status") == "VALID"]
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in valid:
        groups[row["fixture_id"]][row["condition"]] = row
    by_condition = {}
    for condition in study.CONDITIONS:
        selected = [row for row in valid if row["condition"] == condition]
        by_condition[condition] = {measure: describe([row[measure] for row in selected]) for measure in MEASURES}
    pairs = {}
    for complexity in ("LOW", "HIGH"):
        pairs[f"code_mapped_minus_neutral_{complexity.lower()}"] = {
            measure: paired_difference(groups, f"CODE_MAPPED_{complexity}", f"NEUTRAL_{complexity}", measure)
            for measure in MEASURES
        }
    for framing in ("CODE_MAPPED", "NEUTRAL"):
        pairs[f"high_minus_low_{framing.lower()}"] = {
            measure: paired_difference(groups, f"{framing}_HIGH", f"{framing}_LOW", measure)
            for measure in MEASURES
        }
    correlations = {}
    for scope, selected in {
        "all_primary": [row for row in valid if row["condition"] in study.PRIMARY_CONDITIONS],
        **{condition: [row for row in valid if row["condition"] == condition] for condition in study.CONDITIONS},
    }.items():
        if len(selected) < 3 or len({row["passed"] for row in selected}) < 2:
            correlations[scope] = None
            continue
        correlations[scope] = {}
        for measure in ("character_count", "input_token_count", "output_token_count", "edit_distance_from_neutral_baseline"):
            statistic, p_value = spearmanr([row[measure] for row in selected], [int(row["passed"]) for row in selected])
            correlations[scope][measure] = {"spearman_rho": float(statistic), "two_sided_p_descriptive": float(p_value)}
    return {
        "status": "COMPLETE" if valid else "NO_VALID_OBSERVATIONS",
        "measures": list(MEASURES),
        "by_condition": by_condition,
        "paired_differences": pairs,
        "descriptive_length_correctness_correlations": correlations,
        "length_adjusted_gee_diagnostic": diagnostic_length_gee(rows),
        "interpretation_rule": "Length and structure results are descriptive controls; stimuli and primary inference remain frozen regardless of their values.",
    }

