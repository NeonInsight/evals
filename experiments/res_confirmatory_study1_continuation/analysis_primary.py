"""Frozen primary interaction analysis."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import analysis_common as common
import conditional_logit

ALPHA = 0.05
SESOI = 0.15


def gee_fallback(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Frozen fallback, used only after conditional-logit computational failure."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
    except Exception as error:
        return {"status": "COMPUTATIONAL_FAILURE", "reason": f"GEE dependency failure: {error}"}
    records = []
    for fixture_id, group in sorted(groups.items()):
        for condition in common.CELL_ORDER:
            neutral = int(condition.startswith("NEUTRAL"))
            high = int(condition.endswith("HIGH"))
            records.append(
                {
                    "fixture_id": fixture_id,
                    "passed": int(group[condition]["passed"]),
                    "neutral": neutral,
                    "high": high,
                    "neutral_by_high": neutral * high,
                }
            )
    frame = pd.DataFrame(records)
    exog = sm.add_constant(frame[["neutral", "high", "neutral_by_high"]], has_constant="add")
    try:
        model = sm.GEE(
            frame["passed"],
            exog,
            groups=frame["fixture_id"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        fit = model.fit(maxiter=200)
    except Exception as error:
        return {"status": "COMPUTATIONAL_FAILURE", "reason": f"GEE did not fit: {error}"}
    if not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all():
        return {"status": "COMPUTATIONAL_FAILURE", "reason": "GEE estimate or interval is non-finite"}
    if np.max(np.abs(fit.params)) > 100 or np.max(np.abs(fit.bse)) > 100:
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": "GEE separation produced an effectively non-finite estimate or interval",
        }
    terms = {}
    for name in ("neutral", "high", "neutral_by_high"):
        estimate, standard_error = float(fit.params[name]), float(fit.bse[name])
        lower, upper = estimate - 1.959963984540054 * standard_error, estimate + 1.959963984540054 * standard_error
        z = estimate / standard_error
        from scipy.stats import norm

        terms[name] = {
            "log_odds": estimate,
            "standard_error": standard_error,
            "odds_ratio": math.exp(estimate),
            "ci_95_log_odds": [lower, upper],
            "ci_95_odds_ratio": [math.exp(lower), math.exp(upper)],
            "two_sided_wald_p": float(2 * norm.sf(abs(z))),
        }
    return {
        "status": "GEE_FALLBACK_CONVERGED",
        "family": "binary",
        "link": "logit",
        "cluster": "fixture_id",
        "working_correlation": "exchangeable",
        "n_fixtures": len(groups),
        "terms": terms,
    }


def interpret_probability_interval(estimate: float, interval: list[float]) -> dict[str, str]:
    lower, upper = interval
    if estimate > 0 and lower > 0:
        return {
            "category": "CONFIRMATORY_SUPPORT",
            "statement": "The interaction is positive and its 95% interval excludes zero.",
        }
    if upper < 0:
        qualifier = (
            " A practically meaningful contrary-direction interaction is supported or compatible with the interval."
            if lower <= -SESOI
            else ""
        )
        return {
            "category": "CONTRARY_DIRECTION_RESULT",
            "statement": "The interval excludes zero on the negative side, contrary to the preregistered direction." + qualifier,
        }
    if lower > -SESOI and upper < SESOI:
        return {
            "category": "PRACTICALLY_NEGLIGIBLE_INTERACTION",
            "statement": "The entire probability-scale 95% interval lies strictly inside (-0.15, +0.15).",
        }
    if upper < SESOI and lower <= -SESOI:
        return {
            "category": "PREDICTED_SIZE_POSITIVE_INTERACTION_RULED_OUT_CONTRARY_EFFECT_STILL_POSSIBLE",
            "statement": "The study constrains the predicted positive interaction while remaining compatible with a practically meaningful contrary-direction interaction.",
        }
    if lower <= 0 <= upper and upper >= SESOI:
        return {
            "category": "INCONCLUSIVE",
            "statement": "The interval includes zero and effects at or above the preregistered positive SESOI remain unresolved.",
        }
    return {
        "category": "INCONCLUSIVE",
        "statement": "The interval does not meet a stronger preregistered interpretive category.",
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups, incomplete = common.complete_primary_groups(rows)
    if not groups:
        return {
            "status": "INCOMPLETE_NO_PRIMARY_ANALYSIS",
            "complete_primary_fixtures": 0,
            "incomplete_primary_fixtures": incomplete,
        }
    fixture_ids, matrix = common.matrix_from_groups(groups)
    probability = common.bootstrap_probability_effects(matrix)
    conditional = conditional_logit.fit_conditional_logit(matrix)
    fallback = None
    selected_model = "conditional_logistic"
    selected = conditional
    if conditional["status"] != "FIT_CONVERGED":
        fallback = gee_fallback(groups)
        selected_model = "gee_fallback"
        selected = fallback
    cells = {}
    for index, condition in enumerate(common.CELL_ORDER):
        successes = int(matrix[:, index].sum())
        cells[condition] = {
            "passes": successes,
            "n": int(len(matrix)),
            "proportion": successes / len(matrix),
            "clopper_pearson_95_ci": common.exact_binomial_interval(successes, len(matrix)),
        }
    return {
        "status": "COMPLETE" if not incomplete else "COMPLETE_CASE_PRIMARY_WITH_TECHNICAL_MISSINGNESS",
        "alpha": ALPHA,
        "alpha_sidedness": "two-sided",
        "sesoi_probability_interaction": SESOI,
        "coding": "neutral=1, high=1; predicted neutral-by-high interaction is positive",
        "complete_primary_fixtures": len(groups),
        "complete_fixture_ids": fixture_ids,
        "incomplete_primary_fixtures": incomplete,
        "cells": cells,
        "probability_scale": probability,
        "interpretation": interpret_probability_interval(probability["interaction"], probability["interaction_95_ci"]),
        "conditional_logistic": conditional,
        "fallback_triggered": fallback is not None,
        "fallback_reason": conditional.get("reason") if fallback is not None else None,
        "gee_fallback": fallback,
        "selected_primary_model": selected_model,
        "selected_primary_result": selected,
        "matched_four_cell_vectors": [
            {"fixture_id": fixture_id, **dict(zip(common.CELL_ORDER, matrix[index].tolist()))}
            for index, fixture_id in enumerate(fixture_ids)
        ],
    }
