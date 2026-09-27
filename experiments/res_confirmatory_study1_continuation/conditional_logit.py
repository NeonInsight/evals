"""Exact four-cell fixture-stratified conditional logistic regression.

The fixed design has four primary rows per fixture.  Conditioning on each
fixture's success total makes the nuisance fixture intercept disappear.  This
implementation keeps the primary fit reproducible without selecting among
statistical packages after results are visible.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

# Cell order: CODE_MAPPED_LOW, NEUTRAL_LOW, CODE_MAPPED_HIGH, NEUTRAL_HIGH.
# Coding makes the predicted neutral-by-high interaction positive.
DESIGN = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    )
)
TERMS = ("neutral", "high", "neutral_by_high")
SUBSET_STATS = {
    k: np.asarray([DESIGN[list(choice)].sum(axis=0) for choice in itertools.combinations(range(4), k)])
    for k in (1, 2, 3)
}


def _components(beta: np.ndarray, totals: dict[int, int], observed_stat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    log_likelihood = float(beta @ observed_stat)
    gradient = observed_stat.astype(float).copy()
    hessian = np.zeros((3, 3), dtype=float)
    for k, count in totals.items():
        if not count:
            continue
        stats = SUBSET_STATS[k]
        linear = stats @ beta
        weights = np.exp(linear - logsumexp(linear))
        mean = weights @ stats
        centered = stats - mean
        covariance = centered.T @ (weights[:, None] * centered)
        log_likelihood -= count * float(logsumexp(linear))
        gradient -= count * mean
        hessian -= count * covariance
    return log_likelihood, gradient, hessian


def fit_conditional_logit(outcomes: np.ndarray, max_iterations: int = 100) -> dict[str, Any]:
    outcomes = np.asarray(outcomes, dtype=int)
    if outcomes.ndim != 2 or outcomes.shape[1] != 4 or not np.isin(outcomes, (0, 1)).all():
        raise ValueError("outcomes must be a binary fixture-by-four matrix")
    success_totals = outcomes.sum(axis=1)
    informative = (success_totals > 0) & (success_totals < 4)
    used = outcomes[informative]
    totals = {k: int(np.sum(success_totals[informative] == k)) for k in (1, 2, 3)}
    if len(used) < 4:
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": "fewer than four informative fixture strata",
            "n_fixtures": int(len(outcomes)),
            "n_informative_fixtures": int(len(used)),
            "degenerate_all_fail": int(np.sum(success_totals == 0)),
            "degenerate_all_pass": int(np.sum(success_totals == 4)),
        }
    observed_stat = (used @ DESIGN).sum(axis=0)
    beta = np.zeros(3, dtype=float)
    converged = False
    reason = "maximum iterations reached"
    for iteration in range(1, max_iterations + 1):
        value, gradient, hessian = _components(beta, totals, observed_stat)
        information = -hessian
        if not np.isfinite(information).all() or np.linalg.cond(information) > 1e12:
            reason = "singular or ill-conditioned conditional information matrix"
            break
        step = np.linalg.solve(information, gradient)
        if not np.isfinite(step).all():
            reason = "non-finite Newton step"
            break
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta + scale * step
            candidate_value, _, _ = _components(candidate, totals, observed_stat)
            if np.isfinite(candidate_value) and candidate_value >= value - 1e-12:
                beta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            reason = "conditional-likelihood line search failed"
            break
        _, new_gradient, _ = _components(beta, totals, observed_stat)
        if np.max(np.abs(new_gradient)) < 1e-8 or np.max(np.abs(scale * step)) < 1e-9:
            converged = True
            reason = "gradient/step tolerance reached"
            break
        if np.max(np.abs(beta)) > 30:
            reason = "separation produced an unbounded estimate"
            break
    value, gradient, hessian = _components(beta, totals, observed_stat)
    if not converged:
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": reason,
            "iterations": iteration,
            "n_fixtures": int(len(outcomes)),
            "n_informative_fixtures": int(len(used)),
            "degenerate_all_fail": int(np.sum(success_totals == 0)),
            "degenerate_all_pass": int(np.sum(success_totals == 4)),
            "last_coefficients": beta.tolist(),
        }
    try:
        covariance = np.linalg.inv(-hessian)
    except np.linalg.LinAlgError:
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": "final conditional information matrix is singular",
            "n_fixtures": int(len(outcomes)),
            "n_informative_fixtures": int(len(used)),
        }
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": "final covariance is non-finite or non-positive",
            "n_fixtures": int(len(outcomes)),
            "n_informative_fixtures": int(len(used)),
        }
    standard_errors = np.sqrt(np.diag(covariance))
    if np.max(np.abs(beta)) > 30 or np.max(standard_errors) > 30:
        return {
            "status": "COMPUTATIONAL_FAILURE",
            "reason": "separation produced an effectively non-finite estimate or interval",
            "n_fixtures": int(len(outcomes)),
            "n_informative_fixtures": int(len(used)),
            "last_coefficients": beta.tolist(),
            "last_standard_errors": standard_errors.tolist(),
        }
    result_terms = {}
    for index, name in enumerate(TERMS):
        estimate = float(beta[index])
        standard_error = float(math.sqrt(covariance[index, index]))
        lower, upper = estimate - 1.959963984540054 * standard_error, estimate + 1.959963984540054 * standard_error
        z = estimate / standard_error
        result_terms[name] = {
            "log_odds": estimate,
            "standard_error": standard_error,
            "odds_ratio": math.exp(estimate),
            "ci_95_log_odds": [lower, upper],
            "ci_95_odds_ratio": [math.exp(lower), math.exp(upper)],
            "two_sided_wald_p": float(2 * norm.sf(abs(z))),
        }
    return {
        "status": "FIT_CONVERGED",
        "iterations": iteration,
        "convergence_reason": reason,
        "log_likelihood": float(value),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "n_fixtures": int(len(outcomes)),
        "n_informative_fixtures": int(len(used)),
        "degenerate_all_fail": int(np.sum(success_totals == 0)),
        "degenerate_all_pass": int(np.sum(success_totals == 4)),
        "conditional_success_totals": totals,
        "terms": result_terms,
        "covariance": covariance.tolist(),
    }
