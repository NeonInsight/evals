#!/usr/bin/env python3
"""Phase 0A power and minimum-detectable-effect simulation.

Inputs are frozen historical RES observations only.  Correlated Bernoulli
fixtures are generated with a Gaussian copula whose tetrachoric correlations
are estimated from the completed prior four-cell RES study.  The simulation
uses the same fixture-stratified conditional logistic interaction test that is
frozen for the continuation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

import conditional_logit
import study

PRIOR_FOUR_CELL = study.REPO / "experiments/res_confirmatory_study1/results/scored_observations.csv"
PRIOR_V1_CODE = study.CANONICAL_RESULT_PATH
PRIOR_V1_NEUTRAL = study.CANONICAL_NEUTRAL_RESULT_PATH
PRIOR_REPLICATION = study.REPO / "experiments/res_complexity_ladder/results/report.json"
CANDIDATE_NS = (64, 128, 192, 256)
ALPHA = 0.05
TARGET_POWER = 0.80
SESOI = 0.15
POWER_REPLICATES = 3000
MDE_REPLICATES = 1500
POWER_SEED = 2026092711
MDE_SEED = 2026092712

# Cell order is CODE_MAPPED_LOW, NEUTRAL_LOW, CODE_MAPPED_HIGH, NEUTRAL_HIGH.
SCENARIOS = {
    "replication_anchored_zero_symmetric": {
        "cell_probabilities": [0.62, 0.545, 0.50, 0.575],
        "rationale": "Aggregate framing effect is zero; LOW=-0.075 and HIGH=+0.075.",
    },
    "replication_anchored_zero_nearby_positive": {
        "cell_probabilities": [0.60, 0.55, 0.50, 0.60],
        "rationale": "Aggregate framing effect is +0.025; LOW=-0.050 and HIGH=+0.100.",
    },
    "replication_anchored_zero_nearby_negative": {
        "cell_probabilities": [0.62, 0.52, 0.50, 0.55],
        "rationale": "Aggregate framing effect is -0.025; LOW=-0.100 and HIGH=+0.050.",
    },
    "v1_positive_main_effect_allocation": {
        "cell_probabilities": [0.55, 0.625, 0.45, 0.675],
        "rationale": "Positive framing effects at both levels; LOW=+0.075 and HIGH=+0.225.",
    },
}
MDE_GRID = (0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24)


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def historical_inputs() -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(PRIOR_FOUR_CELL)
    wide = frame.pivot(index="fixture_id", columns="condition", values="pass")[["EL", "NL", "EH", "NH"]]
    values = wide.to_numpy(dtype=int)
    cell_rates = values.mean(axis=0)
    totals = values.sum(axis=1)
    code = json.loads(PRIOR_V1_CODE.read_text(encoding="utf-8"))
    neutral = json.loads(PRIOR_V1_NEUTRAL.read_text(encoding="utf-8"))
    replication = json.loads(PRIOR_REPLICATION.read_text(encoding="utf-8"))
    # The report schema is retained verbatim as an input, while the documented
    # generated result is explicitly recorded below for auditability.
    prior = {
        "files": {
            str(PRIOR_FOUR_CELL.relative_to(study.REPO)): study.file_hash(PRIOR_FOUR_CELL),
            str(PRIOR_V1_CODE.relative_to(study.REPO)): study.file_hash(PRIOR_V1_CODE),
            str(PRIOR_V1_NEUTRAL.relative_to(study.REPO)): study.file_hash(PRIOR_V1_NEUTRAL),
            str(PRIOR_REPLICATION.relative_to(study.REPO)): study.file_hash(PRIOR_REPLICATION),
        },
        "prior_four_cell_n": int(len(values)),
        "prior_four_cell_pass_probabilities": dict(zip(("CM_LOW", "N_LOW", "CM_HIGH", "N_HIGH"), cell_rates.tolist())),
        "prior_binary_correlations": np.corrcoef(values.T).tolist(),
        "prior_framing_discordance": {
            "low": float(np.mean(values[:, 0] != values[:, 1])),
            "high": float(np.mean(values[:, 2] != values[:, 3])),
        },
        "prior_complexity_discordance": {
            "code_mapped": float(np.mean(values[:, 0] != values[:, 2])),
            "neutral": float(np.mean(values[:, 1] != values[:, 3])),
        },
        "prior_degenerate_rates": {
            "all_fail": float(np.mean(totals == 0)),
            "all_pass": float(np.mean(totals == 4)),
            "informative": float(np.mean((totals > 0) & (totals < 4))),
        },
        "v1_generated": {
            "code_mapped": sum(r["generated_choice"] == r["ideal"] for r in code["fixtures"]),
            "neutral": sum(r["generated_choice"] == r["ideal"] for r in neutral["fixtures"]),
            "n": 32,
        },
        "fresh_replication_generated": {
            "loaded": 35,
            "neutral": 38,
            "n": 64,
            "paired_result_role": "approximately-null framing-main-effect anchor only",
        },
        "fresh_replication_file_schema": replication.get("schema_version"),
    }
    return values, prior


def tetrachoric_correlation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = values.mean(axis=0)
    joint = (values[:, :, None] * values[:, None, :]).mean(axis=0)
    correlation = np.eye(4)
    for i in range(4):
        for j in range(i):
            threshold_i, threshold_j = norm.ppf(probabilities[i]), norm.ppf(probabilities[j])
            target = joint[i, j]

            def objective(rho: float) -> float:
                covariance = [[1.0, rho], [rho, 1.0]]
                return float(multivariate_normal.cdf([threshold_i, threshold_j], mean=[0, 0], cov=covariance) - target)

            correlation[i, j] = correlation[j, i] = brentq(objective, -0.999, 0.999)
    # Finite historical data can yield a non-positive joint tetrachoric matrix.
    # The eigenvalue floor is fixed prospectively and reported.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    adjusted = np.maximum(eigenvalues, 0.03)
    correlation = eigenvectors @ np.diag(adjusted) @ eigenvectors.T
    scale = np.sqrt(np.diag(correlation))
    correlation = correlation / scale[:, None] / scale[None, :]
    return correlation, eigenvalues


def simulate_once(rng: np.random.Generator, n: int, probabilities: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    latent = rng.multivariate_normal(np.zeros(4), correlation, size=n)
    return (latent < norm.ppf(probabilities)).astype(int)


def scenario_metrics(outcomes: np.ndarray) -> dict[str, Any]:
    totals = outcomes.sum(axis=1)
    return {
        "cell_pass_probabilities": outcomes.mean(axis=0).tolist(),
        "binary_correlations": np.corrcoef(outcomes.T).tolist(),
        "framing_discordance": {
            "low": float(np.mean(outcomes[:, 0] != outcomes[:, 1])),
            "high": float(np.mean(outcomes[:, 2] != outcomes[:, 3])),
        },
        "complexity_discordance": {
            "code_mapped": float(np.mean(outcomes[:, 0] != outcomes[:, 2])),
            "neutral": float(np.mean(outcomes[:, 1] != outcomes[:, 3])),
        },
        "degenerate_rates": {
            "all_fail": float(np.mean(totals == 0)),
            "all_pass": float(np.mean(totals == 4)),
            "informative": float(np.mean((totals > 0) & (totals < 4))),
        },
    }


def power_for(n: int, probabilities: np.ndarray, correlation: np.ndarray, repetitions: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rejections = 0
    converged = 0
    positive = 0
    informative = []
    for _ in range(repetitions):
        outcomes = simulate_once(rng, n, probabilities, correlation)
        fit = conditional_logit.fit_conditional_logit(outcomes)
        informative.append(fit.get("n_informative_fixtures", 0))
        if fit["status"] != "FIT_CONVERGED":
            continue
        converged += 1
        term = fit["terms"]["neutral_by_high"]
        positive += term["log_odds"] > 0
        rejections += term["two_sided_wald_p"] < ALPHA
    return {
        "n": n,
        "repetitions": repetitions,
        "rejections": rejections,
        "achieved_power": rejections / repetitions,
        "monte_carlo_wilson_95_ci": wilson(rejections, repetitions),
        "convergence_rate": converged / repetitions,
        "positive_estimate_rate": positive / repetitions,
        "mean_informative_fixtures": float(np.mean(informative)),
        "mean_informative_fraction": float(np.mean(informative) / n),
    }


def probability_interaction(probabilities: list[float] | np.ndarray) -> float:
    cm_low, n_low, cm_high, n_high = probabilities
    return float((n_high - cm_high) - (n_low - cm_low))


def main() -> dict[str, Any]:
    values, prior = historical_inputs()
    correlation, raw_eigenvalues = tetrachoric_correlation(values)
    scenario_results: dict[str, Any] = {}
    for scenario_index, (name, specification) in enumerate(SCENARIOS.items()):
        probabilities = np.asarray(specification["cell_probabilities"], dtype=float)
        interaction = probability_interaction(probabilities)
        if not math.isclose(interaction, SESOI, abs_tol=1e-12):
            raise ValueError(f"{name} does not implement the frozen SESOI")
        by_n = {}
        for n_index, n in enumerate(CANDIDATE_NS):
            by_n[str(n)] = power_for(
                n,
                probabilities,
                correlation,
                POWER_REPLICATES,
                POWER_SEED + 1000 * scenario_index + n_index,
            )
        diagnostic_rng = np.random.default_rng(POWER_SEED + 9000 + scenario_index)
        diagnostics = scenario_metrics(simulate_once(diagnostic_rng, 200000, probabilities, correlation))
        scenario_results[name] = {
            **specification,
            "probability_scale_interaction": interaction,
            "aggregate_framing_main_effect": float(((probabilities[1] + probabilities[3]) - (probabilities[0] + probabilities[2])) / 2),
            "expected_design_diagnostics": diagnostics,
            "power_by_n": by_n,
        }

    eligible = []
    for n in CANDIDATE_NS:
        if all(result["power_by_n"][str(n)]["monte_carlo_wilson_95_ci"][0] >= TARGET_POWER for result in scenario_results.values()):
            eligible.append(n)
    selected_n = min(eligible) if eligible else max(CANDIDATE_NS)
    selection_rule = (
        "smallest candidate N whose 95% Monte Carlo lower bound is at least 0.80 in every SESOI scenario"
        if eligible
        else "largest evaluated computationally feasible N; 80% SESOI power was not established"
    )

    mde_results = {}
    for n_index, n in enumerate(CANDIDATE_NS):
        grid = {}
        for effect_index, effect in enumerate(MDE_GRID):
            probabilities = np.asarray([0.62, 0.62 - effect / 2, 0.50, 0.50 + effect / 2])
            grid[f"{effect:.2f}"] = power_for(
                n,
                probabilities,
                correlation,
                MDE_REPLICATES,
                MDE_SEED + 1000 * n_index + effect_index,
            )
        passing = [float(effect) for effect, result in grid.items() if result["monte_carlo_wilson_95_ci"][0] >= TARGET_POWER]
        mde_results[str(n)] = {
            "grid": grid,
            "mde_grid_estimate": min(passing) if passing else None,
            "definition": "smallest simulated probability interaction whose Monte Carlo 95% lower bound reaches 0.80",
        }

    payload = {
        "schema_version": "res-cs1-continuation-power-v1",
        "alpha": ALPHA,
        "alpha_sidedness": "two-sided",
        "target_power": TARGET_POWER,
        "sesoi_probability_interaction": SESOI,
        "candidate_n": list(CANDIDATE_NS),
        "power_replicates_per_cell": POWER_REPLICATES,
        "mde_replicates_per_cell": MDE_REPLICATES,
        "seeds": {"power": POWER_SEED, "mde": MDE_SEED},
        "historical_inputs": prior,
        "copula": {
            "method": "Gaussian copula with pairwise tetrachoric correlations from prior four-cell RES outcomes",
            "cell_order": ["CODE_MAPPED_LOW", "NEUTRAL_LOW", "CODE_MAPPED_HIGH", "NEUTRAL_HIGH"],
            "raw_tetrachoric_eigenvalues": raw_eigenvalues.tolist(),
            "fixed_nearest_correlation_eigenvalue_floor": 0.03,
            "correlation_matrix": correlation.tolist(),
        },
        "test": "fixture-stratified conditional logistic neutral-by-high Wald test",
        "scenario_results": scenario_results,
        "mde_results": mde_results,
        "selected_n": selected_n,
        "selection_rule": selection_rule,
        "computational_feasibility_ceiling": 256,
        "feasibility_basis": "At N=256 the fixed design requires 1,280 CPU target-model evaluations in 32 eight-fixture shards; larger Ns were not authorized for this run budget.",
    }
    study.atomic_json(study.ROOT / "power_analysis.json", payload)

    lines = [
        "# Phase 0A power and MDE analysis",
        "",
        f"Primary test: two-sided fixture-stratified conditional logistic interaction, α={ALPHA:.2f}. Target power={TARGET_POWER:.2f}. SESOI={SESOI:.2f} absolute probability interaction.",
        "",
        "Only earlier RES observations were used. The prior four-cell outcomes calibrate within-fixture dependence and degenerate fixture prevalence; the original 17/32 → 26/32 result and fresh 35/64 versus 38/64 replication anchor plausible pass rates and main effects.",
        "",
        "## Prior-data inputs",
        "",
        f"Prior four-cell degenerate rates: all-fail {prior['prior_degenerate_rates']['all_fail']:.3f}, all-pass {prior['prior_degenerate_rates']['all_pass']:.3f}, informative {prior['prior_degenerate_rates']['informative']:.3f}.",
        f"Prior framing discordance: LOW {prior['prior_framing_discordance']['low']:.3f}, HIGH {prior['prior_framing_discordance']['high']:.3f}.",
        "",
        "## Achieved power at SESOI",
        "",
        "| Scenario | Aggregate framing effect | N=64 | N=128 | N=192 | N=256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in scenario_results.items():
        powers = [result["power_by_n"][str(n)]["achieved_power"] for n in CANDIDATE_NS]
        lines.append(
            f"| {name} | {result['aggregate_framing_main_effect']:+.3f} | "
            + " | ".join(f"{power:.3f}" for power in powers)
            + " |"
        )
    lines.extend(
        [
            "",
            "All-pass and all-fail fixtures were simulated at their empirically calibrated prevalence. They remain in nominal N and contribute no conditional-likelihood information. Each scenario's JSON diagnostics report cell probabilities, correlations, framing and complexity discordance, and degenerate rates.",
            "",
            "## MDE grid",
            "",
            "| N | MDE grid estimate |",
            "| ---: | ---: |",
        ]
    )
    for n in CANDIDATE_NS:
        estimate = mde_results[str(n)]["mde_grid_estimate"]
        lines.append(f"| {n} | {estimate if estimate is not None else 'above 0.24'} |")
    lines.extend(
        [
            "",
            "## Frozen sample size",
            "",
            f"Selected N: **{selected_n} fixtures**.",
            "",
            f"Selection rule: {selection_rule}.",
            "",
            "The MDE is a grid estimate under the replication-anchored symmetric allocation and calibrated dependence structure. Monte Carlo uncertainty and model misspecification remain limitations.",
            "",
        ]
    )
    (study.ROOT / "power_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps({"selected_n": result["selected_n"], "selection_rule": result["selection_rule"]}))

