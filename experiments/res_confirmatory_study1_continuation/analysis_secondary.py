"""Frozen secondary decision-margin and perturbation-specificity analyses."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

import analysis_common as common
import study

MARGIN_STRATA = (
    ("abs_m_le_0_75", 0.0, 0.75, True),
    ("0_75_lt_abs_m_le_1_50", 0.75, 1.50, False),
    ("1_50_lt_abs_m_le_3_00", 1.50, 3.00, False),
    ("abs_m_gt_3_00", 3.00, float("inf"), False),
)


def stratum(value: float) -> str:
    absolute = abs(value)
    if absolute <= 0.75:
        return "abs_m_le_0_75"
    if absolute <= 1.50:
        return "0_75_lt_abs_m_le_1_50"
    if absolute <= 3.00:
        return "1_50_lt_abs_m_le_3_00"
    return "abs_m_gt_3_00"


def rank_biserial_smaller(u: float, n_transition: int, n_stable: int) -> float:
    return float(1 - (2 * u) / (n_transition * n_stable))


def h2a(groups: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    transitions, stable = [], []
    fixture_rows = []
    for fixture_id, group in sorted(groups.items()):
        baseline = group["NEUTRAL_HIGH"]
        code = group["CODE_MAPPED_HIGH"]
        changed = bool(baseline["passed"] != code["passed"])
        margin = abs(float(baseline["decision_margin"]))
        (transitions if changed else stable).append(margin)
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "baseline_condition": "NEUTRAL_HIGH",
                "baseline_decision_margin": float(baseline["decision_margin"]),
                "absolute_baseline_margin": margin,
                "transition": changed,
                "margin_stratum": stratum(margin),
            }
        )
    if not transitions or not stable:
        return {
            "status": "INSUFFICIENT_GROUP_VARIATION",
            "transition_n": len(transitions),
            "nontransition_n": len(stable),
            "fixtures": fixture_rows,
        }
    test = mannwhitneyu(transitions, stable, alternative="less", method="auto")
    effect = rank_biserial_smaller(float(test.statistic), len(transitions), len(stable))
    rng = np.random.default_rng(study.BOOTSTRAP_SEED + 1)
    boot = []
    x, y = np.asarray(transitions), np.asarray(stable)
    for _ in range(10000):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        u = mannwhitneyu(xb, yb, alternative="less", method="asymptotic").statistic
        boot.append(rank_biserial_smaller(float(u), len(xb), len(yb)))
    return {
        "status": "COMPLETE",
        "hypothesis": "transition absolute margins are smaller",
        "test": "one-sided Mann-Whitney U",
        "alternative": "transition < nontransition",
        "u": float(test.statistic),
        "one_sided_p": float(test.pvalue),
        "transition_n": len(transitions),
        "nontransition_n": len(stable),
        "transition_median_abs_margin": float(np.median(transitions)),
        "nontransition_median_abs_margin": float(np.median(stable)),
        "rank_biserial_positive_means_transition_smaller": effect,
        "rank_biserial_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_repetitions": 10000,
        "fixtures": fixture_rows,
    }


def h2b(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fixture: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("technical_status") == "VALID":
            by_fixture.setdefault(row["fixture_id"], {})[row["condition"]] = row
    fixture_rows = []
    excluded = []
    for fixture in study.read_jsonl(study.ROOT / "fixtures.jsonl"):
        fixture_id = fixture["fixture_id"]
        group = by_fixture.get(fixture_id, {})
        if not fixture.get("h2b_control_valid", False):
            excluded.append({"fixture_id": fixture_id, "reason": "pre-inference perturbation match invalid"})
            continue
        required = {"NEUTRAL_HIGH", "CODE_MAPPED_HIGH", "NONREL_CONTROL_HIGH"}
        if not required.issubset(group):
            excluded.append({"fixture_id": fixture_id, "reason": "technical missingness", "available": sorted(group)})
            continue
        baseline, relational, control = (
            group["NEUTRAL_HIGH"],
            group["CODE_MAPPED_HIGH"],
            group["NONREL_CONTROL_HIGH"],
        )
        relational_transition = bool(baseline["passed"] != relational["passed"])
        control_transition = bool(baseline["passed"] != control["passed"])
        margin = float(baseline["decision_margin"])
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "shared_baseline_observation_id": baseline["observation_id"],
                "shared_baseline_rendering_hash": baseline["rendering_hash"],
                "baseline_decision_margin": margin,
                "absolute_baseline_margin": abs(margin),
                "margin_stratum": stratum(margin),
                "relational_transition": relational_transition,
                "nonrelational_transition": control_transition,
            }
        )
    relational_only = sum(row["relational_transition"] and not row["nonrelational_transition"] for row in fixture_rows)
    control_only = sum(row["nonrelational_transition"] and not row["relational_transition"] for row in fixture_rows)
    discordant = relational_only + control_only
    p_value = float(binomtest(relational_only, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    strata = {}
    for name, *_ in MARGIN_STRATA:
        selected = [row for row in fixture_rows if row["margin_stratum"] == name]
        strata[name] = {
            "n": len(selected),
            "relational_transitions": sum(row["relational_transition"] for row in selected),
            "nonrelational_transitions": sum(row["nonrelational_transition"] for row in selected),
            "relational_transition_rate": (
                sum(row["relational_transition"] for row in selected) / len(selected) if selected else None
            ),
            "nonrelational_transition_rate": (
                sum(row["nonrelational_transition"] for row in selected) / len(selected) if selected else None
            ),
        }
    table = Counter((row["relational_transition"], row["nonrelational_transition"]) for row in fixture_rows)
    return {
        "status": "COMPLETE" if fixture_rows else "NO_VALID_MATCHED_CONTROLS",
        "test": "one-sided paired exact McNemar",
        "alternative": "relational transition incidence > nonrelational transition incidence",
        "n": len(fixture_rows),
        "excluded": excluded,
        "paired_table": {
            "neither": table[(False, False)],
            "relational_only": relational_only,
            "nonrelational_only": control_only,
            "both": table[(True, True)],
        },
        "one_sided_p": p_value,
        "relational_transition_rate": (
            sum(row["relational_transition"] for row in fixture_rows) / len(fixture_rows) if fixture_rows else None
        ),
        "nonrelational_transition_rate": (
            sum(row["nonrelational_transition"] for row in fixture_rows) / len(fixture_rows) if fixture_rows else None
        ),
        "descriptive_margin_strata": strata,
        "fixtures": fixture_rows,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups, incomplete = common.complete_primary_groups(rows)
    return {
        "h2a_decision_margin_concentration": h2a(groups) if groups else {"status": "NO_COMPLETE_PRIMARY_FIXTURES"},
        "h2b_relational_specificity": h2b(rows),
        "incomplete_primary_fixtures": incomplete,
        "margin_definition": "logit(correct/scored decision token) minus logit(other admissible token) at the first scorable decision token; absolute value used for boundary analyses",
        "margin_strata": [name for name, *_ in MARGIN_STRATA],
    }

