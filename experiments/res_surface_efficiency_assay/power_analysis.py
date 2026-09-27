#!/usr/bin/env python3
"""Prospective paired non-inferiority power calculation for Phase 0."""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import norm

import study

ALPHA = 0.05
TARGET_POWER = 0.80
CANDIDATES = (64, 128, 192, 256)
MINIMUM_DESIGN_DISCORDANCE = 0.20
SENSITIVITY_DISCORDANCE = (0.15, 0.20, 0.25, 0.33)


def achieved_power(n: int, discordance: float, margin: float) -> float:
    """Normal approximation for paired risk-difference non-inferiority.

    Under the design alternative the true compact-minus-verbose accuracy
    difference is zero.  The standard error is approximated by sqrt(q/n),
    where q is total paired discordance.
    """
    z_alpha = float(norm.ppf(1 - ALPHA))
    signal = margin * math.sqrt(n / discordance)
    return float(norm.cdf(signal - z_alpha))


def required_n(discordance: float, margin: float) -> float:
    z_alpha = float(norm.ppf(1 - ALPHA))
    z_power = float(norm.ppf(TARGET_POWER))
    return discordance * ((z_alpha + z_power) / margin) ** 2


def prior_disagreement() -> float:
    path = (
        study.REPO
        / "experiments/res_confirmatory_study1_continuation/results/"
        "posthoc_surface_efficiency_analysis.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    agreement = payload["comparisons"]["code_mapped_low_vs_neutral_low"]["parsed_choice_agreement_rate"]
    return float(1 - agreement)


def main() -> None:
    historical = prior_disagreement()
    design_discordance = max(MINIMUM_DESIGN_DISCORDANCE, historical)
    table = {
        str(n): {
            str(q): achieved_power(n, q, study.NONINFERIORITY_MARGIN)
            for q in SENSITIVITY_DISCORDANCE
        }
        for n in CANDIDATES
    }
    selected = next(
        n for n in CANDIDATES
        if achieved_power(n, design_discordance, study.NONINFERIORITY_MARGIN) >= TARGET_POWER
    )
    if selected != study.SCIENTIFIC_N:
        raise RuntimeError("Study SCIENTIFIC_N does not equal the prospective power selection")
    payload = {
        "status": "PROSPECTIVE_PHASE0_POWER_COMPLETE",
        "analysis": "paired generated-accuracy non-inferiority",
        "null_boundary": -study.NONINFERIORITY_MARGIN,
        "design_alternative": 0.0,
        "alpha": ALPHA,
        "sidedness": "one-sided",
        "target_power": TARGET_POWER,
        "historical_low_representation_disagreement": historical,
        "historical_role": "design motivation only; no current outcomes",
        "design_discordance": design_discordance,
        "design_discordance_rule": "max(0.20, prior LOW representation disagreement)",
        "required_n_continuous": required_n(design_discordance, study.NONINFERIORITY_MARGIN),
        "candidate_n": list(CANDIDATES),
        "selected_n": selected,
        "achieved_power_grid": table,
        "sensitivity_discordance": list(SENSITIVITY_DISCORDANCE),
        "caveat": (
            "The 0.25 and 0.33 discordance scenarios are sensitivity analyses; "
            "they are not used to shrink or expand N after scoring."
        ),
    }
    study.atomic_json(study.ROOT / "power_analysis.json", payload)
    lines = [
        "# Prospective power analysis",
        "",
        "This calculation was completed before target-model inference. The primary generated-accuracy endpoint is the paired compact-minus-verbose risk difference, tested for non-inferiority against −0.10 at one-sided α=.05.",
        "",
        f"The prior LOW representation pair had {historical:.1%} choice disagreement. The design uses max(20%, historical), giving a frozen discordance assumption of {design_discordance:.0%}. At a true difference of zero, the continuous normal-approximation requirement is {payload['required_n_continuous']:.1f} fixtures; the first frozen candidate reaching 80% power is **N={selected}**.",
        "",
        "| N | q=.15 | q=.20 | q=.25 | q=.33 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in CANDIDATES:
        values = table[str(n)]
        lines.append(
            f"| {n} | {values['0.15']:.1%} | {values['0.2']:.1%} | "
            f"{values['0.25']:.1%} | {values['0.33']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The higher-discordance columns are declared sensitivity analyses. N remains 128 regardless of their values or any later observed result. Choice agreement, graph specificity, and margin stability are additional frozen outcomes and do not adapt sample size.",
            "",
        ]
    )
    (study.ROOT / "power_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected_n": selected}, sort_keys=True))


if __name__ == "__main__":
    main()
