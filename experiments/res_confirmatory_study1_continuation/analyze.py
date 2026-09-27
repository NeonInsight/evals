#!/usr/bin/env python3
"""Compile the frozen primary, secondary, and descriptive analyses."""
from __future__ import annotations

import json
from pathlib import Path

import analysis_common as common
import analysis_controls
import analysis_primary
import analysis_secondary
import study

RESULTS = study.ROOT / "results"


def summary_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["fixture_id"], {})[row["condition"]] = row
    output = []
    for fixture in study.read_jsonl(study.ROOT / "fixtures.jsonl"):
        fixture_id = fixture["fixture_id"]
        group = groups.get(fixture_id, {})
        row = {"fixture_id": fixture_id, "h2b_control_valid": fixture["h2b_control_valid"]}
        for condition in study.CONDITIONS:
            observation = group.get(condition)
            prefix = condition.lower()
            row[f"{prefix}_technical_status"] = observation.get("technical_status") if observation else "MISSING"
            row[f"{prefix}_pass"] = observation.get("passed") if observation else None
            row[f"{prefix}_decision_margin"] = observation.get("decision_margin") if observation else None
        output.append(row)
    return output


def write_incomplete(status, rows):
    payload = {
        "status": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
        "run_status": status,
        "observations_present": len(rows),
        "expected_observations": status.get("expected_observations"),
    }
    study.atomic_json(RESULTS / "analysis_status.json", payload)
    (RESULTS / "REPORT.md").write_text(
        "# RES Confirmatory Study 1 — relational-complexity continuation\n\n"
        f"Status: **incomplete**. {len(rows)}/{status.get('expected_observations', 'unknown')} frozen observations are present.\n\n"
        "No confirmatory or secondary inferential result is reported before every predetermined fixture has completed or exhausted its frozen technical retries. Runtime interruption is not an exclusion; the next run resumes the same sequence.\n",
        encoding="utf-8",
    )
    return payload


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_status_path = RESULTS / "run_manifest.json"
    status = json.loads(run_status_path.read_text(encoding="utf-8")) if run_status_path.exists() else {"status": "INCOMPLETE"}
    raw = common.load_observations()
    rows = common.validate_and_join(raw)
    if status.get("status") != "COMPLETE_ATTEMPTED_ALL_FIXTURES":
        result = write_incomplete(status, rows)
        print(json.dumps(result, sort_keys=True))
        return result

    selected = []
    for row in rows:
        selected.append(
            {
                "observation_id": row["observation_id"],
                "fixture_id": row["fixture_id"],
                "condition": row["condition"],
                "framing": row["framing"],
                "complexity": row["complexity"],
                "expected": row["expected"],
                "technical_status": row["technical_status"],
                "technical_failure_reason": row.get("technical_failure_reason"),
                "parsed_choice": row.get("parsed_choice"),
                "passed": row.get("passed"),
                "decision_margin": row.get("decision_margin"),
                "raw_a_logit": row.get("raw_a_logit"),
                "raw_b_logit": row.get("raw_b_logit"),
                "character_count": row["character_count"],
                "input_token_count": row["input_token_count"],
                "output_token_count": row.get("output_token_count"),
                "edit_distance_from_neutral_baseline": row["edit_distance_from_neutral_baseline"],
                **{metric: row[metric] for metric in study.STRUCTURAL_METRICS},
                "rendering_hash": row["rendering_hash"],
                "model": row.get("model"),
                "revision": row.get("revision"),
                "run_id": row.get("run_id"),
            }
        )
    common.write_csv(RESULTS / "scored_observations.csv", selected)
    common.write_csv(RESULTS / "fixture_level_matched_summary.csv", summary_rows(rows))

    primary = analysis_primary.analyze(rows)
    secondary = analysis_secondary.analyze(rows)
    controls = analysis_controls.analyze(rows)
    study.atomic_json(RESULTS / "primary_interaction_analysis.json", common.json_ready(primary))
    study.atomic_json(
        RESULTS / "decision_margin_analysis.json",
        common.json_ready(secondary["h2a_decision_margin_concentration"]),
    )
    study.atomic_json(
        RESULTS / "perturbation_specificity_analysis.json",
        common.json_ready(secondary["h2b_relational_specificity"]),
    )
    study.atomic_json(RESULTS / "length_structural_control_analysis.json", common.json_ready(controls))

    valid = sum(row.get("technical_status") == "VALID" for row in rows)
    invalid = len(rows) - valid
    overall = {
        "status": "COMPLETE",
        "protocol": study.VERSION,
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "attempted_observations": len(rows),
        "valid_observations": valid,
        "technical_invalid_observations": invalid,
        "primary": primary,
        "secondary": secondary,
        "controls": controls,
    }
    study.atomic_json(RESULTS / "results.json", common.json_ready(overall))

    probability = primary.get("probability_scale", {})
    interpretation = primary.get("interpretation", {})
    model_result = primary.get("selected_primary_result", {})
    interaction_term = model_result.get("terms", {}).get("neutral_by_high", {})
    h2a = secondary["h2a_decision_margin_concentration"]
    h2b = secondary["h2b_relational_specificity"]
    lines = [
        "# RES Confirmatory Study 1 — relational-complexity continuation",
        "",
        f"Status: **complete**. {len(rows)} predetermined observations attempted; {valid} valid and {invalid} technically invalid after frozen retries.",
        "",
        "## Preregistered hypotheses",
        "",
        "The primary hypothesis predicted a positive interaction: the Neutral advantage over the frozen v1 code-mapped representation would be larger at HIGH than LOW relational complexity. H2a predicted smaller Neutral-HIGH baseline margins among relational-framing transitions. H2b predicted more transitions from relational framing than from the matched non-relational perturbation sharing the same Neutral-HIGH baseline.",
        "",
        "## Confirmatory primary result",
        "",
    ]
    if probability:
        lines.extend(
            [
                f"FramingEffect_LOW = {probability['framing_effect_low']:+.3f} (95% fixture-bootstrap CI {probability['framing_effect_low_95_ci'][0]:+.3f} to {probability['framing_effect_low_95_ci'][1]:+.3f}).",
                f"FramingEffect_HIGH = {probability['framing_effect_high']:+.3f} (95% fixture-bootstrap CI {probability['framing_effect_high_95_ci'][0]:+.3f} to {probability['framing_effect_high_95_ci'][1]:+.3f}).",
                f"Probability interaction = {probability['interaction']:+.3f} (95% fixture-bootstrap CI {probability['interaction_95_ci'][0]:+.3f} to {probability['interaction_95_ci'][1]:+.3f}).",
                "",
                f"Precommitted category: **{interpretation.get('category', 'unavailable')}**. {interpretation.get('statement', '')}",
            ]
        )
    if interaction_term:
        lines.extend(
            [
                "",
                f"Selected primary model: {primary['selected_primary_model']}. Neutral×HIGH log-odds={interaction_term['log_odds']:+.3f}, OR={interaction_term['odds_ratio']:.3f}, 95% OR CI {interaction_term['ci_95_odds_ratio'][0]:.3f}–{interaction_term['ci_95_odds_ratio'][1]:.3f}, two-sided p={interaction_term['two_sided_wald_p']:.4g}.",
            ]
        )
    lines.extend(["", "## Preregistered secondary results", ""])
    if h2a.get("status") == "COMPLETE":
        lines.append(
            f"H2a: one-sided Mann–Whitney U={h2a['u']:.1f}, p={h2a['one_sided_p']:.4g}; rank-biserial effect={h2a['rank_biserial_positive_means_transition_smaller']:+.3f} (95% bootstrap CI {h2a['rank_biserial_bootstrap_95_ci'][0]:+.3f} to {h2a['rank_biserial_bootstrap_95_ci'][1]:+.3f})."
        )
    else:
        lines.append(f"H2a status: {h2a.get('status')}.")
    lines.append(
        f"H2b: one-sided paired exact McNemar p={h2b.get('one_sided_p')}; relational transition rate={h2b.get('relational_transition_rate')}, non-relational transition rate={h2b.get('nonrelational_transition_rate')}, valid matched controls n={h2b.get('n')}."
    )
    lines.extend(
        [
            "",
            "## Retrospective motivation analysis",
            "",
            "The v1-versus-replication structural comparison was completed before this run and is recorded in `../historical_structural_comparison.md`. It is retrospective and non-confirmatory; it did not select fixtures, thresholds, exclusions, or hypotheses.",
            "",
            "## Descriptive analyses",
            "",
            "Prompt length, output length, edit distance, and all frozen structural metrics are reported in `length_structural_control_analysis.json`. Margin-stratified H2b rates are descriptive. These analyses did not alter any stimulus or primary test.",
            "",
            "## Replicated and novel results",
            "",
            "The historical 17/32→26/32 v1 result and the later 35/64-versus-38/64 replication remain prior evidence. Every estimate from this continuation is a new result from the frozen matched complexity design; it should not be described as independently replicated unless a later frozen study reproduces it.",
            "",
            "## Exploratory interpretations",
            "",
            "No exploratory mechanism claim is designated as confirmatory. Any later label search must use held-out fixtures and remain a separate research phase.",
            "",
            "## Limitations",
            "",
            "This is one pinned model and prompt family. Conditional likelihood can receive little information from all-pass and all-fail fixtures. Surface and structural manipulations cannot uniquely identify an internal mechanism. The study does not test consciousness, subjective experience, self-awareness, or human-equivalent self-representation.",
            "",
        ]
    )
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    study.atomic_json(
        RESULTS / "analysis_status.json",
        {
            "status": "COMPLETE",
            "primary_category": interpretation.get("category"),
            "fallback_triggered": primary.get("fallback_triggered"),
        },
    )
    print(json.dumps({"status": "COMPLETE", "primary_category": interpretation.get("category")}, sort_keys=True))
    return overall


if __name__ == "__main__":
    main()

