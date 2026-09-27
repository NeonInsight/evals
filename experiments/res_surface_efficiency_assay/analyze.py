#!/usr/bin/env python3
"""Compile the frozen primary and secondary surface-efficiency analyses."""
from __future__ import annotations

import json

import analysis_common as common
import analysis_primary
import analysis_secondary
import study

RESULTS = study.ROOT / "results"


def write_incomplete(status: dict, rows: list[dict]) -> dict:
    payload = {
        "status": "INCOMPLETE_NO_CONFIRMATORY_INFERENCE",
        "run_status": status,
        "observations_present": len(rows),
        "expected_observations": status.get("expected_observations"),
    }
    study.atomic_json(RESULTS / "analysis_status.json", payload)
    (RESULTS / "REPORT.md").write_text(
        "# RES surface-form efficiency assay\n\n"
        f"Status: **incomplete**. {len(rows)}/{status.get('expected_observations', 'unknown')} frozen observations are present.\n\n"
        "No confirmatory result is reported before every predetermined fixture has completed or exhausted its frozen technical retries.\n",
        encoding="utf-8",
    )
    return payload


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / "run_manifest.json"
    status = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "INCOMPLETE"}
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
                "surface_form": row["surface_form"],
                "compression_rank": row["compression_rank"],
                "graph_state": row["graph_state"],
                "decision_locus": row["decision_locus"],
                "expected": row["expected"],
                "technical_status": row["technical_status"],
                "parsed_choice": row.get("parsed_choice"),
                "passed": row.get("passed"),
                "decision_margin": row.get("decision_margin"),
                "character_count": row["character_count"],
                "input_token_count": row["input_token_count"],
                "output_token_count": row.get("output_token_count"),
                "character_savings_from_preserved_verbose": row["character_savings_from_preserved_verbose"],
                "input_token_savings_from_preserved_verbose": row["input_token_savings_from_preserved_verbose"],
                "levenshtein_from_preserved_verbose": row["levenshtein_from_preserved_verbose"],
                "graph_edit_distance_from_preserved": row["graph_edit_distance_from_preserved"],
                "edit_distance_from_preserved_same_form": row["edit_distance_from_preserved_same_form"],
                "input_token_delta_from_preserved_same_form": row["input_token_delta_from_preserved_same_form"],
                "rendering_hash": row["rendering_hash"],
            }
        )
    common.write_csv(RESULTS / "scored_observations.csv", selected)
    primary = analysis_primary.analyze(rows)
    secondary = analysis_secondary.analyze(rows)
    study.atomic_json(RESULTS / "primary_surface_efficiency.json", common.json_ready(primary))
    study.atomic_json(RESULTS / "secondary_graph_specificity.json", common.json_ready(secondary))

    valid = sum(row.get("technical_status") == "VALID" for row in rows)
    overall = {
        "status": "COMPLETE",
        "protocol": study.VERSION,
        "model": study.MODEL_ID,
        "revision": study.MODEL_REVISION,
        "attempted_observations": len(rows),
        "valid_observations": valid,
        "technical_invalid_observations": len(rows) - valid,
        "primary": primary,
        "secondary": secondary,
    }
    study.atomic_json(RESULTS / "results.json", common.json_ready(overall))

    compact = primary.get("compact_vs_verbose", {})
    specificity = secondary.get("graph_specificity", {})
    lines = [
        "# RES surface-form efficiency and relational-graph specificity assay",
        "",
        f"Status: **complete**. {len(rows)} predetermined observations attempted; {valid} valid and {len(rows) - valid} technically invalid.",
        "",
        "## Confirmatory primary result",
        "",
        f"Precommitted category: **{primary.get('interpretation_category', 'unavailable')}**.",
    ]
    if compact:
        lines.extend(
            [
                f"Verbose generated accuracy: {compact['baseline_accuracy']:.1%}; symbolic generated accuracy: {compact['variant_accuracy']:.1%}.",
                f"Symbolic-minus-verbose accuracy difference: {compact['difference_variant_minus_baseline']:+.1%}; one-sided 95% lower bound {compact['one_sided_95_lower_bound']:+.1%}; frozen non-inferiority boundary −{study.NONINFERIORITY_MARGIN:.0%}.",
                f"Exact choice agreement: {compact['choice_agreement_rate']:.1%} ({compact['choice_agreements']}/{compact['n']}); mean input-token savings {compact['input_token_savings']['mean']:+.2f}.",
            ]
        )
    lines.extend(["", "## Relational specificity", ""])
    if specificity.get("status") == "COMPLETE":
        lines.append(
            f"At the symbolic surface form, the decision-locus edit changed the generated choice in {specificity['relational_transition_rate']:.1%} of fixtures versus {specificity['nonrelational_transition_rate']:.1%} for the matched marker edit; one-sided paired exact McNemar p={specificity['one_sided_p']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This study tests behavioral fidelity under relation-preserving compression and behavioral sensitivity to a matched decision-locus edit. Character, token, and Levenshtein distances measure surface cost, not semantic distance. Results cannot establish an internal mechanism, consciousness, subjective experience, or human-equivalent self-representation.",
            "",
            "All stimuli, expected answers, comparisons, thresholds, exclusions, and analysis rules were frozen before target-model inference. No adaptive label or paraphrase search is permitted on these fixtures.",
            "",
        ]
    )
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    study.atomic_json(
        RESULTS / "analysis_status.json",
        {"status": "COMPLETE", "primary_category": primary.get("interpretation_category")},
    )
    result = {"status": "COMPLETE", "primary_category": primary.get("interpretation_category")}
    print(json.dumps(result, sort_keys=True))
    return overall


if __name__ == "__main__":
    main()
