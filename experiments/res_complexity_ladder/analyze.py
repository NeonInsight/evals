"""Complete-stage analysis. Exactly one confirmatory test: generated LL vs NN at k=5."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import warnings

import protocol as p
from run import load_checkpoint


def mcnemar_exact(corrections, regressions):
    n = corrections + regressions
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(corrections, regressions) + 1)) / 2 ** n) if n else 1.0


def exact_ci(successes, n, alpha=0.05):
    if not 0 <= successes <= n or n < 1:
        raise ValueError("Invalid binomial count")
    def root(target, tail, increasing):
        low, high = 0.0, 1.0
        for _ in range(65):
            mid = (low + high) / 2
            value = sum(math.comb(n, j) * mid ** j * (1 - mid) ** (n - j) for j in tail)
            if (value < target) == increasing:
                low = mid
            else:
                high = mid
        return (low + high) / 2
    lower = 0.0 if successes == 0 else root(alpha / 2, range(successes, n + 1), True)
    upper = 1.0 if successes == n else root(alpha / 2, range(successes + 1), False)
    return [lower, upper]


def paired(records, left, right, endpoint="generated_correct"):
    indexed = {(r["fixture"]["fixture_id"], r["fixture"]["cell"]): r for r in records}
    table = {"both_correct": 0, "corrections": 0, "regressions": 0, "both_wrong": 0}
    for fid in {r["fixture"]["fixture_id"] for r in records}:
        a, b = indexed[(fid, left)][endpoint], indexed[(fid, right)][endpoint]
        key = ("both_correct" if b else "regressions") if a else ("corrections" if b else "both_wrong")
        table[key] += 1
    table["accuracy_difference_right_minus_left"] = (table["corrections"] - table["regressions"]) / (len(records) // 4)
    return table


def cell_summary(records):
    n = len(records)
    correct = sum(r["generated_correct"] for r in records)
    coverage = sum(r["parsed"] is not None for r in records)
    logits = sum(r["logit_correct"] for r in records)
    return dict(n=n, generated_correct=correct, generated_accuracy=correct / n,
                generated_exact_95_ci=exact_ci(correct, n), parser_coverage=coverage / n,
                strict_parser_coverage=sum(r["strict_parse"] is not None for r in records) / n,
                descriptive_criterion_met=correct / n >= .75 and coverage / n >= .95,
                logit_correct=logits, logit_accuracy=logits / n, logit_exact_95_ci=exact_ci(logits, n),
                raw_zero_margin=sum(r["raw_ab_margin"] == 0 for r in records),
                generated_logit_choice_agreement=sum(r["parsed"] == r["logit_choice"] for r in records) / n,
                near_boundary_fraction=sum(abs(r["raw_ab_margin"]) <= .75 for r in records) / n,
                generated_B_fraction=sum(r["parsed"] == "B" for r in records) / n,
                median_input_tokens=statistics.median(r["input_tokens"] for r in records))


def near_boundary(records):
    indexed = {(r["fixture"]["fixture_id"], r["fixture"]["cell"]): r for r in records}
    margins = {"discordant": [], "concordant": [], "regressions": []}
    for fid in {r["fixture"]["fixture_id"] for r in records}:
        a, b = indexed[(fid, "LL")], indexed[(fid, "NN")]
        kind = "concordant" if a["logit_choice"] == b["logit_choice"] else "discordant"
        margins[kind].append(abs(a["raw_ab_margin"]))
        if a["logit_correct"] and not b["logit_correct"]:
            margins["regressions"].append(abs(a["raw_ab_margin"]))
    regressions = margins["regressions"]
    return dict(threshold_abs_raw_LL_margin=.75,
                median_LL_abs_margin={key: statistics.median(value) if value else None for key, value in margins.items()},
                regressions=len(regressions),
                regressions_with_LL_abs_margin_at_most_threshold=sum(m <= .75 for m in regressions),
                fraction_of_regressions_near_boundary=(sum(m <= .75 for m in regressions) / len(regressions)) if regressions else None,
                interpretation="Preregistered output-margin diagnostic, not evidence of an internal mechanism.")


def random_intercept_model(records):
    """Secondary approximate Bayesian GLMM; no additional confirmatory p-values."""
    import numpy as np
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    frame = pd.DataFrame([dict(correct=int(r["generated_correct"]), fixture_id=r["fixture"]["fixture_id"],
                               label_loaded=int(r["fixture"]["cell"][0] == "L"),
                               task_loaded=int(r["fixture"]["cell"][1] == "L")) for r in records])
    specification = dict(method="BinomialBayesMixedGLM.fit_vb (mean-field variational Bayes)",
                         formula="correct ~ label_loaded * task_loaded + (1 | fixture_id)",
                         fixed_effect_prior_sd=2.0, random_effect_log_sd_prior_sd=.5,
                         note="Secondary localization; approximate marginal posterior intervals, not confirmatory tests.")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = BinomialBayesMixedGLM.from_formula(
                "correct ~ label_loaded * task_loaded", {"fixture": "0 + C(fixture_id)"},
                frame, vcp_p=.5, fe_p=2.0)
            # Explicit deterministic initial values; do not depend on a version-specific RNG API.
            dimension = model.k_fep + model.k_vcp + model.k_vc
            fit = model.fit_vb(mean=np.zeros(dimension), sd=np.full(dimension, .6),
                               minim_opts={"maxiter": 1000})
        if not fit.optim_retvals["success"] or not all(math.isfinite(x) for x in fit.fe_mean):
            return dict(specification, status="FIT_FAILED", reason=str(fit.optim_retvals["message"]))
        estimates = {name: dict(log_odds_mean=float(mean), posterior_sd=float(sd),
                                approximate_95_interval=[float(mean - 1.96 * sd), float(mean + 1.96 * sd)])
                     for name, mean, sd in zip(model.exog_names, fit.fe_mean, fit.fe_sd)}
        return dict(specification, status="FIT_WITH_WARNINGS" if caught else "FIT_OK",
                    effects=estimates, warnings=[str(w.message) for w in caught],
                    fixture_log_sd_mean=float(fit.vcp_mean[0]))
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return dict(specification, status="FIT_FAILED", reason=str(exc))


def analyze_stage(k, records, fit_secondary=True):
    expected = {r["row_id"] for r in p.rows_for(k)}
    observed = [r["fixture"]["row_id"] for r in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Inference requires exactly one observation of every preregistered row")
    cells = {cell: cell_summary([r for r in records if r["fixture"]["cell"] == cell]) for cell in p.CELLS}
    pairs = {f"{a}_to_{b}": paired(records, a, b) for a, b in itertools.combinations(p.CELLS, 2)}
    stratified = {}
    for key in ("ideal", "allow_code", "difficulty", "family", "persona", "peer_congruent"):
        stratified[key] = {cell: {str(value): cell_summary([r for r in records if r["fixture"]["cell"] == cell and r["fixture"][key] == value])
                                for value in sorted({r["fixture"][key] for r in records})} for cell in p.CELLS}
    primary = dict(status="NOT_CONFIRMATORY_AT_THIS_LEVEL")
    if k == 5:
        main = pairs["LL_to_NN"]
        value = mcnemar_exact(main["corrections"], main["regressions"])
        primary = dict(status="COMPLETE", endpoint="generated accuracy", contrast="LL vs NN",
                       test="exact two-sided McNemar", alpha=.05, p_value=value,
                       reject_equal_accuracy=value < .05,
                       neutral_benefit_replicated=value < .05 and main["accuracy_difference_right_minus_left"] > 0)
    return dict(status="COMPLETE_DIAGNOSTIC", factor_count=k, cells=cells,
                primary=primary, generated_pairs_descriptive=pairs,
                logit_pair_descriptive=paired(records, "LL", "NN", "logit_correct"),
                near_boundary=near_boundary(records), exploratory_strata=stratified,
                secondary_model=random_intercept_model(records) if k == 5 and fit_secondary else dict(status="NOT_FIT"))


def compile_report(fit_secondary=True):
    freeze = p.check_freeze()
    stages, all_records = {}, []
    for k in p.COUNTS:
        records = []
        for shard in range(p.COUNTS[k] * 4 // p.SHARD_ROWS):
            records.extend((load_checkpoint(k, shard, freeze) or {}).get("records", []))
        all_records.extend(records)
        stages[str(k)] = analyze_stage(k, records, fit_secondary) if len(records) == p.COUNTS[k] * 4 else dict(
            status="NOT_STARTED" if not records else "INCOMPLETE", completed_prompts=len(records), expected_prompts=p.COUNTS[k] * 4)
    complete = all(v["status"] == "COMPLETE_DIAGNOSTIC" for v in stages.values())
    onset = {}
    for cell in p.CELLS:
        states = {k: stages[str(k)]["cells"][cell]["descriptive_criterion_met"] for k in p.COUNTS if "cells" in stages[str(k)]}
        failed = [k for k, met in states.items() if not met]
        recovered = bool(failed and any(met and k > min(failed) for k, met in states.items()))
        onset[cell] = dict(first_observed_below_criterion=min(failed) if failed else None,
                           nonmonotonic=recovered, complete_ladder=complete,
                           limitation="Observed task-specific screening result, not a universal complexity threshold.")
    report = dict(protocol=p.VERSION, freeze_hash=freeze, stages=stages, onset_descriptive=onset,
                  status="COMPLETE_DIAGNOSTIC" if complete else "INCOMPLETE_NO_GLOBAL_CONCLUSION",
                  frozen_gate="FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED", mechanism_assay_authorized=False,
                  token_ledger=dict(completed_prompt_attempts=len(all_records),
                                    unpadded_input_tokens=sum(r["input_tokens"] for r in all_records),
                                    padded_input_token_slots=sum(r["batch_padded_input_tokens"] for r in all_records),
                                    generated_token_slots=sum(len(r["generated_ids"]) for r in all_records),
                                    excludes="Interrupted uncheckpointed batches; generated slots may include padding after EOS."))
    p.atomic_json(p.ROOT / "results" / "report.json", report)
    lines = ["# RES complexity × semantics — staged report", "", f"Status: **{report['status']}**", "",
             "LL = loaded labels / loaded task; NN = neutral / neutral; NL = neutral labels / loaded task; LN = loaded labels / neutral task.", "",
             "| Factors | Progress | LL generated | NN generated | NL generated | LN generated |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, stage in stages.items():
        if "cells" not in stage:
            cells = ["—"] * 4
            progress = f"{stage['completed_prompts']}/{stage['expected_prompts']} prompts"
        else:
            cells = []
            for cell in p.CELLS:
                value = stage["cells"][cell]
                low, high = value["generated_exact_95_ci"]
                cells.append(f"{value['generated_correct']}/{value['n']} ({low:.1%}–{high:.1%} exact 95% CI)")
            progress = "complete"
        lines.append("| " + " | ".join([k, progress] + cells) + " |")
    primary = stages["5"].get("primary")
    lines.extend(["", "## Interpretation", "",
                  "The sole confirmatory test is the fresh five-factor generated-output LL–NN paired comparison. Lower levels, crossovers, strata, logits, and the random-intercept model are diagnostic; none can rescue a failed primary comparison.", "",
                  "First observed below-criterion counts are descriptive and may be nonmonotonic. Even-factor ties, finite factor identities, and profile difficulty prevent attributing a drop uniquely to complexity.", "",
                  "The original capability gate remains `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`; no mechanism assay is authorized."])
    if primary:
        lines.extend(["", f"Five-factor exact two-sided McNemar p = {primary['p_value']:.6g}; neutral benefit replicated: {primary['neutral_benefit_replicated']}."])
    lines.extend(["", "Machine-readable report includes paired tables, exact intervals, coverage, answer-code and difficulty strata, near-boundary diagnostics, secondary model status, and token accounting.", ""])
    (p.ROOT / "results" / "REPORT.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-secondary", action="store_true", help="Preflight only; do not omit the secondary fit in a completed production report")
    args = parser.parse_args()
    result = compile_report(not args.skip_secondary)
    print(json.dumps({"status": result["status"], "token_ledger": result["token_ledger"]}, indent=2))
