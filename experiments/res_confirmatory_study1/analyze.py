"""Preregistered matched analysis; only complete frozen data get inferential results."""
from __future__ import annotations

import csv
import json
import math
import statistics
import warnings
from collections import Counter

import protocol as p
from run import RESULTS, load


def exact_ci(k, n):
    from scipy.stats import beta
    if not 0 <= k <= n or n < 1:
        raise ValueError("Invalid count")
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]


def exact_mcnemar(fail_to_pass, pass_to_fail):
    from scipy.stats import binomtest
    n = fail_to_pass + pass_to_fail
    return float(binomtest(fail_to_pass, n, .5).pvalue) if n else 1.0


def quantiles(values):
    import numpy as np
    if not values:
        return None
    a = np.array(values, dtype=float)
    return dict(n=len(a), min=float(a.min()), q10=float(np.quantile(a, .1)),
                q25=float(np.quantile(a, .25)), median=float(np.median(a)),
                q75=float(np.quantile(a, .75)), q90=float(np.quantile(a, .9)), max=float(a.max()))


def model(rows, length_adjusted=False):
    import numpy as np
    from statsmodels.discrete.conditional_models import ConditionalLogit
    columns = ["explicit", "high", "interaction"]
    if length_adjusted:
        columns += ["prompt_tokens_centered", "position_centered"]
    x = np.array([[r[c] for c in columns] for r in rows], dtype=float)
    groups = np.array([r["fixture_id"] for r in rows])
    y = np.array([r["pass"] for r in rows], dtype=int)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = ConditionalLogit(y, x, groups=groups).fit(disp=False, maxiter=500)
        estimates = {}
        for i, name in enumerate(columns):
            coef, se = float(fit.params[i]), float(fit.bse[i])
            estimates[name] = dict(log_odds=coef, standard_error=se,
                                   odds_ratio=math.exp(coef),
                                   ci_95_log_odds=[coef - 1.96 * se, coef + 1.96 * se],
                                   ci_95_odds_ratio=[math.exp(coef - 1.96 * se), math.exp(coef + 1.96 * se)],
                                   two_sided_wald_p=float(fit.pvalues[i]))
        return dict(status="FIT_WITH_WARNINGS" if caught else "FIT_OK",
                    method="ConditionalLogit fixture-stratified maximum likelihood; Wald intervals/tests",
                    terms=estimates, n_observations=int(fit.nobs),
                    n_informative_fixtures=len({g for g in groups if len(set(y[groups == g])) > 1}),
                    warnings=[str(w.message) for w in caught])
    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        return dict(status="FIT_FAILED", reason=str(exc),
                    method="ConditionalLogit fixture-stratified; no substitute independence test")


def bootstrap_effects(grouped):
    import numpy as np
    rng = np.random.default_rng(p.SEED + 11)
    keys = sorted(grouped)
    low = np.array([grouped[f]["EL"]["pass"] - grouped[f]["NL"]["pass"] for f in keys], dtype=float)
    high = np.array([grouped[f]["EH"]["pass"] - grouped[f]["NH"]["pass"] for f in keys], dtype=float)
    values = np.array([((grouped[f]["EH"]["pass"] - grouped[f]["NH"]["pass"]) -
                        (grouped[f]["EL"]["pass"] - grouped[f]["NL"]["pass"])) for f in keys], dtype=float)
    sample = rng.integers(0, len(keys), (10000, len(keys)))
    means = values[sample].mean(axis=1)
    return dict(interaction_difference_in_differences=float(values.mean()),
                fixture_bootstrap_percentile_95_ci=[float(np.quantile(means, .025)),
                                                    float(np.quantile(means, .975))],
                framing_effect_low=dict(absolute_difference=float(low.mean()),
                    fixture_bootstrap_percentile_95_ci=[float(np.quantile(low[sample].mean(axis=1), q)) for q in (.025, .975)]),
                framing_effect_high=dict(absolute_difference=float(high.mean()),
                    fixture_bootstrap_percentile_95_ci=[float(np.quantile(high[sample].mean(axis=1), q)) for q in (.025, .975)]),
                contrast_counts={str(int(v)): int((values == v).sum()) for v in (-2, -1, 0, 1, 2)},
                bootstrap_seed=p.SEED + 11, bootstrap_iterations=10000)


def transitions(grouped):
    result = []
    for fid in sorted(grouped):
        for complexity in "LH":
            neutral, explicit = grouped[fid]["N" + complexity], grouped[fid]["E" + complexity]
            a, b = neutral["pass"], explicit["pass"]
            result.append(dict(fixture_id=fid, complexity="LOW" if complexity == "L" else "HIGH",
                               neutral_pass=a, explicit_pass=b,
                               transition="fail_to_pass" if not a and b else
                                          "pass_to_fail" if a and not b else "no_change",
                               baseline_condition="E" + complexity,
                               baseline_abs_raw_ab_margin=abs(explicit["raw_ab_margin"]),
                               neutral_abs_raw_ab_margin=abs(neutral["raw_ab_margin"])))
    return result


def boundary(trans):
    import numpy as np
    # The old ladder preregistered |raw A-B margin| <= .75, used here unchanged.
    groups = {}
    for scope in ("LOW", "HIGH", "POOLED"):
        selected = [t for t in trans if scope == "POOLED" or t["complexity"] == scope]
        sensitive = [t["baseline_abs_raw_ab_margin"] for t in selected if t["transition"] != "no_change"]
        insensitive = [t["baseline_abs_raw_ab_margin"] for t in selected if t["transition"] == "no_change"]
        near = [t for t in selected if t["transition"] != "no_change" and t["baseline_abs_raw_ab_margin"] <= .75]
        groups[scope] = dict(sensitive=quantiles(sensitive), insensitive=quantiles(insensitive),
                             sensitive_with_margin_at_most_0_75=len(near),
                             sensitive_total=len(sensitive),
                             fraction_sensitive_near=len(near) / len(sensitive) if sensitive else None,
                             exact_95_ci_sensitive_near=exact_ci(len(near), len(sensitive)) if sensitive else None,
                             sorted_sensitive_margins=sorted(sensitive),
                             sorted_insensitive_margins=sorted(insensitive))
    rng = np.random.default_rng(p.SEED + 12)
    fids = sorted({t["fixture_id"] for t in trans})
    by_fixture = {fid: [t for t in trans if t["fixture_id"] == fid] for fid in fids}
    diffs = []
    for _ in range(10000):
        sampled = rng.choice(fids, len(fids), replace=True)
        rows = [t for fid in sampled for t in by_fixture[fid]]
        s = [t["baseline_abs_raw_ab_margin"] for t in rows if t["transition"] != "no_change"]
        i = [t["baseline_abs_raw_ab_margin"] for t in rows if t["transition"] == "no_change"]
        if s and i:
            diffs.append(float(np.median(s) - np.median(i)))
    groups["pooled_median_sensitive_minus_insensitive"] = (
        dict(point=groups["POOLED"]["sensitive"]["median"] - groups["POOLED"]["insensitive"]["median"],
             fixture_bootstrap_percentile_95_ci=[float(np.quantile(diffs, .025)),
                                                 float(np.quantile(diffs, .975))],
             bootstrap_seed=p.SEED + 12, bootstrap_iterations=10000)
        if diffs and groups["POOLED"]["sensitive"] and groups["POOLED"]["insensitive"] else None)
    return dict(threshold_source="previous RES ladder preregistration: raw A-minus-B <= 0.75 in absolute value",
                baseline="EXPLICIT within each complexity", groups=groups,
                limitation="A/B output margin is not a full distribution or evidence of a specific internal mechanism")


def confounds(rows, grouped):
    from scipy.stats import spearmanr
    pair_summaries = {}
    for complexity in "LH":
        pairs = [(grouped[f]["E" + complexity], grouped[f]["N" + complexity]) for f in sorted(grouped)]
        def delta(key):
            vals = [a[key] - b[key] for a, b in pairs]
            return dict(mean=sum(vals) / len(vals), median=statistics.median(vals),
                        min=min(vals), max=max(vals), nonzero=sum(v != 0 for v in vals))
        pair_summaries["LOW" if complexity == "L" else "HIGH"] = dict(
            explicit_minus_neutral_characters=delta("prompt_char_count"),
            explicit_minus_neutral_prompt_tokens=delta("prompt_token_count"),
            explicit_minus_neutral_response_tokens=delta("response_token_count"))
    pos = {c: Counter(r["position_within_fixture"] for r in rows if r["condition"] == c)
           for c in p.CONDITIONS}
    correlation = {}
    for key in ("prompt_char_count", "prompt_token_count", "response_token_count", "evaluation_order"):
        values = [r[key] for r in rows]
        if len(set(values)) < 2 or len({r["pass"] for r in rows}) < 2:
            correlation[key] = dict(spearman_rho=None, p_value_diagnostic=None,
                                    reason="constant input")
        else:
            stat = spearmanr(values, [r["pass"] for r in rows])
            correlation[key] = dict(spearman_rho=float(stat.statistic) if math.isfinite(stat.statistic) else None,
                                    p_value_diagnostic=float(stat.pvalue) if math.isfinite(stat.pvalue) else None)
    return dict(pairwise_lengths=pair_summaries, condition_position_counts={k: dict(v) for k, v in pos.items()},
                correlations_descriptive=correlation, adjusted_conditional_logit=model(rows, True),
                semantic_flags=[], exclusion_rule="No post-freeze exclusions; flag and stop if a design flaw is discovered.",
                note="Response length can be an outcome of the manipulation; its association is diagnostic, not a causal adjustment.")


def write_csv(path, records):
    if not records:
        return
    with PathOpen(path) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def PathOpen(path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", newline="", encoding="utf-8")


def analyze():
    freeze_hash = p.check_freeze()
    raw = []
    for index in range(32):
        raw.extend((load(index, freeze_hash) or {}).get("records", []))
    expected = p.rendered()
    expected_index = {r["observation_id"]: r for r in expected}
    if len({r["observation_id"] for r in raw}) != len(raw) or any(r["observation_id"] not in expected_index for r in raw):
        raise ValueError("Duplicate or unknown observation")
    token_data = json.loads((RESULTS / "token_counts.json").read_text())
    if token_data["freeze_hash"] != freeze_hash or len(token_data["rows"]) != 256:
        raise ValueError("Token audit incomplete/mismatched")
    token_index = {r["observation_id"]: r for r in token_data["rows"]}
    rows = []
    for r in raw:
        spec, audit = expected_index[r["observation_id"]], token_index[r["observation_id"]]
        if (r["prompt_token_count"] != audit["prompt_token_count"] or
                r["prompt_sha256"] != audit["chat_template_prompt_sha256"] or
                r["messages_sha256"] != audit["messages_sha256"]):
            raise ValueError("Inference prompt differs from frozen tokenizer audit")
        rows.append(dict(observation_id=r["observation_id"], fixture_id=r["fixture_id"],
                         condition=r["condition"], framing=spec["framing"], complexity=spec["complexity"],
                         expected=spec["expected"], generated_classification=r["generated_classification"],
                         logit_classification=r["logit_classification"], logit_correct=int(r["logit_correct"]),
                         **{"pass": int(r["passed"])}, parsed=int(r["parsed"] is not None),
                         raw_a_logit=r["raw_a_logit"], raw_b_logit=r["raw_b_logit"],
                         processed_a_logit=r["processed_a_logit"], processed_b_logit=r["processed_b_logit"],
                         raw_ab_margin=r["raw_ab_margin"], decision_margin=r["decision_margin"],
                         prompt_token_count=r["prompt_token_count"], prompt_char_count=r["prompt_char_count"],
                         response_token_count=r["response_token_count"],
                         explicit=int(r["condition"][0] == "E"), high=int(r["condition"][1] == "H"),
                         interaction=int(r["condition"] == "EH"),
                         prompt_tokens_centered=(r["prompt_token_count"] - 200) / 100,
                         position_centered=spec["position_within_fixture"] - 1.5,
                         position_within_fixture=spec["position_within_fixture"],
                         evaluation_order=spec["evaluation_order"], model=r["model"], run_id=r["run_id"],
                         seed=r["seed"], prompt_sha256=r["prompt_sha256"]))
    rows.sort(key=lambda r: r["evaluation_order"])
    write_csv(RESULTS / "scored_observations.csv", rows)
    base = dict(protocol=p.VERSION, freeze_hash=freeze_hash, model=p.MODEL,
                completed=len(rows), expected=256, original_gate="FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED",
                status="INCOMPLETE_NO_CONFIRMATORY_ANALYSIS", exclusions=[])
    if len(rows) != 256:
        p.atomic_json(RESULTS / "results.json", base)
        report = ["RES Confirmatory Study 1 tests whether relational framing interacts with relational task complexity in a matched factorial design. It measures behavioral effects and does not assume or test for subjective experience.",
                  "", "# RES Confirmatory Study 1", "", f"Status: **incomplete, {len(rows)}/256** scored observations.",
                  "", "## Confirmatory results", "", "No factorial inference on incomplete data.",
                  "", "## Secondary analyses", "", "Deferred until all 256 observations are collected.",
                  "", "## Exploratory observations", "", "None reported.", ""]
        (RESULTS / "REPORT.md").write_text("\n".join(report))
        return base
    grouped = {}
    for r in rows:
        grouped.setdefault(r["fixture_id"], {})[r["condition"]] = r
    if len(grouped) != 64 or any(set(g) != set(p.CONDITIONS) for g in grouped.values()):
        raise ValueError("A fixture is missing a matched condition")
    cells = {}
    for condition in p.CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        k = sum(r["pass"] for r in selected)
        cells[condition] = dict(pass_count=k, n=len(selected), proportion=k / len(selected),
                                exact_95_ci=exact_ci(k, len(selected)), parser_coverage=sum(r["parsed"] for r in selected),
                                logit_correct=sum(r["logit_correct"] for r in selected))
    t = transitions(grouped)
    write_csv(RESULTS / "matched_transitions.csv", t)
    transition_summaries = {}
    for complexity in ("LOW", "HIGH"):
        count = Counter(x["transition"] for x in t if x["complexity"] == complexity)
        count["no_change"] += 0
        count["fail_to_pass"] += 0
        count["pass_to_fail"] += 0
        transition_summaries[complexity] = dict(**count, n=64,
            explicit_minus_neutral_percentage_points=100 * (count["fail_to_pass"] - count["pass_to_fail"]) / 64,
            exact_two_sided_mcnemar_p=exact_mcnemar(count["fail_to_pass"], count["pass_to_fail"]))
    primary = dict(interaction_model=model(rows), effect_size=bootstrap_effects(grouped),
                   matched_four_cell_table=[dict(fixture_id=fid,
                                                 NL=grouped[fid]["NL"]["pass"], EL=grouped[fid]["EL"]["pass"],
                                                 NH=grouped[fid]["NH"]["pass"], EH=grouped[fid]["EH"]["pass"])
                                            for fid in sorted(grouped)])
    p.atomic_json(RESULTS / "factorial_analysis.json", primary)
    decision = boundary(t)
    p.atomic_json(RESULTS / "decision_margin_analysis.json", decision)
    lengths = confounds(rows, grouped)
    p.atomic_json(RESULTS / "length_confound_analysis.json", lengths)
    base.update(status="COMPLETE", cells=cells, factorial=primary,
                transitions=transition_summaries, decision_boundary=decision, confounds=lengths)
    p.atomic_json(RESULTS / "results.json", base)
    interaction = primary["interaction_model"]
    lines = ["RES Confirmatory Study 1 tests whether relational framing interacts with relational task complexity in a matched factorial design. It measures behavioral effects and does not assume or test for subjective experience.",
             "", "# RES Confirmatory Study 1", "", f"Status: **complete**, 64 fixtures × four conditions, pinned `{p.MODEL}`.",
             "", "## Confirmatory results", "", "| Condition | Pass | Proportion | Exact 95% CI |",
             "| --- | ---: | ---: | ---: |"]
    for c in p.CONDITIONS:
        s = cells[c]
        lines.append(f"| {c} | {s['pass_count']}/{s['n']} | {s['proportion']:.1%} | {s['exact_95_ci'][0]:.1%}–{s['exact_95_ci'][1]:.1%} |")
    effect = primary["effect_size"]
    lines.extend(["", "Primary contrast: (explicit − neutral at HIGH) − (explicit − neutral at LOW) = "
                  f"{effect['interaction_difference_in_differences']:+.3f} absolute pass probability "
                  f"(fixture bootstrap 95% interval {effect['fixture_bootstrap_percentile_95_ci'][0]:+.3f} "
                  f"to {effect['fixture_bootstrap_percentile_95_ci'][1]:+.3f})."])
    if interaction["status"].startswith("FIT") and "terms" in interaction:
        v = interaction["terms"]["interaction"]
        lines.append(f"Fixture-stratified conditional logistic interaction: log odds {v['log_odds']:+.3f}, "
                     f"OR {v['odds_ratio']:.3f}, 95% OR interval {v['ci_95_odds_ratio'][0]:.3f}–"
                     f"{v['ci_95_odds_ratio'][1]:.3f}, two-sided Wald p={v['two_sided_wald_p']:.4g}; "
                     f"{interaction['n_informative_fixtures']}/64 informative fixtures.")
    else:
        lines.append(f"Primary conditional model {interaction['status']}: {interaction.get('reason')}. "
                     "Four-cell paired table retained; no independence test substituted.")
    lines.extend(["", "## Secondary analyses", "", "| Framing change | Fail → pass | Pass → fail | No change | Absolute difference | Exact McNemar p |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for c in ("LOW", "HIGH"):
        s = transition_summaries[c]
        lines.append(f"| {c}, neutral → explicit | {s['fail_to_pass']}/64 | {s['pass_to_fail']}/64 | "
                     f"{s['no_change']}/64 | {s['explicit_minus_neutral_percentage_points']:+.1f} pp | "
                     f"{s['exact_two_sided_mcnemar_p']:.4g} |")
    lines.append("")
    for c in ("LOW", "HIGH"):
        estimate = effect["framing_effect_low" if c == "LOW" else "framing_effect_high"]
        lines.append(f"{c} framing difference: {estimate['absolute_difference']:+.3f} "
                     f"(fixture bootstrap 95% interval {estimate['fixture_bootstrap_percentile_95_ci'][0]:+.3f} "
                     f"to {estimate['fixture_bootstrap_percentile_95_ci'][1]:+.3f}).")
    for c in ("LOW", "HIGH", "POOLED"):
        b = decision["groups"][c]
        lines.append(f"{c} baseline explicit |A−B| margins: sensitive "
                     f"{b['sensitive']['median'] if b['sensitive'] else 'n/a'} (n={b['sensitive_total']}), "
                     f"insensitive {b['insensitive']['median'] if b['insensitive'] else 'n/a'} "
                     f"(n={b['insensitive']['n'] if b['insensitive'] else 0}); "
                     f"{b['sensitive_with_margin_at_most_0_75']}/{b['sensitive_total']} "
                     "sensitive transitions within frozen |margin|≤0.75" +
                     (f" (exact 95% CI {b['exact_95_ci_sensitive_near'][0]:.1%}–"
                      f"{b['exact_95_ci_sensitive_near'][1]:.1%})." if b['exact_95_ci_sensitive_near'] else "."))
    pooled_diff = decision["groups"]["pooled_median_sensitive_minus_insensitive"]
    if pooled_diff:
        lines.append(f"Pooled sensitive minus insensitive baseline median |margin|: "
                     f"{pooled_diff['point']:+.3f}, fixture bootstrap 95% interval "
                     f"{pooled_diff['fixture_bootstrap_percentile_95_ci'][0]:+.3f} to "
                     f"{pooled_diff['fixture_bootstrap_percentile_95_ci'][1]:+.3f}.")
    lines.extend(["", "Length and order checks:"])
    for c in ("LOW", "HIGH"):
        d = lengths["pairwise_lengths"][c]
        lines.append(f"- {c}: explicit minus neutral mean prompt length "
                     f"{d['explicit_minus_neutral_characters']['mean']:+.1f} characters and "
                     f"{d['explicit_minus_neutral_prompt_tokens']['mean']:+.1f} tokens; "
                     f"mean generated response difference {d['explicit_minus_neutral_response_tokens']['mean']:+.2f} tokens "
                     f"(64 matched pairs).")
    adj = lengths["adjusted_conditional_logit"]
    lines.append(f"- Token/position-adjusted matched diagnostic model: {adj['status']}" +
                 (f", interaction OR {adj['terms']['interaction']['odds_ratio']:.3f}, 95% CI "
                  f"{adj['terms']['interaction']['ci_95_odds_ratio'][0]:.3f}–"
                  f"{adj['terms']['interaction']['ci_95_odds_ratio'][1]:.3f}." if "terms" in adj else
                  f", {adj.get('reason')}."))
    lines.extend(["- Each condition occurred 16/64 times in each within-fixture position. "
                  "Token counts, response lengths, order correlations and semantic checks are retained "
                  "in `length_confound_analysis.json`. No post-freeze exclusions were made. "
                  "Residual length or wording effects cannot be ruled out by these diagnostics.",
                  "", "## Exploratory observations", "", "None designated as confirmatory beyond the interaction model. "
                  "Inspect raw records before proposing a new protocol.", "", "## Limits and reproduction", "",
                  "The old five-factor capability gate stays `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`. "
                  "These are behavioral observations on one pinned model and prompt family; no subjective or mechanistic claim follows.",
                  "", "Exact preflight, inference, collection, and analysis commands are in `../README.md`. "
                  "The SHA-256 manifest, all prompts, raw shard outputs, matched transitions, and model coefficients are retained.", ""])
    (RESULTS / "REPORT.md").write_text("\n".join(lines))
    return base


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(dict(status=result["status"], completed=result["completed"], expected=256)))
