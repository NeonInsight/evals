# RES Research Release 0.1 — confirmatory complexity continuation

**Status: release candidate for review.** Prepared from the preserved completed confirmatory run, with surface-efficiency status recorded only as an incomplete snapshot from 2026-09-29.

This directory provides a concise, separately reviewable research release. **The original raw observations, frozen protocol, and successful final report are already committed** on the parent experimental branch under [`experiments/res_confirmatory_study1_continuation/results/`](../../experiments/res_confirmatory_study1_continuation/results/). The subsequent successful collector published a completed report; this directory does not supersede or overwrite it.

Read `REPORT.md` for the narrative, `results_summary.json` and `results_table.csv` for tabular outputs, `analysis_status.json` for the regenerated historical snapshot, and `PROVENANCE.json` for the byte hashes of the **original downloadable ZIP package** (not of the normalized copies of text files in this directory).

## Authoritative sources

- Confirmatory source GitHub Actions run: [36324399210](https://github.com/NeonInsight/evals/actions/runs/36324399210), collected artifact `10936466534`.
- Confirmatory completion: [original committed complete report](../../experiments/res_confirmatory_study1_continuation/results/REPORT.md), [original raw observations](../../experiments/res_confirmatory_study1_continuation/results/observations.jsonl), [original frozen inference JSON](../../experiments/res_confirmatory_study1_continuation/results/primary_interaction_analysis.json), and [original run manifest](../../experiments/res_confirmatory_study1_continuation/results/run_manifest.json).
- Surface-efficiency status snapshot: [run 36499730628](https://github.com/NeonInsight/evals/actions/runs/36499730628), collected artifact `11012009179`; **511/896 stored observations, incomplete** at snapshot time.
- Surface-efficiency continuation was started in [commit 226bfcc](https://github.com/NeonInsight/evals/commit/226bfcc2635f10e06e7ccadfb3b7f53ff3cf2938). Its later results must be tracked separately and must not be retrospectively folded into the frozen snapshot.

The original 2026-10-09 ZIP was verified against its own file-level SHA-256 ledger, and its complete analysis files were byte-for-byte identical to the downloaded GitHub collector artifact. The source `observations.jsonl` contains **640 unique observation IDs from 128 fixtures**. All four cell counts were checked against the machine-readable summary. This is an integrity and reporting audit, **not an independent statistical reimplementation or external replication**.

## Interpretation boundary

The pre-registered *positive* neutral-by-high interaction was not observed; the interaction was significantly **negative** under the frozen coding. The paired relational-perturbation specificity result is positive; the decision-margin prediction is unsupported. All findings are behavioral and specific to the tested model, fixture set, representations and scoring.

This release intentionally changes **no** frozen fixtures, prompts, model calls, scorer code, inferential definitions, or checkpoint data.
