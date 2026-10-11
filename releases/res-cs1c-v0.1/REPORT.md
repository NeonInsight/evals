# RES Current Results Artifact

**Prepared:** 2026-10-09  
**Scope:** Relational–Episodic Self (RES) confirmatory complexity continuation plus the current status of the frozen surface-form efficiency assay.  
**Claim ceiling:** These are behavioral evaluation results. They do **not** establish consciousness, subjective experience, persistent personal identity, or a mechanistic self-model.

## Executive result

The frozen RES Confirmatory Study 1 complexity-continuation run is complete: **128/128 predetermined fixtures and 640/640 observations** are present. The preregistered prediction was a **positive neutral-by-high relational-complexity interaction**. The observed interaction was statistically clear in the **opposite direction**.

On the probability scale, the framing interaction was **-19.53 percentage points** (10,000-bootstrap 95% CI **-30.47 to -8.59 pp**). In the selected conditional-logistic model, the neutral × high interaction had **OR = 0.156** (95% CI **0.051 to 0.481**), **two-sided Wald p = 0.00123**.

This rejects the preregistered directional hypothesis. It does not reduce the study to a null result: a frozen secondary assay found substantially more transitions under relational perturbations than under matched non-relational perturbations, while the preregistered decision-margin account was not supported.

## Data integrity and frozen design

- Protocol: `res-confirmatory-study1-complexity-continuation-v1`
- Final run ID: `36324399210`
- Frozen manifest SHA-256: `41938f71803fecaa17b6c8961343843be25f4c72887d03c32b6f34697d5553ef`
- Completed fixtures: **128/128**
- Stored observations: **640/640**
- All 16 shards report complete.
- Primary model: **conditional logistic regression**; no fallback model was triggered.
- Primary alpha: **0.05, two-sided**.
- Probability-scale bootstrap: **10,000 repetitions**, seed `2026092705`.
- Preregistered probability-interaction SESOI: **0.15**.

The final collector job was marked failed because its stale report/status files still described an earlier 505/640 checkpoint. The preserved final `run_manifest.json` and inferential output files, however, contain the complete 640-observation analysis. This artifact regenerates the human-readable status from those complete frozen outputs without rerunning inference or altering the analysis plan.

## Primary confirmatory test

The frozen coding was `neutral = 1`, `high = 1`; therefore the preregistered prediction was a **positive neutral × high interaction**.

| Condition | Passes | N | Pass rate | Exact 95% CI |
|---|---:|---:|---:|---:|
| Neutral, low complexity | 81 | 128 | 63.3% | 54.3–71.6% |
| Code-mapped, low complexity | 80 | 128 | 62.5% | 53.5–70.9% |
| Neutral, high complexity | 68 | 128 | 53.1% | 44.1–62.0% |
| Code-mapped, high complexity | 92 | 128 | 71.9% | 63.2–79.5% |

### Probability-scale contrasts

- Low-complexity neutral minus code-mapped effect: **+0.78 pp** (95% CI -6.25 to +7.81 pp).
- High-complexity neutral minus code-mapped effect: **-18.75 pp** (95% CI -28.12 to -9.38 pp).
- Difference-in-differences interaction: **-19.53 pp** (95% CI -30.47 to -8.59 pp).

The low-complexity conditions were nearly identical, while at high complexity the code-mapped condition outperformed the neutral condition by **18.75 pp**. That pattern produces the negative interaction.

### Conditional-logistic primary result

The selected conditional-logistic fit converged using **55 informative fixtures** out of 128 matched fixtures. The preregistered interaction term was:

- log-odds coefficient: **-1.857**
- odds ratio: **0.156**
- 95% OR CI: **0.051 to 0.481**
- two-sided Wald p: **0.001229**

**Confirmatory interpretation:** `CONTRARY_DIRECTION_RESULT`. The preregistered positive interaction is not supported; the data support an interaction in the negative direction under the frozen coding and analysis.

## Frozen secondary analyses

### Relational perturbation specificity

A paired exact McNemar test compared whether each fixture transitioned under the relational perturbation versus the matched non-relational perturbation.

- Relational transition rate: **32.81%** (42/128)
- Non-relational transition rate: **7.81%** (10/128)
- Discordant pairs: **34 relational-only vs 2 non-relational-only**
- One-sided exact McNemar p: **9.706e-09**

This supports the narrower claim that the observed behavioral transitions were **selective for relationally structured perturbations** rather than being equally induced by the matched non-relational perturbations used in this assay.

### Decision-margin hypothesis

The frozen margin hypothesis predicted smaller absolute baseline decision margins among fixtures that transitioned.

- Transition fixtures: **n = 42**, median |margin| = **1.812**
- Non-transition fixtures: **n = 86**, median |margin| = **2**
- One-sided Mann–Whitney U: **U = 1672.5**, **p = 0.250**
- Rank-biserial effect (positive = transition margins smaller): **0.074**, bootstrap 95% CI **-0.117 to 0.263**

**Interpretation:** the decision-margin hypothesis is **not supported** in this confirmatory continuation. The observed median difference is small and the uncertainty interval includes effects in both directions.

## What the current evidence supports

The present results support a narrower behavioral statement than the original complexity hypothesis:

> Within this frozen matched-fixture assay, relationally structured perturbations produced behavior changes more often than matched non-relational perturbations, but sensitivity did not increase with relational complexity in the preregistered direction. The primary complexity interaction instead appeared in the opposite direction, driven by a high-complexity advantage for the code-mapped representation relative to the neutral representation.

This is evidence about **behavioral sensitivity to relational representation under the tested task structure**. It does not, by itself, identify the internal mechanism producing that sensitivity.

## Claims not supported by this artifact

This artifact should not be cited as establishing any of the following:

- consciousness, sentience, or subjective experience;
- persistent identity across sessions;
- a mechanistic or causally localized self-representation;
- general relational sensitivity across models, domains, or task families;
- a monotonic relationship between relational complexity and framing sensitivity;
- the decision-margin explanation tested here.

Mechanistic claims would require causal-abstraction/interchange or comparable representation-level evidence against task-state, persona/role, generic-agent, and scaffolding alternatives.

## Surface-form efficiency assay: current status only

The next frozen assay is **not complete** and therefore contributes **no confirmatory inferential result** to this artifact.

- Protocol: `res-surface-efficiency-assay-v1`
- Run ID: `36499730628`
- Frozen manifest SHA-256: `442670e751808d831b1491c7d81f178a07bb9051315fb33808b3c443e7b90ca1`
- Completed fixtures: **73/128** (57.0%)
- Stored observations: **511/896** (57.0%)
- Status: **`INCOMPLETE_RESUMABLE`**

The checkpoints are preserved for a frozen resume. Partial surface-efficiency observations are intentionally not analyzed for the confirmatory endpoint here.

## Reproducibility files

The `machine/` directory contains the exact complete-result files used to build this report:

- `run_manifest.json`
- `primary_interaction_analysis.json`
- `perturbation_specificity_analysis.json`
- `decision_margin_analysis.json`
- `fixture_level_matched_summary.csv`
- `scored_observations.csv`
- `surface_efficiency_run_manifest.json` (status only)

`results_summary.json` provides a compact machine-readable rendering of the conclusions above. `PROVENANCE.json` records run identifiers, frozen-manifest hashes, source artifact metadata, and SHA-256 hashes of the packaged files.

## Citation note

When describing the primary study, report the **contrary-direction interaction**, not merely the individual cell differences. When describing the secondary specificity result, retain its narrower behavioral interpretation and the one-sided paired test specified by the frozen analysis. The incomplete surface-efficiency assay should be reported only as resumable status until its predetermined fixture set is complete or has exhausted the frozen retry rules.
