# RES Confirmatory Study 1 — relational-complexity continuation

Status: **complete**. 640 predetermined observations attempted; 640 valid and 0 technically invalid after frozen retries.

## Preregistered hypotheses

The primary hypothesis predicted a positive interaction: the Neutral advantage over the frozen v1 code-mapped representation would be larger at HIGH than LOW relational complexity. H2a predicted smaller Neutral-HIGH baseline margins among relational-framing transitions. H2b predicted more transitions from relational framing than from the matched non-relational perturbation sharing the same Neutral-HIGH baseline.

## Confirmatory primary result

FramingEffect_LOW = +0.008 (95% fixture-bootstrap CI -0.062 to +0.078).
FramingEffect_HIGH = -0.188 (95% fixture-bootstrap CI -0.281 to -0.094).
Probability interaction = -0.195 (95% fixture-bootstrap CI -0.305 to -0.086).

Precommitted category: **CONTRARY_DIRECTION_RESULT**. The interval excludes zero on the negative side, contrary to the preregistered direction. A practically meaningful contrary-direction interaction is supported or compatible with the interval.

Selected primary model: conditional_logistic. Neutral×HIGH log-odds=-1.857, OR=0.156, 95% OR CI 0.051–0.481, two-sided p=0.001229.

## Preregistered secondary results

H2a: one-sided Mann–Whitney U=1672.5, p=0.2496; rank-biserial effect=+0.074 (95% bootstrap CI -0.117 to +0.263).
H2b: one-sided paired exact McNemar p=9.70612745732069e-09; relational transition rate=0.328125, non-relational transition rate=0.078125, valid matched controls n=128.

## Retrospective motivation analysis

The v1-versus-replication structural comparison was completed before this run and is recorded in `../historical_structural_comparison.md`. It is retrospective and non-confirmatory; it did not select fixtures, thresholds, exclusions, or hypotheses.

## Descriptive analyses

Prompt length, output length, edit distance, and all frozen structural metrics are reported in `length_structural_control_analysis.json`. Margin-stratified H2b rates are descriptive. These analyses did not alter any stimulus or primary test.

## Replicated and novel results

The historical 17/32→26/32 v1 result and the later 35/64-versus-38/64 replication remain prior evidence. Every estimate from this continuation is a new result from the frozen matched complexity design; it should not be described as independently replicated unless a later frozen study reproduces it.

## Exploratory interpretations

No exploratory mechanism claim is designated as confirmatory. Any later label search must use held-out fixtures and remain a separate research phase.

## Limitations

This is one pinned model and prompt family. Conditional likelihood can receive little information from all-pass and all-fail fixtures. Surface and structural manipulations cannot uniquely identify an internal mechanism. The study does not test consciousness, subjective experience, self-awareness, or human-equivalent self-representation.
