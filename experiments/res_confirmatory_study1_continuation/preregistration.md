# RES Confirmatory Study 1 — relational-complexity continuation — preregistration

Frozen before confirmatory target-model inference. This is the next confirmatory continuation of the existing RES Study 1 sequence. It tests behavioral sensitivity to representations of locus of action and relational structure. It does not test consciousness, subjective experience, self-awareness, or literal human-like selfhood.

## Canonical prior representation

The canonical CODE-MAPPED manipulation is **RES Qwen3B five-factor failure-mode ablation**, sequence `qwen3b-five-factor-ablation-v1`, condition `five-count-code-mapped`: generated accuracy 17/32 versus 26/32 for `five-count-neutral`.

- source commit: `698718e47dacc8f0a9f065464a711c8d28d8d317`
- source artifact: `experiments/res_step0_interface_gate/run_five_factor_ablation_shard.py`
- source SHA-256: `0f3e25f844f46bcbca0045320c2ba2ffe7e7fac407c2aefcf74eff9d71deacb0`
- current byte-identical artifact: `experiments/res_confirmatory_study1_continuation/canonical/v1_code_mapped_source.py`
- exact template identifier: `five-count-code-mapped`

The current artifact hash is required to equal the source hash at every preflight. The later LL/NN “loaded” wording and its approximately 35/64 versus 38/64 result are separate and are not merged into this framing manipulation.

## Role of prior data

The hypothesis is prospective and is not a post-hoc explanation for the later replication non-result. Earlier RES results are used only for Phase 0 power/covariance estimation and the explicitly retrospective, non-confirmatory structural comparison in `historical_structural_comparison.md`. That comparison cannot redefine complexity, change fixtures or hypotheses, select fixtures or exclusions, or count as confirmatory evidence.

## Hypotheses

For each complexity level, FramingEffect = P(Neutral pass) − P(CodeMapped pass). The primary probability interaction is FramingEffect_HIGH − FramingEffect_LOW. The directional expectation is FramingEffect_LOW ≥ 0, FramingEffect_HIGH > 0, and a **positive** interaction.

H2a predicts that HIGH-complexity Neutral→CodeMapped transition fixtures have smaller `|margin(Neutral_HIGH)|` than framing-stable fixtures. The frozen test is a one-sided Mann–Whitney U with transition margins predicted smaller; rank-biserial effect and a bootstrap interval are reported.

H2b predicts more transitions for Neutral_HIGH→CodeMapped_HIGH than for Neutral_HIGH→NonRelationalPerturbation_HIGH. Both use the identical Neutral_HIGH observation and margin. The frozen test is a one-sided paired exact McNemar test. Descriptive absolute-margin strata are `≤0.75`, `(0.75,1.50]`, `(1.50,3.00]`, and `>3.00`.

## Sample size and power

Scientific N is **128 fixtures**. Alpha is .05 two-sided, target power .80, and probability-scale interaction SESOI is .15. The power simulation used earlier RES outcomes only, explicitly modeled within-fixture correlation, framing and complexity discordance, and empirical all-pass/all-fail prevalence. Required approximately-zero aggregate framing scenarios were included. The selected-N MDE grid estimate is 0.15. Full assumptions, inputs, seeds, achieved power, Monte Carlo intervals, and diagnostics are in `power_analysis.md` and `power_analysis.json`.

## Matched design and frozen complexity

Every fixture has Neutral×LOW, CodeMapped×LOW, Neutral×HIGH, CodeMapped×HIGH, and a NonRelationalPerturbation×HIGH control. The four primary versions preserve the same five factor values, response mapping, actor records, task decision, and correct code. LOW has CURRENT EXECUTOR and PEER. HIGH preserves that decision while adding REQUESTER→COORDINATOR→CURRENT EXECUTOR delegation and PEER→OBSERVER→REQUESTER record relay.

Frozen metrics are task-relevant actor/entity count, relational-binding count, directed role/delegation count, maximum relational nesting depth, locus-of-action transition count, conjunction/dependency count, characters, and pinned-tokenizer input tokens. Their operational definitions are in `historical_structural_comparison.json`. Longer text alone cannot validate HIGH.

The candidate pool (128) and ordered reserve pool (64) were written and hashed before blinded validation. A/B order was randomized. The validator saw only the frozen question and A/B texts. Initial global concordance was 100.0% against a frozen ≥85% threshold. It used 1 of two allowed cycles, and every final fixture has validated HIGH > LOW ordering. No scored target-model output was used.

## Non-relational perturbation

All controls were generated before inference from the exact Neutral_HIGH rendering by a deterministic fixture-seeded document-marker algorithm with at most two attempts. It preserves actor identity, bindings, delegation, locus semantics, relational complexity, and the answer, while avoiding the v1 factor/task wording. Both absolute token-delta and character-delta magnitudes must match the CodeMapped_HIGH change within ±20%; small changes use absolute tolerances of ±2 tokens and ±20 characters. 128/128 controls are valid. Invalid controls remain in the primary study and are excluded only from H2b. No regeneration is allowed after outputs.

## Scoring, margin, and technical failures

Scoring is deterministic and blinded to condition metadata: exactly one isolated A or B is parsed, then compared with the prespecified correct code. The inference configuration is pinned in `inference_config.json`. At the first generation step, A and B are the finite admissible decision-token set. The signed margin is `logit(correct/scored token) − logit(other admissible token)`; this is the already-frozen mathematically equivalent RES decision-margin implementation. H2a/H2b use `|margin(Neutral_HIGH)|`.

Technical invalidity is limited to invocation failure, truncation preventing scoring, corrupted/missing data, parser failure after frozen retries, scorer failure after retries, or wrong frozen configuration. Each observation receives one initial call plus at most two identical retries for technical failure only. A valid wrong, surprising, ambiguous, ceiling/floor, large-margin, or small-margin result is never substantively excluded or retried. If a primary cell remains technically missing, available raw observations remain and that fixture is excluded only from analyses requiring all four cells. The fifth-cell failure affects only H2b.

## Primary analysis and fallback

The primary model is fixture-stratified conditional logistic regression: `pass ~ neutral + high + neutral×high + strata(fixture_id)`. Coding makes the predicted interaction positive. The two-sided α=.05 interaction coefficient is primary. GEE with binary outcome, logit link, fixture clustering, and exchangeable working correlation is used only when conditional logistic regression cannot produce a finite converged interaction estimate and interval. Model choice never depends on significance.

The transparent probability difference-in-differences and 95% fixture-bootstrap interval are always reported. Precommitted categories are: CONFIRMATORY SUPPORT for a positive estimate whose 95% CI excludes zero; PRACTICALLY NEGLIGIBLE only when the full probability CI lies strictly inside (−.15,+.15); PREDICTED-SIZE POSITIVE RULED OUT, CONTRARY EFFECT STILL POSSIBLE when upper<+.15 and lower≤−.15; INCONCLUSIVE when zero and +.15 remain included; and CONTRARY-DIRECTION when the CI excludes zero negatively.

## Runtime, sharding, and no-feedback boundary

There are 16 deterministic shards of 8 fixtures (40 evaluations). Each fixture checkpoints atomically. A shard approaching the approximately 4.5-hour workflow ceiling finishes its current fixture, flushes, and exits. Resume skips an observation only after rendering hash, model/revision, inference configuration, and scorer configuration all match the freeze; any mismatch stops execution.

After the Phase 0 commit, scored outputs cannot change fixtures, complexity, representation, perturbations, scorer, margin, N, models, alpha, SESOI, strata, exclusions, retries, or hypotheses. All-pass and all-fail fixtures remain in descriptive data even when they provide zero conditional-likelihood information.

## Interpretation boundary

The study can test whether sensitivity to the frozen v1 manipulation changes with relational complexity, whether transitions concentrate near the decision boundary, and whether relational transitions exceed matched generic surface perturbation. It cannot establish the internal mechanism or consciousness, subjective experience, self-awareness, or human-equivalent self-representation. Optimized-label search remains a later held-out exploratory phase.
