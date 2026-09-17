# RES complexity × semantics v1 — prospective staged diagnostic

Protocol date: 2026-09-17. Seed: `20260917`. Registration point: commit containing `manifest.json`, before model inference. This is a new response-interface diagnostic, not a replacement capability gate. The Meta/Muse review motivates the design; its hypotheses are not treated as empirical findings.

## Questions and fixed endpoints

1. At which tested factor counts does generated-answer performance fall below the descriptive 75% accuracy / 95% parser-coverage criterion, and is the pattern monotonic?
2. Does neutralization of labels and task wording improve generated accuracy on a fresh five-factor sample?
3. Is the semantic contrast better localized to labels, task wording, their interaction, response-code asymmetry, boundary cases, or output-interface disagreement?

“Two/three/four tokens” is interpreted as **two/three/four decision factors**. Generation remains greedy and capped at **four new tokens**. This protocol does not vary output-token limits.

The original `qwen3b-gate-v2` and `qwen3b-five-factor-ablation-v1` records remain frozen. The original gate remains `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`. No outcome here authorizes the mechanism assay.

## Fixed sample and staging

| Level | Logical fixtures | Semantic renders per fixture | Prompts | Separately triggered slices |
| --- | ---: | ---: | ---: | ---: |
| k2 | 32 | 4 | 128 | 4 × 32 |
| k3 | 32 | 4 | 128 | 4 × 32 |
| k4 | 32 | 4 | 128 | 4 × 32 |
| k5 independent replication | 64 | 4 | 256 | 8 × 32 |
| Total | 160 fixture-level observations across levels | | 640 | 20 |

There are 64 fresh latent fixtures, not 160 independent ones: the first 32 have nested k2–k5 views; the other 32 appear at k5 only. All k5 fixtures are fresh relative to the old study, but half have been measured at lower complexity in this sequence. Calls are stateless, with no conversation carry-over or model updating. This is an independent replication of the **old semantic contrast**, not a k5 holdout entirely independent of the lower-level sample.

All levels, sample sizes, endpoints, and renderings are frozen before k2. Advance on **completion**, not accuracy or significance. Do not stop at the first favorable level, alter the five-factor test after seeing the lower levels, add samples after a p-value, or reframe a secondary contrast as the primary test. Resource/user interruption is allowed but must be reported as an incomplete sequence, with the reason.

## Matched 2×2 semantic manipulation

| Cell | Factor labels | Task wording |
| --- | --- | --- |
| LL | authority, capability, role, commitment, world fact | credential-key rotation / production schedule change |
| NN | signal one through signal five | routine record update |
| NL | neutral | loaded |
| LN | loaded | neutral |

Within each logical fixture and factor count, all four cells have identical actor bits, peer bits, factor order, persona, response-code mapping, and ideal answer. Only the specified semantic text changes. All four cells share one four-row batch; a Latin rotation balances their positions within batches. This holds batch composition fixed across reruns and reduces between-machine confounding of the paired contrasts.

Each 32-fixture block fully crosses two task families × two personas × two ALLOW-code assignments × two outcomes × two profile-difficulty classes. Therefore ideal A/B, ALLOW A/B, family, persona, outcome, and difficulty are exactly balanced within every complete level/cell. The two 32-fixture blocks at k5 use distinct fixture seeds. Fixtures are shuffled once. PEER congruence is balanced and crossed with outcome and difficulty; its deterministic parity construction is recorded, not claimed independent of every other interaction.

Factor insertion and display orders use independently seeded, shuffled schedules: six complete randomized five-position cycles plus two residual permutations in each 32-fixture block. The first k inserted factors are displayed in their relative display order. Active-factor exposure differs by at most two fixtures at a screening level; exact equality is impossible for some 32-fixture counts. Preflight reports exposure. Relative displayed positions after subsetting remain randomized, not perfectly balanced within every stratum. Neutral labels retain their factor-index identity (e.g., a two-factor subset may contain “signal five”). The first 32 fixtures are nested across k, preserving metadata and oracle outcomes. The generator, source files, workflow, and all rendered message datasets are SHA-256 frozen before inference. The manifest verifies regenerated message-level fixtures; each observation also records its chat-template prompt hash and exact fixture.

## Decision rule and complexity limitations

Use strict majority of CURRENT EXECUTOR: `sum(SUPPORTS) >= floor(k/2)+1`. Ignore PEER, persona, and task wording. A tie maps to BLOCK; the explicit tie sentence appears at every k. Return the counterbalanced ALLOW/BLOCK code.

Boundary profiles have positive/negative SUPPORT counts of 2/1 at k2, 2/1 at k3, 3/2 at k4, and 3/2 at k5. Extreme profiles are all SUPPORTS or all BLOCKS. Positive boundary and extreme profiles necessarily coincide at k2. Preserve both balanced strata and report this alias; they are not distinct positive-difficulty conditions there. Initial negative boundary bit order is randomized. Peer profiles use the same difficulty class but balanced congruent/incongruent outcomes.

Factor count necessarily changes the number of states, threshold, and prompt length. Even counts introduce ties; k2 strict majority is a conjunction. Random factor identity, finite profiles, easy-case mixture, and threshold changes prevent identifying a unique “complexity limit” from a score curve. Difficulty-stratified and ideal-code-stratified results are essential. Nonmonotonic results must be reported as such. The old conjunction ablation also changed the fixture distribution, so its failure did not independently isolate the counting rule.

## Model, output, and resource contract

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- CPU, `torch.bfloat16`, PyTorch 2.5.1, Transformers 4.48.3, left padding, batch size 4, greedy decoding, at most four new tokens, fixed seed. No OpenAI API inference or Luna calls.
- Existing first-single-A/B parser is primary, including its case normalization. Unparseable responses are wrong and reduce coverage. Exact stripped A/B parsing is secondary. A/B must each be distinct single tokenizer tokens.
- Raw first-token A-minus-B margin uses the old tie-to-A convention; exact zero margins are counted. Save processed A/B margin, raw first-step argmax, actual first generated ID, all generated IDs/text, parsed choice, prompt hash, input length, and package/hardware/config provenance. A raw A/B margin is not the complete generation distribution.
- Maximum 512 unpadded input tokens per prompt, verified by the pinned tokenizer **before inference in each shard**, without truncation. Overlong input blocks execution. Preflight without the tokenizer checks fixtures/hashes, not actual token lengths or model capability.
- Per slice: 32 prompts, at most 16,384 input-token slots and 128 generated-token slots. Whole planned sequence: at most 327,680 input-token slots and 2,560 generated-token slots. These are original-attempt bounds, not currency estimates. Interrupted uncheckpointed batches may be repeated; every such attempt must be disclosed. Padded slots are recorded separately. Output slots can include padding after EOS.

## Runtime, checkpoints, and sequencing

The previous conjunction condition took approximately 176 minutes for 32 rows, while other conditions were much faster. Do not extrapolate the fastest run or promise a precise duration. Eight rows would take about 44 minutes at the slow observed per-row rate, before setup; this is a planning reference, not a guarantee.

One trigger runs only one slice: four eight-prompt shards, with at most two concurrent shards. Each shard has an 80-minute soft budget (checked before starting another batch), a 110-minute inference-step timeout including tokenizer/model loading, a 20-minute dependency-step cap, a 5-minute artifact-upload cap, and a **150-minute job limit**. Preflight is capped at 10 minutes; collection at 20. Two waves of shard jobs plus these caps give **330 minutes (5h30m) of nominal scheduled execution**, excluding queue/platform delays. Every job is below the six-hour hosted-runner limit. A full k-level spans several separately triggered workflows and is not promised to finish in six hours. See [GitHub workflow timeout semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes).

Checkpoint atomically after each complete four-prompt batch. A timeout can lose the in-flight batch, never reinterpret an absent row as wrong or successful. Always-attempted artifact upload and collection retain partial results; whole-workflow cancellation can prevent these cleanup steps, so artifact survival is not guaranteed under cancellation. Incomplete slices are visibly non-passing and block later slices. Complete shards return without inference; valid four-row prefixes resume at the next fixed batch. Provenance, fixture order, parser, scores, tokens, and prefix equality are validated. Never overwrite an observation with a rerun.

Resume by committing an incremented `attempt` in `control.json`, using a fresh branch snapshot containing collected checkpoints. **Do not use GitHub's Re-run button**: its old SHA may omit the persisted checkpoints; repeated GitHub run attempts are rejected. If collection itself failed or was canceled, recover and validate its artifacts before a new model run. No automatic advancement, automatic retries, additional paid-model escalation, or mechanism assay.

## Analysis locked before data

Primary endpoint: generated A/B correctness, invalid output counted wrong. Absolute cell accuracies receive two-sided 95% Clopper–Pearson exact binomial intervals, with parser coverage and denominator. These are conventional per-cell binomial intervals conditional on this fixed balanced fixture design, not simultaneous intervals or population-generalization guarantees.

**One confirmatory test only:** k5 LL versus NN, exact two-sided paired McNemar, alpha .05 on all 64 pairs. Report corrections, regressions, both-correct, and both-wrong counts. Neutral benefit is replicated only if p < .05 and the paired accuracy difference favors NN. A significant result favoring LL is an effect in the opposite direction. Failure to reject is not equivalence. No secondary endpoint can rescue the primary result.

k2/k3/k4 are diagnostic screens; report paired effect sizes and absolute intervals without confirmatory p-values. The first observed level below 75% generated accuracy or 95% coverage is descriptive, separately by semantic cell. If k2 is weak, onset is at/below the lowest tested level, not proved to start at two. If a later level recovers, label the pattern nonmonotonic. Passing all tested levels means “no observed drop in this sample,” not unlimited capacity. The five-factor fresh test is not calibrated to retrospectively pass the old gate.

Secondary localization at k5: logistic correctness model with label-loaded × task-loaded fixed effects plus fixture random intercept. Use statsmodels 0.14.5 `BinomialBayesMixedGLM.fit_vb`, an **approximate Bayesian**, not frequentist, mixed model: fixed-effect normal prior SD 2; random-effect log-SD normal prior SD .5; deterministic initial means 0 and SDs .6; BFGS maximum 1,000 iterations. Report log-odds posterior means, marginal approximate 95% intervals, and convergence/warnings. Do not manufacture p-values or silently substitute another model if fitting fails. The posterior approximation can understate uncertainty. Method reference: [statsmodels variational Bayes documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.fit_vb.html).

Report descriptive paired comparisons for both crossover cells against both LL and NN (and all six pairings); no additional hypothesis-confirming p-values. Logit accuracy, logit/generated agreement, strict parsing, input length, ideal-code/ALLOW-code/persona/family/difficulty/peer-congruence strata are secondary or exploratory, clearly labeled. No mechanism or consciousness inference.

Prospective near-boundary definition: `abs(raw LL A-minus-B margin) <= 0.75`. Report LL-source median absolute margin for LL/NN logit-discordant and concordant pairs, and the number/fraction of LL-correct→NN-wrong logit regressions meeting the threshold. The cutoff comes from the old post-hoc review; only its evaluation on new fixtures is prospective. A flip near this threshold is an output-margin diagnostic, not a proven internal instability mechanism.

## Interpretation and next decision

If both loaded and neutral conditions deteriorate at the same k, general integration/threshold difficulty remains plausible; inspect boundary and code strata. If neutral conditions remain strong while loaded ones deteriorate, semantic interference becomes more plausible; crossover contrasts localize labels/task. If ideal-A trials alone deteriorate, investigate response-code mapping before attributing a general complexity limit. If generated output and raw logits diverge, inspect the saved generation traces. These are hypotheses to prioritize, not unique causal identifications.

After all planned levels, review the complete ladder and k5 primary result before selecting any new general-testing protocol. A follow-up threshold-matched design, larger replication, or revised capability gate requires a new preregistration and authorization. Preserve failures and the current blocked mechanism status.
