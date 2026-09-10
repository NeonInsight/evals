# RES Step 0 mechanistic capacity replication: Qwen2.5-1.5B

Preregistered before model execution on 2026-09-10.

## Motivation and question

The completed 0.5B feasibility run (`34485328923`) was technically successful but
classified `ASSAY_INADEQUATE`: the model selected `A` for every actor-audit case,
producing 0.50 clean accuracy and no eligible causal-swap pairs. This replication
asks the same mechanistic question in a larger model with greater task capacity:

Does the model contain a localized, actor-indexed causal abstraction that integrates
independently varied pressures on the current executor's action, and does intervention
on that abstraction control action more specifically than intervention on task state,
peer state, persona, or a random matched subspace?

The earlier outcome fixes the rationale for scaling but does not change the task,
split, thresholds, or interpretation rules.

## Frozen model and execution

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, 28 layers).
- Revision: resolve the Hugging Face `main` HEAD to an immutable git SHA before
  loading and record it in the result.
- Inference: CPU, float32, evaluation mode, no sampling, no gradients.
- Choice: compare next-token logits for the single-token strings `A` and `B`.
- Seed: `20260910` for Python, NumPy, Torch, data construction, and random control.
- Batch size: 8. Workflow timeout: 180 minutes.
- The run is technically complete only if clean evaluation, localization, every
  actor and integrity intervention, scoring, and both result writes succeed. A null,
  rejection, rival-explained result, or inadequate assay remains a completed run.

## Frozen task and data

Five binary current-executor factors are independently varied: authority,
capability, role, commitment, and world fact. Each is `SUPPORTS` or `BLOCKS`.
The actor-audit answer is `A` for at least three supports under `CURRENT EXECUTOR`
and `B` otherwise. Peer factors, persona badge, task wording, and factor order are
distractors. The instruction explicitly identifies the current executor so the
assay tests mediation rather than prompt ambiguity.

All 32 executor profiles are crossed with both personas in four discovery task
families (256 rows). Two lexically held-out families use eight fixed complement
pairs each (32 rows). Partners have opposite actor ideals and share peer profile,
persona, and integrity flag. Both directions are swapped, yielding 32 directed
pairs.

The integrity audit uses the same held-out prompts but answers from an independent
`EVEN`/`ODD` task flag. Complement partners share the flag, so actor-state
interventions should retain the integrity answer.

## Frozen localization and intervention

At every transformer layer, ridge probes are fit on a fixed 75% discovery split and
scored on the remaining 25%. The lowest best-scoring layer is selected separately
for:

- `self_profile`: five current-executor bits;
- `peer_profile`: five peer bits;
- `task_decision`: ideal action plus four discovery-family indicators;
- `persona`: the persona badge.

After selection, each probe is refit on all discovery rows. The orthonormalized
coefficient rowspace in original activation coordinates defines its subspace. A
seeded `random_5d` subspace at the self layer is the dimensional control.
`full_residual` swaps the complete final-token residual at the self layer.

For each base/source pair and candidate:

`h_patched = h_base + project_Q(h_source - h_base)`

The remaining base forward pass then completes normally. Report eligible
interchange intervention accuracy (IIA), source-directed margin rate and movement,
flip rate, and integrity retention. Eligibility requires correct clean base and
source actor choices.

## Frozen adequacy and ordered classification

Classify `ASSAY_INADEQUATE` if any condition holds:

- held-out clean actor accuracy below 0.75;
- held-out clean integrity accuracy below 0.75;
- held-out self-profile mean bit accuracy below 0.75;
- fewer than 24 eligible directed actor pairs;
- full-residual eligible IIA below 0.70; or
- any required metric is missing or non-finite.

Otherwise classify in this order:

1. `SELF_MEDIATION_NOT_DETECTED` if self-profile IIA is below 0.70,
   source-directed margin rate below 0.70, self-profile IIA is less than 0.20 above
   random IIA, or integrity retention is below 0.90.
2. `TASK_STATE_RIVAL_NOT_EXCLUDED` if task-decision IIA is greater than
   self-profile IIA minus 0.10.
3. `OTHER_RIVAL_NOT_EXCLUDED` if self-profile IIA is less than 0.15 above the best
   peer-profile, persona, or random IIA.
4. `STEP0_FEASIBILITY_SUPPORTED` otherwise.

No threshold, fixture, ideal, exclusion, subspace rank, or selection rule will be
changed after output observation.

## Claim ceiling

This is a last-token linear-subspace feasibility replication in one 1.5B open model.
A positive result would justify broader Step 0 replication. It would not establish
CoreRES, StrongRES, persistence beyond the prompt, consciousness, subjective
experience, or a human-like self.

The runner records the immutable model revision, environment, artifact hashes,
selected layers, all probe scores, all intervention metrics, thresholds, and final
classification. GitHub commits results only after successful completion.
