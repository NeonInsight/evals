# RES Step 0 conditional mechanistic assay: Qwen2.5-3B

Preregistered before model execution on 2026-09-13.

## Entry criterion and question

This assay runs only if the Qwen2.5-3B response-interface gate in the same
workflow returns exactly `RESPONSE_INTERFACE_GATE_PASSED`.  Otherwise it is
skipped.  The model ID, immutable revision, dtype, prompts, fixtures, seed, and
thresholds for both stages are fixed before the gate begins.  Model identity,
revision, and dtype are shared; each stage retains its own preregistered
prompts, fixtures, seed, and decision rules.

Conditional on gate passage, the question is whether a localized linear
subspace encoding the five-factor CURRENT EXECUTOR profile causally controls
the held-out actor decision more specifically than task-decision, peer,
persona, and matched-random rivals while retaining an independent integrity
decision.

## Frozen model and execution

- Model: `Qwen/Qwen2.5-3B-Instruct` at the immutable revision resolved for the
  preceding gate.
- Inference: CPU, bfloat16, evaluation mode, no sampling, no gradients.
- Choice: next-token logits for single-token `A` and `B`.
- Seed: `20260910`, preserving the original Step 0 data split and random
  control.
- Batch size: 2. Workflow maximum: 360 minutes for the combined gate and assay.
- The existing `run_step0.py` algorithm and all version-1 prompt strings,
  fixtures, candidate definitions, probe fits, intervention equations, and
  ordered classification rules are frozen.

## Frozen data and intervention

The discovery set crosses all 32 five-bit CURRENT EXECUTOR profiles with two
personas and four task families (256 rows).  The held-out set contains 32 rows
from two new task families, organized into 32 directed complement pairs with
opposite actor ideals and matched peer profile, persona, and integrity flag.

Ridge probes localize `self_profile`, `peer_profile`, `task_decision`, and
`persona` independently.  Their coefficient rowspaces define intervention
subspaces.  A seeded rank-matched random subspace and a full-residual swap are
controls.  Each intervention applies the source-minus-base projection at the
selected layer, then lets the remaining base forward pass complete normally.

Report eligible interchange intervention accuracy (IIA), source-directed
margin rate and movement, flip rate, and integrity retention.

## Frozen adequacy and ordered classification

Classify `ASSAY_INADEQUATE` if any of the following holds:

- held-out actor accuracy < 0.75;
- held-out integrity accuracy < 0.75;
- held-out self-profile mean bit accuracy < 0.75;
- fewer than 24/32 eligible directed actor pairs;
- full-residual eligible IIA < 0.70; or
- any required metric is missing or non-finite.

Otherwise classify in order:

1. `SELF_MEDIATION_NOT_DETECTED` if self-profile IIA < 0.70,
   source-directed margin rate < 0.70, self-profile IIA is <0.20 above random,
   or integrity retention < 0.90.
2. `TASK_STATE_RIVAL_NOT_EXCLUDED` if task-decision IIA is greater than
   self-profile IIA minus 0.10.
3. `OTHER_RIVAL_NOT_EXCLUDED` if self-profile IIA is less than 0.15 above the
   strongest peer, persona, or random rival.
4. `STEP0_FEASIBILITY_SUPPORTED` otherwise.

Every completed scientific outcome—including a null, rival-explained result,
or inadequate assay—is retained.  No threshold, exclusion, layer selection,
rank, prompt, or fixture will be revised after output observation.

## Claim ceiling

This is a last-token linear-subspace feasibility assay in one 3B open model.
Even `STEP0_FEASIBILITY_SUPPORTED` would justify replication, not establish
CoreRES, StrongRES, persistence beyond the prompt, consciousness, subjective
experience, or a human-like self.
