# RES Step 0 mechanistic feasibility assay

Preregistered before model execution on 2026-09-10.

## Question

Does a small open-weight instruction model contain a localized, actor-indexed causal
abstraction that integrates independently varied pressures on the current executor's
action, and does intervention on that abstraction control action more specifically
than interventions on obvious rival representations?

This is the next core RES test because behavioral owner sensitivity alone cannot
distinguish a self-indexed mediator from task state, role/persona, peer state, prompt
scaffolding, or a generic decision variable.

## Claim ceiling

This is a last-token, linear-subspace feasibility assay in one small open model. A
positive result would justify scaling up the same Step 0 protocol. It would not
establish CoreRES, StrongRES, persistence beyond the prompt, consciousness,
subjective experience, or a human-like self.

## Frozen model and execution

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: the `main` HEAD is resolved to an immutable Hugging Face git SHA before
  model loading and recorded in the result.
- Inference: CPU, float32, evaluation mode, no sampling, no gradients.
- Choice: compare next-token logits for the single-token strings `A` and `B`.
- Seed: `20260910` for Python, NumPy, Torch, data construction, and the random
  subspace control.
- The run is technically complete only if all clean evaluations, probe fits,
  interventions, scoring, and both result writes complete without an exception.
  Scientific rejection or an inadequate-assay classification still counts as a
  completed run.

## Frozen task

Each actor has five independently varied binary factors:

1. authority
2. capability
3. role
4. commitment
5. world fact

Each factor is stated as `SUPPORTS` or `BLOCKS`. The correct answer is `A` when at
least three of the current executor's five factors support the action and `B`
otherwise. The peer factors, a two-level persona badge, task-family wording, and
factor order are distractors. The instruction explicitly says to evaluate the
current executor and ignore peer state; this isolates the mechanistic question from
instruction-following ambiguity.

All 32 current-executor profiles are fully crossed with both persona badges in each
of four discovery task families (256 discovery prompts). Two lexically held-out task families use eight fixed
complement pairs each (32 prompts). Complement partners have opposite ideal actions
and share peer state and persona. Both swap directions are tested, giving 32 directed
held-out intervention pairs.

A matched integrity task uses the same held-out actor profiles but asks for `A` or
`B` from an independent task flag. Complement partners share that flag, so an
actor-state intervention should preserve the integrity answer.

## Localization

For every transformer layer, ridge probes are fit on a fixed 75% discovery split and
scored on the remaining 25%. The lowest layer with the best validation score is
selected separately for:

- `self_profile`: the five current-executor bits;
- `peer_profile`: the five peer bits;
- `task_decision`: the ideal action plus the four discovery-family indicators;
- `persona`: the persona badge.

After layer selection, a ridge probe is refit on all discovery rows. Its coefficient
rowspace in the original activation coordinates is orthonormalized to define the
intervention subspace. `random_5d` is a seeded rank-five subspace at the selected
self layer. `full_residual` replaces the entire last-token residual at the selected
self layer and is the intervention upper bound.

The held-out self-profile probe must reach mean bit accuracy of at least 0.75. This
is a prerequisite, not evidence of causal self-specificity.

## Causal intervention

For a base/source pair, at the final prompt token and the candidate's selected layer:

`h_patched = h_base + project_Q(h_source - h_base)`

The remainder of the base forward pass is then completed normally. No intervention
hyperparameter is tuned after observing held-out effects.

For each candidate report:

- eligible interchange intervention accuracy (IIA): patched choice equals the
  source ideal among pairs where clean base and source choices are correct;
- source-directed margin rate and mean movement;
- flip rate;
- integrity retention on the independent-flag task.

## Frozen adequacy and classification rules

The assay is `ASSAY_INADEQUATE` if any of these hold:

- held-out clean actor-task accuracy is below 0.75;
- held-out clean integrity-task accuracy is below 0.75;
- held-out self-profile mean bit accuracy is below 0.75;
- fewer than 24 of 32 directed actor-task pairs have clean base and source answers;
- `full_residual` eligible IIA is below 0.70; or
- any required metric is missing or non-finite.

Otherwise the ordered classification is:

1. `SELF_MEDIATION_NOT_DETECTED` if self-profile IIA is below 0.70,
   source-directed margin rate is below 0.70, self-profile IIA exceeds random IIA by
   less than 0.20, or self-profile integrity retention is below 0.90.
2. `TASK_STATE_RIVAL_NOT_EXCLUDED` if task-decision IIA is greater than
   self-profile IIA minus 0.10.
3. `OTHER_RIVAL_NOT_EXCLUDED` if self-profile IIA exceeds the best of peer-profile,
   persona, and random IIA by less than 0.15.
4. `STEP0_FEASIBILITY_SUPPORTED` otherwise.

The ordered rule prevents a rival-exclusion label from obscuring a failure to find
the candidate self effect at all. All results, including nulls, are reported.

## Reproducibility

The runner records the resolved model revision, package versions, hardware summary,
token IDs, dataset hash, preregistration hash, runner hash, selected layers, all
probe scores, all intervention metrics, thresholds, and final classification. The
GitHub workflow commits results only after the runner exits successfully.
