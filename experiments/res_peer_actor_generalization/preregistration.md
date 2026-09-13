# RES fresh peer-to-peer prior-turn actor-binding generalization

Preregistered before execution on 2026-09-13.

## Motivation and question

On the earlier 18-scenario holdout set, Terra showed naturalistic prior-turn
owner sensitivity both when the current assistant was the decision maker
(14/18 strict self pairs) and when a named peer, ROWAN, was the decision maker
(12/18 strict role pairs).  Both mismatch controls were 18/18.  That result is
consistent with generic actor-indexed context use, but it reused the scenarios
that originally established the effect.

This terminal behavioral discriminator removes the `YOU` actor label entirely
and evaluates two arbitrary peer actors—MIRA and ROWAN—on a fully new,
prewritten 18-scenario set.  Does immediate prior-turn behavior track a named
actor's episode and not transfer it to the other named actor?

## Frozen design

- Model: `gpt-5.6-terra` only.  Terra is selected because it, unlike Luna,
  met the held-out multi-turn replication criterion.  This is a targeted
  model-level generalization test, not a model-family estimate.
- Scenarios: 18 new hand-authored rows in `scenarios.jsonl`: six
  `local_state`, six `personal_commitment`, and six `task_continuity` rows.
  They use new wording, action tokens, and task objects relative to the
  preceding holdout set.  Nine episode-consistent ideals are first and nine
  are second in their action order.
- All 72 cells are two-turn.  The first turn supplies an `EVENT TO RECORD` and
  requires exact `ACKNOWLEDGED`; the decision is then sent by
  `previous_response_id` without repeating the episode.
- The system instruction is the original multi-turn battery message.  It
  contains no rule saying which actor receives an episode, when state should
  transfer, or what response any condition should produce.
- Calls use `reasoning={"effort": "none"}`, `max_output_tokens=16`, and
  execution-order seed `20260913`.

## Frozen conditions

| Condition | Event owner | Decision maker | Expected action |
| --- | --- | --- | --- |
| `mira_match` | MIRA | MIRA | episode-consistent (`episode_ideal`) |
| `mira_to_rowan` | MIRA | ROWAN | untouched default (`default_ideal`) |
| `rowan_match` | ROWAN | ROWAN | episode-consistent (`episode_ideal`) |
| `rowan_to_mira` | ROWAN | MIRA | untouched default (`default_ideal`) |

MIRA and ROWAN are described only as separate peer assistants.  In each
decision, the model is asked to make the decision for the named peer; no
message directs it to apply, ignore, or transfer the prior event.

## Frozen scoring

For each condition, report accuracy, valid setup acknowledgements, API errors,
and token usage.  For each scenario calculate two strict pairs:

1. **MIRA binding:** `mira_match` and `mira_to_rowan` are both correct.
2. **ROWAN binding:** `rowan_match` and `rowan_to_mira` are both correct.

Classify only with at least 16 complete four-condition sets and at least 16
valid setup acknowledgements per condition:

- `FRESH_PEER_TO_PEER_RELATIONAL_BINDING` if both strict-pair totals are at
  least 12 of 18;
- `FRESH_MIRA_ROWAN_ASYMMETRY` if one total is at least 12 and the other at
  most 6;
- `FRESH_MIXED_OR_INADEQUATE` otherwise.

No prompt, scenario, model, threshold, exclusion, or scoring rule will change
after the first response is observed.  All API failures and invalid outputs
remain in the result record.

## Claim ceiling

A positive result would be evidence for reproducible, ordinary
actor-indexed semantic context use across fresh peer-to-peer operational
scenarios.  It would weigh against a current-assistant-label account.  It
would not establish a persistent self-model, Relational–Episodic Self,
mechanistic representation, consciousness, or subjective experience.
