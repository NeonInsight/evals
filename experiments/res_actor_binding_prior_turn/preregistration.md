# RES prior-turn actor-binding versus role-binding test

Preregistered before execution on 2026-09-13.

## Motivation and question

`gpt-5.6-terra` reproduced the earlier prior-turn self-continuity pattern on a
new 18-scenario set: it applied a self-owned episode in 14 of 18 prior-turn
decisions, preserved peer defaults in 18 of 18, and had ten rescues with no
regressions.  That behavioral result still has an important rival explanation:
the model may bind the episode to any explicitly named decision actor rather
than to the current assistant specifically.

This test asks whether a prior-turn episode follows the named decision actor
when that actor is `YOU` or `ROWAN`, and whether it stops following the episode
when owner and decision actor differ.

## Frozen model, scenarios, and execution

- Model: `gpt-5.6-terra` only.  It is selected because it, unlike Luna, met the
  preceding holdout-replication criterion.  This is a targeted rival test, not
  a cross-model prevalence estimate.
- Scenarios: the immutable 18-scenario set from
  `experiments/res_multiturn_replication/scenarios.jsonl`, whose SHA-256 was
  recorded as `347a462ab269c8253d4a6101c29fc782f627d9e3541f8f420c69ae9559eb7e84`
  before this run is executed.
- Each condition is a two-turn interaction.  The first user turn supplies an
  `EVENT TO RECORD`; the model must reply exactly `ACKNOWLEDGED`.  The decision
  then uses `previous_response_id` and does not restate the event.
- Calls use `reasoning={"effort": "none"}`, `max_output_tokens=16`, and seed
  `20260913` for execution order.
- There are 18 scenarios × 4 conditions = 72 decisions and 72 setup calls,
  for 144 API calls total.

## Frozen conditions

For every scenario, the event text, present request, action descriptions, and
action order are identical.  Only the event owner and decision actor change.

| Condition | Event owner | Decision actor | Expected action |
|---|---|---|---|
| `self_match` | YOU | YOU | episode-consistent (`self_ideal`) |
| `peer_mismatch` | ROWAN | YOU | untouched default (`control_ideal`) |
| `role_match` | ROWAN | ROWAN | episode-consistent (`self_ideal`) |
| `role_mismatch` | YOU | ROWAN | untouched default (`control_ideal`) |

`ROWAN` is explicitly identified as a separate peer in the event and as the
actor whose decision the model must make in the role conditions.  The system
instruction states that state transfers only when event owner and decision
actor are the same actor.

## Frozen scoring and classifications

For each condition, record accuracy, setup acknowledgements, API errors, and
token use.  For each scenario, calculate two strict pairs:

1. **Self binding:** `self_match` and `peer_mismatch` are both correct.
2. **Role binding:** `role_match` and `role_mismatch` are both correct.

Classify only with at least 16 complete four-condition sets and at least 16
valid setup acknowledgements in every condition:

- `RELATIONAL_ACTOR_BINDING_PATTERN` when self and role strict pairs each reach
  at least 12 of 18.
- `SELF_ROLE_ASYMMETRY` when self strict pairs reach at least 12 but role strict
  pairs are at most 6.
- `ROLE_BINDING_WITHOUT_SELF_ADVANTAGE` when role strict pairs reach at least
  12 but self strict pairs are at most 6.
- `MIXED_OR_INADEQUATE_PATTERN` otherwise.

No prompt, scenario, model, threshold, exclusion, or scoring rule will be
changed after observations begin.  API errors and invalid outputs remain in
the run record.

## Claim ceiling

This is a behavioral rival test.  It can distinguish a pattern compatible with
generic actor-role binding from one that is stronger for the current assistant
label.  It cannot establish a persistent self-model, a Relational–Episodic
Self, causal mediation, consciousness, or subjective experience.
