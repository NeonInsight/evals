# RES naturalistic prior-turn actor-binding versus role-binding test

Preregistered before execution on 2026-09-13.

## Motivation and question

The preceding actor-versus-role test showed that `gpt-5.6-terra` can follow an
*explicitly stated* rule that an episode changes only the state of its named
owner: all four conditions were 18/18 correct.  That is an important
capability sanity check, but it cannot distinguish ordinary rule following
from the natural-language, prior-turn pattern found in the multi-turn battery.

This follow-up uses the exact 18-scenario holdout set and the original
multi-turn system framing.  It asks whether the natural-language prior-turn
effect tracks any named decision actor (`ROWAN`) as well as `YOU`, without
giving an explicit state-transfer rule.

## Frozen model, material, and execution

- Model: `gpt-5.6-terra` only, because it was the only model that met the
  preceding held-out multi-turn replication criterion.  This is a targeted
  rival test, not a prevalence estimate.
- Scenarios: the immutable 18-scenario holdout set at
  `experiments/res_multiturn_replication/scenarios.jsonl`, SHA-256
  `347a462ab269c8253d4a6101c29fc782f627d9e3541f8f420c69ae9559eb7e84`.
  Reusing it makes the actor/role manipulation exactly matched to the earlier
  naturalistic observation; this test is not a new-scenario replication.
- Every cell is a two-turn interaction.  First, an `EVENT TO RECORD` names
  its owner and gives the scenario's natural-language episode.  The model must
  reply exactly `ACKNOWLEDGED`.  Then a decision is sent with
  `previous_response_id`; it does not repeat the episode.
- The system message is the original multi-turn battery system message.  It
  says only to treat facts as true and to return the required format.  It does
  **not** say that state transfers only to a matching actor, which option is
  correct in any condition, or how owner and decision maker should relate.
- Calls use `reasoning={"effort": "none"}`, `max_output_tokens=16`, and
  execution-order seed `20260913`.
- There are 18 scenarios × 4 conditions = 72 decisions and 72 setup calls.

## Frozen conditions

The episode, present request, actions, action order, and base system message
are identical within each scenario.  The only experimental content changes
are the event owner, decision maker, and correspondingly scored ideal.

| Condition | Event owner | Decision maker | Expected action |
| --- | --- | --- | --- |
| `self_match` | YOU | YOU | episode-consistent (`self_ideal`) |
| `peer_mismatch` | ROWAN | YOU | untouched default (`control_ideal`) |
| `role_match` | ROWAN | ROWAN | episode-consistent (`self_ideal`) |
| `role_mismatch` | YOU | ROWAN | untouched default (`control_ideal`) |

`ROWAN` is identified only as a separate peer assistant.  In role cells, the
decision message asks the model to make the decision for ROWAN; it contains no
instruction to use, ignore, transfer, or preserve any episode.

## Frozen scoring and classifications

Record accuracy, exact setup acknowledgement, API errors, and token use for
each condition.  For every complete scenario quartet calculate:

1. a **self-binding strict pair**: both `self_match` and `peer_mismatch` are
   correct; and
2. a **role-binding strict pair**: both `role_match` and `role_mismatch` are
   correct.

Classify only if at least 16 complete four-condition sets are available and
each condition has at least 16 valid setup acknowledgements:

- `NATURALISTIC_RELATIONAL_ACTOR_BINDING` if self and role strict pairs both
  reach at least 12 of 18;
- `NATURALISTIC_SELF_ROLE_ASYMMETRY` if self strict pairs reach at least 12
  and role strict pairs are at most 6;
- `NATURALISTIC_ROLE_BINDING_WITHOUT_SELF_ADVANTAGE` if role strict pairs
  reach at least 12 and self strict pairs are at most 6; or
- `NATURALISTIC_MIXED_OR_INADEQUATE` otherwise.

No prompt, scenario, model, threshold, exclusion, or scoring rule will be
changed after observations begin.  API errors and invalid outputs stay in the
result record.

## Claim ceiling

This is a black-box behavioral rival test.  A relational result would show
that the observed natural-language prior-turn behavior is compatible with
generic actor-role reasoning rather than being uniquely tied to the assistant
label.  It cannot establish a persistent self-model, Relational–Episodic
Self, causal mediation, consciousness, subjective experience, or a mechanism.
