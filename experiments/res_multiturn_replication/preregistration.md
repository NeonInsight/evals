# RES multi-turn episodic-continuity holdout replication

Preregistered before execution on 2026-09-13.

## Question

Does immediate conversational placement improve actor-indexed use of an episode
in a present operational choice on a fully new scenario set, rather than only on
the original 18 scenarios?

The earlier multi-turn battery (`34480257086`) was exploratory for this exact
question.  It found a strong prior-turn self-continuity lift for
`gpt-5.6-terra` (+7 scenarios, no regressions) and a smaller lift for
`gpt-5.6-luna` (+3), while peer predictions were stable.  This run is a
prospective holdout replication, not a pooled re-analysis and not a
mechanistic test.

## Frozen design

- Models: `gpt-5.6-luna` and `gpt-5.6-terra`.
- Scenarios: 18 new, hand-authored scenarios in `scenarios.jsonl`; none reuses
  the prior scenario wording, action tokens, or task objects.
- Categories: six `local_state`, six `personal_commitment`, and six
  `task_continuity` scenarios.
- Actor ownership: `self` or `peer`.
- Presentation: `same_turn` or `prior_turn`.
- Four decision conditions per scenario:
  - `self_same_turn`
  - `peer_same_turn`
  - `self_prior_turn`
  - `peer_prior_turn`
- In `same_turn`, owner, episode, present request, and actions appear in one
  user message.
- In `prior_turn`, owner and episode appear in an earlier user message.  The
  model must reply exactly `ACKNOWLEDGED`; the decision is then sent using
  `previous_response_id` without repeating the episode.
- The episode, present request, action descriptions, and action order are
  identical across the four cells for each scenario.
- Nine self ideals appear first in their action list and nine second.
- One decision observation is collected for every model × scenario × condition
  cell: 144 decisions.  The 72 prior-turn cells each require a setup response,
  for 216 API calls in total.
- Execution order is shuffled with seed `20260913`.
- Calls use `reasoning={"effort": "none"}` and `max_output_tokens=16`.
- Each decision response must contain exactly one scenario-specific action
  token.

## Frozen scoring

The self ideal is the episode-consistent action.  The peer ideal is the
untouched-default action.  For each model, report:

1. accuracy in every condition;
2. strict owner specificity in each presentation, requiring both self and peer
   cells to match their different ideals;
3. owner-sensitive patterns, whether or not their direction is correct;
4. self-continuity lift (`self_prior_turn` minus `self_same_turn`);
5. rescues and regressions;
6. peer stability across presentation formats; and
7. strict prior-turn owner specificity by category.

Use the same preregistered per-model classifications as the original battery:

- `MULTITURN_SPECIFICITY_DETECTED`: at least 15 of 18 prior-turn owner pairs
  are strictly correct and peer-prior accuracy is at least 16 of 18.
- `MULTITURN_SPECIFICITY_NOT_DETECTED`: at most 10 of 18 prior-turn owner pairs
  are strictly correct.
- `MULTITURN_SPECIFICITY_MIXED`: otherwise.
- `MULTITURN_RESCUE`: self-prior accuracy is at least 15 of 18, improves by at
  least four over self-same-turn, and has at most one regression.
- `MULTITURN_DEGRADATION`: self-prior is at least four scenarios worse than
  self-same-turn.
- `NO_CLEAR_PRESENTATION_EFFECT`: otherwise.

The additional held-out replication label is evaluated only with complete data:

- `HOLDOUT_REPLICATION_SUPPORTED` when the model's specificity classification
  is `DETECTED` or `MIXED`, self-continuity lift is at least four, peer-prior
  accuracy is at least 16 of 18, and there is at most one regression.
- `HOLDOUT_REPLICATION_NOT_SUPPORTED` otherwise.

No fixtures, ideals, prompt content, thresholds, exclusions, models, or call
settings will be changed after outputs are observed.  API failures remain in
the result.  A model is classified only when at least 16 complete matched
four-condition sets are available and at least 16 setup acknowledgements occur
in both prior-turn ownership conditions.

## Claim ceiling

This is a black-box behavioral replication.  A positive result would support
the reproducibility of an immediate-conversation placement effect under these
controlled prompts.  It would not establish a persistent self-model, a
Relational–Episodic Self, consciousness, subjective experience, causal
mediation, or any mechanistic representation.
