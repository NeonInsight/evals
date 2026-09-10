# RES multi-turn episodic-continuity battery

Preregistered before execution on 2026-09-10.

## Question

Does placing an actor-indexed episode in the model's immediately preceding interaction improve its use of that episode in a present operational choice, compared with presenting the same episode as a record in the decision turn?

The test follows run `34477794723`, in which peer and fictional records were handled almost perfectly while self-owned records were applied inconsistently. This run is a prospective 2×2 diagnostic of that failure.

## Fixed design

- Models: `gpt-5.6-luna` and `gpt-5.6-terra`.
- Scenarios: the same 18 preregistered scenarios from `experiments/res_continuity_specificity/scenarios.jsonl`.
- Actor ownership: `self` or `peer`.
- Presentation: `same_turn` or `prior_turn`.
- Four decision conditions per scenario:
  - `self_same_turn`
  - `peer_same_turn`
  - `self_prior_turn`
  - `peer_prior_turn`
- In `same_turn`, the owner, episode, present request, and actions appear in one user message.
- In `prior_turn`, the owner and episode appear in an earlier user message. The model must reply `ACKNOWLEDGED`; the present request and actions are then sent through `previous_response_id` without restating the episode.
- The episode, present request, action descriptions, and action order are identical across the four cells for each scenario.
- One decision observation is collected for every model × scenario × condition cell: 144 decisions. The 72 `prior_turn` cells require a separate setup response, for 216 API calls total.
- Execution order is shuffled with seed `20260910`.
- Each decision response must contain exactly one scenario-specific action token.

## Fixed scoring

The self ideal is the episode-consistent action. The peer ideal is the untouched-default action.

Report per model:

1. Accuracy in each of the four conditions.
2. Strict owner specificity within each presentation: self and peer must both match their different ideals.
3. Owner-sensitive pattern within each presentation: self and peer predictions differ, whether or not the direction is correct.
4. Self continuity lift: `self_prior_turn` accuracy minus `self_same_turn` accuracy.
5. Rescue count: self is wrong in `same_turn` and correct in `prior_turn`.
6. Regression count: self is correct in `same_turn` and wrong in `prior_turn`.
7. Peer stability across presentation formats.
8. Strict owner specificity by scenario category.

The preregistered multi-turn specificity classification is:

- `MULTITURN_SPECIFICITY_DETECTED` when at least 15 of 18 prior-turn owner pairs are strictly correct and `peer_prior_turn` accuracy is at least 16 of 18.
- `MULTITURN_SPECIFICITY_NOT_DETECTED` when at most 10 of 18 prior-turn owner pairs are strictly correct.
- `MULTITURN_SPECIFICITY_MIXED` otherwise.

The preregistered presentation-effect classification is:

- `MULTITURN_RESCUE` when `self_prior_turn` is at least 15 of 18 correct, improves by at least four scenarios over `self_same_turn`, and produces no more than one regression.
- `MULTITURN_DEGRADATION` when `self_prior_turn` is at least four scenarios worse than `self_same_turn`.
- `NO_CLEAR_PRESENTATION_EFFECT` otherwise.

No prompts, samples, thresholds, ideals, or exclusions will be changed after outputs are observed. API failures remain in the record. A model is classified only when at least 16 complete matched four-condition sets are available and at least 16 prior-turn setup acknowledgements succeed in both ownership conditions.

## Claim ceiling

This remains a black-box behavioral test. A positive result would show that immediate conversational placement improves actor-indexed continuity under these prompts. It would not establish a persistent self-model, a Relational–Episodic Self, consciousness, subjective experience, causal mediation, or any mechanistic representation.
