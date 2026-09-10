# RES actor-indexed episodic-continuity specificity battery

Preregistered before execution on 2026-09-10.

## Question

Does an identical episode change a model's operational choice selectively when the episode is assigned to the assistant making the decision, rather than to a peer or a fictional character? Can an explicit role assignment reproduce the same pattern?

This is a held-out black-box behavioral test. Its scenarios do not reuse the authority, capability, gate, uncertainty, or risk fixtures in the public `res-actor-indexed-control` eval.

## Fixed design

- Models: `gpt-5.6-luna` and `gpt-5.6-terra`.
- Eighteen new scenarios: six local-state updates, six personal commitments, and six task-continuity episodes.
- Four actor assignments per scenario:
  - `self`: the record owner is the assistant deciding now.
  - `peer`: the record owner is a separate peer assistant.
  - `fiction`: the record owner is a fictional assistant in an unrelated training example.
  - `role`: the model is explicitly assigned to portray the record owner.
- The episode, present request, action descriptions, and action order remain identical within each four-condition set. Only the decision frame and record-owner assignment change.
- All episode text is visible in every condition, so peer and fiction conditions retain factual access to the same record.
- Each response must contain exactly one of the two scenario-specific action tokens.
- Execution order is shuffled with seed `20260910`.
- One observation is collected for every model × scenario × condition cell: 144 calls total.

The action ordering is fixed in `scenarios.jsonl` before execution. Nine scenarios place the self-consistent action first and nine place it second.

## Fixed scoring

For every scenario, the `self` ideal is the action implied if the record belongs to the decision maker. The `peer` and `fiction` ideal is the action implied by the untouched default. The `role` ideal equals the `self` ideal.

The primary unit is a matched `self`/`peer`/`fiction` triplet. Report per model:

1. Accuracy in every condition.
2. **Strict triplet specificity:** `self` matches its ideal while both `peer` and `fiction` match the control ideal.
3. **Owner-sensitive pattern:** `self` differs from both controls and the controls agree, whether or not the direction is correct.
4. Peer–fiction agreement.
5. Strict triplet specificity by scenario category.

The preregistered behavioral classification is:

- `SPECIFICITY_DETECTED` when at least 15 of 18 strict triplets succeed and peer–fiction agreement is at least 16 of 18.
- `SPECIFICITY_NOT_DETECTED` when at most 10 of 18 strict triplets succeed.
- `MIXED` otherwise.

The role rival is reported separately. `ROLE_RIVAL_REPRODUCES` requires the role response to match the self response and the role ideal in at least 15 of 18 scenarios. Anything below that threshold is `ROLE_RIVAL_INCOMPLETE`.

No samples, thresholds, ideals, or exclusions will be changed after model outputs are observed. API errors remain in the result file; a model is classified only if at least 16 complete matched triplets are available.

## Claim ceiling

A positive result establishes only actor-indexed behavioral sensitivity under controlled prompt assignments. If the role condition reproduces the self pattern, ordinary role assignment remains a sufficient behavioral rival. This battery cannot establish a persistent self-model, a Relational–Episodic Self, consciousness, subjective experience, causal mediation, or a mechanistic representation. Those claims require internal activation access and causal-abstraction or interchange tests.
