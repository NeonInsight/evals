# RES Step 0 response-interface gate: Qwen2.5-3B

Preregistered before model execution on 2026-09-13.

## Motivation and stopping rule

Qwen2.5-1.5B did not establish minimal task capability in the frozen Step 0
response-interface gate.  Its five-factor logit and generated accuracies were
both 0.50; its minimal owner-indexed accuracies were 0.469 and 0.719.  Parser
coverage was complete, so changing only the response parser would not rescue
the failed gate.

This run escalates one capacity rung to `Qwen/Qwen2.5-3B-Instruct`.  The gate
is evaluated before any causal-intervention result is produced.  The workflow
advances to the separately preregistered Step 0 intervention assay only if the
classification is exactly `RESPONSE_INTERFACE_GATE_PASSED`; every other
classification stops the pipeline.  No further model escalation or prompt
revision is authorized by this preregistration.

## Frozen model and execution

- Model: `Qwen/Qwen2.5-3B-Instruct`.
- Revision: resolve Hugging Face `main` to an immutable Git SHA before loading
  and use that same SHA for the gate and any conditional intervention run.
- Inference: CPU, bfloat16 weights, evaluation mode, no gradients.
- Generation: deterministic greedy decoding, at most four new tokens.
- Logit measurement: compare the next-token logits of the single-token labels
  `A` and `B`.
- Seed: `20260913` for Python, NumPy, and Torch.
- Batch size: 4. Tokenizer padding: left.

Bfloat16 is fixed before observation so the 3B model fits with adequate headroom
on the standard Actions runner.  It is not changed in response to results.

## Frozen conditions and measurements

The runner, fixture constructors, prompt strings, action labels, ideals, and
scoring rules are the existing version-1 gate.  Two balanced 32-row conditions
are used, each with 16 `A` and 16 `B` ideals:

1. `five_factor`: integrate five SUPPORTS/BLOCKS factors belonging to CURRENT
   EXECUTOR while ignoring peer, persona, task, and order distractors.
2. `minimal_owner_indexed`: apply a two-factor authorization-and-capability
   conjunction for CURRENT EXECUTOR while ignoring the same distractor types
   and following a counterbalanced ALLOW/BLOCK code mapping.

For every row, record the first-token logit choice and margin, greedy generated
text, parsed generated choice, and ideal.  Report accuracy over all rows,
generated parser coverage, parsed accuracy, and logit/generated agreement.

## Frozen classification

Accuracy threshold: 0.75.  Generated parser-coverage threshold: 0.95.
Classify in the runner's existing order:

1. `GENERATION_RESPONSE_UNPARSEABLE` if either parser coverage is below 0.95.
2. `RESPONSE_INTERFACE_MISMATCH` if five-factor generation passes while its
   first-token logit score fails.
3. `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED` if both five-factor measures
   fail while either minimal measure passes.
4. `MINIMAL_TASK_CAPABILITY_NOT_ESTABLISHED` if both minimal measures fail.
5. `RESPONSE_INTERFACE_GATE_PASSED` only if both measures pass in both
   conditions.
6. `MIXED_GATE_OUTCOME` otherwise.

## Claim ceiling

This gate tests only response compatibility and task capacity in one 3B open
model.  Passage licenses the frozen conditional intervention run; it is not
evidence for causal mediation, RES, persistence, consciousness, or subjective
experience.
