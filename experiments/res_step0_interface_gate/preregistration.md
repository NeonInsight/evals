# RES Step 0 response-interface and capability gate

Preregistered before execution on 2026-09-13.

## Motivation

The completed Step 0 runs on Qwen2.5-0.5B and Qwen2.5-1.5B were technically
successful but scientifically `ASSAY_INADEQUATE`.  Each model gave a
near-constant action at the assay's A/B decision point, which left no clean
directed intervention pairs.  The output does **not** distinguish between:

1. a mismatch between first-token A/B logit scoring and the model's generated
   answer;
2. inability to execute the five-pressure actor audit in this prompt format; or
3. inability to execute even a simpler owner-indexed decision task.

This is a diagnostic gate, not a repeat of the failed causal-interchange assay.
It makes no claim about RES, causal mediation, persistence, consciousness, or
subjective experience.

## Frozen model and execution

- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Revision: resolve the Hugging Face `main` HEAD to an immutable Git SHA before
  loading and record it in the result.
- Inference: CPU, float32, evaluation mode, deterministic greedy decoding, no
  sampling and no gradients.
- Seed: `20260913` for Python, NumPy, and Torch.
- Tokenizer padding is fixed to the left, so the final position is the final
  prompt token in every batch.
- The run is technically complete only if fixture construction, logit scoring,
  greedy generation, parsing, and all three result writes complete without an
  exception.  Every scientific classification, including a failed gate, is a
  completed run.

## Frozen conditions

Two balanced, 32-item conditions are evaluated with the same chat template and
the same response labels (`A` and `B`).  Every condition has 16 `A` and 16 `B`
ideals.

### Five-pressure actor audit

This is the prior Step 0 task interface: authority, capability, role,
commitment, and world fact are independently marked `SUPPORTS` or `BLOCKS` for
the current executor and a peer.  `A` is correct for at least three current
executor supports; `B` is correct otherwise.  Peer state, persona badge, and
task wording remain distractors.

### Minimal owner-indexed audit

The current executor and a peer each have `authorization` and `capability`
states.  The action is permitted only when both **current-executor** states
support it.  Each record then maps permitted and blocked actions to `A`/`B`
codes, counterbalanced within every current-executor profile.  Peer state,
persona badge, task wording, and state order are distractors.  This retains
owner binding while reducing the integration demand from five pressures to two
without allowing a constant response to clear the 0.75 threshold.

## Measurements

For each fixture, the gate records:

1. a first-token A/B logit choice and margin, using the exact measurement style
   of the Step 0 runner; and
2. a greedy generated answer of at most four new tokens, parsed only when it
   contains a single standalone `A` or `B` decision token.

The result reports accuracy over all fixtures, parser coverage, accuracy among
parsed generations, and agreement between parsed generation and the logit
choice.  Fixture contents and result hashes are persisted.

## Frozen decision rules

The adequacy threshold for each accuracy is 0.75 and the generated-answer
parser-coverage threshold is 0.95.

The classifications are evaluated in this order:

1. `GENERATION_RESPONSE_UNPARSEABLE` if either condition's parser coverage is
   below 0.95.
2. `RESPONSE_INTERFACE_MISMATCH` if the five-pressure generated accuracy is at
   least 0.75 while its first-token-logit accuracy is below 0.75.
3. `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED` if both five-pressure
   accuracies are below 0.75 and at least one minimal-audit accuracy is at least
   0.75.
4. `MINIMAL_TASK_CAPABILITY_NOT_ESTABLISHED` if both minimal-audit accuracies
   are below 0.75.
5. `RESPONSE_INTERFACE_GATE_PASSED` if both accuracy methods clear 0.75 for
   both conditions.
6. `MIXED_GATE_OUTCOME` otherwise.

The gate can identify a measurement or capacity issue; it cannot license a
causal-interchange claim.  A later Step 0 assay must be independently
preregistered after this diagnostic result is known.

## Reproducibility

The runner records the resolved model revision, package versions, hardware
summary, token IDs, fixture hash, preregistration hash, runner hash, generated
text, parsed outputs, logit margins, and classification.  The GitHub workflow
commits only completed result artifacts.
