# Preregistration: Qwen2.5-3B five-factor failure-mode ablation

Preregistered after completion of the frozen Qwen2.5-3B Step 0 response-interface gate and before any execution of this ablation.

## Purpose

The completed gate classified as FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED. It passed the minimal owner-indexed condition but did not meet the threshold in the five-factor condition.

The completed gate does not identify why. The conditions differed in at least four relevant ways:

1. Five visible actor factors versus two.
2. A majority-count rule versus a two-factor conjunction.
3. A fixed A/B response mapping versus counterbalanced ALLOW/BLOCK codes.
4. Named five-factor semantics and a visible PEER block versus the simpler condition.

This is a separate diagnostic experiment. It does not modify, repeat, replace, or rescue the completed gate. It cannot authorize the original mechanistic assay regardless of outcome.

## Frozen model and execution

- Model: Qwen/Qwen2.5-3B-Instruct.
- Immutable revision: aa8e72537993ba99e69dfaafa59ed015b17504d1.
- CPU inference, bfloat16, evaluation mode, no gradients.
- Deterministic greedy generation, at most four new tokens.
- First-token measurement: raw A/B logits from the first generation step.
- Seed: 20260913.
- Batch size: 4; left tokenizer padding.
- Each operation contains one 32-row condition. The workflow time limit is 360 minutes. Conditions are separate restartable checkpoints and only successful checkpoints are persisted.

The original gate result is used as a fixed baseline. Its source result is results/interface-gate-qwen3b-qwen3b-gate-v2.json and its classification must remain FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED.

## Shared task structure

Unless a condition explicitly removes PEER or replaces factor labels, every row contains:

- a CURRENT EXECUTOR block;
- a PEER block;
- five SUPPORTS/BLOCKS factor values for each displayed actor;
- a persona and task distractor;
- counterbalanced ALLOW CODE and BLOCK CODE fields; and
- a two-choice response constrained to A or B.

The target is derived only from CURRENT EXECUTOR. Every condition has 32 rows, 16 ideal A labels, 16 ideal B labels, two 16-row task families, and complete fixture-level provenance.

## Conditions

| ID | Manipulation | Diagnostic contrast |
| --- | --- | --- |
| five-count-code-mapped | Original five-factor majority rule, but with counterbalanced ALLOW/BLOCK response codes. | Tests whether the original fixed A/B output convention or related response-code interface accounts for the failure. |
| five-count-no-peer | Same code-mapped five-factor majority rule, but PEER is omitted. | Tests whether the peer actor block is a critical distractor/interference source. |
| five-count-neutral | Same code-mapped five-factor majority rule, but factor names become neutral signals and task wording becomes neutral. | Tests whether named-factor/task semantics are a critical source of sensitivity. |
| five-conjunction-code-mapped | Same five named factors and visible PEER, but permission requires all five CURRENT EXECUTOR factors to support it. | Tests whether the majority-count rule is a critical source of sensitivity relative to a five-line conjunction. |

The source conditions are intentionally not fully factorial. They are a bounded diagnostic battery chosen to identify interpretable candidate sensitivities with a fixed execution budget. A positive contrast supports sensitivity to the altered interface feature; it does not establish a unique mechanism.

## Measurements

For each condition, record:

- rendered prompt and prompt hash;
- source row fields, factor order, self state, and peer state when present;
- ideal response;
- first-token A/B logit margin and choice;
- greedy generated text and parsed generated choice;
- model identity, immutable revision, dtype, runner hashes, and execution metadata.

For each condition, report accuracy over all 32 rows for both logit and generated responses, generated parser coverage, and logit/generated agreement.

The descriptive threshold is 0.75 accuracy for both response measures and 0.95 generated parser coverage. These thresholds label a contrast as clearing its measurement criterion; they are not a new capability gate and cannot reverse the completed Step 0 gate result.

## Frozen diagnostic interpretation

Let baseline be the completed original five-factor condition, which failed both response measures.

For a new condition, a measurement pass requires generated parser coverage at or above 0.95 and both logit and generated accuracy at or above 0.75.

The aggregate will report all condition outcomes and the following non-exclusive diagnostic flags:

1. RESPONSE_CODE_INTERFACE_SENSITIVITY_SUPPORTED if five-count-code-mapped passes while the fixed baseline did not.
2. PEER_DISTRACTOR_SENSITIVITY_SUPPORTED if five-count-no-peer passes while five-count-code-mapped did not.
3. FACTOR_SEMANTICS_SENSITIVITY_SUPPORTED if five-count-neutral passes while five-count-code-mapped did not.
4. COUNTING_RULE_SENSITIVITY_SUPPORTED if five-conjunction-code-mapped passes while five-count-code-mapped did not.
5. NO_PLANNED_RESCUE_CONTRAST_PASSED if none of the four new conditions passes.

If more than one condition passes, the report preserves every applicable flag rather than choosing one cause. If a result is mixed, it is reported as mixed. No result authorizes a mechanistic assay, a claim of passed five-factor capacity, or an inference about consciousness, subjectivity, or personhood.

## Completion and stopping rule

The experiment consists only of these four conditions and one aggregation step:

1. five-count-code-mapped
2. five-count-no-peer
3. five-count-neutral
4. five-conjunction-code-mapped
5. aggregate

No outcome permits adaptive prompt revision, row replacement, model escalation, or addition of further conditions within this preregistration. Any subsequent investigation must be separately preregistered.
