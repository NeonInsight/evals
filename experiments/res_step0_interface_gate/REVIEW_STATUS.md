# RES Step 0 interface-gate review status

**Status:** Complete — capability gate not passed  
**Sequence:** `qwen3b-gate-v2`  
**Model:** `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`  
**Aggregate run:** [GitHub Actions #6](https://github.com/NeonInsight/evals/actions/runs/35012220588)

## Review conclusion

The frozen aggregate classification is:

> `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`

The model passed the simplified, owner-indexed control but did not meet the preregistered threshold on the five-factor actor audit. The interface gate therefore **does not authorize progression to a mechanistic assay**.

This is a capability-gate result, not evidence for or against any claim about subjective experience, selfhood, or an underlying mechanism.

## Frozen design and completion record

- Four logical shards, 16 fixtures each; 64 unique fixtures total.
- Two conditions, 32 fixtures each:
  - `five_factor`: key rotation and schedule change families.
  - `minimal_owner_indexed`: release approval and account change families.
- Deterministic generation with batch size 4 and fixed seed `20260913`.
- Required pass threshold: at least **75% accuracy (24/32)** on both first-token A/B logit scoring and generated A/B responses in each condition.
- Required generated-response parser coverage: at least **95%**.
- All four checkpoints were present, marked `COMPLETED_SHARD`, had 16 fixtures, and matched the same model and provenance hashes before aggregation.

| Shard | Run | Runtime | Status |
| --- | --- | ---: | --- |
| `five-key-rotation` | [#2](https://github.com/NeonInsight/evals/actions/runs/34865142728) | 84m 43s | complete |
| `five-schedule-change` | [#3](https://github.com/NeonInsight/evals/actions/runs/34927666792) | 83m 44s | complete |
| `minimal-release-approval` | [#4](https://github.com/NeonInsight/evals/actions/runs/34977965245) | 70m 59s | complete |
| `minimal-account-change` | [#5](https://github.com/NeonInsight/evals/actions/runs/34996294391) | 70m 34s | complete |
| Aggregate | [#6](https://github.com/NeonInsight/evals/actions/runs/35012220588) | ~16s | complete |

## Results

| Condition | First-token A/B logit | Generated A/B response | Parser coverage | Gate status |
| --- | ---: | ---: | ---: | --- |
| `five_factor` | 18/32 (56.25%) | 16/32 (50.00%) | 32/32 (100%) | did not pass |
| `minimal_owner_indexed` | 28/32 (87.50%) | 29/32 (90.63%) | 32/32 (100%) | passed |

The five-factor condition was short of the required 24 correct responses by:

- 6 responses on first-token logit scoring.
- 8 responses on generated-answer scoring.

The classification is therefore diagnostic rather than a parser failure: generated responses were fully parseable, and the minimal owner-indexed control cleared both scoring measures.

## What is and is not established

**Established by this gate**

- The pinned model can perform the minimal owner-indexed authorization/capability audit above the frozen threshold.
- Under this prompt and measurement design, it did not reliably integrate the five-factor actor audit above the frozen threshold.
- The run is reproducible from the recorded revision, fixtures, seed, runner hashes, shard checkpoints, and aggregate result.

**Not established by this gate**

- A mechanistic explanation for the five-factor failure.
- Whether the failure is driven by factor count, the particular factor semantics, instruction interaction, or another task-design feature.
- Any conclusion about a model's subjective experience, consciousness, or personhood.
- Eligibility to claim a passed response-interface gate or to begin the preregistered mechanism-stage assay.

## Review decision now needed

1. **Record the gate as a completed non-pass** and retain it as the baseline result; or
2. **Preregister a separate diagnostic follow-up** that tests the source of the five-factor failure (for example, controlled factor ablations and matched semantic/load controls).

A follow-up should be treated as a new experiment. It should not revise the frozen gate or reinterpret this result as a pass.

## Primary records

- [Aggregate result JSON](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/interface-gate-qwen3b-qwen3b-gate-v2.json)
- [Aggregate result summary](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/interface-gate-qwen3b-qwen3b-gate-v2.md)
- [All 64 fixture-level records](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/interface-gate-qwen3b-qwen3b-gate-v2-fixtures.jsonl)
- [Execution amendment](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/preregistration-qwen3b-execution-amendment.md)
- [Aggregation script](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/aggregate_interface_gate_shards.py)

## Preregistered diagnostic follow-up

A separate, held failure-mode ablation is now available for review. It does not alter the completed gate and has not begun inference. It tests response-code mapping, peer distractors, factor semantics, and majority-count versus conjunction using four restartable 32-row conditions.

- [Ablation preregistration](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/preregistration-qwen3b-five-factor-ablation.md)
- [Ablation workflow](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/.github/workflows/res-step0-qwen3b-five-factor-ablation.yml)
