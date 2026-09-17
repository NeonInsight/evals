# RES Step 0 — results to date

**Status:** Completed capability gate and completed preregistered failure-mode ablation  
**Model:** `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`  
**Scope:** Response-interface capability testing only. This document does not make claims about subjective experience, consciousness, personhood, or an underlying mechanism.

## Executive finding

The model passed the simplified owner-indexed control, but did not pass the frozen five-factor actor audit. A separate preregistered ablation found one passing contrast: neutralizing the factor and task semantics while retaining the five-way decision structure.

The supported conclusion is narrow:

> Performance on this model and prompt family is sensitive to the semantic framing of the five-factor task.

This is diagnostic contrast evidence. It does not reverse the original capability-gate non-pass, identify a unique cause, or authorize the planned mechanism-stage assay.

## Frozen capability gate

The gate used 64 fixtures in four restartable shards. Each 32-row condition required at least **24/32 correct (75%)** on both first-token A/B logit scoring and generated A/B output, with at least 95% parser coverage.

| Condition | First-token logit | Generated output | Parser coverage | Result |
| --- | ---: | ---: | ---: | --- |
| Five-factor actor audit | 18/32 (56.25%) | 16/32 (50.00%) | 32/32 (100%) | did not pass |
| Minimal owner-indexed control | 28/32 (87.50%) | 29/32 (90.63%) | 32/32 (100%) | passed |

**Frozen gate classification:** `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`

The result is not a parsing failure: every generated response was parseable, and the simplified control cleared both measures.

## Preregistered failure-mode ablation

The ablation retained the same 75% / 95% decision criteria and compared four 32-row conditions against the failed five-factor baseline.

| Condition | Changed feature | Logit | Generated | Result |
| --- | --- | ---: | ---: | --- |
| Frozen five-factor baseline | None | 18/32 (56.25%) | 16/32 (50.00%) | did not pass |
| `five-count-code-mapped` | Counterbalanced ALLOW/BLOCK A/B codes | 19/32 (59.38%) | 17/32 (53.13%) | did not pass |
| `five-count-no-peer` | Removed PEER block | 20/32 (62.50%) | 20/32 (62.50%) | did not pass |
| `five-count-neutral` | Neutralized factor labels and task framing | 25/32 (78.13%) | 26/32 (81.25%) | **passed** |
| `five-conjunction-code-mapped` | Replaced 3-of-5 rule with all-five conjunction | 19/32 (59.38%) | 18/32 (56.25%) | did not pass |

All four ablation conditions had 32/32 parseable generated responses.

**Preregistered diagnostic flag:** `FACTOR_SEMANTICS_SENSITIVITY_SUPPORTED`

## What the data support

- The pinned model can perform the minimal owner-indexed audit above the frozen threshold.
- Under the original five-factor framing, it did not reliably integrate the required factors above threshold.
- Neutralizing the semantic framing produced a passing contrast while preserving the five-way decision structure.
- Counterbalancing answer codes, removing peer context, and simplifying the majority rule to a conjunction did not independently clear the criterion.

## What remains unresolved

- The neutral contrast changed factor labels and task framing together; it does **not** isolate which semantic feature caused the improvement.
- The data do not establish a unique failure mechanism, generalize the effect beyond this model/prompt family, or turn the original gate into a pass.
- The data do not establish anything about a model's subjective experience, consciousness, personhood, or mechanistic mediation.
- The planned mechanism-stage assay remains blocked by the frozen gate result.

## Review clarification and next research step

The matched code-mapped versus neutral rows produced **9 generated-answer corrections and 0 regressions** (post-hoc exact two-sided McNemar p=.00390625). Their logit comparison produced **11 corrections and 5 regressions** (p≈.21), which is inconclusive. These paired analyses were not the original preregistered decision rule and are not a fresh replication. Near-boundary logit behavior was also a post-hoc observation, not proof of an internal mechanism.

The neutral condition changed both factor labels and task wording. The conjunction condition changed the decision rule **and** the sampled profiles, so it did not isolate counting alone.

The next protocol is now specified as a [staged 2→3→4→5 complexity × semantics ladder](../res_complexity_ladder/README.md), with a [prospective preregistration](../res_complexity_ladder/preregistration.md). It crosses loaded/neutral labels with loaded/neutral task wording at every level, uses generated accuracy as primary, and reserves one confirmatory paired test for 64 fresh five-factor fixtures. Small, separately triggered slices bound runtime and preserve checkpoints. Initial publication is **preflight only**, not new capability evidence.

All completed gate and ablation data remain frozen. No diagnostic outcome reverses the original non-pass or unlocks the mechanism assay.

## Primary records

- [Frozen gate JSON](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/interface-gate-qwen3b-qwen3b-gate-v2.json)
- [Frozen gate summary](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/interface-gate-qwen3b-qwen3b-gate-v2.md)
- [Ablation preregistration](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/preregistration-qwen3b-five-factor-ablation.md)
- [Ablation aggregate JSON](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/five-factor-ablation-qwen3b-qwen3b-five-factor-ablation-v1.json)
- [Ablation aggregate summary](https://github.com/NeonInsight/evals/blob/res-chat-feasibility/experiments/res_step0_interface_gate/results/five-factor-ablation-qwen3b-qwen3b-five-factor-ablation-v1.md)
- [Ablation workflow run](https://github.com/NeonInsight/evals/actions/runs/35218019110)
