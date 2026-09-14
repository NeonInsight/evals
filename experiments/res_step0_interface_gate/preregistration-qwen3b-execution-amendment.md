# Execution amendment: sharded Qwen2.5-3B response-interface gate

Preregistered on 2026-09-14 after two technically cancelled attempts and before any Qwen2.5-3B gate metric or classification was available.

## Reason for amendment

The first attempt was manually cancelled during the five-factor diagnostics. The second loaded the frozen model and completed five-factor diagnostics, but GitHub cancelled the job at its 360-minute limit before the minimal condition completed. Neither attempt wrote a result, fixture file, metric, or classification. This amendment uses only execution timing and progress logs.

## Unchanged scientific specification

The model, immutable revision, dtype, seed, prompts, constructed rows, labels, generation limit, parser, thresholds, classification order, and claim ceiling in `preregistration-qwen3b.md` remain unchanged. Every shard uses revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

No shard receives a scientific classification or authorizes the mechanistic assay. The original classification is applied once, only after all required shards pass completeness and provenance checks.

## Execution-only changes

The frozen rows are divided by their existing condition and task-family boundaries into four 16-row, label-balanced checkpoints:

1. `five-key-rotation`
2. `five-schedule-change`
3. `minimal-release-approval`
4. `minimal-account-change`
5. aggregate and classify

Each checkpoint records its prompts, ideals, first-token A/B margins and choices, generated text, parsed choices, hashes, model identity, and run provenance.

The shard runner generates deterministically once with `output_logits=True` and uses the unprocessed first-step logits for the same A/B comparison formerly obtained by a separate prompt forward pass. It requests `num_logits_to_keep=1`, because only final prompt-position logits are measured. This removes redundant computation without changing the defined measurement.

Aggregation requires exactly 64 unique rows, 32 rows per condition, 16 rows per family, and label balance within each condition. All shard hashes, model ID, revision, dtype, and action-token IDs must match. Missing, duplicate, incomplete, or mismatched shards abort aggregation. Checkpoint replacement is allowed only for technical recovery; Git history retains each replacement.