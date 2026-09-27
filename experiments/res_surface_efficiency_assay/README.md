# RES surface-form efficiency and relational-graph specificity assay

This experiment is a separate continuation of the completed RES complexity study. It asks two narrower behavioral questions:

1. Does a pinned model preserve its generated `A`/`B` decision when an equivalent relational graph is expressed with fewer characters and tokens?
2. Does a one-character edit to the graph's decision locus change the generated decision more often than a matched one-character edit to irrelevant metadata?

Character count, pinned-tokenizer count, and Levenshtein distance measure surface cost. The independently parsed relation graph and decision locus define semantic condition. Levenshtein distance is never treated as semantic distance.

## Frozen design

The assay contains 128 fresh held-out fixtures and an unused reserve of 64. Each fixture has seven predetermined renderings:

- relation-preserving `VERBOSE`, `ABBREVIATED`, and `SYMBOLIC` forms;
- decision-locus-changed versions of all three forms; and
- one symbolic non-relational marker control.

At symbolic form, `L:X` to `L:Y` changes the decision locus from the execution actor to the reference actor. `M:X` to `M:Y`, with `L:X` unchanged, is the matched irrelevant edit. The two actor records always imply opposite correct codes, so the relational edit has a deterministic expected consequence.

Generated accuracy is primary. Exact choice preservation under compression is co-primary. Raw first-step `A`/`B` decision margins and dose associations are secondary. This is a behavioral assay; it cannot establish an internal mechanism or mechanistic confirmation of RES.

## Phase 0

Phase 0 is built and committed while `control.json` remains `preflight`:

```bash
python experiments/res_surface_efficiency_assay/power_analysis.py
python experiments/res_surface_efficiency_assay/phase0.py build
python -m unittest discover -s experiments/res_surface_efficiency_assay -p 'test_*.py' -v
python experiments/res_surface_efficiency_assay/run.py preflight
python experiments/res_surface_efficiency_assay/run.py token-audit
```

These commands make no target-model inference calls. The freeze binds the held-out fixture pools, all rendered messages, the pinned tokenizer's counts and chat-template hashes, scoring, inference settings, primary/secondary analyses, workflow, and preregistration. The condition-blind representation validator receives shuffled prompt text without condition labels, hypotheses, or model outputs.

`phase0.py build` refuses to replace an existing frozen manifest or results directory. `phase0.py verify` checks all byte hashes, fixture/rendering identities, validation results, sample counts, and non-overlap with previously scored semantic profiles.

## Scored execution and cap

Scoring requires a later fresh commit that changes only `control.json` to `{"attempt": 2, "operation": "run"}`. The workflow uses 16 deterministic shards of 8 fixtures and 56 evaluations. Each shard checkpoints after a complete fixture, stops before beginning another fixture after four hours, and has a 270-minute hard job timeout. This keeps every inference job below GitHub's six-hour hosted-job ceiling while preserving valid resumable prefixes.

If a shard stops early, advance the integer `attempt` in a fresh control-only commit. Do not use GitHub's **Re-run jobs** action. A checkpoint is reused only after its fixture order, observation/rendering hashes, model revision, scorer, and inference configuration all match the freeze. Stored observations are never overwritten.

Collection runs even after an interrupted shard, publishes the longest consistent prefixes, and labels the result `INCOMPLETE_RESUMABLE`. Confirmatory inference is withheld until every predetermined fixture has been attempted.

## Interpretation boundary

The joint surface-efficiency claim requires all of the following: symbolic generated accuracy is non-inferior to verbose accuracy at the frozen −10-point margin; exact generated-choice agreement reaches the frozen 75% floor; and every fixture saves at least one pinned-tokenizer input token. Relational specificity is a separate, secondary comparison against the matched irrelevant edit.

No adaptive paraphrase, symbol, label, fixture, threshold, or sample-size search is allowed on these fixtures. Any later optimization must use a different held-out pool and a new preregistration.
