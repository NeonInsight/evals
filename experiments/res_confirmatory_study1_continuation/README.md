# RES Confirmatory Study 1 — relational-complexity continuation

This directory continues the existing Study 1 sequence with the refined frozen protocol. The completed earlier 64-fixture run remains historical evidence. This continuation uses the original v1 `five-count-code-mapped` representation, adds prospective power and blinded complexity validation, and adds the matched non-relational specificity control.

## Phase 0

Phase 0 must be built and committed while `control.json` remains `preflight`:

```bash
cd experiments/res_confirmatory_study1_continuation
python power_analysis.py
python phase0.py build
cd ../..
python -m unittest discover -s experiments/res_confirmatory_study1_continuation -p 'test_*.py' -v
python experiments/res_confirmatory_study1_continuation/run.py preflight
python experiments/res_confirmatory_study1_continuation/run.py token-audit
```

`phase0.py build` refuses an existing manifest or results directory. `phase0.py verify` checks every entry in `hashes.sha256`, every fixture/rendering hash, validation thresholds, the two-attempt perturbation limit, and byte identity of the canonical v1 artifact.

No target-model inference is performed by the power, fixture, validation, perturbation, historical-comparison, freeze, test, preflight, or tokenizer-audit commands.

## Scored execution

After the complete Phase 0 freeze is committed and pushed, a separate commit may change only `control.json` to `{"attempt": 2, "operation": "run"}`. The workflow launches 16 deterministic shards. Each shard contains eight fixtures and five renderings per fixture, checkpoints after every fixture, and stops cleanly near the wall-clock ceiling.

Resume an incomplete run by incrementing `attempt` in a new commit. Do not use the GitHub “Re-run jobs” button on an old commit. Before skipping stored work, the runner verifies the rendering hash, model/revision, inference configuration, scorer configuration, and prefix order. Any mismatch stops the run.

The collector writes raw `observations.jsonl`, `run_manifest.json`, the fixture-level matched summary, primary and secondary analyses, structural controls, and `REPORT.md`. An incomplete run receives no confirmatory inference and remains resumable.

## Interpretation

This is a behavioral experiment on one pinned model and prompt family. It does not test consciousness, subjective experience, self-awareness, or human-equivalent self-representation. Optimized-label search remains a separate later phase using held-out fixtures.
