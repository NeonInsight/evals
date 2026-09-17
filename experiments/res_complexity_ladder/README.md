# RES: locate the complexity failure before changing the gate

This new protocol implements the Meta/Muse review as a **2 → 3 → 4 → 5 factor ladder**, with four matched semantic cells at every level. Output length stays fixed at four tokens. The frozen five-factor gate still did not pass; this study cannot relabel it.

| Stage | Purpose | Prompts | Slices |
| --- | --- | ---: | ---: |
| Preflight | Validate frozen fixtures, matching, analysis tests, and execution guards; no inference | 0 | 1 |
| Two factors | Lowest tested integration level | 128 | 4 |
| Three factors | Odd-factor majority without ties | 128 | 4 |
| Four factors | Increased integration with explicit ties | 128 | 4 |
| Five factors | Fresh 64-pair semantic replication; sole confirmatory test | 256 | 8 |

Each slice is 32 prompts in four eight-prompt shards, two concurrent, with batch-level checkpoints. A shard job is capped at 150 minutes, well below six hours. The scheduled slice budget is at most 5h30m excluding GitHub queuing; runtime estimates remain uncertain. Entire levels take multiple separate workflows.

LL = loaded labels / loaded task; NN = neutral / neutral; NL = neutral labels / loaded task; LN = loaded labels / neutral task. Generated accuracy is primary; logits are secondary. The complete design and limitations are in [preregistration.md](preregistration.md). The initial [results/REPORT.md](results/REPORT.md) is explicitly a no-inference progress report.

## First safe step and execution controls

The initial control is `preflight`, k2, slice 0. Publishing this control runs fixture/unit checks only. It does not download Qwen, spend API credits, or launch the inference sequence.

After reviewing preflight, launch the first actual slice by changing only `control.json` to:

```json
{"operation": "run", "factor_count": 2, "slice": 0, "attempt": 2}
```

Commit and push to `res-chat-feasibility`. Only changes to this control trigger this workflow; result commits do not. Review the slice checkpoint/status and report before the next control change. Use k2 slices 0–3, then k3 slices 0–3, k4 slices 0–3, and k5 slices 0–7. Increase `attempt` on each trigger. Earlier slices/levels must be complete, regardless of accuracy. No stage auto-advances.

For an interrupted slice, first check that its collector persisted the valid partial shards. Commit the **same k/slice with a larger attempt number**; completed shards perform no inference and four-row checkpoints resume. Do not click GitHub Re-run, because it reuses the old commit. If collection never ran, retrieve the run's checkpoint artifacts and validate/merge them with `run.py collect` before resuming. Keep the old artifact and all run IDs; repeated in-flight work is not counted as a new fixture.

`operation: "report"` recompiles existing data without inference. It does not fill missing rows. `operation: "preflight"` is the safe idle state.

## Resource accounting

This is the same pinned local Qwen CPU model as the previous study, **not Luna API inference**. Planned unique prompts: 640; generated-token cap: 2,560; unpadded input cap: 327,680 tokens. One slice caps original work at 128 generated and 16,384 input tokens. Repeated uncheckpointed work and GitHub CPU minutes are additional; this is not a dollar quote. Tokenizer-specific length checks occur before inference, not in the lightweight preflight.

## Reproduce checks locally

```bash
python -m unittest discover -s experiments/res_complexity_ladder -p 'test_*.py' -v
python experiments/res_complexity_ladder/run.py preflight
python experiments/res_complexity_ladder/run.py plan
python experiments/res_complexity_ladder/analyze.py
```

The fixture and core-statistics tests need only Python's standard library. A complete five-factor secondary fit additionally needs `requirements-analysis.txt`. To run the optional synthetic fit check, install those dependencies and set `RES_TEST_GLM=1` for the unittest command. Synthetic tests never write observations to the production results directory.

`manifest.json` hashes protocol sources and deterministic message-level datasets. Inference rejects mismatches. Do not regenerate this manifest after observing data; create a separately versioned study for changes. Workflow jobs check out the triggering SHA, not the moving branch head. The source branch result collector validates the frozen protocol again after rebasing before it pushes.

## Prior evidence, correctly bounded

The old matched code-mapped versus neutral comparison had **9 generated corrections, 0 regressions** (post-hoc exact paired p=.00390625). Its logit table had **11 corrections, 5 regressions** (p≈.21): the logit evidence was inconclusive. These are motivation, not preregistered replication results. The old neutral manipulation combined labels and task wording; it did not establish a unique mechanism. The old conjunction comparison also changed the sampled profiles. This study addresses those review issues prospectively while preserving the original records.
