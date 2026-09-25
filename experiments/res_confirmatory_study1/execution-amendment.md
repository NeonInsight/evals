# RES Confirmatory Study 1 — pre-inference execution amendment

2026-09-25, after the initial freeze commit `851a4db885048c14f24ef7842eedda11e15e59b4`.

GitHub Actions preflight run `36155173482` stopped in `token-audit`: Transformers 4.48.3 raised `NameError: Extension is not defined` while compiling the chat template because the tokenizer-only installation omitted `jinja2`. The `shard` and `collect` jobs were skipped. **Zero scored model calls occurred.** No outputs were inspected and no hypothesis, gate, exclusion, fixture, rendering, order, model, or analysis code was changed in response to results.

Execution repair: pin `jinja2==3.1.6` in the tokenizer-only preflight installation and document it in the README. The source-hashed manifest was revised while control remained `preflight`, and the original was retained as `manifest-preflight-1.json`. All 64 canonical fixture SHA-256 values and all 256 rendered-prompt SHA-256 values must compare equal between manifests. The revised manifest is the inference freeze point. A second preflight must succeed before a separate `run` control commit can start scoring.
