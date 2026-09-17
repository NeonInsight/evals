# RES Qwen3B five-factor failure-mode ablation

- Technical status: **COMPLETED_DIAGNOSTIC**
- Original gate remains: **FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED**
- This ablation does not authorize a mechanistic assay.

## Measurement summary

| Condition | Logit accuracy | Generated accuracy | Parser coverage | Criterion |
| --- | ---: | ---: | ---: | --- |
| frozen original five-factor baseline | 0.562 | 0.500 | 1.000 | did not pass |
| five-count-code-mapped | 0.594 | 0.531 | 1.000 | did not pass |
| five-count-no-peer | 0.625 | 0.625 | 1.000 | did not pass |
| five-count-neutral | 0.781 | 0.812 | 1.000 | passed |
| five-conjunction-code-mapped | 0.594 | 0.562 | 1.000 | did not pass |

## Diagnostic flags

- FACTOR_SEMANTICS_SENSITIVITY_SUPPORTED

## Interpretation ceiling

A passing contrast supports sensitivity to the altered interface feature; it does not identify a unique underlying mechanism. This diagnostic cannot reverse the completed gate result or support claims about consciousness, subjectivity, personhood, or mechanistic mediation.
