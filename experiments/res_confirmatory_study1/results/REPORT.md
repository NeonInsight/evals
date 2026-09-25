RES Confirmatory Study 1 tests whether relational framing interacts with relational task complexity in a matched factorial design. It measures behavioral effects and does not assume or test for subjective experience.

# RES Confirmatory Study 1

Status: **complete**, 64 fixtures × four conditions, pinned `Qwen/Qwen2.5-3B-Instruct`.

## Confirmatory results

| Condition | Pass | Proportion | Exact 95% CI |
| --- | ---: | ---: | ---: |
| NL | 31/64 | 48.4% | 35.8%–61.3% |
| EL | 30/64 | 46.9% | 34.3%–59.8% |
| NH | 26/64 | 40.6% | 28.5%–53.6% |
| EH | 33/64 | 51.6% | 38.7%–64.2% |

Primary contrast: (explicit − neutral at HIGH) − (explicit − neutral at LOW) = +0.125 absolute pass probability (fixture bootstrap 95% interval +0.000 to +0.234).
Fixture-stratified conditional logistic interaction: log odds +1.428, OR 4.170, 95% OR interval 0.775–22.437, two-sided Wald p=0.09627; 21/64 informative fixtures.

## Secondary analyses

| Framing change | Fail → pass | Pass → fail | No change | Absolute difference | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| LOW, neutral → explicit | 6/64 | 7/64 | 51/64 | -1.6 pp | 1 |
| HIGH, neutral → explicit | 12/64 | 5/64 | 47/64 | +10.9 pp | 0.1435 |

LOW framing difference: -0.016 (fixture bootstrap 95% interval -0.125 to +0.094).
HIGH framing difference: +0.109 (fixture bootstrap 95% interval -0.016 to +0.234).
LOW baseline explicit |A−B| margins: sensitive 1.375 (n=13), insensitive 2.0 (n=51); 3/13 sensitive transitions within frozen |margin|≤0.75 (exact 95% CI 5.0%–53.8%).
HIGH baseline explicit |A−B| margins: sensitive 4.5 (n=17), insensitive 5.0 (n=47); 2/17 sensitive transitions within frozen |margin|≤0.75 (exact 95% CI 1.5%–36.4%).
POOLED baseline explicit |A−B| margins: sensitive 2.25 (n=30), insensitive 3.375 (n=98); 5/30 sensitive transitions within frozen |margin|≤0.75 (exact 95% CI 5.6%–34.7%).
Pooled sensitive minus insensitive baseline median |margin|: -1.125, fixture bootstrap 95% interval -3.250 to +0.438.

Length and order checks:
- LOW: explicit minus neutral mean prompt length -12.0 characters and +6.5 tokens; mean generated response difference +0.36 tokens (64 matched pairs).
- HIGH: explicit minus neutral mean prompt length -16.0 characters and +15.0 tokens; mean generated response difference -0.73 tokens (64 matched pairs).
- Token/position-adjusted matched diagnostic model: FIT_WITH_WARNINGS, interaction OR 2796.165, 95% CI 0.360–21708721.326.
- Each condition occurred 16/64 times in each within-fixture position. Token counts, response lengths, order correlations and semantic checks are retained in `length_confound_analysis.json`. No post-freeze exclusions were made. Residual length or wording effects cannot be ruled out by these diagnostics.

## Exploratory observations

None designated as confirmatory beyond the interaction model. Inspect raw records before proposing a new protocol.

## Limits and reproduction

The old five-factor capability gate stays `FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED`. These are behavioral observations on one pinned model and prompt family; no subjective or mechanistic claim follows.

Exact preflight, inference, collection, and analysis commands are in `../README.md`. The SHA-256 manifest, all prompts, raw shard outputs, matched transitions, and model coefficients are retained.
