# Phase 0A power and MDE analysis

Primary test: two-sided fixture-stratified conditional logistic interaction, α=0.05. Target power=0.80. SESOI=0.15 absolute probability interaction.

Only earlier RES observations were used. The prior four-cell outcomes calibrate within-fixture dependence and degenerate fixture prevalence; the original 17/32 → 26/32 result and fresh 35/64 versus 38/64 replication anchor plausible pass rates and main effects.

## Prior-data inputs

Prior four-cell degenerate rates: all-fail 0.344, all-pass 0.328, informative 0.328.
Prior framing discordance: LOW 0.203, HIGH 0.266.

## Achieved power at SESOI

| Scenario | Aggregate framing effect | N=64 | N=128 | N=192 | N=256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| replication_anchored_zero_symmetric | +0.000 | 0.494 | 0.852 | 0.965 | 0.992 |
| replication_anchored_zero_nearby_positive | +0.025 | 0.473 | 0.847 | 0.962 | 0.994 |
| replication_anchored_zero_nearby_negative | -0.025 | 0.485 | 0.848 | 0.971 | 0.997 |
| v1_positive_main_effect_allocation | +0.150 | 0.459 | 0.866 | 0.978 | 0.997 |

All-pass and all-fail fixtures were simulated at their empirically calibrated prevalence. They remain in nominal N and contribute no conditional-likelihood information. Each scenario's JSON diagnostics report cell probabilities, correlations, framing and complexity discordance, and degenerate rates.

## MDE grid

| N | MDE grid estimate |
| ---: | ---: |
| 64 | 0.22 |
| 128 | 0.15 |
| 192 | 0.12 |
| 256 | 0.1 |

## Frozen sample size

Selected N: **128 fixtures**.

Selection rule: smallest candidate N whose 95% Monte Carlo lower bound is at least 0.80 in every SESOI scenario.

The MDE is a grid estimate under the replication-anchored symmetric allocation and calibrated dependence structure. Monte Carlo uncertainty and model misspecification remain limitations.
