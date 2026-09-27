# Prospective power analysis

This calculation was completed before target-model inference. The primary generated-accuracy endpoint is the paired compact-minus-verbose risk difference, tested for non-inferiority against −0.10 at one-sided α=.05.

The prior LOW representation pair had 18.0% choice disagreement. The design uses max(20%, historical), giving a frozen discordance assumption of 20%. At a true difference of zero, the continuous normal-approximation requirement is 123.7 fixtures; the first frozen candidate reaching 80% power is **N=128**.

| N | q=.15 | q=.20 | q=.25 | q=.33 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 66.3% | 55.7% | 48.2% | 40.0% |
| 128 | 89.9% | 81.2% | 73.2% | 62.7% |
| 192 | 97.3% | 92.7% | 87.0% | 77.9% |
| 256 | 99.4% | 97.3% | 94.0% | 87.3% |

The higher-discordance columns are declared sensitivity analyses. N remains 128 regardless of their values or any later observed result. Choice agreement, graph specificity, and margin stability are additional frozen outcomes and do not adapt sample size.
