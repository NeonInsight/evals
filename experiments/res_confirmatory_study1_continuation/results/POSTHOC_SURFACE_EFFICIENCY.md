# Post-hoc surface-form efficiency analysis

**Status: descriptive and post-hoc.** This does not alter or replace the frozen confirmatory analysis.

Efficiency is split into two observable pieces: (1) compactness, measured by signed character and pinned-token savings, and (2) behavioral fidelity, measured by exact choice agreement, correctness transitions, and decision-margin shift. Levenshtein distance remains a surface-edit measure; equality of the six frozen structural metrics is only an operational check that the matched pair retained the same frozen structural signature.

| Matched representation pair | n | mean token savings | choice agreement | pass transitions | accuracy change |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_mapped_low_vs_neutral_low | 128 | +8.00 | 82.0% | 23 | -0.8% |
| code_mapped_high_vs_neutral_high | 128 | +8.00 | 67.2% | 42 | +18.8% |
| nonrel_control_high_vs_neutral_high | 128 | -7.00 | 92.2% | 10 | +3.1% |

Positive savings means the variant is shorter than its Neutral baseline. A shorter prompt with an unchanged choice is evidence of surface efficiency for this pinned model and fixture, not evidence that the two strings are semantically interchangeable in general.

The existing frozen prompts do not instantiate contrasts such as 'happy' versus an emoji or a controlled paraphrase family. Testing that idea directly requires a new preregistered fixture set with relation-preserving paraphrases, relation-changing counterfactuals, and matched non-relational controls.
