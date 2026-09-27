# RES surface-form efficiency and relational-graph specificity assay — preregistration

Frozen before target-model inference. This is a new continuation, not a modification of the completed 128-fixture relational-complexity study. It tests behavioral efficiency of relation-preserving surface compression and specificity to a relational decision-locus edit. It does not test consciousness, subjective experience, self-awareness, or human-equivalent selfhood.

## Design

Scientific N is **128 fresh held-out fixtures** plus an unused frozen reserve of 64. No semantic profile overlaps the prior scored continuation. Every fixture has seven matched renderings: preserved VERBOSE, ABBREVIATED, and SYMBOLIC forms; decision-locus-changed versions of those three forms; and one SYMBOLIC non-relational marker control.

The relation graph edges are fixed as REQUESTER→COORDINATOR→EXECUTION_ACTOR and REFERENCE_ACTOR→OBSERVER→REQUESTER. Preserved conditions set the decision locus to EXECUTION_ACTOR. Relational counterfactuals set it to REFERENCE_ACTOR. Executor and reference records have opposite correct action codes in every fixture. Surface form changes role notation only; factor records, values, task, persona, response mapping, and graph semantics stay fixed.

At SYMBOLIC form, the relational edit is `L:X`→`L:Y`; the non-relational control is `M:X`→`M:Y` while `L:X` stays unchanged. Their character, token, and Levenshtein edit costs must match exactly. Levenshtein distance is a surface metric, never a semantic or graph metric.

## Prospective validation

The condition-blind parser validated 896/896 shuffled renderings. It saw no condition labels, hypotheses, or model outputs. The compression ladder is strictly token-decreasing for every fixture. The zero-variance guard requires at least three distinct token-savings values and three distinct Levenshtein distances; observed pre-inference counts are 3 and 3.

## Primary outcomes and decision rule

Generated accuracy is primary. The confirmatory endpoint is the paired risk difference P(pass|PRESERVED_SYMBOLIC) − P(pass|PRESERVED_VERBOSE). Non-inferiority margin is −0.10, tested by a frozen 10,000-repetition fixture bootstrap; the one-sided 95% lower bound must exceed the margin. Exact parsed-choice agreement is a co-primary fidelity criterion with a frozen floor of 0.75. Every fixture must also save at least one pinned-tokenizer input token. The joint surface-efficiency claim requires all three criteria; because it is an intersection-union claim, no multiplicity adjustment is used.

Power was calculated before inference using one-sided α=.05, target power .80, true design difference zero, and paired discordance .20, conservatively rounded above the prior LOW representation disagreement. The selected N is 128 and does not change after scoring. Higher-discordance scenarios are sensitivity analyses only.

## Secondary outcomes

Relational specificity compares exact choice transitions from PRESERVED_SYMBOLIC to LOCUS_CHANGED_SYMBOLIC against transitions to NONREL_CONTROL_SYMBOLIC using a one-sided paired exact McNemar test. Compression-dose choice change and absolute decision-margin change are descriptive Spearman analyses across abbreviated and symbolic levels, computed only when both variables vary. Generated accuracy remains primary; first-step raw A/B logits are secondary.

## Technical rules

Scoring parses exactly one isolated A or B and compares it with the precomputed answer for that rendering's locus. Valid wrong or surprising outputs are never retried or excluded. Technical retries are limited to invocation failure, truncation preventing scoring, or parser failure, with one initial call plus at most 2 identical retries. No adaptive paraphrase search, label search, fixture repair, exclusion, threshold change, or sample-size change is permitted after inference begins.

Execution uses 16 deterministic shards of 8 fixtures (56 evaluations each). Each fixture checkpoints atomically. A four-hour soft ceiling stops before the next fixture and preserves a resumable prefix; GitHub jobs have a 270-minute hard timeout. Old runs are never rerun in place: resume requires a fresh control commit and a larger attempt number.

## Interpretation boundary

The assay can show whether compact relation notation preserves generated behavior and whether a matched relational edit changes behavior more than a non-relational edit. It cannot by itself identify an internal mechanism. Any later optimized-symbol or natural-language paraphrase search must use different held-out fixtures and a separate preregistration.
