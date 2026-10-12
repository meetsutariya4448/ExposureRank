# Decision memo

Keep the smoothed item baseline as a serious default comparator: the registered MF advantage is small and inconclusive. Compare models on exactly the same user/candidate population and retain missing feedback as unobserved. Refuse a strong conclusion when superiority depends on the sampling denominator, outcome-observation mechanism, a near tie, or uncertain heavy-weight correction.

- mf minus item: 0.001009, 97.5% interval [-0.001274, 0.003339].
- neural minus mf: -0.001548, 97.5% interval [-0.004729, 0.001640].

The outcome/candidate experiments measure evaluation fragility while trained scores remain fixed. Report each condition with MC bounds; do not choose a spectacular sampled winner. Known-probability HT helps estimate matching linear rewards only under this explicit simulator design. It does not repair arbitrary normalized ranking metrics or identify a causal online improvement. An online test would require a new randomized-exposure design and real operational outcomes.

Attribution/license boundaries: DATA_NOTICE.md.
