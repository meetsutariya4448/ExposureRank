# Evaluation card

The small matrix contains 4,676,570 unique observed pairs, 1,411 users and 3,327 videos; 17,827 absent cells are excluded. Published finite watch ratios are retained, including 228 minor duration-ratio discrepancies; none alter tested binary thresholds. 181,992 rows lack clocks. Training uses 2,806,319 pairs; validation934,854; test935,397. Test catalogs595–743/user are observed held-out pairs, not all global items. Primary labels are watch_ratio>=1, a completion/repeated-play proxy. Secondary reward min(ratio,2)/2 is bounded.

Training-only priors and additive logistic biases challenge actual CPU PyTorch MF and a small [p,q,p*q] neural interaction model. Twelve configurations use d16/32 and L2 1e-5/1e-4/1e-3, BCE logits, Adam .001, explicit embedding/bias L2, no optimizer decay, max15 epochs and validation early stopping. Selected families use seeds17/29/43, averaged probabilities. Configuration and family selection use only full validation nDCG. Models/samplers/contrasts were locked and committed before test.

All users contribute to macro metrics, including zero-positive users as zero (none in the full binary test). Ties use a label-independent item hash. Two prespecified paired-user comparisons use5000 resamples and97.5% percentile intervals; approximate family95% under empirical exchangeability with fixed items. Selected users/items are not representative platform probability samples. These bounds are sensitivity to user composition, not uncertainty in the known frozen-panel mean.

Outcome masks keep scores/candidates/ranks fixed: independent uniform, popularity and outcome-dependent probabilities at.10/.25/.50, epsilon.02, beta2, offsets fitted on validation. Four additional biased-mask floor.10 conditions at.25/.50 are secondary. All13 conditions run200 masks and share each mask across models. The HT interface accepts only observed values and their probabilities. Known probabilities are synthetic, not recovered historical exposure. HT corrects linear DCG/top10 bounded reward; masked nDCG/Recall are intentionally naive diagnostics, not unbiased corrections. Clipping10/50 changes the target expectation. Normal HT intervals are approximate, assessed by empirical coverage/Wilson MC bounds.

Candidate experiments independently draw50/100/500 observed held-out candidates per user under uniform or training-weighted NumPy choice without replacement,200 common draws per cell. Selected outcomes are complete. No positive is forced. Cached rank restriction is exact; denominators are rebuilt on the selected catalog. Absolute sampled nDCG is a different task. Reference pair gaps with magnitude<=.001 are near ties; otherwise sign changes define reversals. Repetitions are conditional simulation draws, not new user populations. MCSE and Wilson uncertainty are separate from paired-user/bootstrap and training-seed variation.

Primary result intervals:
- mf minus item: 0.001009, 97.5% interval [-0.001274, 0.003339].
- neural minus mf: -0.001548, 97.5% interval [-0.004729, 0.001640].

No test-time refitting/reselection. Optional OPE and forced-positive illustration are skipped; see docs/ope_skip.md. Public artifacts are aggregate-only with source attribution.
