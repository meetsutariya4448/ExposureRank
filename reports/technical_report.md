# ExposureRank technical report

## Decision and findings

There is no clear evidence of a learned-model advantage over the strongest simple baseline on this fixed test. The comparison intervals include zero. This supports retaining the item baseline as a serious comparator, rather than assuming extra parameters improve the decision. It does not establish equivalence or an online effect.

```csv
model,ndcg10,recall10,recall20,catalog_coverage
global,0.3034025960193612,0.0145207740848951,0.0298296725516049,0.0309588217613465
item,0.8664610193020882,0.0442934874379195,0.0850540157711142,0.0318605350165314
bias,0.8658319782684739,0.0443536821276894,0.084890210025493,0.0318605350165314
mf,0.8674695381296799,0.0444018499959394,0.0850955948865926,0.0450856627592425
neural,0.8659210878543806,0.0442736282116983,0.084806474123723,0.0324616771866546
```

- mf minus item: 0.001009, 97.5% interval [-0.001274, 0.003339].
- neural minus mf: -0.001548, 97.5% interval [-0.004729, 0.001640].

MF/item ordering reversed in 692 of 1800 primary mechanism/density mask repetitions combined, and 118 of 1200 candidate repetitions combined. Conditions have different laws, so these totals are inventory summaries rather than pooled probability estimates; per-condition rates and Wilson bounds are in the reversal tables. The reference MF/item gap is only slightly above the prespecified .001 near-tie cutoff. Other near-tied model pairs remain explicitly classified.

## Dataset, methods and inference

The small matrix contains 4,676,570 unique observed pairs, 1,411 users and 3,327 videos; 17,827 absent cells are excluded. Published finite watch ratios are retained, including 228 minor duration-ratio discrepancies; none alter tested binary thresholds. 181,992 rows lack clocks. Training uses 2,806,319 pairs; validation934,854; test935,397. Test catalogs595–743/user are observed held-out pairs, not all global items. Primary labels are watch_ratio>=1, a completion/repeated-play proxy. Secondary reward min(ratio,2)/2 is bounded.

Training-only priors and additive logistic biases challenge actual CPU PyTorch MF and a small [p,q,p*q] neural interaction model. Twelve configurations use d16/32 and L2 1e-5/1e-4/1e-3, BCE logits, Adam .001, explicit embedding/bias L2, no optimizer decay, max15 epochs and validation early stopping. Selected families use seeds17/29/43, averaged probabilities. Configuration and family selection use only full validation nDCG. Models/samplers/contrasts were locked and committed before test.

All users contribute to macro metrics, including zero-positive users as zero (none in the full binary test). Ties use a label-independent item hash. Two prespecified paired-user comparisons use5000 resamples and97.5% percentile intervals; approximate family95% under empirical exchangeability with fixed items. Selected users/items are not representative platform probability samples. These bounds are sensitivity to user composition, not uncertainty in the known frozen-panel mean.

Outcome masks keep scores/candidates/ranks fixed: independent uniform, popularity and outcome-dependent probabilities at.10/.25/.50, epsilon.02, beta2, offsets fitted on validation. Four additional biased-mask floor.10 conditions at.25/.50 are secondary. All13 conditions run200 masks and share each mask across models. The HT interface accepts only observed values and their probabilities. Known probabilities are synthetic, not recovered historical exposure. HT corrects linear DCG/top10 bounded reward; masked nDCG/Recall are intentionally naive diagnostics, not unbiased corrections. Clipping10/50 changes the target expectation. Normal HT intervals are approximate, assessed by empirical coverage/Wilson MC bounds.

Candidate experiments independently draw50/100/500 observed held-out candidates per user under uniform or training-weighted NumPy choice without replacement,200 common draws per cell. Selected outcomes are complete. No positive is forced. Cached rank restriction is exact; denominators are rebuilt on the selected catalog. Absolute sampled nDCG is a different task. Reference pair gaps with magnitude<=.001 are near ties; otherwise sign changes define reversals. Repetitions are conditional simulation draws, not new user populations. MCSE and Wilson uncertainty are separate from paired-user/bootstrap and training-seed variation.

## Linear correction and failure diagnostics

See linear_correction.csv for matching-target bias/RMSE, empirical versus analytic/estimated variance, ESS/max weights, clipping fraction and empirical coverage. Do not compare HT DCG error to naive nDCG error as if they shared a target. Estimates may exceed the reference reward domain; they are not clipped. Low observed weights/support may create inaccurate normal intervals; measured coverage is reported without a blanket nominal guarantee. The deterministic full-observation endpoint has exactly zero mask variance.

## Resources and reproducibility

Sixteen actual learned-model runs (12 configurations plus four additional selected-seed runs) took 146.92 measured training seconds in total. Maximum recorded process RSS was 0.898 GiB on macOS-26.6.2-arm64-arm-64bit, PyTorch CPU; MPS was not used. Setup, source audit, baseline optimization, simulations, report generation and clean reconstruction are separate timings. See evidence/mf_preflight.json, runs.json and stage manifests. Final clean reproduction status is in evidence/reproduction.json; it must pass before the project is declared complete.

## Limits and publication

Historical2020 static observed-cell benchmark; no future session, satisfaction, causal engagement, real historical propensities or production lift. Data selection/blocked missing cells and shared items limit generalization. Three seeds are a small fitting sensitivity, not a complete training uncertainty distribution. Macro top10 preferences concentrate on a small item set. Coverage of recommended items and statistical intervals are different quantities. Numerical watch-ratio discrepancies have unknown cause; recomputed/capped/threshold sensitivities are reported without test-driven target changes.

KuaiRec authors, https://kuairec.com/, CIKM2022, DOI10.5281/zenodo.18164998. Metadata CC BY versus embedded CC BY-SA conflict is unresolved. Conservatively attribute dataset-derived reports/tables/figures under CC BY-SA4.0, indicate transformations, and exclude raw logs, maps, per-user metrics, scores, embeddings and checkpoints. Original code MIT. This is local preparation; no public publication or rights clearance for model redistribution.
