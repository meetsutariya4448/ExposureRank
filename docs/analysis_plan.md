# Frozen scientific protocol

Binding source: the approved 19-document ExposureRank blueprint, especially documents 03–08 and 18. This protocol is committed before model fitting. Prior research saw aggregate label distributions and audited exact source/split counts; this is not a blinded new dataset.

Questions: compare trained rankers on common observed-cell holdouts; isolate synthetic feedback observation effects; separately isolate candidate-universe effects; measure linear known-probability corrections and distinguish user, seed and simulator uncertainty.

Exact source archive SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c. Small matrix only, valid published ratios retained; numerical discrepancy and missing-clock policies are frozen. Pair hash and warm support rules, labels, compact model grid, seeds and evaluation designs are in configs/*.json.

Baselines: Beta(1,1) global prior, item completion rate with 20 prior observations, additive logistic bias with L2 .001. MF logit mu+bu+ci+pu dot qi; neural adds concatenated/product embedding ReLU32/ReLU16/dropout.1 interaction. Penalize embeddings and biases; intercept and neural linear weights are not penalized, matching the specified embedding/bias penalty. Use Adam with explicit L2, optimizer weight_decay=0. Center additive bias intercept for identification. Twelve learned configurations, seed17, validation-selected three-seed probability means. Ties within1e-4 use smaller dimension/larger regularization; baseline global/item/bias order; family MF first.

Primary macro all-user nDCG10; primary contrasts selected learned minus selected simple and neural minus MF. Freeze artifacts and all design offsets before final test. Paired user bootstrap5000,97.5% percentile bounds, approximate two-contrast family95%; empirical exchangeability/items fixed, no platform representativeness. Reference means are fixed-panel quantities.

Masking: three registered mechanisms × three densities ×200 independent Bernoulli draws; validation density offsets, epsilon.02, beta2. Fixed candidates/ranks, shared masks. Estimators receive observed records only. HT applies to linear DCG and bounded top10 reward; no normalized-ratio unbiasedness. Report bias/RMSE/MCSE, Wilson reversal and coverage bounds, variance/zero variance and ESS. Floor.10/clipping10,50 secondary if executed.

Candidate sampling: uniform and training-weighted without replacement ×50/100/500 ×200 draws; shared subsets, complete selected outcomes, no forced positives. Restrict frozen ranks, recalculate subset denominators. Full deterministic endpoint once. Reversal defined against reference gap magnitude>.001; otherwise near tie. No joint mask/candidate primary experiment.

Resource preflight uses training/validation only, CPU-first, max8GiB extra RSS and projected eight-hour registered grid. No paid compute or favorable data subset. Fresh final-state reproduction uses exact membership/config/order and numeric tolerances atol1e-7/rtol1e-5/metric1e-6. Reports may remain valid under negative findings.
