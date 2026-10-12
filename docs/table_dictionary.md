# Table and uncertainty dictionary

All metrics use the same registered user set and the relevant declared candidate catalog. The five models are global/item/bias/MF/neural. Full-reference values are deterministic for the frozen observed test panel. Public tables are model/condition/repetition aggregates, never real-user rows.

| Table | Interpretation |
|---|---|
| full_reference.csv | Macro full observed held-out nDCG/Recall/DCG/reward, item-union recommendation coverage, recommendation-share concentration |
| primary_contrasts.csv | Two prespecified paired-user nDCG differences;97.5% empirical percentile intervals from5000 shared user draws |
| seed_sensitivity.csv | Three individual fitting seeds/family; descriptive variability, not a large-sample training uncertainty distribution |
| mask_results.csv | Naive missing-as-zero metric error versus the same fixed reference catalog; ranks/candidates unchanged |
| linear_correction.csv | HT/clipped estimates versus matching linear DCG/reward truth; bias/RMSE and mask-design variance/coverage diagnostics |
| candidate_results.csv | Mean metrics on a **different selected-catalog task**; generic bias/RMSE columns are arithmetic shifts/RMS differences from full-reference values, not estimator bias/RMSE for an identical statistical target |
| candidate_reversals.csv | Within-task paired model gaps and conditional draw variability; generic bias/RMSE columns quantify gap shifts across tasks, not a justified full-catalog correction |
| *_reversals.csv | Sign changes only when full reference gap magnitude>.001; near ties kept separate. Wilson95% MC bounds from200 conditional draws |
| *_user_uncertainty.csv | Paired user resampling of each user's mean-over-simulator differences; distinct from repetition-level MC uncertainty |
| target_sensitivity.csv | Fixed ranks evaluated under registered thresholds/caps/recomputed reward; no refitting or target selection |
| order_stability.csv | Descriptive complete leaderboard Kendall correlation, separate from the prespecified near-tie reversal rule |

Do not interpret a higher/lower sampled nDCG as improved/worsened trained-model accuracy on the full catalog. Candidate sample difficulty and ideal/positive denominators change. We report those shifts to expose that interpretation error. Selection weights in candidate sampling are not marginal inclusion probabilities; no HT correction is performed there.

Coverage names differ: catalog_coverage measures the union of recommended videos divided by the observed test-item union; coverage in linear_correction is the observed frequency of approximate normal design intervals containing the known linear reference. Monte Carlo uncertainty in that frequency does not prove exact nominal interval coverage.

Repetitions recur on the same fixed users, not new populations. The user bootstrap assumes empirical exchangeability with items fixed; it does not establish representative platform inference. Known mask probabilities belong to a declared simulator, not original historical logging.
