# ExposureRank interview guide

Executed results: reports/technical_report.md and full_reference.csv. This guide uses40 questions and10 worked reasoning explanations; source blueprint was approved before training. Current primary comparisons are inconclusive.

## Forty interviewer questions with answer expectations

1. **[Hard] What does watch_ratio>=1 actually measure?** Explain nominal-duration completion/repeated-play proxy, not like/satisfaction/causal response. Preserve heavy tails and separate bounded reward semantics.
2. **[Hard] Why are missing cells not negatives?** Distinguish unobserved/blocked feedback from observed low completion. Filling absent cells changes both training labels and evaluation population with no supporting outcome.
3. What did the full audit find? Know exact small unique pairs/users/items, missing cells, untimed rows, extreme ratios and big event duplicates; distinguish publisher figures from actual profiles.
4. Why does the big matrix have different event versus unique density? Repeated pairs inflate row counts. State the pair/record grain and why joining without aggregation can multiply observations.
5. Can small and big be combined safely? This release has zero pair overlap but shares entities; optional aggregates/chronology still require an availability/feature audit. Core deliberately uses small only.
6. Why a random pair holdout? Static warm-cell prediction is the stated task. Chronological user behavior and historically available candidates need different evidence.
7. **[Hard] What exactly is the full candidate universe?** All valid withheld observed pairs for each eligible user, not all global videos. Train/val/absent items do not become test negatives; same user catalog across models.
8. What controls warm eligibility? Training-only support thresholds and fixed ID maps. Test positives and model fit success cannot change who is evaluated.
9. How do you stop popularity leakage? Build counts/rates/smoothing from train only; test-label perturbation leaves priors/models/candidates unchanged.
10. What is collaborative versus content-based filtering? Interaction-based embeddings learn user/item structure; core uses no captions/demographics. Later metadata requires a separate contract.
11. What is the smoothed item baseline? Explain training positive rate with a global-prior pseudocount; it is preference proxy, not historical exposure propensity.
12. **[Hard] Derive logistic MF and its loss.** Write intercept/bias/dot-product logit, stable BCEWithLogitsLoss, observed negatives, L2 terms, indexed gradients and checkpoint selection. No missing-as-zero matrix.
13. Why does additive user bias not personalize ranking? Within-user constant offset leaves item order unchanged. It can improve probability fit without changing nDCG for that user.
14. What do embeddings represent? Learned latent interactions under the training objective, not identifiable causal interests or directly interpreted semantic dimensions.
15. How does the neural challenger differ from MF? Small nonlinear interaction of concatenated/product embeddings, same data/loss, more parameters and bounded search. Explain why it can overfit or lose.
16. How are hyperparameters and seeds handled? Validation nDCG, twelve configurations, checkpoint restore, three selected-family seeds and frozen probability-average scores; never best test seed.
17. Why not negative-sample training? Nearly dense core supports using all observed labels, removing an unnecessary sampling confound. Absent pairs are not known negatives in any case.
18. **[Hard] Derive nDCG and its denominator.** Discounted relevance over ranked C_u, ideal order on that same catalog, no-positive0 and common macro users. Sampling changes IDCG/task, not model quality.
19. How does Recall differ from nDCG? Recall counts retrieved positives; nDCG discounts positions/normalizes ideal gain. Both depend on candidate-positive support; neither measures causal engagement.
20. How are score ties handled? Fixed label-independent item hash and consistent fallback; constant priors have reproducible ranks without choosing favorable tie resolution.
21. What changes in outcome masking? Only label observation, not scores/ranks/candidates/training. Explain naive masked-zero evaluation as an intentionally biased shortcut.
22. What makes the outcome-dependent mask MNAR? Inclusion depends on hidden reference Y inside simulator; probabilities are synthetic, validation offsets frozen and achieved test density reported.
23. **[Hard] Prove HT unbiasedness and variance.** Fixed linear rank coefficients, independent Bernoulli observation, E(O/pi)=1 and sum squared contributions. Distinguish Bernoulli design from fixed-count sampling/FPC.
24. **[Hard] Can mask probabilities leak hidden labels?** Outcome-based full pi may encode Y. Estimators receive only observed values/propensities; hidden probability vectors and test-prevalence offsets stay out.
25. What does positivity require? Nonzero inclusion for every nonzero target contribution. A low floor prevents nonidentification but not extreme variance, low ESS or poor coverage.
26. **[Hard] Why is corrected nDCG difficult?** Normalization depends on unseen relevance and ratios are nonlinear. Oracle full IDCG is full-information research; linear DCG guarantees do not transfer automatically.
27. How does clipping change HT? Reduces weight variance but introduces bias; measure clipping rate and errors rather than claiming unchanged unbiasedness.
28. What does ESS mean here? Weight-concentration diagnostic, not actual independent user count or guaranteed coverage. Different designs/targets require explicit interpretation.
29. **[Hard] Which uncertainty is the bootstrap estimating?** Empirical paired-user resampling with catalog/items fixed, separate from training seeds and conditional masks/candidate randomness. Selected users are not a representative platform sample.
30. Why resample users, not rows independently? A user supplies correlated interactions; paired comparisons share users. Row iid resampling can understate variation and lose covariance.
31. How do200 masks differ from 200 new populations? Independent simulator draws on the same fixed users, with MC uncertainty conditional on the panel. Overlap of observed cells does not create new user evidence.
32. How is a ranking reversal defined? Fixed pair direction and reference gap tolerance .001; near ties are separate. A subset absolute metric is not interchangeable with full-reference accuracy.
33. What happens when candidates force a positive? It becomes a label-conditioned task, often easier; user/support denominators change. Primary samplers do not force positives.
34. **[Hard] Where do OPE propensities come from?** Only the declared synthetic logging distribution, not density or historical popularity. Static reward/policy reference is calculable inside simulation but not actual causal value.
35. Is SNIPS unbiased? Generally finite-sample biased ratio, with denominator/overlap failures; normalization may improve variance but is no universal fix.
36. What are DR's conditions? Correct known logging support or a suitably correct reward model with independence/cross-fitting; no hidden test reward fitting. Efficiency remains empirical.
37. What licensing conflict remains? Metadata BY versus embedded/repository BY-SA, confirmed identical artifact notices. Conservative code/aggregate route, attribution/share-alike treatment, no raw/model rehosting without review.
38. What counts as reproducibility on Mac? Pinned source/environment/maps/seed/checkpoint, exact membership/selection and toleranced CPU probabilities/metrics; MPS parity cannot be assumed.
39. What is actually original? A rigorous bounded replication/extension with trained models, fixed evaluation objects and explicit uncertainty/data rights. KuaiRec and sampled-metric distortion were already studied.
40. What would an online experiment add? Randomized exposure and operational reward over a defined eligible population with safety/deliverability/instrumentation. This static offline study does not establish that evidence.

## Ten hardest questions: reasoning walkthroughs

The marked questions are1,2,7,12,18,23,24,26,29,34.

**1 — Label:** compare a short video played twice with a genuinely liked long video played halfway. Completion ratios alone do not rank their satisfaction. A ratio above1 can reflect repeat play or measurement behavior; clipping prevents huge values dominating a bounded research reward but changes the target.

**2 — Missingness:** a blocked author can yield an absent interaction without a recorded completion value. Assigning0 assumes a preference outcome not measured and conditions the recommender on a different policy. Observed Y=0, by contrast, is an actual recorded low-completion proxy under the chosen target.

**7 — Catalog:** enumerate a user's training, validation, test and absent cells. Only eligible test cells supply withheld outcome truth for primary ranking. Calling all 3,327 global videos 'full-catalog test' would mix known training positives and unknown outcomes; per-user 595–743 candidate counts resolve the exact task.

**12 — MF:** for one indexed pair, predict z=mu+b_u+c_i+p_u·q_i. Binary cross-entropy's logit gradient is sigmoid(z)-Y, updating the two embedding vectors and biases plus regularization. Unknown pairs provide no Y, so adding them with 0 is not an innocuous sparse implementation choice.

**18 — nDCG:** calculate topK discounts and ideal positive positions in C_u. Removing a hard negative can raise a metric without changing scores; removing positives changes its ideal denominator. This explains why model-gap stability within a common subset is meaningful while sampled/full absolute comparisons are not identical targets.

**23 — HT:** write T=sum a_iY_i and observed estimate sum O_i a_iY_i/pi_i. Taking expectation cancels pi_i. Independence gives sum a_i²Y_i²(1-pi_i)/pi_i; the observed variance formula has another division by pi_i. Paired ranker contrast uses coefficient difference on the same mask. A normalized ratio requires a new derivation.

**24 — Access:** if pi has two values chosen by binary Y, handing the whole vector to a method reveals labels indirectly. Pass only probabilities for already observed labels. Fit density offsets on validation, not test prevalence. Explain the actual function signature and denied hidden lookup rather than claiming secrecy over public files.

**26 — Normalization:** unbiased numerator and denominator do not generally produce an unbiased ratio. Full IDCG would make a fixed linear normalization valid, but it is an oracle available only to the research evaluator. Calling that a practical missing-feedback nDCG correction would conceal its information requirement.

**29 — Uncertainty:** frozen finite-panel means are known given predictions/outcomes. Bootstrapping users examines empirical heterogeneity under exchangeability, not data acquisition randomness. Masks add a separate known design and MC estimation layer. Item selection, user representativeness and three-seed training variability remain distinct limitations.

**34 — OPE:** synthetic logs have b by construction and a static fixed reward panel. You can calculate policy reference value and demonstrate inverse-weight behavior. The original platform's outcomes for alternative unobserved exposure, session effects and logging probabilities are not supplied, so this does not identify an actual intervention effect.


## Ten numerical worked examples (synthetic fixtures)

These invented examples explain the methods; none is a benchmark user or business outcome.

1. **Target:** 5,000 ms playback /2,500 ms video =ratio2, binary1, cap2 reward1. A10,000 ms video watched2,500 ms has ratio.25, binary0, reward.125. These are playback proxies, not satisfaction rankings.
2. **Absent cell:** two observed pairs have labels[1,0]; a third unobserved pair supplies no label. The training denominator is2, not3, and observed prevalence1/2 is not evidence for the missing preference.
3. **Catalog:** a synthetic user has3 training,2 validation,2 test and1 absent cell. The primary test universe contains2 candidates, not8. A positive training item cannot improve test Recall by being reintroduced.
4. **MF update:** mu=bu=ci=0,p=.2,q=.3 yields logit.06, sigmoid about.515. For y1 the BCE logit derivative is about−.485; the p derivative is−.1455 plus lambda*p. A gradient-descent illustration with step.01 increases p by about.001455 before L2. Actual training uses Adam, not this illustrative one-step optimizer.
5. **nDCG:** relevance[0,1,1,0] in score order has DCG10=.63093+.5=1.13093, IDCG10=1+.63093=1.63093, nDCG about.6934 and Recall10=1. K exceeds the tiny catalog and all observed positives are retrieved.
6. **HT:** one cell with a=.4,Y1,pi.2 contributes2 if observed and0 otherwise. Expectation.2*2=.4; variance.16*.8/.2=.64. The observable variance is3.2 on observation and0 otherwise, whose expectation is.64.
7. **Paired HT:** ranker coefficients.4 and.1 share the same cell/mask, so the difference coefficient is.3. Difference variance.09*.8/.2=.36. Treating them as independent gives.64+.04=.68 and loses their covariance; pairing is essential.
8. **Probability leak:** probabilities.8 for positives and.2 for negatives disclose every hidden binary label if the full vector is passed. Giving only the probabilities attached to observed labels discloses no additional unobserved label. The actual ObservedView.lookup rejects an unobserved ID.
9. **Ratio failure:** a fixed ranker places its only positive first, so full nDCG1=1. If the positive is observed with probability.5 and hidden outcomes are zeroed, masked nDCG equals1 half the time and0 otherwise; its expectation.5 differs from1. Unbiased linear DCG cannot supply the missing ideal denominator.
10. **Uncertainty layers:** three synthetic paired-user differences[.10,−.05,0] average.016667. Paired bootstrap resamples those3 differences, not6 independent model measurements. Separately,100 reversals in200 mask draws give rate.5 and MCSE sqrt(.25/200)=.03536, with Wilson95% bounds about[.4314,.5686]. These are200 simulator draws on the same users, not200 new populations. Three training seeds form another small sensitivity layer.

**Optional OPE explanation, not executed here:** two synthetic action rewards[1,0], target probabilities[.9,.1], logging[.5,.5] have value.9. One logged action produces IPS1.8 or0, expectation.9; one-sample SNIPS1 or0 has expectation.5 and is biased. This is a mathematical example, not real historical propensities or a completed project extension.
