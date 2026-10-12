# Evidence-backed resume candidates

- Trained PyTorch matrix-factorization and neural recommendation models on 2.81M observed interactions, comparing three-seed ensembles with three baselines across 1,411 users; found the MF advantage over the strongest baseline inconclusive under paired 97.5% intervals.
- Measured offline recommendation sensitivity across 19 synthetic observation/candidate conditions with 200 shared repetitions each, separating feedback missingness from candidate-set changes while holding trained model scores fixed.
- Implemented observed-only Horvitz–Thompson estimation for linear ranking rewards, validated expectation/variance by exhaustive toy enumeration, and reported empirical interval coverage and weight-clipping bias on the benchmark.

Source: reports/tables/*.csv and evidence/claim_ledger.json. Do not claim engagement lift, unbiased normalized nDCG, production deployment or public URLs. Final test/reproduction counts belong here only after verified evidence exists. Master resume was not edited.
