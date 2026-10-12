# ExposureRank blueprint conformance audit

This is a conformance review against the original ExposureRank planning documents, especially binding specification18 and contracts03–07/11. The original v1.0.0 tag and release remain intact. See evidence/blueprint_conformance.json for executed checks and evidence/reproduction.json for the new clean rebuild. This audit preserves the original model selection, checkpoints, score lock, primary contrasts and random streams.

## Findings and repairs

The original implementation completed the core experiments but omitted three specified diagnostics. The audit added them rather than treating the original completion statement as sufficient evidence:

1. Retain per-user, per-repetition primary-contrast differences. There are now19 private condition tensors, each200×2×1411. Only aggregate summaries enter the release.
2. Report explicit95% Monte Carlo intervals for conditional simulator means, alongside MCSE, Wilson reversal/coverage bounds and the separate paired-user bootstrap. These intervals quantify simulation precision on the fixed reference panel, not platform generalization.
3. Run the registered discrepancy-exclusion reward sensitivity. It excludes41 test pairs out of the228 flagged source discrepancies, retains all1411 test users, and restricts the existing fixed ranks. No models are refitted or targets selected.

Additional checks include validation/test predictive log loss, descriptive three-seed mean/SD/range, finite-probability and index validation, the declared zero-variance popularity fallback, SQL NULL rejection, an independent scalar ranking-metric reference and an actual masked-nDCG bias counterexample. Candidate summary columns now say difference/shift from the full task rather than suggesting estimator bias for the same task.

## Requirement and evidence matrix

| Blueprint requirement | Executed evidence |
|---|---|
| Exact source, grain and restricted rights (02/10/18) | Archive publisher MD5/SHA256 and source member hash; complete SQL audits;4,676,570 distinct observed pairs;17,827 absent cells excluded; rights register and DATA_NOTICE |
| Static targets and pair-hash60/20/20 split (03) |2,806,319 train /934,854 validation /935,397 test;1411 users /3327 items;2000 independently selected actual hash assignments rechecked; no test-positive eligibility |
| Common warm candidate universe and tie rules (03/05) | Minimum training supports1905/user and756/item; common595–743 held-out candidates; independent short/zero-positive/tied-score fixtures |
| Three train-only baselines (04) | Recomputed priors and exact validation scores; fixed additive-bias penalty and convergence evidence; no big-matrix training |
| Actual bounded MF/neural training (04) | Six grid configurations per family; selected seeds17/29/43; all16 checkpoint hashes, parameter counts and validation scores rechecked; original fit revision retained |
| Validation-only selection and pretest lock (03/04) | Recomputed tie preferences, ensemble probabilities and primary contrast selection; assert_lock; protocol→fit→lock Git ancestry; test-label perturbation audit |
| Exact full-reference metrics and paired inference (05) | Full reference and both5000-draw97.5% bootstrap contrasts recomputed; independent scalar metric fixtures; all-user/observed-catalog denominators |
| Frozen outcome-mask laws and linear correction (06) | Every validation-calibrated offset recomputed;13×200 shared masks; observed-only access boundary; exhaustive HT expectation/variance and shared-contrast tests; clipping/floor/coverage diagnostics |
| Separate complete-outcome candidate tasks (07) | Six×200 shared draws, q50/100/500, exact full endpoint, direct-sort restriction tests; no forced positives or candidate HT claim |
| Distinct uncertainty layers and sensitivities (03–07/11) | User bootstrap, seed summaries and conditional Monte Carlo intervals separate; retained19 contrast tensors; thresholds/caps/recomputed and excluded-ratio diagnostics |
| Honest reports and safe walkthroughs (12/15) | Six SVG/PNG scientific figures,26 recomputed headline claims, technical report/memo/cards, two executed aggregate/synthetic notebooks,40 interview questions and worked examples |
| Clean rebuild and restricted local release (11/18) | Fresh clone and pinned environment, complete retraining/simulation, exact choices/scores/draw identities, private tensors and declared numeric-table tolerance; release inventory/checksums and exclusions |

## Scientific conclusion preserved

MF versus the item baseline and neural versus MF remain inconclusive under their registered paired intervals. There is no demonstrated neural superiority, equivalence or online engagement lift. Masking and candidate sampling are artificial evaluation experiments on a static historical observed panel; candidate metrics answer different catalog tasks. HT guarantees concern fixed-rank linear targets under known simulator probabilities, not unbiased normalized nDCG or real historical exposure correction.

## Scope and limitations

Mandatory research gates are the completion claim. Selected polish includes three seeds, sensitivity diagnostics, two executed notebooks and a clean rebuild. This does **not** claim every possible polished extension is built.

- E10 synthetic OPE is explicitly optional and skipped.
- The aggregate interactive viewer is deferred to the separate presentation step described in document12; an aggregate payload alone is not a website or completed viewer. The document18 polished acceptance line includes a viewer, so full presentation polish remains outstanding.
- Positive-anchored candidate illustrations, optional two-level bootstrap, MPS and MLflow are not executed or listed as implemented tools.
- Publication and expanded raw/model distribution remain separate. The unresolved BY/BY-SA license conflict is disclosed; real logs, maps, per-user outputs, embeddings and checkpoints are excluded from the local release.

The independent conformance script is scripts/blueprint_audit.py. Its run-lineage record explicitly labels input/checkpoint linkage as retrospective verification, not a fabricated fit-time log. The original reproduction record is retained as evidence/reproduction_v1.0.json; the current record verifies the scientific sources changed by this audit.
