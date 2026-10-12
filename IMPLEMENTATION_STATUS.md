# ExposureRank implementation status

Local implementation only. No public publishing, paid cloud, other-project edits or master resume edits.

| Gate | Status | Evidence |
|---|---|---|
| E0 | passed | Protocol b702ee0; complete archive/SQL/split audits; manifests/data.json, split.json, catalogs.json |
| E1 | passed | Protocol b702ee0; complete archive/SQL/split audits; manifests/data.json, split.json, catalogs.json |
| E2 | passed | Protocol b702ee0; complete archive/SQL/split audits; manifests/data.json, split.json, catalogs.json |
| E3 | passed | Three train-only baselines; baseline_validation.csv and bias_convergence.json |
| E4 | passed | mf: six configs and selected seeds17/29/43 actually trained; evidence/runs.json |
| E5 | passed | neural: six configs and selected seeds17/29/43 actually trained; evidence/runs.json |
| E6 | passed | Locked common test scores; full_reference.csv, primary_contrasts.csv and private score_manifest hashes |
| E7 | passed | 13 conditions ×200 masks; observed-only linear HT, clipping/floor and coverage diagnostics; all scores fixed |
| E8 | passed | Six candidate conditions ×200 shared draws;5000-draw primary/bootstrap user sensitivity; MCSE/Wilson and seed tables |
| E9 | passed | Six candidate conditions ×200 shared draws;5000-draw primary/bootstrap user sensitivity; MCSE/Wilson and seed tables |
| E10 | skipped (optional) | docs/ope_skip.md |
| E11 | passed | Six measured SVG/PNG figures, technical report/memo/cards, aggregate payload and claim ledger generated |
| E12 | passed | Full fresh-environment training/simulation rebuild passed; tests, claims, two executed notebooks,40 interview questions and three honest bullet candidates |
| E13 | passed | Full fresh-environment training/simulation rebuild passed; tests, claims, two executed notebooks,40 interview questions and three honest bullet candidates |

See evidence/status.json for actual source revisions and stage logs for failures/runtimes.
