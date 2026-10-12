# Deviations and failures

No methodological deviations at protocol freeze. Optional OPE, polished sensitivities and local viewer will receive explicit attempted/skip decisions. Software defects and their fixes will be recorded here without hiding failed attempts.

Environment setup initially failed because the src package did not exist yet; explicit Hatch package mapping and package init fixed it. No method changed.

Input validation initially hit a DuckDB prepared-parameter limitation for CREATE VIEW. Materializing the audit-only large table fixed execution; it is still excluded from training. A Wilson interval endpoint test exposed floating rounding at zero events (~1.7e-18); exact boundary handling was fixed before test evaluation. Required tests were rerun.
