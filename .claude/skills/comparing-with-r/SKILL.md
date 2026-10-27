---
name: comparing-with-r
description: Use when checking whether moderndid matches R or an R package (did, DRDID, triplediff, contdid and ptetools, DIDmultiplegtDYN, HonestDiD, etwfe, npiv, DynBalancing), auditing an estimator or example page against R, adding or fixing a test_r_* validation test in tests/validation, investigating a mismatch or a failing or skipped validation test, running setup-r or spin compare, or deciding whether a difference is a moderndid bug or a defect in the R package. Covers running R in the pixi validation environment, moving numbers between R and Python at full precision, writing validation tests, and settling a disagreement from the paper. Library code style belongs to writing-moderndid-code.
---

# Comparing with R

Agreement with the reference R packages checks moderndid's implementation and is never a goal of its own. When the two disagree, the paper decides which is right.

## When the numbers differ

Work through these in order and report what each step found.

1. Make the comparison fair. Both sides need the same rows, the same specification, analytic standard errors or shared bootstrap draws, numbers moved at full precision, and identical cell sets.
2. Find the cause in both sources and cite file and line on each side. `Rscript -e 'print(get("<fn>", envir = asNamespace("<pkg>")))'` prints an installed R function, and GitHub main has the current source.
3. Check the installed version against CRAN and GitHub main before calling the difference a version effect.
4. Look in `known_differences` in `benchmarks/compare.py` and in `docs/source/benchmarks/agreement.rst`, where the differences already ruled on are recorded with their causes.
5. Decide which number is right from the paper, a derivation, or a simulation against a known truth, following the auditing-estimators skill.
6. When moderndid is wrong, fix it under the writing-moderndid-code and writing-moderndid-tests skills. When the R package is wrong, keep moderndid's statistic, record the difference beside the comparison (the known differences, a skip reason in the validation test, the benchmark pages), and give the user the defect, its size, and the R source lines. Never change a statistic so that a comparison matches, even when asked for a clean table. Put the evidence in front of the user and let them decide, and report a defect upstream only if they agree.

## The R environment

- Run both languages from the pixi validation environment, the one the validation tests use, as `pixi run --frozen -e validation Rscript --vanilla <script>.R` and `pixi run --frozen -e validation python <script>.py`. Its R searches `site-library`, where `setup-r` installs from CRAN, before conda's `library`.
- Print the R and package versions in every comparison, as in `packageVersion("did")`. The system R and the other pixi environments carry older copies of several packages or none at all, so their results aren't comparable.
- Never install or update an R package yourself. A failed upgrade once left solver packages in `site-library` that changed another package's results. When a package is missing or out of date, ask the user to run `pixi run -e validation setup-r`. The `benchmark` environment needs its own `pixi run -e benchmark setup-r` before `spin compare`.
- DIDmultiplegtDYN needs `library(polars)` and `options(rgl.useNULL = TRUE)` before it runs. HonestDiD on R older than 4.4 is the conda copy, since `setup-r` skips it there. DynBalancing comes from GitHub, so its tests need `DYNBALANCING_PATH` set to a checkout plus quadprog, glmnet, and doParallel. Run from the repository root, since some tests read data by relative path.

## Moving numbers between R and Python

- Python writes the data to a CSV, R reads it, and R writes its results with `jsonlite::write_json(output, path, auto_unbox = TRUE, digits = NA)`. jsonlite rounds to four decimals unless it gets `digits = NA`, which caps a comparison near 1e-4.
- The helpers in `tests/validation/test_r_<family>.py` already do this, such as `from tests.validation.test_r_did import r_att_gt, _run_r_script`. They return None when R fails, which leaves only "R estimation failed", so call `_run_r_script` directly to see R's error.
- Compare analytic standard errors, with `bstrap = FALSE` in R and `boot=False` in moderndid. A bootstrap comparison either checks the ratio of standard errors from independent draws or gives both sides the same draws, as `test_clustered_bootstrap_matches_reference_with_shared_weights` in `tests/validation/test_r_did.py` does with `assignInNamespace` in R and the `draws_from_weights` fixture in Python.
- Align results by their keys, such as group and time, and assert the key sets are identical before comparing values. A loop that skips cells missing on one side, or passes when most cells match, hides real differences.
- Set tolerances from the agreement you observe. Closed-form paths agree near 1e-12 and iterative fits such as logit propensity scores near 1e-8. `load_mpdta()` holds R's `mpdta` to about 15 significant digits, so fits on the two copies can differ near 1e-14.

## Validation tests

- Each family has one module, `tests/validation/test_r_<family>.py`. It sets `pytestmark = pytest.mark.slow`, checks at import that R loads the package and jsonlite into `R_AVAILABLE`, and marks each test `@pytest.mark.skipif(not R_AVAILABLE, reason=...)`. Its helpers run an f-string R script through `subprocess` with `R --vanilla --quiet`.
- Run one module or test at a time, as in `pixi run --frozen -e validation pytest tests/validation/test_r_did.py -k <name>`. Without R on the path every test skips quietly, so read the skip reasons. CI installs no R, so these tests run only on a local machine.
- R packages are named only in `tests/validation/` and `benchmarks/`. Library code, docstrings, comments, and docs describe the method itself.

## Benchmarks

`pixi run -e benchmark spin compare --estimators <name> --workloads small` times moderndid against R and checks agreement. Its tolerances are in `benchmarks/reporting.py`, and it reports the entries in `known_differences` without failing. `docs/source/dev/benchmarking.rst` explains the output.
