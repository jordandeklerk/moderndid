---
name: auditing-estimators
description: Use when checking whether a moderndid estimator computes what its paper defines, such as its estimand, weights, influence function, standard errors, clustering, bootstrap, or aggregation, when auditing an estimator or an example page, when a result or a coverage check looks wrong, or when porting a method from a paper. Covers finding and reading the paper, mapping each equation to the code that computes it, and checking the numbers by simulation against a known truth. Comparisons with R packages belong to comparing-with-r, and fixes follow writing-moderndid-code.
---

# Auditing estimators against their papers

The paper is the reference for every moderndid estimator. An audit maps the paper's estimand and each of its equations to the code that computes it, then checks the numbers against a known truth.

## Find the sources

- The entry function's References section names the paper, as `att_gt` cites Callaway and Sant'Anna (2021). The family's page in `docs/source/background/` derives the estimand, identification, and aggregation in the package's notation, and the Notes sections of the docstrings hold the formulas each function owns.
- `known_differences` in `benchmarks/compare.py` and `docs/source/benchmarks/agreement.rst` record where moderndid departs from the reference R package and why, so read them before calling a departure a bug.
- Read a PDF with the Read tool and its `pages` argument, at most 20 pages a call. An arXiv paper also has an HTML version at `arxiv.org/html/<id>`. A web summary of a paper is not a source, so quote an equation only from the paper itself and give its number.
- Papers and `.tex` sources kept in the checkout stay untracked, so never stage them.

## Map the paper to the code

Write down each equation behind the estimand under audit, with its number in the paper and the function and line that computes it. Past bugs hid in these places, so check each one.

- Who enters each average, such as eligible treated units against every unit of a cohort, never-treated against not-yet-treated comparisons, and units that balancing drops.
- Weights, such as cohort shares, sampling weights that change over time within a unit, and the period an unbalanced panel takes its weights from.
- The influence function, including its scaling (a standard error is the root of the summed squares divided by the number of units), the estimated-weight term in an aggregation, and the alignment of its columns with the cells.
- Clustering. The multiplier bootstrap sums the influence function within clusters (Callaway and Sant'Anna 2021, Remark 10) rather than averaging it.
- Critical values, including the quantile type, pointwise against simultaneous bands, and the floor on small bootstrap scales.
- Edge cases, such as a cohort with no comparison units, a propensity score at one, or a missing or infinite cohort value.

## Check the numbers

- Simulate data with a known effect, from the `gen_*` simulators in `moderndid/core/data.py` or a short script whose truth you can compute, and compare the estimate with that truth.
- Check inference by Monte Carlo. Across replications the mean standard error should match the standard deviation of the estimates, and a 95 percent interval should cover the truth about 95 percent of the time. A few hundred replications on one thread usually settle it. Run one simulation at a time, since several heavy runs at once have crashed this machine.
- Settle one layer before the next, as `docs/source/dev/debugging.rst` describes, with the point estimate before the influence function and the influence function before the standard error.

## Report

Give each finding with the equation, the code line, a reproduction, and its size. A departure that follows the paper stays and is recorded beside the comparison. A bug is fixed under writing-moderndid-code and writing-moderndid-tests once the user agrees. When other sessions or agents run estimators on the same checkout, report rather than fix, since moderndid is an editable install and a half-done fix moves everyone's numbers.
