# ModernDiD benchmarks

The benchmark suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io/)
to track how long the estimators take across revisions. A separate command runs
the same workloads through the R packages that implement each method. It
reports how long each side takes and how far moderndid's estimates and standard
errors sit from R's.

## Running benchmarks

Install the locked benchmark environment from the repository root and register
the machine before your first run. The environment also includes R for the
comparison described below.

```console
pixi install --frozen -e benchmark
pixi run -e benchmark benchmark-machine
pixi run -e benchmark bench --quick
```

`spin bench` measures the current checkout in the active environment and prints
results without saving a history. Add `--quick` to run each case once, `-t` to
select a module, class, method, or workload, and `--compare` to measure two
committed revisions in separate ASV environments.

```console
pixi run -e benchmark spin bench -t bench_estimators.ATTgt --quick
pixi run -e benchmark spin bench --compare main HEAD -t bench_estimators.ATTgt
```

Since revision comparisons install only committed package source, commit
library changes before you compare them. The benchmark suite and its
configuration always come from the current checkout and define both runs.
`asv.conf.json` pins the dependencies of the managed environments.
`spin bench --help` lists every option.

`spin asv` runs the lower-level ASV commands that keep a result history.

```console
pixi run -e benchmark spin asv run --show-stderr HEAD
```

Generated environments and results live in `.asv/` and are ignored by Git.
Machine metadata lives in `~/.asv-machine.json`.

## What is measured

- `bench_estimators` times complete calls to `att_gt`, `drdid`, `ipwdid`,
  `ordid`, `ddd`, `cont_did`, and `did_multiplegt`, including their
  preprocessing and inference. The two-period estimators keep the workload's
  units and covariates over two periods.
- `bench_aggregation` times `aggte` and `agg_ddd` on a result fitted during
  setup.

The `baseline` workload in `cases.py` has 1,000 units observed over six periods
with two treated cohorts. Other workloads raise the units to 10,000, the
periods to 12, or the cohorts to five, add four covariates, or turn on the
bootstrap with 999 draws. The `small` workload with 400 units and the `short`
workload with four periods keep quick runs fast. Every timing limits the math
libraries to one thread to keep runs on different machines comparable.

ASV lists each parameter combination of a method as its own row. A timing is
the sample median followed by half the interquartile range. A large second
number means unstable samples that deserve a rerun.

## Comparing with R

`spin compare` runs moderndid and the R package that implements each method on
the same seeded workload. It then reports both timings and how far the results
are apart.

```console
pixi run -e benchmark setup-r
pixi run -e benchmark benchmark-r
pixi run -e benchmark spin compare --estimators att_gt aggte --workloads small baseline
```

The references come from did for `att_gt` and `aggte`, DRDID for `drdid`,
`ipwdid`, and `ordid`, triplediff for `ddd` and `agg_ddd`, contdid for
`cont_did`, and DIDmultiplegtDYN for `did_multiplegt`. `setup-r` installs each
of these from CRAN into R's site library and updates it whenever CRAN has a
newer release. Since R searches that library before conda's, the newer copies
take precedence without overwriting conda's files. A reference whose package is
missing or fails to load shows up as unavailable instead of stopping the run.

Both sides warm up once and then time the same number of calls on one thread.
R times only the estimator call, after its package and the data are loaded. The
report gives the median of each side, their ratio, and the largest gap in
estimates and standard errors over the cells both sides report. Estimates match
when each gap is within 1e-6 + 1e-5 × |R| and standard errors when it is within
1e-6 + 1e-3 × |R|. The command exits with an error when any case differs.

Three cases need some care when you read the report.

- Since both `cont_did` implementations bootstrap their dose-response standard
  errors with different random draws, only their estimates are compared.
- R's contdid stops on the `small` workload because rounding leaves its overall
  weights about 1e-16 short of the exact sum of one it requires. With that
  check relaxed, R's estimates match moderndid's on that workload too.
- The triplediff package matches some cohorts to the wrong units in the
  overall standard error of its group aggregation. The did package had the
  same problem until release 2.3. The comparison leaves that one number out for
  `agg_ddd` and still compares the estimate.

## Writing benchmarks

Timing modules live in the `benchmarks/` folder inside this directory and are
named `bench_*.py`. Workload definitions live in `cases.py`, shared setup
helpers in `common.py`, and the R references in `references.py`. Generate data
and warm up the estimator in `setup`. A `time_` method then measures only the
operation. Estimator benchmarks time the complete public call, including
preprocessing and inference. Aggregation benchmarks fit the estimator during
setup and time only the aggregation.

Keep `params` and `param_names` stable across revisions. Increase the
benchmark's `version` when its workload, setup, or measured operation changes.
Check discovery and run the affected module before requesting review.

```console
pixi run -e benchmark benchmark-check
pixi run -e benchmark bench --quick -t small
```
