# ModernDiD benchmarks

The benchmark suite uses [Airspeed Velocity (ASV)](https://asv.readthedocs.io/)
to track how long the estimators take across revisions. A separate command runs
the same workloads through the R packages that implement each method. It
reports how long each side takes and how far moderndid's estimates and standard
errors sit from R's.

The [benchmarking guide](../docs/source/dev/benchmarking.rst) explains which
outputs each timing comparison checks. The commands here help you measure a
change during development. The [Comparisons with
R](../docs/source/r_comparisons/index.rst) section of the documentation checks
each estimator against R on application data without timing it.

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
  `ordid`, `ddd`, `cont_did`, and `did_multiplegt`. Each call covers its
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

To keep a comparison with its environment and individual timing samples,
save a JSON report and a CSV export. `--strict` also makes missing references,
failed calls, and incomplete output coverage produce a nonzero exit status.
An explicitly unsupported workload is recorded as skipped. Both exports are
written before the status check so that an unsuccessful run remains available
for inspection.

```console
pixi run -e benchmark spin compare --workloads baseline --repeats 5 --strict \
    --output baseline.json --csv baseline.csv
```

The report records expected and compared output counts separately. A match
requires the complete set of expected finite comparisons within tolerance.
Missing labels or nonfinite values produce partial coverage. Undefined
standard errors for normalized reference effects are recorded with their
labels. Read those details alongside the maximum differences.

Where a reference package computes an output differently from the method's
paper, the report records that output as a known difference with its largest
gap. The printed report explains each one below the table. A known
difference never fails a run. A missing or nonfinite value among the known
differences still makes the coverage partial.

- For `cont_did`, only the overall ATT and ACRT estimates are held to the
  tolerances. The contdid package evaluates the dose curves on a basis whose
  boundary knots are the grid's endpoints rather than each cohort's observed
  dose range. Its influence functions also hold the cohort weights fixed,
  rescale the cohort shares to sum to one, and scale each ACRT cell by its
  size instead of its treated count. The 40 curve estimates and the 42
  standard errors computed from the returned influence functions are
  therefore known differences. The full bootstrap still runs inside the
  timer.
- For `agg_ddd`, triplediff counts every unit of an enabling cohort in the
  cohort shares. Since only the eligible units receive treatment, moderndid
  counts only them. Because every cohort of the generated panels is half
  eligible, the two weightings give the same estimates. The influence
  function's term for the estimated shares still differs for an output that
  averages two or more cohorts. The standard errors of such outputs are
  known differences. With two or more cohorts in the panel, these are the
  overall standard error of each aggregation and the event-study standard
  errors at event times observed for two or more cohorts. For the group
  aggregation, triplediff also builds that term from group-time cohort
  labels instead of the distinct cohorts. The estimates, the standard error
  of each cohort, and the event-study standard errors at event times
  observed for one cohort stay held to the tolerances.

Triple-difference checks also include pre-treatment estimates. The reference
validation suite compares the bootstrap rule itself by feeding the same
multiplier draws to both languages.

R's contdid stops on the `small` workload because rounding leaves its overall
weights about 1e-16 short of the exact sum of one it requires. The runner
records that reference failure and leaves the installed R package unchanged.

## Writing benchmarks

Timing modules live in the `benchmarks/` folder inside this directory and are
named `bench_*.py`. Workload definitions live in `cases.py`, shared setup
helpers in `common.py`, and the R references in `references.py`. Generate data
and warm up the estimator in `setup`. A `time_` method then measures only the
operation. Estimator benchmarks time the complete public call with its
preprocessing and inference. Aggregation benchmarks fit the estimator during
setup and time only the aggregation.

Keep `params` and `param_names` stable across revisions. Increase the
benchmark's `version` when its workload, setup, or measured operation changes.
Check discovery and run the affected module before requesting review.

```console
pixi run -e benchmark benchmark-check
pixi run -e benchmark bench --quick -t small
```
