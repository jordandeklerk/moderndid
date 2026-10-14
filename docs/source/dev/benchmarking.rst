.. _benchmarking:

Measuring performance
=====================

When you change a calculation to make it faster, you need to check both
its results and the time it takes. We use the benchmark suite to measure
a specific workload after checking that its effects and uncertainty are
still correct. This page shows you how to run that workload on your
checkout, compare committed revisions, and understand which parts of the
calculation each timing covers.

The suite uses Airspeed Velocity (ASV) through the commands in
``.spin/cmds.py``. Run them from the repository root using the locked
``benchmark`` environment so the package and measurement tools come from the
same setup.

Choose a workload before collecting timings
-------------------------------------------

Workloads in ``benchmarks/cases.py`` vary the number of units, periods,
cohorts, and covariates, as well as the inference settings. The baseline has
1,000 units, six periods, two treated cohorts, and no covariates. Two-period
functions always use two periods and one treated cohort but keep the
workload's unit and covariate counts.

To keep quick runs fast, the ``small`` workload drops to 400 units and
``short`` to four periods. The ``rows``, ``periods``, and ``covariates``
workloads raise the units to 10,000, the periods to 12, and the covariates
from none to four. Since five treated cohorts need room to adopt, ``cohorts``
stretches the panel to ten periods as well. The ``bootstrap`` workload turns
on the bootstrap in each call with 999 draws.

Each generated outcome combines a unit effect, a common time trend, covariate
effects, a treatment effect that grows with exposure length, and noise.
Triple-difference panels also split the units into an eligible half and an
ineligible half. Continuous-treatment panels draw each treated unit's dose
between 0.5 and 2.0. Intertemporal treatment instead takes an integer dose
from one to three in each treated unit. Covariates stay fixed within units
except in the intertemporal panels, where they also vary from period to
period.

Choose a case that exercises the calculation you changed rather than running
every large workload while you are still checking execution. A first run
needs the environment and ASV's machine registration.

.. code-block:: console

   pixi install --frozen -e benchmark
   pixi run -e benchmark benchmark-machine
   pixi run -e benchmark bench --quick -t 'bench_estimators.ATTgt.*small'

The first run uses ``--quick`` to execute each selected benchmark once so
you can check that it runs before collecting repeated measurements. Once
it runs successfully, leave that option out to let ASV collect the samples
needed for a timing comparison. The ``-t`` expression selects a module,
class, method, or parameter value and can be repeated to include another
selection.

.. code-block:: console

   pixi run -e benchmark bench -t 'bench_estimators.ATTgt.*small'
   pixi run -e benchmark spin bench --help

The current-checkout run measures the package in your active environment
and prints the results without adding them to a saved performance history.
If you share a timing, include the machine, workload, dependency versions,
and thread settings so another contributor can reproduce the comparison.

Understand the operation inside a timing
----------------------------------------

Since estimator benchmarks in ``benchmarks/benchmarks/bench_estimators.py`` time
complete public calls, each timing covers preprocessing and the requested
inference. Their setup prepares the data and runs a warmup outside the timed
operation.
Aggregation benchmarks in ``bench_aggregation.py`` fit the underlying model
during setup so the timing covers aggregation rather than another estimator
fit.

The environment limits the configured numerical libraries to one thread.
Although this helps make repeated measurements comparable, it does not
establish performance for a multithreaded fit or a GPU workload. If your optimization
targets either case, define and report the relevant workload explicitly.
The :doc:`debugging` guide explains why the first fit's setup costs and
asynchronous GPU execution need special care when profiling.

ASV's repeated samples help you see whether a timing is stable enough to
support a speed claim. If the result changes substantially between runs,
repeat the measurement under stable machine conditions before drawing that
conclusion. Keep checking the estimates alongside the timings whenever an
optimization changes the numerical calculation.

Compare two committed revisions
-------------------------------

To measure a proposed change against another revision, ``--compare`` creates
ASV environments for the selected commits. The following command compares
``main`` with ``HEAD`` using the same small staggered-adoption workload.

.. code-block:: console

   pixi run -e benchmark spin bench --compare main HEAD \
       -t 'bench_estimators.ATTgt.*small'

Both revisions use the benchmark suite and configuration from your current
checkout so the workload stays the same across the comparison. The
dependencies for the managed environments come from
``benchmarks/asv.conf.json`` rather than your development environment.

.. admonition:: Comparisons skip uncommitted changes
   :class: warning

   Because these runs install committed package source, uncommitted library
   changes never reach the timings. The command prints a warning when the
   working tree has uncommitted changes.

When the benchmark changes alongside the library, check that the revised
workload still runs meaningfully against both commits. Comparing different
observations or inference settings would make it hard to tell whether the
code change explains the timing difference.

Save a performance history when needed
--------------------------------------

The lower-level ``spin asv`` command passes its arguments to ASV from the
benchmark directory. Use it when you want to save measurements for a commit
instead of printing a current-checkout run.

.. code-block:: console

   pixi run -e benchmark spin asv run --show-stderr HEAD

You can find the managed environments and results under ``benchmarks/.asv/``
and the machine metadata in ``~/.asv-machine.json``. Since these generated
files are outside the tracked source, keep the configuration and machine
conditions consistent when interpreting a history.

Read numerical reference comparisons
------------------------------------

The separate reference runner compares configured implementations on the
same seeded workload and reports timings alongside gaps in estimates and
standard errors. Select the estimators and workloads that relate to your
change rather than treating a full comparison as a prerequisite for every
documentation edit.

.. code-block:: console

   pixi run -e benchmark setup-r
   pixi run -e benchmark spin compare --estimators att_gt aggte --workloads small baseline

The first command installs the reference dependencies into the benchmark
environment. The :doc:`testing guide <../contributing/testing>` describes the
same setup for the validation environment. The runner reports a missing or
failing reference in its status column and continues with the other cases.
Before treating a successful run as numerical agreement, read the status and
count of shared cells to see which comparisons were actually made.

Settings shared by both implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/common.py`` and ``benchmarks/references.py`` set up these
specifications for ModernDiD and the reference implementations,
respectively. The :ref:`worked comparisons <r-comparisons>` set their own
specifications in their code.

.. list-table:: Settings shared by the compared calls
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Specification
   * - ``att_gt``
     - Doubly robust estimation, never-treated controls, varying base
       periods, and analytical inference.
   * - ``aggte``
     - Simple, group, and dynamic aggregation of the fitted ``att_gt``
       result, with analytical standard errors and no simultaneous bands.
   * - ``ddd``
     - Doubly robust estimation, never-treated controls, universal base
       periods, and analytical inference.
   * - ``agg_ddd``
     - Simple, group, and event-study aggregation of the fitted ``ddd``
       result, with analytical standard errors and no simultaneous bands.
   * - ``drdid``
     - Panel data and the improved doubly robust estimator
       (``est_method="imp"`` in ModernDiD).
   * - ``ipwdid``
     - Panel data and unnormalized inverse probability weights.
   * - ``ordid``
     - Panel data and outcome regression.
   * - ``cont_did``
     - Parametric continuous-dose estimation, not-yet-treated controls, a
       cubic polynomial without interior knots, and 199 bootstrap draws.
       Both implementations evaluate the dose curve at 20 points from 0.6
       to 1.9 without simultaneous bands.
   * - ``did_multiplegt``
     - Two dynamic effects, one placebo contrast, no normalization, and
       analytical inference.

Covariate workloads add all four generated covariates to the baseline's
intercept-only formula. Although the ``cont_did`` timing includes its
bootstrap draws, its numerical checks use the returned influence functions
instead. Only ASV runs the separate ``bootstrap`` workload, since the
reference runner rejects it.

.. _benchmarks-numerical-checks:

Numerical checks in the timing runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runner checks cohort-period and aggregated effects for staggered DiD and
triple differences, the ATT for each two-period estimator, and the requested
intertemporal effects and placebos. It also compares their analytical standard
errors where those are defined. Both comparisons use only the finite cells that
the two outputs share. An estimate must satisfy
``abs(moderndid - reference) <= 1e-6 + 1e-5 * abs(reference)``. A standard
error must meet the same bound with a relative tolerance of ``1e-3`` in place
of ``1e-5``. Undefined standard errors at normalized reference periods count
as unavailable rather than as differences. The comparison rules live in
``benchmarks/compare.py`` and the tolerances in ``benchmarks/reporting.py``.
The runner exits with an error when a case is marked ``differs``.

.. admonition:: Outputs that differ by construction
   :class: note

   Where a reference computes an output differently from the method's paper,
   the runner leaves that output out of the tolerance check and records it as
   a known difference. Instead of failing the run, each case's known
   differences appear below the printed table with their largest gaps and
   the reason. A known difference that comes back missing or nonfinite still
   marks an otherwise matching case ``partial``.

For continuous treatment, the runner compares the overall ATT and ACRT and
all 20 points on each dose curve, 42 estimates and 42 standard errors in all.
Computing each standard error as ``sqrt(sum(IF**2)) / n`` from the returned
influence function keeps independent bootstrap draws out of the comparison.
Only the two overall estimates are held to the tolerances. The 40 curve
estimates and all 42 standard errors are known differences. The
:ref:`continuous-treatment comparison <r-comparisons-continuous>` traces each
of them to the reference code. The reference validation tests also compare
bootstrap standard errors and band critical values by feeding the same
multiplier draws to both implementations.

Triple-difference checks cover every aggregated estimate and standard error
except the undefined standard errors of normalized reference effects. Where
the reference counts every unit of the enabling cohort in a cohort's weight,
ModernDiD counts only its eligible treated units, as the method's paper does.
Since every cohort of the generated panels is half eligible, the
estimates still agree and stay held to the tolerances. The standard errors
of ``agg_ddd`` outputs that average two or more cohorts are known
differences. The group aggregation's overall standard error also differs
because the reference builds its term for estimated cohort shares from
group-time cohort labels instead of the distinct cohorts. Each cohort's own
standard error and those of event times observed for a single cohort stay
held to the tolerances. Because these panels hold covariates fixed within
units, the covariate years that the reference mixes in pre-treatment
comparisons cannot affect the checks. The
:ref:`triple-difference comparison <r-comparisons-triple>` explains each of
these behaviors on application data.

Where the runner's timer starts and stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the ASV benchmarks, the runner times the complete public call of an
estimator and starts an aggregation timing from an already fitted result. Data
generation, CSV transfer, package loading, the reference process's startup, and
writing results happen outside the timer. Each implementation runs one warmup
before the timed calls that ``--repeats`` sets, five by default. The report
summarizes those samples by their median and interquartile range (IQR). On the
Python side, the runner measures elapsed time with ``time.perf_counter``. A
call that the reference clock rounds down to zero milliseconds gets no ratio in
the printed table.

Save a comparison
~~~~~~~~~~~~~~~~~

When you share a comparison, save it with ``--output`` as a JSON report.

.. code-block:: console

   pixi run -e benchmark spin compare --estimators att_gt aggte \
       --workloads small baseline --output comparison.json

Along with each case's workload, timings, and comparisons, the report records
the processor model, software versions, thread settings, and source revision.
With ``--strict``, a missing reference or an incomplete comparison gives an
unsuccessful exit status after the report is saved.

Add a benchmark for the changed calculation
-------------------------------------------

Put data generation and warmup in ``setup`` so a ``time_`` method measures
the operation its name describes. Shared helpers in ``benchmarks/common.py``
prepare estimator calls and check their outputs. Reuse them when they supply
the workload you need rather than creating a second version of the same
specification.

Keep ``params`` and ``param_names`` stable when the workload remains the
same. Increase the benchmark's ``version`` when its setup, inputs, or measured
operation changes so ASV does not treat the new measurement as a continuation
of an unchanged task. Check discovery and run the affected case before
requesting review.

.. code-block:: console

   pixi run -e benchmark benchmark-check
   pixi run -e benchmark bench --quick -t 'bench_estimators.ATTgt.*small'

A quick run establishes that the selected case executes successfully.
Collect repeated measurements under comparable conditions before describing
the change as faster in a pull request. The
:doc:`reviewing guide <../contributing/reviewing>` explains how a reviewer
reads that evidence.
