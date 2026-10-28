.. _benchmarking:

Measuring performance
=====================

When you change a calculation to make it faster, you need evidence that it runs
faster on the work it was meant to speed up and that its results stay the same.
Because timings move with the machine, the data, and the thread settings as much
as with the code, that evidence needs a fixed workload measured the same way on
both sides of the change. This guide shows you how to find where a fit spends
its time, how to time a workload on your checkout, and how to compare two
committed revisions. Its last section checks the estimates against reference
implementations while timing both.

Find where the time goes
------------------------

Before you change anything, a profile shows which calls take the time. Python's
``cProfile`` records the time of each function together with the time of the
calls it makes. We'll profile the staggered difference-in-differences
estimator :func:`~moderndid.att_gt` on the minimum wage data after a first call
that pays one-time costs such as imports.

.. code-block:: python

   import cProfile
   import pstats

   import moderndid as did

   data = did.load_mpdta()
   spec = {
       "yname": "lemp",
       "tname": "year",
       "idname": "countyreal",
       "gname": "first.treat",
       "xformla": "~ lpop",
   }

   # The first call pays one-time costs, such as imports, that the profile should leave out.
   did.att_gt(data, **spec)

   profiler = cProfile.Profile()
   profiler.runcall(did.att_gt, data, **spec)
   pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(11)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

                  56175 function calls (55995 primitive calls) in 0.054 seconds

            Ordered by: cumulative time
            List reduced from 726 to 11 due to restriction <11>

            ncalls  tottime  percall  cumtime  percall filename:lineno(function)
                 1    0.000    0.000    0.054    0.054 att_gt.py:30(att_gt)
                 1    0.000    0.000    0.043    0.043 compute_att_gt.py:25(compute_att_gt)
                 1    0.000    0.000    0.043    0.043 parallel.py:10(parallel_map)
                12    0.000    0.000    0.043    0.004 compute_att_gt.py:685(_process_gt_cell_did)
                12    0.000    0.000    0.043    0.004 compute_att_gt.py:173(run_att_gt_estimation)
                24    0.000    0.000    0.027    0.001 generalized_linear_model.py:1092(fit)
                24    0.002    0.000    0.027    0.001 generalized_linear_model.py:1282(_fit_irls)
                12    0.000    0.000    0.024    0.002 compute_att_gt.py:527(run_drdid)
                12    0.000    0.000    0.024    0.002 drdid_panel.py:37(drdid_panel)
                12    0.000    0.000    0.017    0.001 compute_att_gt.py:288(_cell_fails_screen)
                12    0.000    0.000    0.017    0.001 drdid_panel.py:244(_compute_propensity_score)

The 24 logit fits counted in the ``ncalls`` column take about half of the call.
For each of its 12 effects, one per cohort and year, ``att_gt`` fits one logit
to screen for overlap and another for the propensity score of the doubly robust
estimator. The call counts stay the same on every machine even though the times
don't. The profiler sees calls into compiled libraries such as NumPy and polars
but not the work inside them.

.. admonition:: GPU timings stop before the device does
   :class: warning

   CuPy returns control before the device finishes its work. The :doc:`GPU
   guide <../user_guide/gpu>` shows how to synchronize the device around a
   timed call.

Choose a workload
-----------------

Each timing in the benchmark suite runs on a workload, a generated panel that
``benchmarks/cases.py`` rebuilds from a fixed seed so that every run times the
same data. The suite in ``benchmarks/`` times the estimators with Airspeed
Velocity (ASV), a tool that repeats each timing and can compare two commits.

- ``baseline`` has 1,000 units observed over six periods, two treated cohorts,
  and no covariates.
- ``rows``, ``periods``, and ``covariates`` raise the units to 10,000, the
  periods to 12, and the covariates from none to four.
- ``cohorts`` has five treated cohorts and stretches the panel to ten periods to
  give them room to adopt.
- ``bootstrap`` turns on the bootstrap with 999 draws, or 199 for
  ``did_multiplegt``.
- ``small`` drops to 400 units for quick checks.

The two-period estimators ``drdid``, ``ipwdid``, and ``ordid`` keep a workload's
units and covariates over two periods with one treated cohort. Three more
workloads keep the baseline panel and change an estimation choice instead.
``ipw`` and ``reg`` switch ``att_gt`` and ``ddd`` to inverse probability
weighting or outcome regression. The ``normalized`` workload asks
``did_multiplegt`` for its normalized effects rather than the default ones.
Each class in ``benchmarks/benchmarks/bench_estimators.py`` lists the workloads
it times in its ``params``. The ``short`` workload of four periods appears in
none of them and serves the reference runner.

Every panel adds up a unit effect, a common time trend, covariate effects,
noise, and a treatment effect that grows with time since adoption.
Triple-difference panels also mark half the units eligible for treatment.
Continuous and intertemporal panels draw a dose for each treated unit. Pick the
workload that exercises the calculation you changed rather than every large one
while you're still checking that it runs.

Time the workload on your checkout
----------------------------------

The quickest timing runs a benchmark against the code in your checkout. The
project runs ASV through spin, a command runner whose commands for this
repository live in ``.spin/cmds.py``. The locked ``benchmark`` environment
provides both tools along with pixi tasks for the common commands. From the
repository root, install that environment and register your machine before the
first run, since ASV stores a description of the machine with each result.

.. code-block:: console

   pixi install --frozen -e benchmark
   pixi run -e benchmark benchmark-machine
   pixi run -e benchmark bench --quick -t 'bench_estimators.ATTgt.*small'

The ``bench`` task runs ``spin bench`` on the package in your active
environment and prints the results without saving them. With ``--quick`` each
selected benchmark runs once, enough to show that it works but not enough to
time it. Each ``-t`` option adds a regular expression matched against benchmark
names and workloads, such as the one above that picks the ``small`` workload of
``att_gt``. Once the quick run works, leave out ``--quick`` to collect the
repeated samples a timing needs.

.. code-block:: console

   pixi run -e benchmark bench -t 'bench_estimators.ATTgt.*small'
   pixi run -e benchmark spin bench --help

.. admonition:: Read the spread before the median
   :class: tip

   ASV prints each timing as the median of its samples followed by half their
   interquartile range. A large second number means unstable samples that need
   a rerun under steadier conditions before they can support a speed claim.

Understand the operation inside a timing
----------------------------------------

Estimator benchmarks in ``bench_estimators.py`` time a complete public call,
from preprocessing through the requested inference. Their setup builds the
panel and makes one warm-up call outside the timer. Aggregation benchmarks in
``bench_aggregation.py`` fit the estimator during setup and time only the call
to ``aggte`` or ``agg_ddd``.

To keep timings from different machines comparable, the benchmark environment
limits the numerical libraries and polars to one thread each. A
single-threaded timing doesn't show how a fit with ``n_jobs`` above 1 or a fit
on a GPU performs. If your change targets either, define that workload yourself
and report its settings with the result. When you share a timing, include the
machine, workload, dependency versions, and thread settings so another
contributor can repeat it.

Compare two committed revisions
-------------------------------

To measure a change against ``main``, ``spin bench --compare`` builds the
package at each of the two commits and installs each build in turn into an
environment that ASV manages. It then runs the same benchmarks on both builds
and compares their timings.

.. code-block:: console

   pixi run -e benchmark spin bench --compare main HEAD -t 'bench_estimators.ATTgt.*small'

Both runs use the benchmark suite from your current checkout to keep the
workload the same across the comparison. The managed environment takes its
dependency versions from ``benchmarks/asv.conf.json`` rather than from your
development environment. ASV calls a benchmark faster or slower only when its
samples differ beyond their noise and its time changes by more than the factor
that ``--factor`` sets, 1.05 by default.

.. admonition:: Comparisons skip uncommitted changes
   :class: warning

   Because these runs install committed package source, uncommitted library
   changes never reach the timings. The command prints a warning when the
   working tree has uncommitted changes.

If you change a benchmark along with the library, check that the revised
workload still means the same thing on both commits. Otherwise a timing
difference may come from the workload rather than from your code.

To keep a history of timings rather than a one-off comparison, save the
results of a commit through ``spin asv``. It passes its arguments straight to
ASV from the ``benchmarks`` folder.

.. code-block:: console

   pixi run -e benchmark spin asv run --show-stderr HEAD

ASV keeps its environments and saved results in ``benchmarks/.asv/``, a folder
Git ignores. The description of your machine that ASV records lives in
``~/.asv-machine.json``.

Add a benchmark for the changed calculation
-------------------------------------------

If no workload exercises your change, add a benchmark that does. ASV times each
method whose name starts with ``time_`` and runs the class's ``setup`` first,
outside the timer. Put data generation and the warm-up call in ``setup`` so the
``time_`` method measures only the operation its name describes. Reuse the
helpers in ``benchmarks/common.py`` that prepare estimator calls and check their
outputs rather than writing a second version of the same specification.

ASV gives each value in a class's ``params`` a timing of its own. Keep
``params`` and ``param_names`` stable as long as the workload stays the same.

.. admonition:: Increase the version with the workload
   :class: important

   ASV compares a benchmark's results across commits only while its
   ``version`` stays the same. Increase ``version`` when the setup, inputs, or
   measured operation changes so that old timings never stand in for new ones.

Before requesting review, check that ASV discovers the suite and run the
affected case.

.. code-block:: console

   pixi run -e benchmark benchmark-check
   pixi run -e benchmark bench --quick -t 'bench_estimators.ATTgt.*small'

Because a quick run shows only that the case executes, collect repeated samples
under comparable conditions before you describe the change as faster in a pull
request. The :doc:`reviewing guide <../contributing/reviewing>` explains how a
reviewer reads that evidence.

Check the numbers against reference implementations
---------------------------------------------------

The reference runner, ``spin compare``, checks whether the package agrees with
an independent implementation of each method on the same seeded workload. It
reports how long each side takes and the largest gaps between their estimates
and standard errors.

.. code-block:: console

   pixi run -e benchmark setup-r
   pixi run -e benchmark spin compare --estimators att_gt aggte --workloads small baseline

The first command installs the reference packages into the benchmark
environment.

.. admonition:: A completed run isn't agreement
   :class: warning

   The runner records a missing or failing reference in the status column and
   moves on to the other cases. Check each case's status and its count of
   compared cells before you read a run as agreement.

Settings shared by both implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/common.py`` and ``benchmarks/references.py`` set up the calls on
the two sides of each comparison. Covariate workloads add all four generated
covariates on both sides. The runner rejects the ``bootstrap`` workload because
its comparisons use analytical standard errors.

.. list-table:: Settings shared by the compared calls
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Specification
   * - ``att_gt`` and ``aggte``
     - Doubly robust estimation unless the workload names another method,
       never-treated controls, varying base periods, and analytical inference.
       ``aggte`` adds the simple, group, and dynamic aggregations without
       simultaneous bands.
   * - ``ddd`` and ``agg_ddd``
     - Doubly robust estimation unless the workload names another method,
       never-treated controls, universal base periods, and analytical
       inference. ``agg_ddd`` adds the simple, group, and event-study
       aggregations without simultaneous bands.
   * - ``drdid``, ``ipwdid``, and ``ordid``
     - Panel data with the improved doubly robust estimator
       (``est_method="imp"``), unnormalized inverse probability weights, and
       outcome regression.
   * - ``cont_did``
     - Parametric dose estimation with not-yet-treated controls, a cubic
       polynomial without interior knots, and a dose curve at 20 points from
       0.6 to 1.9 without simultaneous bands.
   * - ``did_multiplegt``
     - Two dynamic effects, one placebo, and analytical inference, with
       normalized effects only on the ``normalized`` workload.

.. _benchmarks-numerical-checks:

Numerical checks in the timing runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runner compares the cohort-period and aggregated effects of staggered DiD
and triple differences, the ATT of each two-period estimator, and the effects
and placebos of intertemporal treatment. It compares their analytical standard
errors as well, in each case over the finite cells that both outputs share. An
estimate must satisfy
``abs(moderndid - reference) <= 1e-6 + 1e-5 * abs(reference)``. A standard
error must meet the same bound with a relative tolerance of ``1e-3`` in place
of ``1e-5``. Undefined standard errors at normalized reference periods count as
unavailable rather than as differences.

A case reads ``match`` only when every expected comparison is finite and within
tolerance. Missing keys or nonfinite values make an otherwise matching case
``partial``. A gap beyond tolerance marks it ``differs`` and makes the runner
exit with an error. The comparison rules live in ``benchmarks/compare.py`` and
the tolerances in ``benchmarks/reporting.py``.

.. admonition:: Outputs that differ by construction
   :class: note

   Where a reference computes an output differently from the method's paper,
   the runner leaves that output out of the tolerance check and records it as
   a known difference. Instead of failing the run, each case's known
   differences appear below the printed table with their largest gaps and the
   reason. A known difference that comes back missing or nonfinite still marks
   an otherwise matching case ``partial``.

For continuous treatment, the runner compares the overall ATT and ACRT and all
20 points on each dose curve, 42 estimates and 42 standard errors in all.
Computing each standard error as ``sqrt(sum(IF**2)) / n`` from the returned
influence function keeps independent bootstrap draws out of the comparison.
Only the two overall estimates are held to the tolerances. The 40 curve
estimates and all 42 standard errors are known differences. The validation
tests check the bootstrap standard errors and band critical values separately
by feeding the same multiplier draws to both implementations.

For triple differences, the standard errors of ``agg_ddd`` outputs that average
two or more cohorts are known differences. Every estimate, each cohort's own
standard error, and the standard errors of event times observed for a single
cohort stay held to the tolerances. The estimates agree only because every
cohort of the generated panels is half eligible. A workload whose cohorts
differed in their eligible share would add its multi-cohort estimates to the
known differences. The :ref:`continuous-treatment <r-comparisons-continuous>`
and :ref:`triple-difference <r-comparisons-triple>` comparisons trace each
known difference to the reference code.

Where the runner's timer starts and stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the ASV benchmarks, the runner times the complete public call of an
estimator and starts an aggregation's timing from an already fitted result.
Data generation, CSV transfer, package loading, the reference process's
startup, and writing results happen outside the timer. Each side makes one
warm-up call before the timed calls that ``--repeats`` sets, five by default.
The printed table reports the median of each side's timed calls. A call that
the reference's clock rounds down to zero milliseconds gets no ratio there. The
saved report also keeps every sample and their interquartile range.

Save a comparison
~~~~~~~~~~~~~~~~~

To share a comparison, save it as a JSON report with ``--output``.

.. code-block:: console

   pixi run -e benchmark spin compare --estimators att_gt aggte \
       --workloads small baseline --output comparison.json

Along with each case's workload, timings, and comparisons, the report records
the processor model, software versions, thread settings, and source revision.
With ``--strict``, a missing reference or an incomplete comparison also gives an
unsuccessful exit status after the report is saved. The :ref:`worked
comparisons <r-comparisons>` check the same estimators on application data
without timing them.
