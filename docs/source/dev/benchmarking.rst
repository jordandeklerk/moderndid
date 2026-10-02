.. _benchmarking:

============
Benchmarking
============

We use Airspeed Velocity (ASV) to measure estimator runtime across revisions.
The suite covers staggered DiD, triple DiD, continuous DiD, intertemporal DiD,
and aggregation. Its workloads vary the number of units, periods, treatment
cohorts, covariates, and bootstrap iterations. A second command runs the same
workloads through the R packages that implement each method and reports how
the speed and the results compare.

Run the commands on this page from the repository root. The ``benchmark``
Pixi environment contains the package and benchmark tools. Its dependencies
are recorded in ``pixi.lock``.

Running the current checkout
----------------------------

Install the locked environment and register the machine before your first run.
Use ``--quick`` to run each benchmark once and check that it executes.

.. code-block:: console

   pixi install --frozen -e benchmark
   pixi run -e benchmark benchmark-machine
   pixi run -e benchmark bench --quick

Without ``--quick``, ASV collects repeated timing samples. Use ``-t`` to select
a module, class, method, or workload profile. Repeat ``-t`` to select more than
one expression.

.. code-block:: console

   pixi run -e benchmark spin bench -t bench_estimators.ATTgt --quick
   pixi run -e benchmark spin bench -t bench_aggregation

``spin bench`` measures the current checkout in the active environment.
These runs print results without adding them to the performance history.
``spin bench --help`` lists the available options.

Comparing revisions
-------------------

Use ``--compare`` to measure two committed revisions in separate ASV
environments. For example, compare ``main`` and ``HEAD`` on the same machine.

.. code-block:: console

   pixi run -e benchmark spin bench --compare main HEAD -t bench_estimators.ATTgt

ASV installs the package source from each selected commit. Uncommitted library
changes are excluded. The benchmark code and configuration come from your
current checkout and define the workload for both revisions. The managed
environments use the dependency versions specified in
``benchmarks/asv.conf.json``.

Keeping a performance history
-----------------------------

Use ``spin asv run`` to save measurements for a revision. ``spin asv`` passes
its arguments to ASV from the benchmark directory.

.. code-block:: console

   pixi run -e benchmark spin asv run --show-stderr HEAD

ASV stores environments and results under ``benchmarks/.asv/``. Machine
metadata is stored in ``~/.asv-machine.json``. Generated files are ignored by
Git. Keep the machine and dependency configuration consistent when comparing
saved results, since changes to either can affect runtime.

Comparing with R
----------------

``spin compare`` runs moderndid and the R package behind each method on the
same seeded workload. It reports the median time of each side, their ratio,
and the largest gap in estimates and standard errors over the cells both sides
report. The benchmark environment includes R. Its ``setup-r`` task installs the
packages that conda-forge doesn't carry.

.. code-block:: console

   pixi run -e benchmark setup-r
   pixi run -e benchmark benchmark-r
   pixi run -e benchmark spin compare --estimators att_gt aggte --workloads small baseline

The references come from did, DRDID, triplediff, contdid, and
DIDmultiplegtDYN. A reference whose package is missing or fails to load is
reported as unavailable instead of stopping the run. The command exits with an error only when an
estimate or standard error falls outside its tolerance. The
`benchmarks README <https://github.com/jordandeklerk/moderndid/blob/main/benchmarks/README.md>`__
explains the tolerances and the cases that need care when you read the report.

Writing and checking benchmarks
-------------------------------

Timing modules live in ``benchmarks/benchmarks/``. Workload definitions live
in ``benchmarks/cases.py`` and shared setup helpers live in
``benchmarks/common.py``. Put data generation and warmup in ``setup``. A
``time_`` method should contain the operation you want to measure.

Estimator timings include the complete public function call, including its
preprocessing and inference. Aggregation timings build the estimator result
in setup so they measure aggregation alone. The R call behind each comparison
lives in ``benchmarks/references.py``.

Keep ``params`` and ``param_names`` stable so ASV can follow a workload across
revisions. Increase the benchmark's ``version`` when its inputs, setup, or
measured operation change. Check discovery and run the module you edited
before requesting review.

.. code-block:: console

   pixi run -e benchmark benchmark-check
   pixi run -e benchmark bench --quick -t small

A quick run checks execution. Collect repeated measurements on a quiet machine
when you need to assess a change in performance.
