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
cohorts, and covariates, as well as the inference settings. Choose a case
that exercises the calculation you changed rather than running every large
workload while you are still checking execution. A first run needs the
environment and ASV's machine registration.

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

Estimator benchmarks in ``benchmarks/benchmarks/bench_estimators.py`` time
complete public calls, including preprocessing and the requested inference.
Their setup prepares the data and runs a warmup outside the timed operation.
Aggregation benchmarks in ``bench_aggregation.py`` fit the underlying model
during setup so the timing covers aggregation rather than another estimator
fit.

The environment limits the configured numerical libraries to one thread.
Although this helps make repeated measurements comparable, it does not
establish performance for a multithreaded fit or a GPU workload. If your optimization
targets either case, define and report the relevant workload explicitly.
The :doc:`debugging` guide explains why compilation and asynchronous GPU
execution need special care when profiling.

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

Because these runs install committed package source, they don't include
uncommitted library changes. Both revisions use the benchmark suite and
configuration from your current checkout so the workload stays the same
across the comparison. The dependencies for the managed environments come
from ``benchmarks/asv.conf.json`` rather than your development environment.

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

   pixi run -e benchmark spin compare --estimators att_gt aggte --workloads small baseline

You'll need to set up the reference dependencies first, as described in
the :doc:`testing guide <../contributing/testing>`. The runner reports a
missing or failing reference as unavailable and continues with the other
cases. Before treating a successful run as numerical agreement, read the
status and count of shared cells to see which comparisons were actually made.

The runner exits with an error when a case is marked ``differs``. It compares
finite cells shared by both outputs and leaves some standard errors out of
the comparison. In particular, continuous-treatment bootstrap standard errors
are excluded, as is the overall standard error for group aggregation of triple
differences. The current exclusions and tolerances live in
``benchmarks/compare.py``; they limit what the report can verify.

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
the change as faster in a pull request.
