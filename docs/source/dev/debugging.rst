.. _debugging:

Debugging an analysis
=====================

When an estimate changes unexpectedly, we need to find out where the
difference comes from before deciding whether the code or the expectation
should change. It may come from the observations entering the comparison,
the effect calculation, or the way uncertainty is calculated. This guide
follows a failing test through those possibilities and shows you how to
inspect the compiled and GPU paths when they're involved.

Run the commands from the repository root in the development environment
described in :doc:`../contributing/guide`. The :doc:`architecture` guide maps
the calculation to its source files if you need help finding the next place
to inspect.

Reproduce the smallest failing comparison
-----------------------------------------

Start with the test that exposed the problem so you retain its data,
specification, and expected result. You can select a single test by its node
identifier rather than running an entire estimator's suite. The following
command selects an existing test of the minimum wage fit.

.. code-block:: console

   pixi run -e dev pytest tests/did/test_att_gt.py::test_att_gt_basic_functionality -vv -x

Replace that identifier with the failing case and add ``--pdb`` to inspect
its state at the first failure. The debugger's ``p`` command reads a value,
``n`` advances within the current function, and ``s`` steps into a call. An
IDE debugger can inspect the same test if it uses the development environment's
Python interpreter.

Before reducing the data, preserve the treatment cohorts and untreated
comparisons that trigger the failure. Removing a cohort, a cluster, or a
missing observation can remove the condition you are trying to investigate.
For a stochastic calculation, record the seed and recreate the random
generator for each comparison rather than reusing a generator whose state
has already advanced.

If a parallel fit fails, repeat it with ``n_jobs=1`` where that argument is
supported so you can inspect one comparison at a time. A difference
between sequential and threaded runs is a reason to examine shared state
and backend propagation rather than assume the numerical formula is wrong.

Read the warnings alongside the failure
---------------------------------------

Warnings can explain why a fit uses fewer observations or cannot estimate a
comparison. The filters in ``tests/conftest.py`` suppress selected numerical
and dependency warnings during tests. Library ``UserWarning`` messages remain
visible unless a particular test filters them. To inspect the warnings that
the usual run hides, request the default warning behavior explicitly.

.. code-block:: console

   pixi run -e dev pytest tests/did/test_att_gt.py::test_att_gt_basic_functionality -vv -W default

The warning gives you a place to start looking before changing the
calculation. If rows were dropped, check the required columns to see which
observations remain in the sample. If the warning concerns overlap, look
at the treated and comparison observations used by that particular fit.

Separate the effect from its uncertainty
----------------------------------------

A mismatch in standard errors can arise even when the fitted effects agree.
We first compare the observations, cohort and period labels, and point
estimates before examining influence functions, clustering, and critical
values. For ``att_gt``, a small fit without bootstrap standard errors or
simultaneous bands gives us a useful starting point.

.. code-block:: python

   import moderndid as did

   data = did.load_mpdta()
   spec = dict(
       data=data,
       yname="lemp",
       tname="year",
       idname="countyreal",
       gname="first.treat",
       xformla="~lpop",
       control_group="nevertreated",
       base_period="universal",
       boot=False,
       cband=False,
       n_jobs=1,
       random_state=42,
   )
   result = did.att_gt(**spec)

Before comparing two arrays of effects, inspect ``result.groups`` and
``result.times`` alongside ``result.att_gt`` so you know that each position
refers to the same comparison. The same care applies to the
``influence_func`` rows because they must represent the same units or
observations on both sides. Once those contributions agree, you can follow
the variance calculation and aggregation weights to understand any remaining
difference in confidence limits. After finding the cause in this simpler
fit, return to the original bootstrap and clustering settings.

The :doc:`testing guide <../contributing/testing>` explains how to choose
assertions for these quantities. An absolute tolerance matters near zero
where a relative tolerance alone provides little room for rounding error.
Choose both from the quantity's scale and the reference calculation rather
than assigning one tolerance to every estimate and standard error.

.. admonition:: Preserve the statistical comparison
   :class: important

   Clipping propensity scores, adding regularization, or changing the retained
   sample can change the estimator. Establish why a calculation fails before
   introducing one of these changes as a numerical fix.

Inspect the compiled CPU path
-----------------------------

Some numerical helpers have Numba implementations that compile on first use.
To step through their Python bodies, set ``NUMBA_DISABLE_JIT`` before starting
the interpreter. We'll use a multiplier-bootstrap test here because it
reaches the numerical helper; an analytical fit can bypass that code.

.. code-block:: console

   NUMBA_DISABLE_JIT=1 pixi run -e dev pytest tests/did/test_mboot.py::test_basic_functionality -vv -x

With compilation disabled, you can use ordinary breakpoints to inspect the
decorated functions as Python code. Keep in mind that this need not be the
same implementation that runs when Numba cannot be imported. In
``core/numba_utils.py``, for example, installing Numba selects the compiled
function bodies; disabling compilation runs those bodies as Python rather
than restoring the earlier fallback definitions.

If the test fails only with compilation enabled, you've narrowed the
search to that execution path without yet establishing that the compiler
is at fault. Start by inspecting argument dtypes, array shapes, contiguous
layout, and the operations accepted by the compiled function. A
``TypingError`` traceback helps you locate the operation and inferred types
that compilation could not resolve.

If you suspect a compiled cache, point ``NUMBA_CACHE_DIR`` to an empty scratch
directory for a fresh run. This avoids deleting caches throughout the checkout
or the development environment.

.. code-block:: console

   NUMBA_CACHE_DIR=/tmp/moderndid-numba-debug pixi run -e dev pytest \
       tests/did/test_mboot.py::test_basic_functionality -vv -x

Use a new directory when repeating this probe so the comparison actually
starts without a previous compiled cache. After the diagnosis, rerun the
affected tests with normal compilation enabled to check the supported path.

Compare supported CPU and GPU calculations
------------------------------------------

Before comparing backends, check that the calculation you're investigating
supports GPU execution. The :doc:`GPU user guide <../user_guide/gpu>` explains
which estimators support it and what you need to install. If you explicitly
request the CuPy backend when CuPy or a CUDA device is unavailable, the
request raises an error rather than silently becoming a CPU fit.

On a machine with a working CUDA installation, reuse the same specification
to compare the CPU and GPU effects. The context manager restores the previous
backend when its block ends.

.. code-block:: python

   import numpy as np

   cpu = did.att_gt(**spec, backend="numpy")
   with did.use_backend("cupy"):
       gpu = did.att_gt(**spec)

   difference = np.asarray(cpu.att_gt) - np.asarray(gpu.att_gt)
   print(np.max(np.abs(difference)))

This calculation checks the effects without treating one fixed precision
threshold as appropriate for every problem. If the discrepancy matters at
the scale of your application, compare the nuisance fits and influence
functions before adding bootstrap randomness. NumPy arrays in these result
fields have already returned to the CPU even when the internal fit used CuPy.

If allocation fails on the GPU, start by checking the array sizes and how
often data move between host and device. Calls to ``set_backend("cupy")``
and ``use_backend("cupy")`` attempt to configure an allocator when the
optional RMM dependency is available, though they do not supply additional
device memory or guarantee that RMM was installed. Before reducing the
workload, preserve the comparison that causes the failure so you can still
investigate it in the smaller fit.

Profile the workload you intend to improve
------------------------------------------

Once the result is correct, profiling can tell you whether time is spent
preparing data, fitting comparisons, or drawing inference. Save the small
``att_gt`` script above as ``profile_analysis.py`` and profile it from the
development environment.

.. code-block:: console

   pixi run -e dev python -m cProfile -s cumulative profile_analysis.py

The report includes imports and the first fit's setup costs. To profile
a fit after its warmup, start the profiler around that call rather than
around the whole script.

.. code-block:: python

   import cProfile

   did.att_gt(**spec)
   profiler = cProfile.Profile()
   profiler.runcall(did.att_gt, **spec)
   profiler.print_stats(sort="cumulative")

Python profiling shows calls into compiled helpers but does not reveal
the work inside every compiled kernel. Timings
with JIT disabled describe a different execution path and should not be used
to predict the compiled path's bottlenecks.

For GPU timings, account for asynchronous execution by synchronizing the
device around the measured operation. Otherwise the host can finish timing
before the device finishes its work. The :doc:`benchmarking` guide describes
the project's CPU workloads and revision comparisons once you are ready to
measure a proposed change consistently.
