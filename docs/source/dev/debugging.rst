.. _debugging:

=================
Debugging Guide
=================

**ModernDiD** combines several technologies (Polars DataFrames, Numba JIT
compilation, CuPy GPU arrays, and Dask/Spark distributed computing) that
each have their own debugging characteristics and common failure modes.

General strategies
==================

Start simple
------------

When a test fails or you get unexpected results, first isolate the problem.

1. **Run the failing test in isolation**::

      pytest tests/did/test_att_gt.py::test_specific_case -vv

2. **Check if the failure is deterministic.** Run it a few times. Flaky
   failures often point to race conditions (in parallel code) or
   insufficient numerical tolerances (in stochastic tests).

3. **Reduce the problem.** If a test uses a large dataset, try reducing the
   number of units or periods. If a test uses bootstrap, try running without
   it first (``boot=False``).

Reading test output
-------------------

Test output includes suppressed warnings by default (configured in
``pyproject.toml``). If you suspect a warning is relevant, run with all
warnings visible::

   pytest tests/did/test_att_gt.py -W default -vv

For assertion failures on numerical results, the output will show the
expected and actual values. Pay attention to whether the discrepancy is in
the point estimate (likely a logic bug) or the standard error (likely a
numerical precision or bootstrap issue).

Numerical issues
================

Floating-point precision
------------------------

The most common class of bugs in econometric software is numerical precision.
Symptoms include tests passing on one platform but failing on another,
results that differ slightly between runs, and ``RuntimeWarning: overflow
encountered`` or ``invalid value encountered`` messages.

To diagnose, add intermediate logging statements or use a debugger to inspect
values at key points in the computation. Look for very large or very small
intermediate values that could overflow or underflow, division by quantities
that could be near zero, and matrix operations on near-singular matrices.

Common fixes include using ``np.clip`` to bound propensity scores away from
0 and 1, using ``scipy.linalg.solve`` instead of explicit matrix inversion,
adding ``atol`` and ``rtol`` parameters to ``np.testing.assert_allclose``
that match the expected precision of the computation, and checking symmetry
and positive semi-definiteness of variance-covariance matrices before using
them.

Tolerance selection
-------------------

When a test fails with a numerical mismatch, don't just loosen tolerances
until it passes. Instead, understand *why* the results differ.

- Deterministic code should match to high precision
  (``rtol=1e-5, atol=1e-6``).
- Standard errors with analytical formulas may have slightly lower precision
  (``rtol=1e-3, atol=1e-4``) due to intermediate rounding.
- Bootstrap results are inherently stochastic. Use ratio-based checks (e.g.,
  ``assert 0.7 < se_ratio < 1.3``) or compare distributions rather than
  point values.
- Cross-language validation (Python vs R) may show small differences due to
  different linear algebra backends or floating-point operation ordering.

Debugging Numba-compiled code
=============================

**ModernDiD** uses Numba for JIT compilation of performance-critical loops in
`numba_utils.py <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/core/numba_utils.py>`__,
`didcont numba.py <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/didcont/numba.py>`__, and
`didhonest numba.py <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/didhonest/numba.py>`__. These functions use ``@nb.njit`` with
``cache=True`` and often ``parallel=True``.

Disabling JIT for debugging
----------------------------

Numba-compiled functions cannot be stepped through with a normal Python
debugger. To disable JIT and run the pure-Python fallback, set the
environment variable before running tests::

   NUMBA_DISABLE_JIT=1 pytest tests/did/test_att_gt.py -vv

With JIT disabled, you can use ``pdb``, ``breakpoint()``, or your IDE's
debugger to step through the code. Performance will be much slower, so use
a small dataset.

**ModernDiD**'s Numba functions are written with pure-Python fallback paths.
The dispatch pattern in ``moderndid/core/numba_utils.py`` checks
``HAS_NUMBA`` and falls back to plain NumPy implementations when Numba is
unavailable. This means

- If a test passes with ``NUMBA_DISABLE_JIT=1`` but fails without it, the
  bug is in the Numba-compiled version specifically
- If it fails both ways, the bug is in the shared logic

Stale cache issues
------------------

Numba caches compiled functions to disk. If you change a Numba-decorated
function and the test still uses the old behavior, clear the cache::

   find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
   find . -name "*.nbi" -delete 2>/dev/null
   find . -name "*.nbc" -delete 2>/dev/null

Or disable caching temporarily by setting::

   NUMBA_DISABLE_CACHING=1 pytest ...

Type errors in nopython mode
-----------------------------

Numba's ``nopython`` mode (the default for ``@nb.njit``) requires that all
types can be inferred at compile time. If you see
``numba.core.errors.TypingError``, it usually means you are passing a Python
object that Numba can't handle (e.g., a dict with mixed-type values, a Polars
Series, or a custom class), using a NumPy function that Numba doesn't support,
or there is a type mismatch between function arguments and the expected types.

The error message will point to the specific line and show the inferred types.
Compare them with what you intended.

Debugging CuPy and GPU code
============================

GPU-accelerated code lives in `cupy <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/cupy>`__ and uses a backend
dispatch pattern. The active backend is controlled via context variable::

   from moderndid.cupy.backend import use_backend

   with use_backend("cupy"):
       result = att_gt(data=df, ...)

Common GPU issues
-----------------

**CuPy not found.** If ``import cupy`` fails, the code automatically falls
back to NumPy. Check your CUDA installation::

   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

**Out of memory.** GPU memory is more limited than system RAM. Symptoms
include ``cupy.cuda.memory.OutOfMemoryError``. Reduce the dataset size or
batch size. The RMM memory pool (initialized automatically by
``set_backend("cupy")``) helps with memory fragmentation but doesn't
increase total memory.

**Results differ between CPU and GPU.** Small floating-point differences
(< 1e-6) are normal due to different operation ordering and fused
multiply-add instructions on GPU. Larger differences suggest a bug in
the GPU code path.

**Comparing CPU and GPU results.** To isolate GPU-specific issues, run the
same computation on both backends and compare step by step::

   import numpy as np
   from moderndid.cupy.backend import use_backend, to_numpy

   # Run on CPU
   result_cpu = att_gt(data=df, boot=False)

   # Run on GPU
   with use_backend("cupy"):
       result_gpu = att_gt(data=df, boot=False)

   # Compare
   np.testing.assert_allclose(
       result_cpu.att_gt, to_numpy(result_gpu.att_gt), rtol=1e-5
   )

Debugging distributed execution
================================

Dask and Spark tests can be harder to debug because computation is deferred
and distributed across workers.

Dask
----

**View the task graph.** For Dask computations, you can visualize what will
be computed before triggering execution::

   import dask
   result = dask_att_gt(ddf, ...)  # returns a delayed result
   dask.visualize(result, filename="task_graph.png")

**Use a local cluster with a single worker.** This serializes execution and
makes errors easier to trace::

   from dask.distributed import Client
   client = Client(n_workers=1, threads_per_worker=1)

**Check worker logs.** When running with a distributed client, exceptions on
workers may not surface as clearly. Use the Dask dashboard
(``http://localhost:8787`` by default) to inspect worker logs and task
states.

**Timeouts.** Dask tests use ``--timeout=120`` in CI. If a test hangs
locally, run it with a timeout to get a traceback::

   pytest tests/dask/ --timeout=60 -vv

Spark
-----

**Java version.** Spark requires Java 17+. Check with ``java -version``.
If you see ``UnsupportedClassVersionError``, your Java version is too old.

**Driver memory.** Spark allocates limited driver memory by default. For
large test fixtures, you may need to increase it::

   export SPARK_DRIVER_MEMORY=4g

**Verbose logging.** Spark is noisy by default. To focus on your code's
output, set the Spark log level::

   spark.sparkContext.setLogLevel("WARN")

**Serialization errors.** If you see ``PicklingError`` or
``SerializationException``, it means Spark tried to serialize an object
that can't be sent to workers. This usually happens when a closure captures
a non-serializable object (like a database connection or a compiled Numba
function).

Test failure patterns
=====================

Here are common test failure patterns and what they typically indicate.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Likely cause
   * - ``AssertionError`` on point estimates
     - Logic bug in estimation, incorrect data transformation, or
       wrong group/time filtering
   * - ``AssertionError`` on standard errors only
     - Influence function calculation error, incorrect degrees of freedom,
       or clustering implementation bug
   * - ``RuntimeWarning: overflow encountered``
     - Propensity scores near 0/1, very large treatment effects, or
       insufficient trimming
   * - Test passes locally, fails in CI
     - Platform-dependent floating-point behavior, missing dependency in
       CI environment, or random seed not properly set
   * - Test passes alone, fails when run with other tests
     - Shared mutable state between tests, or fixture scope issue (a
       ``module``-scoped fixture being modified)
   * - ``TypingError`` from Numba
     - Type mismatch in Numba-compiled function arguments
   * - ``TimeoutError`` in distributed tests
     - Deadlock, excessive data shuffling, or the driver materializing
       too much data
   * - R validation test fails after code change
     - Likely a regression that changed estimation results. Investigate
       carefully before loosening tolerances, as these tests verify that
       Python matches the reference R packages

Using a debugger
================

For non-Numba, non-distributed code, standard Python debugging works well.

**With pytest**, add the ``--pdb`` flag to drop into the debugger on the
first failure::

   pytest tests/did/test_att_gt.py::test_specific_case -vv --pdb

Use ``n`` (next), ``s`` (step into), ``p variable`` (print), and ``c``
(continue) to navigate.

**With breakpoints in code**, insert ``breakpoint()`` at the line you want
to inspect, then run the test normally. Python will drop into the debugger
at that point.

**With an IDE**, most editors (VS Code, PyCharm) can run pytest with their
built-in debugger. Set breakpoints visually and use the IDE's variable
inspector.

Profiling
=========

Before optimizing code, profile it to identify the actual bottleneck. A
function that *looks* slow may account for a fraction of total runtime,
while the real bottleneck may be somewhere unexpected.

Finding CPU bottlenecks
-----------------------

The built-in ``cProfile`` module works well for getting a high-level view
of where time is spent::

   python -m cProfile -s cumtime -c "from moderndid import att_gt; att_gt(data=df)" 2>&1 | head -30

For a more granular view, ``line_profiler`` shows time spent on each line
within a function. Install it with ``pip install line_profiler``, then
decorate the function you want to profile with ``@profile`` and run::

   kernprof -lv your_script.py

To profile within a test, use pytest-benchmark (already in the test
dependencies) to get reliable timing with warmup and multiple iterations::

   pytest tests/did/test_att_gt.py -k "test_specific" --benchmark-only

Measuring memory usage
-----------------------

For memory-intensive operations (large influence function matrices, bootstrap
resampling), ``memory_profiler`` shows line-by-line memory allocation. Install
with ``pip install memory_profiler``, decorate with ``@profile``, and run::

   python -m memory_profiler your_script.py

For a quick check of peak memory without instrumenting code, use the ``/usr/bin/time``
utility (note the full path to avoid the shell builtin)::

   /usr/bin/time -l python -c "from moderndid import att_gt; att_gt(data=large_df)"

The "maximum resident set size" field shows peak memory in bytes.

Profiling Numba-compiled code
------------------------------

Standard Python profilers cannot see inside Numba-compiled functions. To
profile Numba code, temporarily disable JIT (``NUMBA_DISABLE_JIT=1``) and
profile the pure-Python fallback. The hot spots in the pure-Python version
will correspond to the same hot spots in the JIT version, even though absolute
timings differ.

Profiling GPU code
-------------------

For CuPy GPU profiling, use NVIDIA's ``nsys`` profiler to see kernel execution
times and memory transfers::

   nsys profile python your_gpu_script.py

The most common performance issue with GPU code is excessive data transfer
between CPU and GPU. Look for repeated ``to_device()`` and ``to_numpy()``
calls within loops.
