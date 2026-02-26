.. _gpu:

===========================
GPU Acceleration with CuPy
===========================

ModernDiD can offload numerical operations to NVIDIA GPUs via
`CuPy <https://cupy.dev/>`_. When the GPU backend is active, matrix
operations in the two-period doubly robust estimators (weighted least
squares, logistic IRLS, influence function computation) run on the GPU
using cuBLAS and cuSOLVER, which can substantially reduce runtime for
large datasets on powerful GPUs.


Requirements
------------

You need an NVIDIA GPU with CUDA support and a CuPy installation that
matches your CUDA toolkit version.

Install the GPU extra:

.. code-block:: bash

    uv pip install 'moderndid[gpu]'

This installs ``cupy-cuda12x``. If your system uses a different CUDA
version, install the appropriate CuPy wheel directly and then install
ModernDiD without the extra:

.. code-block:: bash

    uv pip install cupy-cuda11x
    uv pip install moderndid

Verify the installation:

.. code-block:: python

    import moderndid as did

    print(did.HAS_CUPY)  # True if CuPy is available


Enabling the backend
--------------------

The GPU backend is opt-in. Pass ``backend="cupy"`` to
:func:`~moderndid.att_gt` or :func:`~moderndid.ddd` to run a single
call on the GPU. The backend activates only for that call and reverts
automatically when it returns:

.. code-block:: python

    import moderndid as did

    result = did.att_gt(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        backend="cupy",
    )

For multiple consecutive GPU calls, you can either set the backend
globally or use the :func:`~moderndid.use_backend` context manager:

.. code-block:: python

    # Option 1: global setting
    did.set_backend("cupy")
    result1 = did.att_gt(...)
    result2 = did.ddd(...)
    did.set_backend("numpy")  # revert when done

    # Option 2: context manager (reverts automatically)
    from moderndid import use_backend

    with use_backend("cupy"):
        result1 = did.att_gt(...)
        result2 = did.ddd(...)

All three approaches are thread-safe and compose correctly with
``n_jobs > 1``. When ``data`` is a Dask or Spark DataFrame,
``backend="cupy"`` enables GPU-accelerated linear algebra on worker GPUs
(see :ref:`Combining GPU and Dask <gpu-dask-workers>` and
:ref:`Combining GPU and Spark <gpu-spark-workers>`).

If CuPy is installed but no GPU is available, ``backend="cupy"``
raises a ``RuntimeError`` with an actionable message. If CuPy is not
installed at all, it raises an ``ImportError``.

To check the active backend at any point:

.. code-block:: python

    xp = did.get_backend()
    print(xp.__name__)  # "numpy" or "cupy"


What gets accelerated
---------------------

The GPU backend accelerates the low-level numerical operations inside
the two-period estimators that :func:`~moderndid.att_gt` and
:func:`~moderndid.ddd` call for each group-time cell, for both panel
and repeated cross-section data with any ``est_method``.

- **Weighted least squares** (``reg``, ``dr``) — Design matrix
  multiplication, normal equation solve via cuSOLVER, and fitted value
  computation via cuBLAS.

- **Logistic IRLS** (``ipw``, ``dr``) — Iteratively reweighted least
  squares for the propensity score model. Each iteration runs sigmoid
  evaluation, Gram matrix accumulation, and a linear solve on the GPU.

- **Influence function computation** — All matrix algebra in the
  influence function (inverse Hessians, score products, weighted sums)
  runs on GPU arrays. Results transfer back to CPU only at function
  boundaries.

- **Multiplier bootstrap** — Random Mammen weight generation and the
  batched matrix multiply for bootstrap replication run on the GPU.
  Draws are batched to stay within a configurable memory budget (1 GB
  by default) so that large bootstrap runs do not exhaust GPU memory.

- **Cluster aggregation** — Scatter-add operations to aggregate
  influence functions at the cluster level use GPU kernels.

These operations are dominated by dense linear algebra (matrix
multiplication, triangular solves) that maps well to GPU hardware.
The group-time loop, cell scheduling, and aggregation logic remain
on the CPU.

The continuous treatment estimator (:func:`~moderndid.cont_did`),
the intertemporal estimator (:func:`~moderndid.did_multiplegt`), and
the sensitivity analysis module (:func:`~moderndid.honest_did`) do
not use the GPU backend. These estimators operate on small matrices
(spline bases, per-group comparisons, and LP constraints respectively)
where GPU kernel launch and data transfer overhead would exceed any
computation benefit.


When it helps
-------------

GPU acceleration provides the largest speedups when the per-cell
sample sizes are large enough to saturate the GPU. This typically
means thousands of units per group-time cell, multiple covariates
producing larger design matrices, and doubly robust estimation
(``est_method="dr"``) which runs both outcome regression and propensity
score estimation per cell.

For small datasets (a few hundred units per cell), the overhead of
transferring data to and from the GPU can outweigh the computation
savings. In those cases, the CPU backend is faster.

The benefit also depends on estimation method. Doubly robust estimation
performs roughly twice as much linear algebra per cell as pure regression
or pure IPW, so the GPU speedup is more pronounced with ``est_method="dr"``.
Bootstrap inference multiplies the work by ``biters``, making the GPU
advantage larger when ``boot=True`` with many iterations.


How data moves between CPU and GPU
-----------------------------------

ModernDiD handles data transfer automatically. You do not need to create
CuPy arrays yourself.

1. Input data (Polars or pandas DataFrames) is preprocessed on the CPU
   as usual.
2. During the tensor construction step, arrays are transferred to the
   GPU in bulk using :func:`~moderndid.cupy.to_device`.
3. All cell-level computation runs on GPU arrays.
4. Results (ATT estimates, influence functions) are transferred back to
   the CPU using :func:`~moderndid.cupy.to_numpy` before being stored
   in the result object.

Because the bulk transfer happens once and results transfer once, the
CPU-GPU communication overhead is small relative to the computation.


Memory management
-----------------

CuPy uses a memory pool by default. Allocated GPU memory is cached
for reuse rather than returned to the OS after each operation. This
means ``nvidia-smi`` may show high memory usage even when arrays have
been freed. This is expected behavior and does not indicate a memory
leak.

If you run ModernDiD alongside other GPU workloads, you can limit
the memory pool size with the ``CUPY_GPU_MEMORY_LIMIT`` environment
variable:

.. code-block:: bash

    export CUPY_GPU_MEMORY_LIMIT="4G"      # absolute limit
    export CUPY_GPU_MEMORY_LIMIT="50%"     # percentage of total GPU memory

You can also free cached blocks explicitly between runs:

.. code-block:: python

    import cupy as cp

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

If an estimator call exhausts GPU memory, ModernDiD raises a
``MemoryError`` with a message suggesting you reduce the problem size
or switch back to ``backend='numpy'``. The bootstrap implementation
batches draws to stay within a 1 GB GPU allocation per batch, but very
large influence function matrices can still exceed available memory.


GPU device selection
--------------------

If your machine has multiple GPUs, CuPy uses device 0 by default.
All computation runs on a single GPU; ModernDiD does not split work
across devices. To select a different GPU, wrap the call in a CuPy
device context:

.. code-block:: python

    import cupy as cp
    import moderndid as did

    with cp.cuda.Device(1):
        result = did.att_gt(
            data=data, yname="y", tname="time",
            idname="id", gname="group", backend="cupy",
        )

For multi-GPU parallelism, use a distributed backend to pin one worker
or executor per GPU. See :ref:`Combining GPU and Dask <gpu-dask-workers>`
(using ``dask-cuda``) or :ref:`Combining GPU and Spark <gpu-spark-workers>`
(using Spark GPU resource scheduling).


Benchmarking correctly
----------------------

GPU execution is asynchronous. Standard Python timing
(``time.perf_counter``, ``%timeit``) measures only the time to
*launch* GPU kernels, not the time for them to complete. For accurate
benchmarks, synchronize the GPU before taking timestamps:

.. code-block:: python

    import cupy as cp
    import time

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    result = did.att_gt(
        data=data, yname="y", tname="time",
        idname="id", gname="group", backend="cupy",
    )

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

The first call in a process incurs one-time overhead from CUDA context
initialization and kernel compilation. CuPy caches compiled kernels in
``~/.cupy/kernel_cache``, so subsequent calls in the same or later
sessions are faster.


.. _gpu-dask-workers:

Combining GPU and Dask
----------------------

The GPU backend and the Dask distributed backend can be combined.
Pass ``backend="cupy"`` to :func:`~moderndid.att_gt` or
:func:`~moderndid.ddd` with a Dask DataFrame to run partition-level
linear algebra on worker GPUs. The low-level functions
:func:`~moderndid.dask.dask_att_gt` and :func:`~moderndid.dask.dask_ddd`
also accept the ``backend`` parameter:

.. code-block:: python

    import dask.dataframe as dd
    import moderndid as did

    ddf = dd.read_parquet("panel_data.parquet")

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        backend="cupy",
    )

When ``backend="cupy"`` is active, each worker converts its partition
arrays to CuPy after building them from pandas. All Gram matrix
accumulation, IRLS iterations, and influence function computation run on
the worker's GPU. Results are converted back to NumPy before leaving the
worker, so driver-side aggregation (tree-reduce, precomputation) stays
on the CPU.

CuPy must be installed on every worker. For multi-GPU machines, use
``dask-cuda`` with a ``LocalCUDACluster`` to pin one worker per GPU:

.. code-block:: python

    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    cluster = LocalCUDACluster()
    client = Client(cluster)

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        backend="cupy",
    )

The ``set_backend`` / ``use_backend`` context manager does **not**
propagate to Dask worker processes. Always use the ``backend`` parameter
on the estimator call instead.

The following example shows a complete workflow where we connect to a multi-GPU
cluster, read data, run estimation, and clean up.

.. code-block:: python

    import dask.dataframe as dd
    from dask.distributed import Client, wait
    from dask_cuda import LocalCUDACluster

    import moderndid as did

    # Start one worker per GPU
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Read and persist input data
    ddf = dd.read_parquet("panel_data.parquet").persist()
    wait(ddf)

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2",
        est_method="dr",
        backend="cupy",
    )

    # Post-estimation stays the same
    event_study = did.aggte(result, type="dynamic")
    did.plot_event_study(event_study)

    client.close()
    cluster.close()

When you need explicit control over the client (for example on
Databricks or a managed cluster), use the low-level
:func:`~moderndid.dask.dask_att_gt` entry point which accepts a
``client`` parameter directly.


.. _gpu-spark-workers:

Combining GPU and Spark
-----------------------

The GPU backend and the Spark distributed backend can be combined.
Pass ``backend="cupy"`` to :func:`~moderndid.att_gt` or
:func:`~moderndid.ddd` with a PySpark DataFrame to run partition-level
linear algebra on executor GPUs. The low-level functions
:func:`~moderndid.spark.spark_att_gt` and :func:`~moderndid.spark.spark_ddd`
also accept the ``backend`` parameter:

.. code-block:: python

    from pyspark.sql import SparkSession
    import moderndid as did

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    sdf = spark.read.parquet("panel_data.parquet")

    result = did.att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        backend="cupy",
    )

When ``backend="cupy"`` is active, partition arrays are converted to CuPy
after collection. All Gram matrix accumulation, IRLS iterations, and
influence function computation run on the GPU. Results are converted back
to NumPy before being stored in the result object.

CuPy must be installed on the driver (and on executors if using Spark's
``mapInPandas`` GPU paths). On GPU-enabled Spark clusters (e.g., Databricks
ML Runtime with GPU instances, or YARN with GPU resource scheduling),
configure executors with GPU resources:

.. code-block:: python

    spark = (
        SparkSession.builder
        .master("yarn")
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "1")
        .getOrCreate()
    )

    result = did.att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        backend="cupy",
    )

The ``set_backend`` / ``use_backend`` context manager does **not**
propagate to Spark executor processes. Always use the ``backend`` parameter
on the estimator call instead.


.. _gpu-troubleshooting:

Troubleshooting
---------------

**"CuPy is not installed"** when calling ``set_backend("cupy")``

The most common cause is installing the generic ``cupy`` package, which
tries to compile from source. Instead, install a prebuilt wheel that
matches your CUDA driver version:

.. code-block:: bash

    uv pip install cupy-cuda12x   # CUDA 12.x
    uv pip install cupy-cuda11x   # CUDA 11.x

Run ``nvidia-smi`` to check which CUDA version your driver supports.
After installing, restart your Python process (or notebook runtime)
before importing ModernDiD (CuPy availability is checked once at import
time).

**"cudaErrorInsufficientDriver"**

The installed CuPy wheel expects a newer CUDA version than your driver
provides. Check ``nvidia-smi`` and switch to the matching wheel.

**"No CUDA GPU is available"**

Make sure ``nvidia-smi`` shows a device. In cloud notebooks, verify that
a GPU runtime is selected.


Next steps
----------

- :ref:`Quickstart <quickstart>` covers estimation options, aggregation
  types, and visualization for local workflows.
- :doc:`distributed` describes the Dask and Spark backends for datasets that
  exceed single-machine memory.
- The :ref:`Examples <user-guide>` section walks through each estimator
  end-to-end with real and simulated data.
