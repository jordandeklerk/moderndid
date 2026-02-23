.. _distributed:

======================
Distributed Estimation
======================

ModernDiD includes distributed backends for :func:`~moderndid.att_gt` and
:func:`~moderndid.ddd` on top of `Dask <https://www.dask.org/>`_ and
`Apache Spark <https://spark.apache.org/>`_. Both backends scale from one
machine to a full cluster and return the same result object types as local
estimation, so your post-estimation workflow stays the same.

This page focuses on practical usage. For internal implementation details, see
:ref:`Distributed Backend Architecture <distributed-architecture>`.


When to use distributed estimation
----------------------------------

Distributed estimation is most useful when the dataset does not fit in memory
on one machine, when local runtime is too long for your iteration cycle, when
you need to run many model specifications over the same large dataset, or when
you already maintain a Dask or Spark cluster for analytics workloads.

Distributed execution has scheduler and communication overhead. If local
estimation is fast and memory safe, local execution is often simpler and easier
to debug.


Requirements
------------

Install the extra for the backend you plan to use.

.. code-block:: console

    uv pip install moderndid[dask]     # Dask backend
    uv pip install moderndid[spark]    # Spark backend

The ``all`` extra includes both, so ``uv pip install moderndid[all]``
also works. ModernDiD cannot set up your cluster environment for you.
Creating clusters, managing workers/executors, and building distributed
DataFrames is the responsibility of the user. Once you have a Dask or Spark
DataFrame, ModernDiD handles the rest.

If you are new to either framework:

- `10 Minutes to Dask <https://docs.dask.org/en/stable/10-minutes-to-dask.html>`_
  and the `Distributed documentation <https://distributed.dask.org/en/stable/>`_
- `PySpark Getting Started <https://spark.apache.org/docs/latest/api/python/getting_started/index.html>`_
  and the `Spark documentation <https://spark.apache.org/docs/latest/>`_


Quick start
-----------

The distributed backend activates automatically when ``data`` is a Dask
or PySpark DataFrame. If ``data`` is any other type, the local estimator
runs instead. This lets you move from local development to cluster execution
without rewriting estimator arguments.

**Dask**

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
        xformla="~ x1 + x2",
        est_method="dr",
    )

    # Post-estimation code is unchanged
    event_study = did.aggte(result, type="dynamic")
    did.plot_event_study(event_study)

**Spark**

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
        xformla="~ x1 + x2",
        est_method="dr",
    )

    event_study = did.aggte(result, type="dynamic")
    did.plot_event_study(event_study)

The same pattern works for triple differences with both backends.

.. code-block:: python

    result = did.ddd(
        data=ddf,       # or sdf
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
    )

Return objects are the same classes as local estimation, so downstream
utilities like ``aggte``, ``agg_ddd``, ``plot_event_study``, and related
plotting functions continue to work without changes.


Interfaces
----------

You can use distributed estimation through two API layers. The high-level
wrappers ``att_gt`` and ``ddd`` are the recommended entry points. The
low-level functions give you explicit control over the client or session.

**Dask**

High-level wrappers do not expose a ``client`` argument. Creating a
``Client`` registers it as the global default, so estimator calls pick it
up automatically. The low-level functions ``dask_att_gt`` and ``dask_ddd``
in :mod:`moderndid.dask` accept a ``client`` argument directly.

.. code-block:: python

    import moderndid as did
    from moderndid.dask import dask_att_gt
    from dask.distributed import Client

    # High-level (uses the global default client)
    client = Client("scheduler-address:8786")
    result_a = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

    # Low-level (accepts client explicitly)
    result_b = dask_att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group", client=client)

**Spark**

High-level wrappers do not expose a ``spark`` argument. ModernDiD uses the
active session or creates a local one automatically. The low-level functions
``spark_att_gt`` and ``spark_ddd`` in :mod:`moderndid.spark` accept a
``spark`` argument directly.

.. code-block:: python

    import moderndid as did
    from moderndid.spark import spark_att_gt
    from pyspark.sql import SparkSession

    # High-level (uses the active session)
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    result_a = did.att_gt(data=sdf, yname="y", tname="time", idname="id", gname="group")

    # Low-level (accepts spark session explicitly)
    result_b = spark_att_gt(data=sdf, yname="y", tname="time", idname="id", gname="group", spark=spark)


Current support and limits
--------------------------

The distributed path supports both panel and repeated cross-section data
for any number of time periods. All standard estimation options work in
distributed mode, including ``control_group``, ``anticipation``,
``base_period``, and ``est_method`` (callable
``est_method`` is not supported). When ``boot=True``, the multiplier
bootstrap runs fully distributed with Mammen two-point weights generated
on workers and tree-reduced to the driver.

The ``n_jobs`` parameter is not used in distributed mode. For ``ddd``,
``boot_type`` is also ignored.


Preparing input data
--------------------

The distributed estimators expect a Dask or PySpark DataFrame in long format.
Include all required columns in the same frame and keep ``time`` and treatment
group columns numeric. For panel data (``panel=True``), ensure unit
identifiers are stable across periods and prefer one record per unit-period.
For repeated cross-section data (``panel=False``), each row is an independent
observation and unit identifiers are not required.

Reading large datasets
^^^^^^^^^^^^^^^^^^^^^^

**Dask.** When the full dataset is too large to build in driver memory, write
it to Parquet in batches and read it back as a Dask DataFrame. This keeps
driver memory constant regardless of total dataset size.

.. code-block:: python

    import gc

    import dask
    import dask.dataframe as dd
    import moderndid as did
    from dask.distributed import Client

    client = Client()   # or connect to an existing scheduler
    n_workers = len(client.scheduler_info()["workers"])

    N_TOTAL = 100_000_000
    CHUNK_SIZE = 500_000
    N_CHUNKS = N_TOTAL // CHUNK_SIZE
    BATCH_SIZE = n_workers * 2          # chunks written per round
    PARQUET_PATH = "/tmp/panel_data"

    # Define a delayed function that builds one chunk
    @dask.delayed
    def _generate_chunk(chunk_id, n):
        dgp = did.gen_did_scalable(
            n=n, dgp_type=1, n_periods=10, n_cohorts=6,
            n_covariates=30, panel=True, random_state=chunk_id,
        )
        df = dgp["data"].to_pandas()
        df["id"] = df["id"] + chunk_id * n   # ensure globally unique IDs
        return df

    # Write in batches so the driver never holds the full dataset
    meta = _generate_chunk(0, 10).compute()   # schema for Dask

    for batch_start in range(0, N_CHUNKS, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N_CHUNKS)
        chunks = [_generate_chunk(i, CHUNK_SIZE)
                  for i in range(batch_start, batch_end)]
        ddf_batch = dd.from_delayed(chunks, meta=meta)
        ddf_batch.to_parquet(
            PARQUET_PATH,
            append=(batch_start > 0),
            engine="pyarrow",
        )
        del ddf_batch, chunks
        gc.collect()

    # Read back as a distributed DataFrame
    ddf = dd.read_parquet(PARQUET_PATH, engine="pyarrow")

**Spark.** Spark reads data lazily, so you can point directly at Parquet,
CSV, or any Spark-supported source and the data stays distributed across
executors without staging batches on the driver.

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    sdf = spark.read.parquet("hdfs:///data/panel_data.parquet")

Partition layout
^^^^^^^^^^^^^^^^

Partition layout affects both runtime and memory. Extremely small partitions
increase scheduler overhead, and extremely large partitions increase
worker/executor memory pressure. If you plan to run multiple model
specifications over the same data, repartition once and persist/cache before
the first fit.

.. code-block:: python

    # Dask
    ddf = ddf.repartition(npartitions=64)

    # Spark
    sdf = sdf.repartition(64)

Persist and sanity-check
^^^^^^^^^^^^^^^^^^^^^^^^

Both frameworks evaluate lazily, so heavy work can execute later than
expected, including inside estimator calls. A good pattern is to
persist/cache and sanity-check the input before fitting. This boundary
separates data-pipeline failures from estimator failures, which makes
debugging faster.

**Dask**

.. code-block:: python

    from dask.distributed import wait

    ddf = ddf.persist()
    wait(ddf)

    print(ddf.columns)
    print(ddf[["id", "time", "group"]].head())

**Spark**

.. code-block:: python

    sdf = sdf.cache()
    sdf.count()  # force materialization

    print(sdf.columns)
    sdf.select("id", "time", "group").show(5)


From local to distributed
-------------------------

A practical workflow is to develop and validate your specification locally
on a sample, then scale up to the full dataset by swapping in a distributed
DataFrame. The estimator arguments stay the same. Only the input type
changes.

.. code-block:: python

    import polars as pl
    import moderndid as did

    shared_args = dict(
        yname="y", tname="time", idname="id", gname="group",
        xformla="~ x1 + x2", est_method="dr", control_group="nevertreated",
    )

    # Develop locally on a sample
    sample = pl.read_parquet("panel_data.parquet").sample(n=10_000, seed=42)
    local_result = did.att_gt(data=sample, **shared_args)

    # Scale with Dask
    import dask.dataframe as dd
    ddf = dd.read_parquet("panel_data.parquet")
    dask_result = did.att_gt(data=ddf, **shared_args)

    # Or scale with Spark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    sdf = spark.read.parquet("panel_data.parquet")
    spark_result = did.att_gt(data=sdf, **shared_args)


Connecting to a cluster
-----------------------

**Dask.** If no client exists, ModernDiD creates a local client
automatically. For multi-node runs, create a ``Client`` pointing at your
scheduler. Creating the client registers it as the global default.

.. code-block:: python

    import moderndid as did
    from dask.distributed import Client

    client = Client("tcp://scheduler-host:8786")

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
    )

This works with any Dask-compatible scheduler endpoint. If you need to
direct calls to a specific client (for example, when multiple clients are
active), use ``client.as_current()``.

For local development with controlled resources, create a ``LocalCluster``
explicitly to set worker count and memory limits.

.. code-block:: python

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="4GB")
    client = Client(cluster)

    result = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

**Spark.** If no active session exists, ModernDiD creates a local Spark
session automatically. For cluster runs, create a ``SparkSession`` pointing
at your cluster manager.

.. code-block:: python

    from pyspark.sql import SparkSession
    import moderndid as did

    spark = (
        SparkSession.builder
        .master("yarn")
        .appName("moderndid-estimation")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "4")
        .getOrCreate()
    )

    sdf = spark.read.parquet("hdfs:///data/panel_data.parquet")

    result = did.att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
    )

This works with any Spark-compatible cluster manager (standalone, YARN,
Mesos, Kubernetes). For Databricks, the ``SparkSession`` is pre-configured
and available as ``spark`` in notebooks, so you can pass Spark DataFrames
directly to estimators without additional setup.

For local development with controlled resources, configure executor
settings explicitly.

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    result = did.att_gt(data=sdf, yname="y", tname="time", idname="id", gname="group")


Distributed-specific parameters
-------------------------------

In addition to standard estimator arguments, distributed execution uses a
small set of controls.

``client`` **(Dask only)**
    A ``distributed.Client`` for low-level calls (``dask_att_gt``,
    ``dask_ddd``). If omitted, ModernDiD first tries ``Client.current()`` and
    creates a local client when none is active.

``spark`` **(Spark only)**
    A ``pyspark.sql.SparkSession`` for low-level calls (``spark_att_gt``,
    ``spark_ddd``). If omitted, ModernDiD first tries
    ``SparkSession.getActiveSession()`` and creates a local session when none
    is active.

``n_partitions``
    Number of partitions per cell computation. The default equals total
    worker threads (Dask) or Spark's default parallelism (Spark), and the
    backend can increase it when estimated partition design matrices would be
    too large.

``max_cohorts``
    Maximum number of treatment cohorts processed concurrently. Defaults to
    the number of workers (Dask) or executor cores (Spark). Lower values
    reduce peak memory because fewer cohort-wide tables are active at once.
    Higher values can improve throughput on memory-rich clusters.

``backend``
    Set to ``"cupy"`` to run partition-level linear algebra on worker GPUs.
    See :ref:`Combining GPU and Dask <gpu-dask-workers>` and
    :ref:`Combining GPU and Spark <gpu-spark-workers>` for setup details.

.. code-block:: python

    from moderndid.dask import dask_att_gt

    result = dask_att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2",
        est_method="dr",
        client=client,
        n_partitions=64,
        max_cohorts=4,
    )

.. code-block:: python

    from moderndid.spark import spark_att_gt

    result = spark_att_gt(
        data=sdf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2",
        est_method="dr",
        spark=spark,
        n_partitions=64,
        max_cohorts=4,
    )

Start with defaults and record runtime and peak memory. Increase
``n_partitions`` if workers are idle for long stretches, or reduce it if
scheduler overhead dominates task time. Reduce ``max_cohorts`` when workers
approach memory limits, and increase it gradually when memory headroom is
large.


Supported estimation features
-----------------------------

All standard estimator arguments work in distributed mode with the same
interface as local estimation.

- **Bootstrap** (``boot=True``) — Mammen two-point weights generated on
  workers and tree-reduced to the driver. ``cband`` defaults to ``False``
  (local defaults to ``True``); set ``cband=True`` explicitly for uniform
  bands.
- **Clustered SEs** (``clustervars``) — one-way and two-way clustering
  supported. Pass a list, not a bare string.
- **Repeated cross-sections** (``panel=False``) — fully supported for both
  ``att_gt`` and ``ddd``.
- **Unbalanced panels** (``allow_unbalanced_panel``) — supported. Default
  ``False`` logs a warning with the number of dropped units.
- **Sampling weights** (``weightsname``) — supported. Weight column must be
  present in the distributed DataFrame.
- **GPU on workers** (``backend="cupy"``) — runs partition-level linear
  algebra on worker GPUs. See :ref:`Combining GPU and Dask <gpu-dask-workers>`
  and :ref:`Combining GPU and Spark <gpu-spark-workers>` for setup.


Running multiple specifications
-------------------------------

When running multiple specifications over the same data, persist/cache the
distributed DataFrame once and reuse it across calls. This avoids re-reading
and re-shuffling the data for each specification.

.. code-block:: python

    from dask.distributed import Client, wait

    client = Client("tcp://scheduler-host:8786")
    ddf = dd.read_parquet("panel_data.parquet").persist()
    wait(ddf)

    specifications = [
        {"est_method": "dr", "control_group": "nevertreated"},
        {"est_method": "dr", "control_group": "notyettreated"},
        {"est_method": "reg", "control_group": "nevertreated"},
    ]

    results = {}
    for spec in specifications:
        results[str(spec)] = did.att_gt(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            xformla="~ x1 + x2",
            **spec,
        )


Monitoring the cluster
----------------------

For long-running jobs, the cluster dashboard provides real-time visibility
into task progress, worker memory, and the task stream.

- **Dask**: Access the dashboard at ``http://scheduler-host:8787/status``
  after creating a client.
- **Spark**: Access the Spark UI at ``http://driver-host:4040`` after
  creating a session. On Databricks, the Spark UI is available directly
  from the cluster page.


Reproducibility
---------------

Distributed reproducibility is best-effort. Set ``random_state`` whenever you
need stable bootstrap draws, then keep cluster conditions as constant as
possible. Results can still vary across runs when worker count, threads per
worker, partition placement, floating-point reduction order, or concurrent
cluster workloads change.

For reproducibility-sensitive comparisons, fix the cluster size and worker
hardware, fix the partition count and input file layout, set the estimator
``random_state`` explicitly, and run on a quiet cluster.


Next steps
----------

- :ref:`Quickstart <quickstart>` covers estimation options, aggregation
  types, and visualization for local workflows.
- :doc:`gpu` describes GPU acceleration for local and distributed workloads.
- :ref:`Estimator Overview <estimator-overview>` surveys all available
  estimators and their distributed support.
- The :ref:`Examples <user-guide>` section walks through each estimator
  end-to-end with real and simulated data.
- For architecture-level details on reduction patterns, memory strategy, and
  execution decomposition, see
  :ref:`Distributed Backend Architecture <distributed-architecture>`.
