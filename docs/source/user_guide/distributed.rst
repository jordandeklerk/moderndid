.. _distributed:

================================
Distributed Estimation with Dask
================================

ModernDiD includes distributed backends for :func:`~moderndid.att_gt` and
:func:`~moderndid.ddd` on top of `Dask <https://www.dask.org/>`_ and
`Dask Distributed <https://distributed.dask.org/>`_. The distributed backend
scales from one machine to many workers and returns the same result object
types as local estimation, so your post-estimation workflow stays the same.

This page focuses on practical usage. For internal implementation details, see
:ref:`Distributed Backend Architecture <distributed-architecture>`.


When to use distributed estimation
----------------------------------

Distributed estimation is most useful when the dataset does not fit in memory
on one machine, when local runtime is too long for your iteration cycle, when
you need to run many model specifications over the same large dataset, or when
you already maintain a Dask cluster for analytics workloads.

Distributed execution has scheduler and communication overhead. If local
estimation is fast and memory safe, local execution is often simpler and easier
to debug.


Requirements
------------

Install the Dask extra before running distributed estimators.

.. code-block:: console

    uv pip install moderndid[dask]

The ``all`` extra includes ``dask``, so ``uv pip install moderndid[all]``
also works. The distributed path expects a Dask DataFrame input and a
working ``distributed`` scheduler.

ModernDiD cannot set up your Dask environment for you. Creating clusters,
managing workers, and building Dask DataFrames is the responsibility of the
user. Once you have a Dask DataFrame, ModernDiD handles the rest. If you
are new to Dask, the
`10 Minutes to Dask <https://docs.dask.org/en/stable/10-minutes-to-dask.html>`_
guide and the
`Distributed documentation <https://distributed.dask.org/en/stable/>`_
are good starting points.


Quick start
-----------

The distributed backend activates automatically when ``data`` is a Dask
DataFrame. If ``data`` is any other type, the local estimator runs instead.
This lets you move from local development to cluster execution without
rewriting estimator arguments.

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

The same pattern works for triple differences.

.. code-block:: python

    result = did.ddd(
        data=ddf,
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
low-level functions ``dask_att_gt`` and ``dask_ddd`` in
:mod:`moderndid.dask` give you explicit control over the Dask client.

High-level wrappers do not expose a ``client`` argument. Creating a
``Client`` registers it as the global default, so estimator calls pick it
up automatically. The low-level functions accept a ``client`` argument
directly.

.. code-block:: python

    import moderndid as did
    from moderndid.dask import dask_att_gt
    from dask.distributed import Client

    # High-level entry point (uses the global default client)
    client = Client("scheduler-address:8786")
    result_a = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

    # Low-level entry point (accepts client explicitly)
    result_b = dask_att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group", client=client)


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

The distributed estimators expect a Dask DataFrame in long format. Include
all required columns in the same frame and keep ``time`` and treatment group
columns numeric. For panel data (``panel=True``), ensure unit identifiers
are stable across periods and prefer one record per unit-period. For
repeated cross-section data (``panel=False``), each row is an independent
observation and unit identifiers are not required.

Staging large datasets in batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the full dataset is too large to build in driver memory, write it to
Parquet in batches and read it back as a Dask DataFrame. This keeps driver
memory constant regardless of total dataset size.

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

The same pattern applies when your raw data comes from a database, an API,
or any other source that must be fetched incrementally. Replace the body of
``_generate_chunk`` with whatever logic produces a single pandas DataFrame,
and the rest of the pipeline stays the same.

Partition layout
^^^^^^^^^^^^^^^^

Partition layout affects both runtime and memory. Extremely small partitions
increase scheduler overhead, and extremely large partitions increase worker
memory pressure. If you plan to run multiple model specifications over the
same data, repartition once and persist before the first fit.

.. code-block:: python

    ddf = ddf.repartition(npartitions=64)

Persist and sanity-check
^^^^^^^^^^^^^^^^^^^^^^^^

Dask is lazy by default, so heavy work can execute later than expected,
including inside estimator calls. In practice this means upstream ETL issues
may surface during ``att_gt`` or ``ddd`` even when your data construction
code appears to run without errors. A good pattern is to persist and
sanity-check the input before fitting.

.. code-block:: python

    from dask.distributed import wait

    ddf = ddf.persist()
    wait(ddf)

    # Lightweight checks before estimation
    print(ddf.columns)
    print(ddf[["id", "time", "group"]].head())

This boundary separates data-pipeline failures from estimator failures,
which makes debugging faster.


From local to distributed
-------------------------

A practical workflow is to develop and validate your specification locally
on a sample, then scale up to the full dataset by swapping in a Dask
DataFrame. The estimator arguments stay the same. Only the input type
changes.

.. code-block:: python

    import polars as pl
    import dask.dataframe as dd
    import moderndid as did

    shared_args = dict(
        yname="y", tname="time", idname="id", gname="group",
        xformla="~ x1 + x2", est_method="dr", control_group="nevertreated",
    )

    # Develop locally on a sample
    sample = pl.read_parquet("panel_data.parquet").sample(n=10_000, seed=42)
    local_result = did.att_gt(data=sample, **shared_args)

    # Scale to the full dataset
    ddf = dd.read_parquet("panel_data.parquet")
    dist_result = did.att_gt(data=ddf, **shared_args)


Connecting to a cluster
-----------------------

If no client exists, ModernDiD creates a local client automatically, which
is convenient for development. For multi-node runs, create a ``Client``
pointing at your scheduler. Creating the client registers it as the global
default, so estimator calls use it automatically.

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

For more predictable runtime in autoscaling environments, warm the cluster
and persist input data before launching long model runs.


Distributed-specific parameters
-------------------------------

In addition to standard estimator arguments, distributed execution uses a
small set of controls.

``client``
    A ``distributed.Client`` for low-level calls (``dask_att_gt``,
    ``dask_ddd``). If omitted, ModernDiD first tries ``Client.current()`` and
    creates a local client when none is active.

``n_partitions``
    Number of Dask partitions per cell computation. The default equals total
    worker threads, and the backend can increase it when estimated partition
    design matrices would be too large.

``max_cohorts``
    Maximum number of treatment cohorts processed concurrently. Defaults to
    the number of Dask workers. Lower values reduce peak memory because fewer
    cohort-wide tables are active at once. Higher values can improve
    throughput on memory-rich clusters.

``progress_bar``
    Enables a ``tqdm`` progress bar in multi-period runs. Each tick corresponds
    to one group-time cell.

``backend``
    Set to ``"cupy"`` to run partition-level linear algebra on worker GPUs.
    See :ref:`Combining GPU and Dask <gpu-dask-workers>` for setup details.

.. code-block:: python

    from moderndid.dask import dask_att_gt

    # Low-level entry point (accepts client explicitly)
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
        progress_bar=True,
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
  present in the Dask DataFrame.
- **GPU on workers** (``backend="cupy"``) — runs partition-level linear
  algebra on worker GPUs. See
  :ref:`Combining GPU and Dask <gpu-dask-workers>` for setup.


Running multiple specifications
-------------------------------

When running multiple specifications over the same data, persist the Dask
DataFrame once and reuse it across calls. This avoids re-reading and
re-shuffling the data for each specification.

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

For long-running jobs, the Dask dashboard provides real-time visibility into task
progress, worker memory, and task stream. The ``progress_bar`` parameter
provides a simpler alternative that works in notebooks and scripts.

.. code-block:: python

    from moderndid.dask import dask_att_gt

    result = dask_att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        client=client,
        progress_bar=True,    # Track cell completion
    )


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
