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

The ``all`` extra does not include ``dask``. Install it explicitly when you
plan to use distributed estimation. The distributed path expects a Dask
DataFrame input and a working ``distributed`` scheduler.


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

High-level wrappers do not expose a ``client`` argument. To run them on a
specific cluster, set the client as current via ``client.as_current()``.

.. code-block:: python

    import moderndid as did
    from moderndid.dask import dask_att_gt
    from dask.distributed import Client

    # Recommended high-level entry point
    result_a = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

    # High-level wrapper with an explicit existing client
    client = Client("scheduler-address:8786")
    with client.as_current():
        result_b = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

    # Equivalent explicit distributed entry point
    result_c = dask_att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group", client=client)


Current support and limits
--------------------------

The distributed path supports both panel data and repeated cross-section
data for any number of time periods. When a Dask DataFrame is passed to
``att_gt`` or ``ddd``, the fully distributed cell-level backend handles
all designs, including two-period data. Set ``panel=False`` for repeated
cross-section data where different individuals are observed in each period.

The distributed-specific tuning controls are ``n_partitions``,
``max_cohorts``, ``progress_bar``, and ``backend`` on high-level
wrappers, plus ``client`` on the low-level functions. High-level
wrappers pick up the active client via ``client.as_current()``.

All standard estimation options that affect identification work in
distributed mode, including ``control_group``, ``anticipation``,
``base_period``, and ``est_method`` (string values only). When ``boot=True``,
the multiplier bootstrap runs fully distributed with Mammen two-point weights
generated on workers and tree-reduced to the driver, so bootstrap inference
scales with the cluster. One default differs from local estimation.
``cband`` defaults to ``False`` in distributed mode (local ``att_gt``
defaults to ``True``), so set ``cband=True`` explicitly if you want uniform
confidence bands.

Some parameters accepted by local wrappers are not yet active in
distributed dispatch. ``att_gt`` with Dask input does not support callable
``est_method`` or ``n_jobs``. ``ddd`` with Dask input ignores
``boot_type`` and ``n_jobs``. The ``cluster`` parameter in multi-period
distributed ``ddd`` is accepted for API compatibility and result metadata
but does not change the inference path.


Preparing input data
--------------------

The distributed estimators expect a Dask DataFrame in long format. Include
all required columns in the same frame and keep ``time`` and treatment group
columns numeric. For panel data (``panel=True``), ensure unit identifiers
are stable across periods and prefer one record per unit-period. For
repeated cross-section data (``panel=False``), each row is an independent
observation and unit identifiers are not required. Loading from Parquet is
the most reliable path because Parquet preserves column types and supports
efficient partition pruning.

.. code-block:: python

    import dask.dataframe as dd

    cols = ["id", "time", "group", "partition", "y", "x1", "x2"]
    ddf = dd.read_parquet("data/panel/*.parquet", columns=cols)

    # Optional light cleaning before estimation
    ddf = ddf.dropna(subset=["id", "time", "group", "y"])

Partition layout affects both runtime and memory. Extremely small partitions
increase scheduler overhead, and extremely large partitions increase worker
memory pressure. If you plan to run multiple model specifications over the same data,
repartition once and persist before the first fit.

.. code-block:: python

    ddf = ddf.repartition(npartitions=64)

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
DataFrame. The estimator arguments stay the same.

.. code-block:: python

    import polars as pl
    import dask.dataframe as dd
    import moderndid as did

    # Step 1: develop locally on a sample
    sample = pl.read_parquet("panel_data.parquet").sample(n=10_000, seed=42)

    local_result = did.att_gt(
        data=sample,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2",
        est_method="dr",
        control_group="nevertreated",
    )

    local_es = did.aggte(local_result, type="dynamic")
    did.plot_event_study(local_es)

    # Step 2: scale to the full dataset
    ddf = dd.read_parquet("panel_data.parquet")

    dist_result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~ x1 + x2",
        est_method="dr",
        control_group="nevertreated",
    )

    dist_es = did.aggte(dist_result, type="dynamic")
    did.plot_event_study(dist_es)


Connecting to a cluster
-----------------------

If no client exists, ModernDiD creates a local client automatically, which
is convenient for development. For multi-node runs, create a client for your
scheduler and run the estimator within ``client.as_current()``.

.. code-block:: python

    import moderndid as did
    from dask.distributed import Client

    client = Client("tcp://scheduler-host:8786")

    with client.as_current():
        result = did.att_gt(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
        )

This pattern works with any Dask-compatible scheduler endpoint, including
Kubernetes, Coiled, and Databricks. On Databricks, you can build the client
from ``dask_databricks``.

.. code-block:: python

    from dask.distributed import Client
    from dask_databricks import DatabricksCluster

    cluster = DatabricksCluster()
    client = Client(cluster)

For local development with controlled resources, create a ``LocalCluster``
explicitly to set worker count and memory limits.

.. code-block:: python

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="4GB")
    client = Client(cluster)

    with client.as_current():
        result = did.att_gt(data=ddf, yname="y", tname="time", idname="id", gname="group")

    client.close()
    cluster.close()

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
        progress_bar=True,
    )

Start with defaults and record runtime and peak memory. Increase
``n_partitions`` if workers are idle for long stretches, or reduce it if
scheduler overhead dominates task time. Reduce ``max_cohorts`` when workers
approach memory limits, and increase it gradually when memory headroom is
large.


Bootstrap inference
-------------------

Bootstrap inference works in distributed mode with the same interface as
local estimation. The multiplier bootstrap generates Mammen two-point
weights on workers and tree-reduces them back to the driver.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        boot=True,
        biters=1000,
        alp=0.05,
        random_state=42,
    )

    event_study = did.aggte(result, type="dynamic")

Set ``cband=True`` if you want uniform confidence bands. The distributed
default is ``cband=False`` to reduce computation, while local ``att_gt``
defaults to ``cband=True``.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        boot=True,
        biters=1000,
        cband=True,
        random_state=42,
    )


Clustered standard errors
-------------------------

The distributed DiD estimator supports clustered standard errors through
the ``clustervars`` parameter. Pass a list of one or two column names to
cluster on.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        boot=True,
        biters=1000,
        clustervars=["state_id"],
        random_state=42,
    )

Two-way clustering is also supported.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        boot=True,
        biters=1000,
        clustervars=["state_id", "industry_id"],
        random_state=42,
    )

The cluster variable columns must be present in the Dask DataFrame. Pass
``clustervars`` as a list, not a bare string. Passing a string raises a
``TypeError``.


Repeated cross-section data
----------------------------

For repeated cross-section designs where different individuals are observed
in each period, set ``panel=False``.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        panel=False,
    )

The same option works for triple differences.

.. code-block:: python

    result = did.ddd(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="dr",
        panel=False,
    )


Unbalanced panels
-----------------

When panel data has units that do not appear in every time period, the
distributed estimator can either drop those units (the default) or keep
them. Set ``allow_unbalanced_panel=True`` to retain all units.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        allow_unbalanced_panel=True,
    )

With the default ``allow_unbalanced_panel=False``, the estimator logs a
warning showing how many units were dropped for not appearing in all
periods.


Weights
-------

Sampling weights are supported in distributed mode through the
``weightsname`` parameter.

.. code-block:: python

    result = did.att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        est_method="dr",
        weightsname="sample_weight",
    )

The weight column must be present in the Dask DataFrame.


GPU acceleration on workers
---------------------------

Pass ``backend="cupy"`` to run partition-level linear algebra on worker
GPUs. For multi-GPU machines, use ``dask-cuda`` with a
``LocalCUDACluster`` to pin one worker per GPU. See
:ref:`Combining GPU and Dask <gpu-dask-workers>` for setup, code
examples, and memory management details.


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
    with client.as_current():
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

For long-running jobs, the Dask dashboard (typically at
``http://scheduler-host:8787``) provides real-time visibility into task
progress, worker memory, and task stream. The ``progress_bar`` parameter
provides a simpler alternative that works in notebooks and scripts.

ModernDiD also provides a cluster monitor that periodically logs memory
and task statistics.

.. code-block:: python

    from moderndid.dask import dask_att_gt
    from moderndid.dask._utils import monitor_cluster

    stop = monitor_cluster(client, interval=15)

    result = dask_att_gt(
        data=ddf,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        client=client,
        progress_bar=True,
    )

    stop()


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


Tuning and troubleshooting
--------------------------

When runtime or memory does not match expectations, use the symptom to guide
tuning. If workers show low utilization while tasks are short, increase
``n_partitions`` to expose more parallel work. If the scheduler is busy but
workers do little useful work, reduce ``n_partitions`` to lower scheduling
overhead. If workers hit memory limits or spill frequently, reduce
``max_cohorts`` and consider increasing partition count. If runtime grows
sharply with ``notyettreated`` controls, try ``nevertreated`` when
identification assumptions allow it. If the progress bar appears stalled,
check the dashboard task stream and worker memory for a skewed partition or
overloaded worker.

Common error messages and their first checks:

``Columns not found in Dask DataFrame: [...]``
    Verify column names and upstream renaming logic before estimation.

``Callable est_method is not supported for Dask inputs. Use 'dr', 'reg', or 'ipw'.``
    Use one of the built-in string methods for distributed runs.

``No valid (g,t) cells found.``
    Check treatment timing, control-group choice, and pre-period availability.

``clustervars must be a list of strings, not a string.``
    Wrap the cluster variable in a list, e.g. ``clustervars=["state_id"]``.

Operational issues can also appear in cluster environments. If the client
connection fails, verify the scheduler address and network routing. If workers
restart mid-run, check memory limits and reduce concurrency with
``max_cohorts``. If there is a long startup before the first cell, persist
input data and wait for workers to reach steady state before calling the
estimator.

A reliable scaling workflow is to validate the specification locally on a
small sample first, then move to Dask with the same estimator arguments.
Persist and inspect the Dask DataFrame before fitting, run one baseline
distributed fit with default tuning, then adjust ``n_partitions`` and
``max_cohorts`` based on cluster telemetry. Keep configuration fixed for
reproducibility-sensitive comparisons, and save aggregated outputs for
auditability.

For architecture-level details on reduction patterns, memory strategy, and
execution decomposition, see
:ref:`Distributed Backend Architecture <distributed-architecture>`.


Next steps
----------

- :ref:`Quickstart <quickstart>` covers estimation options, aggregation
  types, and visualization for local workflows.
- :doc:`gpu` describes GPU acceleration for single-machine workloads.
- :ref:`Estimator Overview <estimator-overview>` surveys all available
  estimators and their distributed support.
- The :ref:`Examples <user-guide>` section walks through each estimator
  end-to-end with real and simulated data.
