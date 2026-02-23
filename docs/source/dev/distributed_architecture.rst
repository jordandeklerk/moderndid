.. _distributed-architecture:

.. _spark-architecture:

======================
Distributed Computing
======================

This document describes the internal design of ModernDiD's distributed
backends for `Dask <https://www.dask.org/>`_ and
`Apache Spark <https://spark.apache.org/>`_. Both backends share the same
algorithmic decomposition; only the communication primitives differ. The
sections below use Dask code examples for concreteness.
:ref:`Spark-specific mechanics <spark-mechanics>` are described at the end.

For usage documentation, see the :ref:`Dask guide <distributed>` and the
:ref:`Spark guide <spark>`.


Scope and design rule
=====================

This page is intended for contributors and advanced users who want to
understand how distributed estimation is implemented. It covers execution
decomposition for ``att_gt`` and ``ddd``, distributed nuisance estimation,
aggregation, and memory-management strategies for large panels.

The distributed backends are designed around a single rule: **Never
materialize the full dataset on any single machine.** All computation
happens on workers via partition-level sufficient statistics. Only small
summary matrices return to the driver.


Execution modes
===============

The Dask entry points ``dask_att_gt`` and ``dask_ddd`` unconditionally
delegate to their multi-period implementations ``dask_att_gt_mp`` and
``dask_ddd_mp``. The multi-period path handles both two-period and
staggered designs.

.. code-block:: python

    # Entry-point dispatch (simplified from dask_att_gt / dask_ddd)
    from ._did_mp import dask_att_gt_mp

    return dask_att_gt_mp(
        client=client,
        data=data,
        ...
    )


Cell-level decomposition
========================

Both :func:`~moderndid.att_gt` and :func:`~moderndid.ddd` decompose estimation
into independent group-time cells ``(g, t)``. Each cell computes a doubly
robust ATT.

.. math::

    ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g].

Each cell needs three components.

- A propensity score model :math:`P(G = g \mid X)`
- An outcome regression model :math:`E[\Delta Y \mid X, G \neq g]`
- Per-unit influence function values for inference

In the local estimator, computing a single cell means subsetting the panel
to the relevant units and time periods, running logistic regression for the
propensity score, running WLS for the outcome regression, and computing the
influence function, all in memory on one machine.

In the distributed backend, each of these steps is decomposed into
per-partition operations that execute on workers. The key here is that the
logistic regression and WLS problems can be expressed entirely in terms
of *sufficient statistics*, small :math:`k \times k` matrices and
:math:`k`-vectors (where :math:`k` is the number of covariates) that can be
computed independently on each partition and then summed. The driver only
ever sees these small summaries, never the raw data.

.. code-block:: python

    # Partition-level sufficient statistics (from moderndid.dask._gram)
    XtWX_local, XtWy_local, n_local = partition_gram(X_part, W_part, y_part)


Distributed nuisance estimation
===============================

The nuisance models are the most expensive part of each cell. Both models are
fit without collecting raw observations on the driver.

**Propensity score via distributed IRLS.** The propensity score is a
logistic regression of treatment status on covariates. ModernDiD fits it
using iteratively reweighted least squares (IRLS), a Newton-Raphson
algorithm where each iteration reduces to a weighted least squares
problem.

Each IRLS iteration follows five steps.

1. The driver broadcasts the current coefficient vector :math:`\beta^{(t)}`
   to all workers. This is a :math:`k`-vector.
2. Each worker computes its local linear approximation. For partition
   :math:`j`, this means computing the predicted probabilities
   :math:`\mu_i = 1/(1 + e^{-X_i \beta})`, the working weights
   :math:`W_i = \mu_i(1 - \mu_i)`, and the working response
   :math:`z_i = X_i\beta + (D_i - \mu_i) / W_i`.
3. Each worker forms the local Gram matrix :math:`X_j^T W_j X_j` (a
   :math:`k \times k` matrix) and the local score vector
   :math:`X_j^T W_j z_j` (a :math:`k`-vector).
4. These per-partition matrices are tree-reduced (see :ref:`tree-reduce-aggregation`) to form
   the global Gram matrix and score vector on the driver.
5. The driver solves the :math:`k \times k` normal equations
   :math:`\beta^{(t+1)} = (X^TWX)^{-1}X^TWz` and broadcasts the updated
   coefficients.

This repeats until convergence. At each step, only :math:`k`-vectors and
:math:`k \times k` matrices travel between workers and the driver. With
20 covariates, each worker sends a 21-by-21 matrix and a 21-vector,
regardless of partition row count.

.. code-block:: python

    # IRLS update shape (conceptual)
    beta = np.zeros(k)
    for _ in range(max_iter):
        part_futures = [
            client.submit(_irls_local_stats, part, beta)  # returns (XtWX_j, XtWz_j, n_j)
            for part in partitions
        ]
        XtWX, XtWz, _ = tree_reduce(client, part_futures, combine_fn=_sum_gram_pair)
        beta_new = np.linalg.solve(XtWX, XtWz)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

**Outcome regression via distributed WLS.** The outcome regression fits a
weighted least squares model of the outcome change :math:`\Delta Y = Y_t -
Y_{t-1}` on covariates :math:`X` among control units (:math:`D = 0`).
Unlike the propensity score, WLS is not iterative. Each worker computes
:math:`X_j^T W_j X_j` and :math:`X_j^T W_j y_j` in a single pass, these
are tree-reduced, and the driver solves the normal equations once. This
requires just one round of communication.

.. code-block:: python

    # One-pass distributed WLS solve
    XtWX, XtWy, _ = distributed_gram(client, partitions)
    gamma = solve_gram(XtWX, XtWy)

.. _tree-reduce-aggregation:

Tree-reduce aggregation
=======================

Both IRLS and WLS produce one :math:`k \times k` Gram matrix per partition.
These must be summed into a single global matrix before the driver can solve
the normal equations. The default partition count equals total worker threads
(see ``get_default_partitions``), so a cluster with 64 threads, for example, produces 64
partition-level matrices. Naive reduction does 63 sequential pairwise adds
on the driver, which serializes the critical path and limits throughput.

ModernDiD uses a tree-reduce pattern with configurable fan-in
(``split_every=8`` by default). Futures are reduced in batches on workers,
then recursively combined. With 64 partitions and fan-in 8, this produces
9 reduction tasks instead of 63 pairwise additions. The pattern is used for
Gram aggregation, global statistics, and bootstrap sums.

.. code-block:: python

    # Tree-reduce API shape
    result = tree_reduce(client, futures, combine_fn=_sum_gram_pair, split_every=8)


Wide-pivot optimization
=======================

For ``nevertreated`` control groups, the distributed backend includes an
optimization that dramatically reduces the number of Dask shuffle
operations.

Consider cohort :math:`g=5` with periods 1 through 10. The cohort has cells
``(5,1), (5,2), ..., (5,10)``. A naive implementation performs one distributed
shuffle join per cell to merge post and pre outcomes by unit ID. That means
10 joins for one cohort.

The wide-pivot optimization removes this redundancy. It builds one wide
DataFrame per cohort with one row per unit and one ``_y_{period}`` column for
every period needed by any cell in the cohort. For cohort :math:`g = 5` with
cells requiring periods 1 through 4, the wide DataFrame looks like:

.. code-block:: text

    ┌────────┬───────┬──────┬──────┬──────┬──────┬──────┬──────┐
    │ id     │ group │ x1   │ x2   │ _y_1 │ _y_2 │ _y_3 │ _y_4 │
    ├────────┼───────┼──────┼──────┼──────┼──────┼──────┼──────┤
    │ 1      │ 5     │ 0.3  │ -0.1 │ 1.2  │ 1.5  │ 1.8  │ 2.4  │
    │ 2      │ 0     │ -0.5 │ 0.7  │ 0.9  │ 1.1  │ 1.3  │ 1.4  │
    │ 3      │ 5     │ 0.1  │ 0.4  │ 1.0  │ 1.3  │ 1.7  │ 2.1  │
    │ ...    │ ...   │ ...  │ ...  │ ...  │ ...  │ ...  │ ...  │
    └────────┴───────┴──────┴──────┴──────┴──────┴──────┴──────┘

Each cell then selects its post and pre outcome columns (e.g.
``_y_4 - _y_3`` for cell ``(5, 4)``) without any additional shuffle.

After this one shuffle join, each cell uses column selection for pre and post
outcomes. No additional worker-to-worker movement is required.

For ``notyettreated`` controls, each cell can require a different control set
at time :math:`t`. The backend therefore falls back to per-cell streaming and
builds a separate merged DataFrame per cell. This is the main reason
``nevertreated`` is often faster in distributed runs.

.. code-block:: python

    # Cohort-wide pivot for nevertreated controls (from _utils)
    wide_dask = prepare_cohort_wide_pivot(
        client=client,
        dask_data=dask_data,
        g=g,
        cells=compute_cells,
        time_col=time_col,
        group_col=group_col,
        id_col=id_col,
        y_col=y_col,
        covariate_cols=covariate_cols,
        n_partitions=n_partitions,
        extra_cols=extra_cols,       # e.g. [partition_col] for DDD
    )


Cohort-level parallelism
========================

Treatment cohorts are processed in parallel via a ``ThreadPoolExecutor``,
controlled by the ``max_cohorts`` parameter. Within each cohort, cells are
processed sequentially since they share the same wide-pivoted DataFrame (or,
for ``notyettreated``, because they operate on overlapping subsets of the
data).

The outer level provides coarse-grained parallelism across cohorts. The inner
level provides fine-grained parallelism across worker partitions for each cell.
This two-level design improves utilization because one cohort can make progress
while another waits on a reduction.

The default ``max_cohorts`` equals the number of Dask workers. Lower values
reduce peak memory because fewer cohort-wide tables are active at once.
Higher values increase concurrent work and can improve throughput on
memory-rich clusters.

.. code-block:: python

    # Cohort-level concurrency pattern (simplified)
    # DiD uses _process_did_cohort_cells, DDD uses _process_cohort_cells
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_cohort_fn, cells, ...): g
            for g, cells in cohort_cells.items()
        }
        for future in as_completed(futures):
            cohort_results = future.result()
            attgt_list.extend(cohort_results)


Influence function streaming
============================

After computing the ATT for a cell, each worker computes per-unit influence
function values. These values are needed for inference. The influence
function matrix has shape ``(n_units, n_cells)``, where each column stores
the influence function contributions for one ``(g, t)`` cell. For large
datasets (millions of units and dozens of cells), this matrix can be
substantial.

Instead of gathering all partitions at once, the backend streams influence
function values one partition at a time using ``as_completed``. When a future
resolves, values are written into the corresponding matrix column and then
released. Peak driver memory stays bounded by one partition-sized chunk.

.. code-block:: python

    # Streaming influence function gathering (conceptual)
    scale = n_units / n_cell
    for future in as_completed(if_futures):
        ids_part, if_part = future.result()
        if_scaled = scale * if_part
        indices = np.searchsorted(unique_ids, ids_part)
        inf_func_mat[indices, cell_index] = if_scaled
        del ids_part, if_part, if_scaled  # free immediately


End-to-end data flow
====================

For a single ``(g, t)`` cell with doubly robust estimation and
``nevertreated`` controls, the high-level flow follows six steps.

1. Build and persist a cohort-level wide table across workers.
2. Prepare cell-specific post/pre outcome partitions and materialize futures.
3. Estimate nuisance parameters with distributed IRLS and distributed WLS.
4. Tree-reduce partition-level sufficient statistics to compute ``ATT(g,t)``.
5. Stream partition-level influence function values into the shared IF matrix.
6. After all cells, compute variance and standard errors on the driver and
   return the same result type as local estimation.

Steps 1 and 2 contain the shuffle-heavy work. Steps 3 through 5 primarily use
task submission and tree-reduce over compact statistics.


Distributed bootstrap
=====================

When ``boot=True``, multiplier bootstrap runs in distributed mode. Each worker
generates Mammen two-point weights and computes local contributions
:math:`\sum \psi_i v_i`. Per-partition ``(B, k)`` matrices are tree-reduced to
the driver, which computes standard errors and critical values.

This avoids transmitting unit-level bootstrap weights over the network.
Workers only need random seeds.

.. code-block:: python

    # Distributed multiplier bootstrap for DDD path
    bres, se_boot, crit_val = distributed_mboot_ddd(
        client=client,
        inf_func_partitions=inf_parts,
        n_total=n_units,
        biters=biters,
        alpha=alpha,
        random_state=random_state,
    )


Memory management
=================

The distributed backend includes several mechanisms to keep memory usage
bounded on both the driver and workers.


Auto-tuned partitions
---------------------

The default partition count equals total worker threads. If estimated
per-partition design matrices would exceed 500 MB, ``auto_tune_partitions``
increases partition count automatically. You can override this with
``n_partitions``.

.. code-block:: python

    # Partition count initialization and auto-tuning
    if n_partitions is None:
        n_partitions = get_default_partitions(client)

    k = len(covariate_cols) + 1 if covariate_cols else 1
    n_partitions = auto_tune_partitions(n_partitions, n_units, k)


Memory-mapped influence functions
---------------------------------

The influence function matrix has shape ``(n_units, n_cells)``. At
100 million units and 50 cells, it is about 40 GB. When the matrix exceeds
1 GB (``MEMMAP_THRESHOLD``), the backend stores it as a temporary memory-mapped
file. The file is removed after estimation finishes.

.. code-block:: python

    # Memory-map IF matrix when dense allocation would be large
    mat_bytes = n_units * n_cols * 8
    if mat_bytes > MEMMAP_THRESHOLD:
        fd, memmap_path = tempfile.mkstemp(suffix=".dat", prefix="did_inf_")
        os.close(fd)
        inf_func_mat = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=(n_units, n_cols))
    else:
        inf_func_mat = np.zeros((n_units, n_cols))


Chunked variance-covariance
---------------------------

Computing :math:`V = \Psi^T \Psi / n` requires a full pass over the influence
function matrix. For more than 10 million rows, the product is computed in
chunks of 1 million rows. Chunk-level partial products are then summed.

.. code-block:: python

    # Chunked vcov path for very large influence-function matrices
    V = chunked_vcov(inf_func_trimmed, n_units)


.. _spark-mechanics:

Spark backend mechanics
=======================

The Spark backend (``moderndid.spark``) implements the same algorithmic
decomposition described above. The entry points ``spark_att_gt`` and
``spark_ddd`` delegate to ``spark_att_gt_mp`` and ``spark_ddd_mp``
respectively. Cell-level decomposition, nuisance estimation, wide-pivot
optimization, cohort-level parallelism, influence function streaming, and
memory management all follow the same design. Only the communication
primitives differ.

The table below maps each Dask primitive to its Spark equivalent.

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Operation
     - Dask
     - Spark
   * - Task submission
     - ``client.submit()`` per partition
     - ``mapInPandas`` across DataFrame partitions
   * - Broadcast
     - ``client.scatter(beta, broadcast=True)``
     - ``SparkContext.broadcast(beta)``
   * - Small-result reduction
     - Custom ``tree_reduce`` with fan-in
     - ``collect()`` + driver-side ``_reduce_gram_list``
   * - Large-result reduction
     - Custom ``tree_reduce`` with fan-in
     - ``RDD.treeReduce(depth=3)``
   * - IF streaming
     - ``as_completed(futures)``
     - ``toLocalIterator(prefetchPartitions=True)``
   * - Caching
     - ``persist()`` + ``wait()``
     - ``.cache()`` + ``.count()``
   * - Default partitions
     - Total worker threads
     - ``spark.sparkContext.defaultParallelism``

**Gram collection.** For IRLS and WLS, each executor computes its local
Gram matrix via ``mapInPandas``, serializes it with pickle, and the driver
collects the small binary results with ``collect()``. Gram matrices are
tiny (kilobytes), so driver-side collection is efficient.

.. code-block:: python

    # mapInPandas + collect pattern (from spark._gram)
    result_df = cached_df.mapInPandas(_compute_gram_udf, schema=out_schema)
    rows = result_df.collect()
    gram_list = [pickle.loads(row["gram_bytes"]) for row in rows]
    XtWX, XtWy, n = _reduce_gram_list(gram_list)

**IRLS broadcast.** Each IRLS iteration broadcasts :math:`\beta` via
``SparkContext.broadcast()`` and destroys the broadcast variable after
collection to avoid memory leaks.

.. code-block:: python

    beta_bc = sc.broadcast(beta)
    rows = cached_df.mapInPandas(_irls_udf, schema=out_schema).collect()
    beta_bc.destroy()

**Bootstrap via RDD treeReduce.** Per-partition ``(B, k)`` bootstrap
matrices are larger than Gram matrices, so the Spark backend uses
``RDD.treeReduce()`` instead of ``collect()`` to avoid materializing
all intermediate results on the driver at once.

.. code-block:: python

    # From spark._bootstrap
    rdd = sc.parallelize(partitions_with_seeds)
    rdd = rdd.map(lambda args: _local_bootstrap(*args))
    total_result = rdd.treeReduce(_sum_bootstrap_pair, depth=3)

**Cache management.** Cohort-wide DataFrames are cached with ``.cache()``
and unpersisted immediately after processing all cells in the cohort to
prevent stale cached tables from consuming executor memory.
