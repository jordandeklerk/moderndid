.. _dev-spark:

Fits on Spark
=============

The walkthroughs of the :doc:`panel <shards>` and :doc:`row <row_shards>`
engines run every pass through ``LocalExecutor`` on shards held in memory. On
Spark, the same drivers run the same passes with a ``SparkExecutor`` over a
table that holds the data. We'll follow a fit from ``att_gt`` to that table and
the tasks of each pass before turning to the tests that keep the two paths in
agreement and the rules for code that runs on Spark.

Running the passes on Spark
---------------------------

``spark_att_gt`` builds a table from the data and gives a ``SparkExecutor``
over it to the driver that the returned configuration calls for.

.. code-block:: python

   # From spark_att_gt, where data is a Spark DataFrame.
   built = (
       build_unit_table(data, config)
       if panel
       else build_row_table(data, config, draws=boot or cband)
   )
   table, config, cohort_counts, n_obs, cluster_counts, whole_clusters, cached = built

   # A panel in which some unit misses a period takes the estimators of repeated cross-sections.
   n_covariates = len(_covariate_names(config))
   if config.panel:
       read_shard = partial(
           shard_from_arrow, n_periods=config.time_periods_count, n_covariates=n_covariates
       )
       fit = fit_att_gt_on_shards
   else:
       read_shard = partial(row_shard_from_arrow, n_covariates=n_covariates)
       fit = fit_att_gt_on_row_shards
   executor = SparkExecutor(table, read_shard, whole_clusters=whole_clusters)
   result = fit(
       executor, config, cohort_counts, random_state, estimation_params, cluster_counts=cluster_counts
   )

``att_gt`` recognizes a Spark DataFrame by its class without importing PySpark
and hands it to ``spark_att_gt`` before any step that reads the data.
Otherwise ``check_columns`` would convert a PySpark 4.2 DataFrame through its
``__arrow_c_stream__`` and collect the whole table to the driver.
``spark_att_gt`` builds its ``DIDConfig`` and ``estimation_params`` by hand
rather than through the in-memory preprocessing.

.. admonition:: A new option needs the Spark path
   :class: warning

   Unless a new ``att_gt`` option also gets a place in ``spark_att_gt``, a
   Spark fit ignores it without an error. Since the parity tests in
   ``tests/did/`` never call ``spark_att_gt``, only the tests in
   ``tests/spark/`` catch that gap.

``build_unit_table`` runs the in-memory ``validate`` stage and
``ColumnSelector`` on an empty frame with the data's schema and repeats the
later ``transform`` steps in Spark SQL. There, one aggregation and one shuffle
gather each unit's rows into one row of the unit table. With
``allow_unbalanced_panel=True``, the table keeps every unit that has a kept
row. Once some unit misses a period, the row of each unit also lists the
indices of the periods it was observed in. The returned configuration then has
``panel`` False, as the in-memory preprocessing would set it.

``build_row_table`` repeats the preprocessing of repeated cross-sections the
same way and keeps each observation in a separate row. Besides the
aggregations that preprocessing needs, it shuffles the rows in only two cases.
One numbers the copies of identical rows for an unclustered fit that draws.
The other keeps each cluster in one partition when a clustered fit has many
clusters.

New data for each unit or row also needs a column that a builder writes and
that ``shard_from_arrow`` or ``row_shard_from_arrow`` reads into the shard.

.. admonition:: Mirror preprocessing changes on Spark
   :class: important

   Since neither builder runs the in-memory pipeline on the rows, a change to
   a ``transform`` step in memory needs the same change in each builder that
   repeats the step.

The fit persists the table where the session allows it and unpersists it
however the fit ends, an interrupt included. A session that refuses to persist
gets a local checkpoint instead. If it refuses that too, a warning says that
each pass rebuilds the table from the data.

.. _dev-spark-tasks:

Tasks of a Spark pass
---------------------

``SparkExecutor.run`` applies a kernel to each partition of the table through
``mapInArrow``. Each task reads its partition in slices of at most 4096 table
rows. The ``read_shard`` function that the executor pickles along with the
kernel turns every slice into a shard. ``row_shard_from_arrow`` reads both
tables of the row engine and tells them apart by the unit table's ``unit``
column. Because a row of the unit table holds a whole unit, a slice never
splits a unit, even in an unbalanced panel. Each task returns its statistics
as one row. Each row also carries the task's moderndid and Python versions for
the driver to check. If a mismatched executor raised an error instead, the
driver would see a ``PythonException`` rather than one that says what to
install.

When the table keeps every cluster within one partition, as it does for a fit
with many clusters, each task finalizes its own keyed sums first. With few
clusters, the table spreads each cluster over the partitions instead. The
driver then finalizes the keyed sums once it has combined every row. Since each
task's keyed sums hold a row of :math:`z_i` for every cluster it touched, both
builders spread clusters only while their approximate count times a bound on
the length of :math:`z_i` stays within :math:`2^{17}`, about a megabyte of
sums.

On the first pass and after any pass whose tasks returned more than 16 rows, a
grouped stage combines the rows of the :math:`P` tasks in about
:math:`\sqrt{P}` groups before the driver collects them. Because choosing
:math:`k` for ``repartition(k)`` would take an extra job under Spark Connect to
count the partitions, the stage groups the task numbers by their integer square
root instead.

.. admonition:: Fold rows in task order
   :class: important

   Like both stages, a new step that combines the rows of tasks folds them in
   the order of their task numbers. Otherwise its sums would change with the
   order in which Spark delivers the rows.

Agreement with the in-memory fit
--------------------------------

What a fit on shards repeats, and how exactly, depends on what each number is
built from.

- The cell plan and the bootstrap signs depend only on the values in the data
  and the seed. They're the same for every partitioning and every rerun.
- The effects, standard errors, and bootstrap results come from sums that add
  up in a different order on each partitioning. They can differ in their last
  bits from one partitioning to the next and, apart from the bootstrap, from
  the in-memory fit. Since the in-memory bootstrap draws its signs from one
  random stream in the order of the units, its results differ by the noise of
  the draws.
- A decision within rounding of its threshold, such as the overlap screen at
  0.999 or a rank check on nearly collinear covariates, can go the other way
  when the sums add up in another order. Covariates so small or large that
  their squares leave the range of floating point can also make the engines
  disagree about which columns are collinear.

Because the two paths share the cell plan, the in-memory checks, and
``_finish_att_gt`` but none of the estimators' algebra, agreement between them
checks the engines' sums. Like the walkthroughs of the two engines,
``tests/did/test_shards.py`` runs the panel engine through ``LocalExecutor``
against the in-memory ``att_gt`` and ``aggte``. Its option matrix of
estimators, comparison groups, base periods, weights, clusters, and
anticipation runs on three shards and compares the warnings in order. Each
estimator also runs on one and seven shards to vary the split.
``tests/did/test_row_shards.py`` does the same for the row engine on both of
its layouts. The :ref:`testing guide <testing-spark>` covers the end-to-end
tests that run on Spark itself, such as ``tests/spark/test_att_gt.py``. A new
option needs a case in the option matrices of all three files.

Besides data near those thresholds, the parity tests work around the few
behaviors below that the engines don't copy. A new warning from a dependency
either gets a copy on the shard path or joins this list.

- Of NumPy's RuntimeWarnings inside statsmodels, only the overflow in a
  logit's exponential has a copy. The row engine alone replays it, once for
  each iteration as in memory.
- The ``SingularMatrixWarning`` that statsmodels 0.15 raises for a numerically
  rank-deficient design, such as one with a covariate far smaller than the
  others, has no copy.
- When every comparison row of one period is trimmed, the in-memory estimators
  of repeated cross-sections divide by a zero mean weight and warn about the
  invalid division before the cell fails. The row engine fails the same cell
  without that warning.
- A Spark fit lists the cohorts of its small-groups warning in ascending order
  and stops before its first job on weights or covariates that aren't numbers.

For a few in-memory checks, the drivers build stand-in arrays from the kernels'
statistics, such as the treated and comparison weight totals for
``_check_cell_weights``.

.. admonition:: Keep the shard path in step
   :class: important

   A change to the in-memory estimators needs the same change on the shard
   path. A check that starts reading more of its input than its stand-in holds
   needs a new kernel statistic.

.. _dev-spark-rules:

Rules for Spark code
--------------------

The tests through ``LocalExecutor`` never reach finalizing in each task, the
grouped stage, the PySpark 3.5 paths, the version check, or the cached table. A
change to any of them needs a test in ``tests/spark/``. Code that runs on Spark
keeps to six rules that the ``LocalExecutor`` tests mostly can't check.

- It calls only DataFrame APIs that Spark Connect supports and never
  ``sparkContext``, RDDs, or broadcast variables.
- It runs on PySpark 3.5 as well as on PySpark 4. Code that needs a newer API
  keeps a 3.5 path beside it and picks one by feature, never by version. A
  version check would miss the newer APIs that Databricks adds to its PySpark
  3.5. ``SparkExecutor.run``, for example, checks the grouped data for
  ``applyInArrow``. Where PySpark 3.5 takes only a literal, such as the value
  that ``array_position`` looks up, the code passes no Column. In library code,
  only the PySpark floor check in ``spark_att_gt`` reads the version.
- It never sets a session configuration because serverless compute on
  Databricks lets a session set only a few of them.
- Kernels are module-level functions with NamedTuple states because
  ``SparkExecutor`` pickles a kernel and the class of its state by reference.
  Since ``LocalExecutor`` never pickles, a pickle round-trip test in
  ``tests/did/`` checks every kernel of the panel engine.
- Since a ``SparkExecutor`` built with ``whole_clusters=True`` finalizes the
  keyed sums in each task, its table must keep every cluster whole. Both
  builders do so by hashing their shuffle on the cluster alone and merging
  only whole partitions afterwards. A repartition by anything else, or a
  ``REBALANCE`` hint, would split clusters and miscount them without an error.
- The kernels keep their working memory bounded through the slices that each
  task reads and through blocks of units and draws. A clustered task's sums
  still grow by one row for each cluster. Serverless compute on Databricks
  needs that bound because Databricks documents a cap of 1 GB on the memory of
  each Python UDF.

The :ref:`testing guide <testing-spark>` shows how to run the tests in
``tests/spark/`` on a local cluster and through Spark Connect before a change
to any of this code goes in.
