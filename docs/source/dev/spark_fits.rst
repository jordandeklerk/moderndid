.. _dev-spark:

Fits on Spark
=============

Moving a fit onto Spark changes where its passes run but not what they
compute. The walkthroughs of the :doc:`panel <shards>` and :doc:`row
<row_shards>` engines run every pass through ``LocalExecutor`` on shards held
in memory. On Spark, ``fit_att_gt_on_shards`` and ``fit_att_gt_on_row_shards``
run the same passes through a ``SparkExecutor``. It sends each kernel to
Spark's own executors, the worker processes that run on each machine. They
apply the kernel to every partition of a table built from the data.

The figure below follows one fit from the user's DataFrame to its result, the
same path that the sections after it take in detail.

.. container:: mdid-diagram

   .. raw:: html
      :file: diagrams/spark-fit.svg

.. rst-class:: mdid-caption

One fit on Spark. Steps 03 to 05 repeat once for each pass of the fit.

Running the passes on Spark
---------------------------

A Spark fit starts in the same ``att_gt`` call that fits data in memory. When
the function receives a Spark DataFrame, it hands the data to
``spark_att_gt``. The lines below, from that function, do the two things a
Spark fit needs before its first pass. They first call ``build_unit_table`` for
a panel, balanced or not, or ``build_row_table`` for repeated cross-sections to
lay out the table that every pass reads. They then give a ``SparkExecutor``
over that table to the fit function that the builder's returned configuration
calls for.

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

The hand-off comes before any step that reads rows, because the in-memory
checks would otherwise collect the data. ``check_columns``, the check that every
named column exists, would convert a PySpark 4.2 DataFrame through its
``__arrow_c_stream__`` and pull the whole table to the driver. ``att_gt``
therefore recognizes a Spark DataFrame by its class, without importing
PySpark. ``spark_att_gt`` then builds its ``DIDConfig`` and
``estimation_params`` by hand rather than through the in-memory preprocessing.

.. admonition:: A new option needs the Spark path
   :class: warning

   Unless a new ``att_gt`` option also gets a place in ``spark_att_gt``, a
   Spark fit ignores it without an error. Only the tests in ``tests/spark/``
   catch that gap, because the parity tests in ``tests/did/`` never call
   ``spark_att_gt``.

A builder has to repeat the in-memory preprocessing without collecting a
single row. ``build_unit_table`` does it by separating the steps that need only
the data's schema from those that read its rows. The in-memory ``validate``
stage and ``ColumnSelector`` need only the names and types of the columns. The
builder runs them unchanged on an empty polars frame with the data's schema.
Because the later ``transform`` steps read rows, the builder repeats them in
Spark SQL. There, one aggregation and one shuffle gather each unit's rows in
long format into one row of the unit table.

With ``allow_unbalanced_panel=True``, the table keeps every unit that has a
kept row. Once some unit misses a period, the row of each unit also lists the
indices of the periods it was observed in. The returned configuration then has
``panel`` False, as the in-memory preprocessing would set it.

``build_row_table`` repeats the preprocessing of repeated cross-sections the
same way and keeps each observation in a separate row. Since a shuffle moves
rows across the network, the builder takes one in only two cases besides the
aggregations that preprocessing needs. One numbers the copies of identical
rows for an unclustered fit that draws. The other keeps each cluster in one
partition when a clustered fit has many clusters. New data for each unit or
row also needs a column that a builder writes and that the shard's reader,
``shard_from_arrow`` or ``row_shard_from_arrow``, turns into an array of the
shard.

.. admonition:: Mirror preprocessing changes on Spark
   :class: important

   Neither builder runs the in-memory pipeline on the rows. A change to a
   ``transform`` step in memory therefore needs the same change in each builder
   that repeats the step.

To keep Spark from rebuilding the table out of the data for every pass, the
fit persists it in the memory and on the disks of Spark's executors where the
session allows. The fit unpersists the table however it ends, an interrupt
included. A session that refuses to persist, such as serverless compute on
Databricks, gets a local checkpoint instead, a copy of the built rows that the
executors keep. If it refuses that too, a warning says that each pass rebuilds
the table from the data.

.. _dev-spark-rules:

Code that runs on Spark has to work in every kind of session a user may have.
That includes Spark Connect, where a Python session reaches a remote Spark
server through DataFrame operations alone, and serverless compute on
Databricks.

.. admonition:: Write Spark code for every kind of session
   :class: important

   Spark code calls only the DataFrame operations that Spark Connect supports,
   never ``sparkContext``, RDDs, or broadcast variables. It never sets a session
   configuration, because serverless compute lets a session set only a few.
   Code that needs an API newer than PySpark 3.5 keeps a 3.5 path and picks
   between them by feature, since Databricks adds newer APIs to its own PySpark
   3.5.

.. _dev-spark-tasks:

Tasks of a Spark pass
---------------------

A pass on Spark has the same job as a pass through ``LocalExecutor``. It runs
the kernel on every shard and combines the partials into one.
``SparkExecutor.run`` hands the kernel to ``mapInArrow``, the DataFrame method
that runs a Python function on the Arrow batches of every partition. Each task
reads its partition in slices of at most 4096 table rows. The ``read_shard``
function that ``SparkExecutor`` pickles along with the kernel turns every slice
into a shard. Because a row of the unit table holds a whole unit, a slice never
splits a unit, even in an unbalanced panel.

.. admonition:: Define kernels at module level
   :class: important

   Pickling sends a kernel and the class of its state by reference. Each kernel
   is therefore a module-level function with a NamedTuple state that an
   executor can import. A pickle round-trip test in ``tests/did/`` checks every
   kernel of the panel engine, because ``LocalExecutor`` never pickles.

Together with blocks of units and draws inside the kernels, the slices keep a
task's working memory within the 1 GB that Databricks documents for each
Python UDF on serverless compute.

Combining starts inside each task, as the task merges the partials of its
slices and returns them as one row. The row also carries the task's moderndid
and Python versions. The driver checks them because an error raised on an
executor with other versions would reach it as PySpark's generic
``PythonException`` rather than as an error that says what to install.

Clustered fits face a trade-off at this step, because a keyed sum can be
finalized only once every unit of its cluster has arrived. When every cluster
sits whole in one partition, as it does for a fit with many clusters, each task
finalizes its own keyed sums first. Keeping clusters whole would run a fit with
few clusters as a few uneven tasks, though. The table of such a fit therefore
spreads each cluster over the partitions. Each task then returns a row of
summed :math:`z_i` for every cluster it touched. The driver finalizes the keyed
sums once it has combined every row. To bound those rows, both builders spread
clusters only while their approximate count times a bound on the length of
:math:`z_i` stays within :math:`2^{17}`, about a megabyte of sums.

.. admonition:: Repartitioning can split clusters
   :class: warning

   A ``SparkExecutor`` built with ``whole_clusters=True`` needs a table that
   keeps every cluster in one partition. Both builders hash their shuffle on
   the cluster alone and merge only whole partitions afterwards. A repartition
   by anything else, or a ``REBALANCE`` hint, would split clusters and miscount
   them without an error.

The rows of the tasks still have to reach the driver. With hundreds of
partitions, collecting every task's row would leave the driver to receive and
add up hundreds of large rows on its own. A grouped stage therefore combines
the rows of the :math:`P` tasks in about :math:`\sqrt{P}` groups on the
executors first. The driver then collects about :math:`\sqrt{P}` rows rather
than :math:`P`. The stage runs on the first pass and after any pass whose tasks
returned more than 16 rows. It groups the task numbers by their integer square
root because a chosen number of groups would take an extra job to count the
partitions under Spark Connect.

.. admonition:: Fold rows in task order
   :class: important

   Like the grouped stage and the driver, a new step that combines the rows of
   tasks folds them in the order of their task numbers. Otherwise its sums
   would change with the order in which Spark delivers the rows.

Agreement with the in-memory fit
--------------------------------

Every change to the engine gets checked against the in-memory fit. Telling a
bug from an expected difference starts with what each number is built from.
The cell plan and the bootstrap signs depend only on the data and the seed and
repeat exactly on every partitioning and rerun. The effects and standard errors
come from sums that add up in a different order on each partitioning. They can
differ in their last bits from one partitioning to the next and from the
in-memory fit. The bootstrap results also differ from the in-memory ones by the
noise of the draws, because the in-memory bootstrap takes its signs from one
random stream in the order of the units.

.. admonition:: Decisions near a threshold can flip
   :class: note

   A decision within rounding of its threshold, such as the overlap screen at
   0.999 or a rank check on nearly collinear covariates, can go the other way
   when the sums add up in another order. Covariates so small or large that
   their squares leave the range of floating point can also make the engines
   disagree about which columns are collinear.

Because the two paths share the cell plan, the in-memory checks, and
``_finish_att_gt`` but none of the estimators' algebra, agreement between them
checks the engines' sums. ``tests/did/test_shards.py`` runs the panel engine
through ``LocalExecutor`` against the in-memory ``att_gt`` and ``aggte``. Its
option matrix of estimators, comparison groups, base periods, weights,
clusters, and anticipation runs on three shards and compares the warnings in
order. Each estimator also runs on one and seven shards to vary the split.
``tests/did/test_row_shards.py`` does the same for the row engine on both of
its layouts. The end-to-end tests in ``tests/spark/test_att_gt.py`` repeat the
comparison on Spark itself. A new option needs a case in the option matrices of
all three files. The parity tests also allow for the few warnings that the
shard path doesn't copy, such as the ``SingularMatrixWarning`` that statsmodels
0.15 raises for a numerically rank-deficient design. A new warning from a
dependency either gets a copy on the shard path or joins those exceptions.

A few of the in-memory checks that both paths share expect arrays that a fit
on shards never holds. For those, the two fit functions build stand-in arrays
from the kernels' statistics, such as the treated and comparison weight totals
that ``_check_cell_weights`` reads.

.. admonition:: Keep the shard path in step
   :class: important

   A change to the in-memory estimators needs the same change on the shard
   path. A check that starts reading more of its input than its stand-in holds
   needs a new kernel statistic.

The tests through ``LocalExecutor`` never reach the persisted table, the keyed
sums that each task finalizes, the grouped stage, the version check, or the
PySpark 3.5 paths. A change to any of them needs a test in ``tests/spark/``. The
:ref:`testing guide <testing-spark>` shows how to run those tests on a local
cluster and through Spark Connect.
