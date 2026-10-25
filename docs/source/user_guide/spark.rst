.. _spark:

Distributed estimation on Spark
===============================

If your panel already sits in a Spark table or holds more units than one
machine can keep in memory, collecting it for an in-memory fit can be slow or
impossible. You can instead pass the Spark DataFrame to
:func:`~moderndid.att_gt` just as you'd pass a pandas or polars DataFrame.
Spark's executors then gather each unit's rows together and send the driver
only statistics such as sums and counts whose size doesn't depend on the number
of units. The driver, the Python process that runs your code, turns those into
the same :class:`~moderndid.MPResult` that an in-memory fit returns.

The question we'll keep coming back to is whether a Spark fit gives you the
answer you'd get in memory. The county data from the :ref:`staggered DiD
example <example_staggered_did>` is small enough to fit both ways and lets the
first fit on this page answer that question directly. Because your own table
may be too large for that, we'll then show how to check a sample of it before
the full fit. We'll also point out where to collect the rows for an in-memory
fit when you need more than balanced panels and the three estimators built into
``att_gt``.

When a Spark fit pays off
-------------------------

Even when your data already sits in Spark, a Spark fit isn't always the faster
choice. Every step of the estimator runs as Spark jobs that take a fixed time
to schedule whether the panel holds a thousand units or a hundred million. When
we timed the county fit below on a laptop running Spark in local mode, it took
about two seconds on Spark against a few hundredths of a second in memory. A
Spark fit pays off once collecting the table to the driver would take a long
time or run out of memory. That fixed cost is then small next to the cost of
moving the rows.

.. admonition:: Fit a small panel in memory
   :class: tip

   Passing ``data.toArrow()`` to ``att_gt``, or ``data.toPandas()`` on
   open-source PySpark 3.5, collects a small Spark DataFrame for an in-memory
   fit that accepts every argument. The :ref:`CPU parallelism guide <scaling>`
   shows how ``n_jobs`` then spreads that fit over the cores of your machine.

Getting ModernDiD onto the executors
------------------------------------

Since every executor imports ModernDiD to work through its share of the units,
the package needs to be installed wherever the executors run and not only on
the driver. The tabs below show the shortest route for each place you might run
Spark. The :ref:`Spark setup guide <spark-setup>` goes through each route in
detail and lists the setups we've tested.

.. tab-set::

   .. tab-item:: Databricks notebook

      Install ModernDiD in the notebook's first cell and restart Python in the
      next. On classic compute in dedicated access mode, that notebook-scoped
      install reaches the executors as well.

      .. code-block:: python

         %pip install moderndid==<version>

      .. code-block:: python

         dbutils.library.restartPython()

   .. tab-item:: Your machine

      The ``spark`` extra installs PySpark along with ModernDiD on your
      machine. The setup guide names the Java versions that Spark needs.

      .. code-block:: console

         python -m pip install "moderndid[spark]"

      .. code-block:: python

         from pyspark.sql import SparkSession

         # Start Spark on this machine with two worker threads.
         spark = SparkSession.builder.master("local[2]").getOrCreate()

   .. tab-item:: Cluster or Spark Connect

      A cluster you manage needs ModernDiD in the Python of every node or in a
      packed environment that you ship to the executors. If the cluster runs a
      Spark Connect server, your session connects to its address.

      .. code-block:: python

         from pyspark.sql import SparkSession

         # Connect to the cluster's Spark Connect server.
         spark = SparkSession.builder.remote("sc://host:15002").getOrCreate()

.. admonition:: Match versions on the driver and executors
   :class: important

   Each executor's Python needs the same ModernDiD version and Python minor
   version as the driver. If an executor has no ModernDiD or a different
   version, the fit stops with an error that quotes the import failure or
   names both versions. A different Python stops it even sooner with PySpark's
   own error.

Fitting the county data both ways
---------------------------------

To compare the two fits on equal terms, we'll fit the county data both ways
with one specification. Leaving out the staggered DiD example's bootstrap gives
both fits analytic standard errors that should agree up to rounding.

.. code-block:: python

   import moderndid as did
   from pyspark.sql import SparkSession

   # active returns the session that's already running, such as a notebook's.
   spark = SparkSession.active()

   # The county data as a polars DataFrame and as a Spark DataFrame.
   local = did.load_mpdta()
   data = spark.createDataFrame(local.to_pandas())

   spec = dict(
       # The columns for log teen employment, the year, the county, and its cohort.
       yname="lemp",
       tname="year",
       idname="countyreal",
       gname="first.treat",
       # Never-treated comparisons, doubly robust and adjusted for log population.
       control_group="nevertreated",
       xformla="~lpop",
       est_method="dr",
       # Measure effects from the year before adoption and seed the band's draws.
       base_period="universal",
       random_state=7,
   )

   # The same call fits the data in memory and on the executors.
   in_memory = did.att_gt(local, **spec)
   result = did.att_gt(data, **spec)

Joining the two results on cohort and year lines up each in-memory estimate and
standard error with its Spark counterpart in the column ending in ``_spark``.
For the 2004 cohort printed below, you should see each pair of columns match.

.. code-block:: python

   # Pair each estimate and standard error with its Spark counterpart.
   columns = ["group", "time", "att", "se"]
   pairs = did.to_df(in_memory).select(columns).join(
       did.to_df(result).select(columns), on=["group", "time"], suffix="_spark"
   )
   print(pairs.drop_nans().filter(group=2004))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (4, 6)
         ┌───────┬──────┬───────────┬──────────┬───────────┬──────────┐
         │ group ┆ time ┆ att       ┆ se       ┆ att_spark ┆ se_spark │
         │ ---   ┆ ---  ┆ ---       ┆ ---      ┆ ---       ┆ ---      │
         │ i64   ┆ i64  ┆ f64       ┆ f64      ┆ f64       ┆ f64      │
         ╞═══════╪══════╪═══════════╪══════════╪═══════════╪══════════╡
         │ 2004  ┆ 2004 ┆ -0.01453  ┆ 0.022129 ┆ -0.01453  ┆ 0.022129 │
         │ 2004  ┆ 2005 ┆ -0.076422 ┆ 0.028671 ┆ -0.076422 ┆ 0.028671 │
         │ 2004  ┆ 2006 ┆ -0.140448 ┆ 0.035378 ┆ -0.140448 ┆ 0.035378 │
         │ 2004  ┆ 2007 ┆ -0.106904 ┆ 0.032886 ┆ -0.106904 ┆ 0.032886 │
         └───────┴──────┴───────────┴──────────┴───────────┴──────────┘

Across all twelve effects, the estimates and standard errors of the two fits
differ by less than :math:`10^{-16}`. That's no more than the rounding you'd
get from adding the same numbers in a different order. A fit that runs at all
also tells you that ModernDiD reached the executors that ran its tasks. On a
cluster, though, a panel as small as the county data may keep all of that work
on a single executor. If the call fails instead, look up its message in the
:ref:`troubleshooting section <spark-troubleshooting>` of the setup guide.

.. admonition:: Tested on a Databricks cluster
   :class: note

   Our test cluster on Azure Databricks ran Runtime 16.4 LTS ML on four
   Standard_E16ads_v5 workers with 64 cores in all. On a generated panel of 100
   million units over ten periods, a billion rows, a doubly robust fit without a
   band took about 200 seconds. A joint test found its estimates consistent with
   the effects we planted in the panel (p = 0.107). Fits of 200,000 units under
   seven specifications also matched their in-memory counterparts to within
   about :math:`10^{-14}` in relative terms.

What a Spark fit does differently
---------------------------------

Most of what you'd ask of ``att_gt`` in memory works on Spark too. Sampling
weights, a cluster variable besides ``idname``, both comparison groups, both
base periods, and anticipation all work with any of the three built-in
estimators. A few arguments aren't supported for a Spark DataFrame, though.
Rather than let a long fit fail partway through, ``att_gt`` raises a ValueError
for each of them before Spark runs a job and tells you what to pass instead.

- ``panel=False`` (repeated cross-sections)
- ``allow_unbalanced_panel=True`` (units missing from some periods)
- A callable ``est_method`` (anything besides ``"dr"``, ``"ipw"``, and ``"reg"``)
- Any ``backend`` (the executors always compute with NumPy)
- ``n_jobs`` other than 1 (Spark already runs the work in parallel)

If your data really is a repeated cross-section, an in-memory fit with
``panel=False`` needs its rows on the driver. So does an in-memory fit that
uses a callable ``est_method`` or passes ``allow_unbalanced_panel=True`` to
keep the units missing from some periods. Any other function that takes data,
such as :func:`~moderndid.ddd`, :func:`~moderndid.cont_did`, or
:func:`~moderndid.core.panel.diagnose_panel`, needs the rows on the driver too.
You can collect the rows with the same ``toArrow()`` or ``toPandas()`` call as
a small panel.

The result looks just like an in-memory one unless you reach for its unit-level
values. Since an in-memory result's ``influence_func``, ``G``, and, in a
weighted fit, ``weights_ind`` each hold a value for every unit, all three are
None on a Spark result. In their place, ``influence_moments`` holds an
:class:`~moderndid.InfluenceMoments` with the sums over units that ``aggte``
reads. Because the result holds only NumPy arrays and plain Python values that
don't grow with the number of units, you can pickle it and aggregate it later
in a Python process without PySpark.

.. _spark-bootstrap:

Why the bootstrap standard errors differ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see the one place where a Spark fit's numbers differ from those of an
in-memory fit, we'll refit both ways with the 10,000 seeded draws that the
staggered DiD example uses.

.. code-block:: python

   # Refit both ways with 10,000 seeded bootstrap draws.
   boot = spec | {"boot": True, "biters": 10000}
   in_memory_boot = did.att_gt(local, **boot)
   result_boot = did.att_gt(data, **boot)

   # Pair the estimates and standard errors as before.
   boot_pairs = did.to_df(in_memory_boot).select(columns).join(
       did.to_df(result_boot).select(columns), on=["group", "time"], suffix="_spark"
   )
   print(boot_pairs.drop_nans().filter(group=2004))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (4, 6)
         ┌───────┬──────┬───────────┬──────────┬───────────┬──────────┐
         │ group ┆ time ┆ att       ┆ se       ┆ att_spark ┆ se_spark │
         │ ---   ┆ ---  ┆ ---       ┆ ---      ┆ ---       ┆ ---      │
         │ i64   ┆ i64  ┆ f64       ┆ f64      ┆ f64       ┆ f64      │
         ╞═══════╪══════╪═══════════╪══════════╪═══════════╪══════════╡
         │ 2004  ┆ 2004 ┆ -0.01453  ┆ 0.022457 ┆ -0.01453  ┆ 0.023326 │
         │ 2004  ┆ 2005 ┆ -0.076422 ┆ 0.029357 ┆ -0.076422 ┆ 0.030115 │
         │ 2004  ┆ 2006 ┆ -0.140448 ┆ 0.037045 ┆ -0.140448 ┆ 0.037629 │
         │ 2004  ┆ 2007 ┆ -0.106904 ┆ 0.034483 ┆ -0.106904 ┆ 0.033868 │
         └───────┴──────┴───────────┴──────────┴───────────┴──────────┘

Although the estimates still match to every printed digit, the standard errors
now differ slightly from one fit to the other. For the 2004 cohort in 2004, the
standard error is 0.0225 in memory and 0.0233 on Spark. Across all twelve
effects, the two fits' standard errors differ by at most about 0.001.

This happens because the two fits use the seed differently. In each draw, the
multiplier bootstrap gives each unit, or each cluster in a clustered fit, a
random sign. An in-memory fit takes those signs from one random stream in the
order of its units. Since a Spark table keeps no fixed order of its rows, a
Spark fit computes each sign from the seed, the draw, and the identifier of the
unit or cluster instead. The two fits therefore end up with different but
equally valid draws from the same bootstrap distribution. The gap between their
standard errors is the noise of a finite number of draws and shrinks as you
raise ``biters``.

Because the signs follow the identifiers rather than where each unit sits, a
Spark fit gives the same standard errors, up to rounding, however Spark splits
the data. The same holds whether the fit runs on classic Spark or through Spark
Connect. The simultaneous bands of the default ``cband=True`` come from the
same draws and don't depend on the split either. The :ref:`architecture guide
<architecture-shard-draws>` shows how a Spark fit computes the signs.

.. admonition:: The seed of an unseeded fit
   :class: note

   Without a ``random_state``, a fit picks its own seed and stores it in
   ``influence_moments.random_state`` so that a later Spark fit can reproduce
   the draws.

.. _spark-prepare:

Preparing your table
--------------------

Your own panel will most likely come from a table that ``spark.read.table``
loads or from a query. If you work with a pandas-on-Spark DataFrame instead,
convert it with its ``to_spark()`` method before you fit it. From PySpark 4.2
on, ``att_gt`` would otherwise quietly collect it to the driver and fit it in
memory.

.. admonition:: Save an expensive query first
   :class: tip

   Since a fit reads the data twice, it also runs the query behind the data
   twice. When that query is expensive, such as a join of large tables, save
   its result to a table before the fit or persist it where the session allows.

Like an in-memory fit, a Spark fit reads long data with one row for each unit
in each period. Since the fit checks the column types in the table's schema
before it reads a single row, we'll print that schema first.

.. code-block:: python

   # The county data stands in for a table that spark.read.table would load.
   data.printSchema()

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         root
          |-- year: long (nullable = true)
          |-- countyreal: long (nullable = true)
          |-- lpop: double (nullable = true)
          |-- lemp: double (nullable = true)
          |-- first.treat: long (nullable = true)
          |-- treat: long (nullable = true)

You can see that each column of the county data holds a long or a double. A
Spark fit needs numbers in the sampling weights and covariates as well as in
the outcome, period, unit, and cohort columns that any fit requires. If any of
them holds something else, the fit stops before Spark runs a job.

The decimal columns that warehouse tables often hold pass that check too. A
Spark fit reads a decimal unit identifier or cluster as a 64-bit integer. Its
type must therefore have no fraction digits and at most 18 digits, such as
decimal(18,0).
Even when it holds only small whole numbers, a column of a wider type such as
decimal(38,0) fails that check until you cast it to bigint.

Since both a Spark fit and an in-memory fit read decimal outcomes, covariates,
and weights as doubles, those columns need no cast. Rows that you collect with
``toPandas()`` or ``toArrow()`` for the sample check below therefore give the
same estimates in memory as on Spark.

As in an in-memory fit, a transformed covariate or a derived cluster goes into
the data as a column of its own. You can make such a column with Spark's column
functions without collecting any rows. With the state code below, passing
``clustervars=["state"]`` would cluster the standard errors by state.

.. code-block:: python

   from pyspark.sql import functions as F

   # The leading digits of a county FIPS code are its state's code.
   data = data.withColumn("state", F.floor(F.col("countyreal") / 1000))

If your table stores the unit identifier as text, the type check stops the fit
and asks you to convert it. Spark's ``xxhash64`` function turns such an
identifier into a 64-bit integer.

.. code-block:: python

   # Casting the county number to text mimics an identifier stored as text.
   text = data.withColumn("county", F.col("countyreal").cast("string"))
   coded = text.withColumn("county_code", F.xxhash64("county"))

Passing the coded column as ``idname="county_code"`` gives the same estimates
as ``countyreal``. Because the bootstrap signs follow the identifiers, its
bootstrap standard errors move slightly. If two identifiers ever shared a code,
they would repeat (unit, period) pairs and stop the fit with an error rather
than quietly merge two units into one.

Because a Spark fit takes only balanced panels, it drops every unit that misses
a period and warns you how many it dropped. Counting those units before the fit
tells you how much of your table it will use.

.. code-block:: python

   # Count the counties observed in fewer years than the panel has.
   n_years = data.select("year").distinct().count()
   incomplete = data.groupBy("countyreal").count().filter(F.col("count") < F.lit(n_years))
   print(incomplete.count())

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         0

With no county missing a year, the fit keeps all 500 of them. On your own
table, the fit would drop every unit that this query counts and any unit with a
missing value in a column it reads.

.. _spark-sample-check:

Checking a sample before the full fit
-------------------------------------

Your own table comes with no in-memory fit to compare against. You can make one
by fitting a sample of its units both on Spark and in memory. Sampling single
rows would leave units incomplete and drop them from both fits. We'll instead
filter on a hash of the identifier so that each unit keeps or loses all of its
rows together.

.. code-block:: python

   import numpy as np

   # Keep about one county in four along with all of its rows.
   sample = data.filter(F.xxhash64("countyreal") % 4 == 0)

   # Fit the sample on Spark and again in memory after collecting it.
   on_spark = did.att_gt(sample, **spec)
   in_memory = did.att_gt(sample.toPandas(), **spec)

   # Stop with an error if the two fits disagree.
   np.testing.assert_allclose(on_spark.att_gt, in_memory.att_gt)
   np.testing.assert_allclose(on_spark.se_gt, in_memory.se_gt)

The filter keeps 120 of the 500 counties in the county data and all five years
of each. Both fits of that sample agree on every estimate and standard error to
about :math:`10^{-16}`. A panel of millions of units needs a larger divisor to
keep the sample small enough to collect. Once the sample agrees, the same call
on the whole table gives you the full fit.

.. admonition:: Leave the bootstrap out of the check
   :class: important

   Because the two fits draw different bootstrap signs, their bootstrap
   standard errors differ by design and would fail the second assertion. Run the check
   with a specification that leaves out ``boot=True``, such as ``spec``.

Continuing the analysis
-----------------------

Since :func:`~moderndid.aggte`, :func:`~moderndid.to_df`, and the plots take a
Spark fit's result the same way they take an in-memory one, the analysis goes
on just as it would in memory. Below, we average the effects into an event
study and bound its adoption-year effect with :func:`~moderndid.honest_did`
from the ``didhonest`` extra. Because restarting Python clears the fitted
result, add that extra to the first cell's ``%pip install`` on Databricks.

.. code-block:: python

   # Average the effects by event time and bound the adoption-year effect.
   event_study = did.aggte(result, type="dynamic")
   sensitivity = did.honest_did(event_study, event_time=0, sensitivity_type="smoothness")

The :ref:`sensitivity analysis example <example_honest_did>` shows how to read
the intervals that ``honest_did`` returns. The only difference you might run
into is that ``aggte`` can't make new bootstrap draws without unit-level values
and reuses the draws that the fit stored instead.

.. admonition:: Ask for enough bootstrap draws up front
   :class: tip

   Set ``biters`` in ``att_gt`` to the most draws any later aggregation will
   use. Asking ``aggte`` for more draws than the fit stored, for another seed,
   or for unit-level draws from a clustered fit raises a ValueError that says
   what to refit.

If you need a contrast that ``aggte`` doesn't report, such as the difference
between two effects, the tip in the :ref:`influence functions background
<background-influence-functions-contrasts>` gives its standard error for a
Spark fit.

Tuning a large fit
------------------

How long a fit takes depends on the estimator, the clustering, the bootstrap
draws, and a few Spark settings you can change. Outcome regression is the
quickest of the three estimators because it estimates no propensity scores.

Of Spark's own settings, the one you're most likely to change is
``spark.sql.shuffle.partitions``. It sets how many partitions Spark makes when
it gathers each unit's rows together and defaults to 200 (``auto`` on
serverless compute). On one machine, 200 partitions can take longer than
everything else a small fit does. Setting the value to a few per core, as
below, shortens that step.

.. code-block:: python

   # A few shuffle partitions per core suit a small panel on one machine.
   spark.conf.set("spark.sql.shuffle.partitions", "8")

On a laptop, that setting brings the county fit on Spark from about two seconds
down to under one. A cluster calls for the same few partitions per core, such
as 192 for 64 cores.

Even with the default ``boot=False``, the default ``cband=True`` makes a fit
draw ``biters`` random signs for each unit or cluster and keep their sums for
the bands of later aggregations. Their cost grows in step with the number of
draws you ask for. When no aggregation will need them, passing
both ``boot=False`` and ``cband=False`` skips the draws and leaves every
aggregation with analytic standard errors.

.. admonition:: Use fewer draws for very large designs
   :class: tip

   With about a thousand group-time effects, such as 25 cohorts over 40
   periods, 10,000 draws can push the sums bound for the driver past the 1 GiB
   that ``spark.driver.maxResultSize`` allows by default in open-source Spark.
   Before Spark 4.1, Spark Connect also needs each task's sums to fit in one
   message of at most 128 MiB by default. Fewer draws bring the sums back under
   whatever limits your setup uses.

When a clustered fit has few clusters, it spreads the units of each cluster
over every core. A fit clustered by 50 states therefore keeps all of your cores
busy, however unequal the states are in size. In exchange, each core sends the
driver a row of sums for every cluster it touched. The fit keeps each cluster
on one core instead once the number of clusters, multiplied by the number of
treated cohorts and by one more than the number of periods, passes about
130,000. With five treated cohorts over ten periods, that happens at about
2,300 clusters. A cluster that holds most of the units then makes every step of
the estimator wait on the core working through it.

For a look at how a Spark fit computes its estimates and draws without
collecting the units, the :ref:`architecture guide <architecture-shards>`
follows one from the inside.
