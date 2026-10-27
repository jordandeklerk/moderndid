.. _spark:

Distributed estimation on Spark
===============================

If your panel already sits in a Spark table or holds more units than one
machine can keep in memory, collecting it for an in-memory fit can be slow or
impossible. You can instead pass the Spark DataFrame to
:func:`~moderndid.att_gt` just as you'd pass a pandas or polars DataFrame.
Spark's executors then work through the rows and send the driver only
statistics such as sums and counts whose size doesn't depend on the number of
units. The driver, the Python process that runs your code, turns those into the
same :class:`~moderndid.MPResult` that an in-memory fit returns.

The question we'll keep coming back to is whether a Spark fit gives you the
answer you'd get in memory. The county data from the :ref:`staggered DiD
example <example_staggered_did>` is small enough to fit both ways and lets the
first fit on this page answer that question directly. Fits of repeated
cross-sections and of an unbalanced panel made from the county data answer the
same question for data that doesn't follow every unit through every period.
Because your own table may be too large to fit both ways, we'll then show how
to check a sample of it before the full fit. We'll also point out where to
collect the rows for an in-memory fit when you need an estimator other than the
three built into ``att_gt``.

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
estimators. You can also fit :ref:`repeated cross-sections and unbalanced
panels <spark-layouts>` on Spark by passing ``panel=False`` or
``allow_unbalanced_panel=True``. A few arguments aren't supported for a Spark
DataFrame, though. Rather than let a long fit fail partway through, ``att_gt``
raises a ValueError for each of them before Spark runs a job and tells you what
to pass instead.

- A callable ``est_method`` (anything besides ``"dr"``, ``"ipw"``, and ``"reg"``)
- Any ``backend`` (the executors always compute with NumPy)
- ``n_jobs`` other than 1 (Spark already runs the work in parallel)

A fit with a callable ``est_method`` therefore needs the rows on the driver.
So does any other function that takes data, such as :func:`~moderndid.ddd`,
:func:`~moderndid.cont_did`, or :func:`~moderndid.core.panel.diagnose_panel`.
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

Rerunning a Spark fit with the same ``random_state``, or splitting its table
differently, gives the same results up to rounding, because only the order in
which the sums add up changes. The bootstrap standard errors still differ
slightly from those of an in-memory fit, as the next section explains. A choice
within rounding of its cutoff can also go the other way between two fits, such
as whether a covariate that almost duplicates another drops out or whether a
cell fails because a propensity score reaches 0.999.

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

A Spark fit of repeated cross-sections without clusters computes each row's
sign from the row's own values rather than from an identifier. Two identical
rows, such as matching answers to a survey, still draw separate signs.
Because each sign follows its unit, row, or cluster rather than where it sits,
a Spark fit gives the same standard errors, up to rounding, however Spark
splits the data. The same holds whether the fit runs on classic Spark or
through Spark Connect. The simultaneous bands of the default ``cband=True``
come from the same draws and don't depend on the split either. The developer
guide to :ref:`random draws on shards <dev-shard-draws>` shows how a Spark fit
computes the signs.

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
bootstrap standard errors move slightly. If two identifiers of a balanced panel
ever shared a code, they would repeat (unit, period) pairs and stop the fit
with an error rather than quietly merge two units into one. Two units of an
unbalanced panel that share a code, though, merge into one without an error if
they have no period in common, belong to the same cohort, and, in a clustered
fit, sit in the same cluster.

Unless you pass ``allow_unbalanced_panel=True``, a Spark fit balances the panel
just as an in-memory fit does. It drops every unit that misses a period and
warns you how many it dropped. Counting those units before the fit tells you
how much of your table it will use.

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
missing value in a column it reads. You can keep those units instead by fitting
:ref:`an unbalanced panel <spark-layouts>` with
``allow_unbalanced_panel=True``.

.. _spark-layouts:

Repeated cross-sections and unbalanced panels
---------------------------------------------

Many tables that call for Spark don't track the same units through every
period. When you tell ``att_gt`` how such a table was collected, it estimates
each effect by applying the repeated cross-section form of the estimator to the
rows of that effect's two periods. We'll make one table in each layout from the
county data and fit it both ways with ``spec``.

Repeated cross-sections
^^^^^^^^^^^^^^^^^^^^^^^

Surveys and logs of transactions often observe different people in each
period rather than following the same ones. If your table holds such repeated
cross-sections, passing ``panel=False`` makes every row an observation of its
own. The standard errors then treat the rows as independent of one another
unless you group them with ``clustervars``, such as by the state that sets the
policy you study. An ``idname`` is optional here and never groups the rows into
units. Passing one still makes the fit drop every row of an identifier that has
a row already treated in the first period.

Because every county appears in all five years, passing ``countyreal`` as
``idname`` would make the fit warn that the rows look like panel data. We'll
leave the identifier out and let the 2,500 rows stand in for repeated
cross-sections, as if each year had sampled new counties. You can again check
that each pair of columns matches and compare the standard errors with those of
the panel fit from before.

.. code-block:: python

   # Fit the 2,500 rows as repeated cross-sections without a unit identifier.
   repeated = spec | {"panel": False, "idname": None}
   in_memory_repeated = did.att_gt(local, **repeated)
   result_repeated = did.att_gt(data, **repeated)

   # Pair the estimates and standard errors as before.
   repeated_pairs = did.to_df(in_memory_repeated).select(columns).join(
       did.to_df(result_repeated).select(columns), on=["group", "time"], suffix="_spark"
   )
   print(repeated_pairs.drop_nans().filter(group=2004))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (4, 6)
         ┌───────┬──────┬───────────┬──────────┬───────────┬──────────┐
         │ group ┆ time ┆ att       ┆ se       ┆ att_spark ┆ se_spark │
         │ ---   ┆ ---  ┆ ---       ┆ ---      ┆ ---       ┆ ---      │
         │ i64   ┆ i64  ┆ f64       ┆ f64      ┆ f64       ┆ f64      │
         ╞═══════╪══════╪═══════════╪══════════╪═══════════╪══════════╡
         │ 2004  ┆ 2004 ┆ -0.01453  ┆ 0.08974  ┆ -0.01453  ┆ 0.08974  │
         │ 2004  ┆ 2005 ┆ -0.076422 ┆ 0.092438 ┆ -0.076422 ┆ 0.092438 │
         │ 2004  ┆ 2006 ┆ -0.140448 ┆ 0.095452 ┆ -0.140448 ┆ 0.095452 │
         │ 2004  ┆ 2007 ┆ -0.106904 ┆ 0.089051 ┆ -0.106904 ┆ 0.089051 │
         └───────┴──────┴───────────┴──────────┴───────────┴──────────┘

Each estimate and standard error matches its Spark counterpart once more. The
estimates also equal those of the panel fit on this balanced data. You can see
what treating the rows as unrelated costs in the standard errors, though. Since
nothing pairs a county's years anymore, they come out about three to five times
as large as the panel fit's.

Unbalanced panels
^^^^^^^^^^^^^^^^^

Panels of customers or administrative records often lose units over time as
accounts close or records lapse. You can keep those units in the fit by
passing ``allow_unbalanced_panel=True`` instead of letting the default
balancing drop them. The standard errors still treat each unit, not each row,
as independent of the others. A cluster variable therefore has to stay fixed
within each unit, just as in a balanced panel. On a panel without gaps, you get
the balanced fit whether or not you pass the option. The :doc:`data guide
<data>` explains what keeping the incomplete units assumes.

A filter on a hash of the county and year drops about one row in five and
leaves 1,999 of the 2,500 rows. A balanced fit of those rows would keep only
the 153 counties observed in all five years and drop the other 347. Both fits
below pass ``allow_unbalanced_panel=True`` to keep all 500 counties. Their
columns should pair up as closely as in the earlier fits.

.. code-block:: python

   # Drop about one county-year in five to leave gaps in the panel.
   thinned = data.filter(F.xxhash64("countyreal", "year") % 5 != 0)

   # Keep the counties that miss a year and fit the panel both ways.
   unbalanced = spec | {"allow_unbalanced_panel": True}
   in_memory_unbalanced = did.att_gt(thinned.toPandas(), **unbalanced)
   result_unbalanced = did.att_gt(thinned, **unbalanced)

   # Pair the estimates and standard errors as before.
   unbalanced_pairs = did.to_df(in_memory_unbalanced).select(columns).join(
       did.to_df(result_unbalanced).select(columns), on=["group", "time"], suffix="_spark"
   )
   print(unbalanced_pairs.drop_nans().filter(group=2004))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (4, 6)
         ┌───────┬──────┬───────────┬──────────┬───────────┬──────────┐
         │ group ┆ time ┆ att       ┆ se       ┆ att_spark ┆ se_spark │
         │ ---   ┆ ---  ┆ ---       ┆ ---      ┆ ---       ┆ ---      │
         │ i64   ┆ i64  ┆ f64       ┆ f64      ┆ f64       ┆ f64      │
         ╞═══════╪══════╪═══════════╪══════════╪═══════════╪══════════╡
         │ 2004  ┆ 2004 ┆ 0.025362  ┆ 0.035346 ┆ 0.025362  ┆ 0.035346 │
         │ 2004  ┆ 2005 ┆ -0.056018 ┆ 0.076207 ┆ -0.056018 ┆ 0.076207 │
         │ 2004  ┆ 2006 ┆ -0.090955 ┆ 0.0475   ┆ -0.090955 ┆ 0.0475   │
         │ 2004  ┆ 2007 ┆ -0.097656 ┆ 0.040683 ┆ -0.097656 ┆ 0.040683 │
         └───────┴──────┴───────────┴──────────┴───────────┴──────────┘

You can see each in-memory column match its Spark counterpart here as well.
Across all twelve effects of either layout, the estimates and standard errors
on Spark differ from those in memory by less than :math:`10^{-15}`.

What the two layouts cost
^^^^^^^^^^^^^^^^^^^^^^^^^

You should expect both layouts to take longer than a balanced panel of the same
size. Since each of their effects reads only the rows of its own two periods,
it fits models of its own. A balanced fit instead estimates its models once for
all the effects of a cohort that share an earlier period and comparison units.
Repeated cross-sections take the longest because every one of their rows counts
as a unit.

A fit of repeated cross-sections never takes the shuffle that gathers each
unit's rows together, though. Without clusters, it shuffles the rows once for
the bootstrap signs that the default ``cband=True`` draws. Passing
``cband=False`` with the default ``boot=False`` leaves out that shuffle along
with the draws.

.. admonition:: Timed on local Spark
   :class: note

   On an Apple M2 Pro laptop with a local Spark 4.2 cluster of two single-core
   executors and eight shuffle partitions, we ran two sessions of five timed
   fits of the balanced panel and of each layout after a warm-up and took the
   median of all ten. Doubly robust fits of the county data with ``spec`` and
   its default band took about 0.8 seconds each. With the same settings and
   four covariates, fits of a panel that :func:`~moderndid.gen_did_scalable`
   generated with 20,000 units, ten periods, and eight treated cohorts took
   about 2.5 seconds for the balanced panel, 3.1 seconds for the unbalanced
   panel left by dropping one row in five, and 5.4 seconds for the same 200,000
   rows as repeated cross-sections.

.. _spark-sample-check:

Checking a sample before the full fit
-------------------------------------

Your own table comes with no in-memory fit to compare against. You can make one
by fitting a sample of its units both on Spark and in memory. Sampling single
rows of a panel would leave almost every unit incomplete for a balanced fit to
drop. We'll instead filter on a hash of the identifier so that each unit keeps
or loses all of its rows together. Hashing each row's values instead of an
identifier gives repeated cross-sections a sample for the same check.

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
it shuffles the rows, as a panel fit does to gather each unit's rows together,
and defaults to 200 (``auto`` on serverless compute). On one machine, 200
partitions can take longer than everything else a small fit does. Setting the
value to a few per core, as below, shortens that step.

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
collecting the units, the developer guide to :ref:`fits on shards
<dev-shards>` follows one from the inside.
