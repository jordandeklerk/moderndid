.. _spark:

Distributed estimation on Spark
===============================

If your panel already sits in a Spark table or holds more units than one
machine can keep in memory, collecting it into one Python process for an
in-memory fit can be slow or impossible. You can instead pass the Spark
DataFrame to :func:`~moderndid.att_gt` just as you'd pass a pandas or polars
DataFrame. Spark's executors, the processes it runs on the cluster's worker
machines, read the rows and reduce them to sums and counts whose size doesn't
depend on the number of units. The driver, the Python process that runs your
code, turns those sums into the same :class:`~moderndid.MPResult` that an
in-memory fit returns.

The question we'll keep asking is whether a Spark fit gives the answer you'd
get in memory. The county data from the :ref:`staggered DiD example
<example_staggered_did>` is small enough to fit both ways. You'll see the two
fits agree, find the one place where they differ, and end with the same check
on a sample of your own table.

When a Spark fit pays off
-------------------------

A Spark fit isn't always the faster choice. Every step of the estimator runs as
a Spark job, a round of work handed to the executors that takes a fixed time to
schedule however small the panel is. In local mode, where the driver and the
executors share one machine, the county fit below took about two seconds
against a few hundredths of a second in memory. A Spark fit pays off once
collecting the table would be slow or run out of memory.

.. admonition:: Fit a small panel in memory
   :class: tip

   Passing ``data.toArrow()`` to ``att_gt``, or ``data.toPandas()`` on
   open-source PySpark 3.5, collects a small Spark DataFrame for an in-memory
   fit that accepts every argument. The :ref:`CPU parallelism guide <scaling>`
   shows how ``n_jobs`` then spreads that fit over your machine's cores.

Fitting the county data both ways
---------------------------------

Because every executor imports ModernDiD for its share of the units, the
package has to be installed wherever the executors run before the first fit.
The :ref:`setup guide <spark-setup>` shows how on Databricks, on a cluster you
manage, and on your own machine.

.. admonition:: Match versions on the driver and executors
   :class: important

   Each executor's Python needs the same ModernDiD version and Python minor
   version as the driver. Otherwise the fit stops with an error that names the
   mismatch or quotes the failed import.

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

Across all twelve effects, the two fits differ by less than :math:`10^{-16}`,
no more than the rounding you'd get from adding the same numbers in another
order. A fit that runs at all also shows that ModernDiD reached every executor
that worked on it. That may not be every executor you have, because Spark hands
a table out in partitions, chunks of rows that each run as a separate task. A
panel as small as the county data may fill one partition and reach one
executor. If the call fails instead, look up its message in the
:ref:`troubleshooting section <spark-troubleshooting>` of the setup guide.

.. admonition:: Tested on a Databricks cluster
   :class: note

   On Azure Databricks, four workers with 64 cores in all fit a generated panel
   of 100 million units over ten periods, a billion rows, in about 200 seconds
   with a doubly robust fit and no band. A joint test found those estimates
   consistent with the effects we planted (p = 0.107). Fits of 200,000 units
   under seven specifications matched their in-memory counterparts to about
   :math:`10^{-14}` in relative terms.

What a Spark fit does differently
---------------------------------

Most of what you'd ask of ``att_gt`` in memory works on Spark too, from
sampling weights and clustering to both comparison groups, both base periods,
anticipation, and all three built-in estimators. Other functions that take
data, such as :func:`~moderndid.ddd`, still need the rows on the driver, where
the same ``toArrow()`` call puts them.

.. admonition:: Three arguments don't apply on Spark
   :class: note

   A callable ``est_method`` needs the rows on the driver. The ``backend`` and
   ``n_jobs`` arguments have no role on Spark, because the executors compute
   with NumPy and already run the work in parallel. Passing any of the three
   raises a ValueError before Spark runs a job.

The result looks just like an in-memory one until you reach for its unit-level
values. In memory, ``aggte`` reads each unit's :ref:`influence function
<background-influence-functions>`, cohort, and sampling weight from
``influence_func``, ``G``, and ``weights_ind``. A Spark result leaves all three
as None and keeps the sums over units that ``aggte`` needs in an
:class:`~moderndid.InfluenceMoments` instead. Since nothing in it grows with
the number of units, you can pickle it and aggregate it later without PySpark.

Rerunning a Spark fit or repartitioning its table changes only the order in
which its sums add up. That can tip a choice within rounding of its cutoff,
such as whether a nearly duplicate covariate drops out. The one difference
you'll see in practice lies in the bootstrap standard errors.

.. _spark-bootstrap:

Why the bootstrap standard errors differ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see that difference, we'll refit both ways with the 10,000 seeded draws that
the staggered DiD example uses.

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

The estimates still match to every printed digit. The standard errors now
differ by up to about 0.001 across the twelve effects, such as 0.0225 in memory
against 0.0233 on Spark for the 2004 cohort in 2004.

The difference comes from how the two fits use the seed. In every draw, the
:ref:`multiplier bootstrap <background-did-bootstrap>` gives each unit, or each
cluster in a clustered fit, a random sign of plus or minus one and multiplies
its influence function by that sign. The spread of the results over many draws
gives the standard errors. An in-memory fit takes the signs from one random
stream in the order of its units. A Spark table has no such order, since its
rows can fall into different partitions on every run. A Spark fit instead
computes each sign from the seed, the draw, and the identifier of the unit or
cluster, much as a hash turns the same input into the same output every time.
Both are valid draws from the same bootstrap distribution. The gap between
their standard errors shrinks as you raise ``biters``.

.. _spark-prepare:

Preparing your table
--------------------

Your own panel will most likely come from a table that ``spark.read.table``
loads or from a query. Convert a pandas-on-Spark DataFrame with its
``to_spark()`` method first, because from PySpark 4.2 on ``att_gt`` would
quietly collect it to the driver and fit it in memory.

.. admonition:: Save an expensive query first
   :class: tip

   A Spark DataFrame holds the query that makes its rows rather than the rows
   themselves. A fit therefore runs that query each time it reads the data.
   Save an expensive query, such as a join of large tables, to a table before
   the fit.

Like an in-memory fit, a Spark fit reads long data with one row for each unit
in each period. We'll print the table's schema first, because the fit checks
its column types before it reads a single row.

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

Every column of the county data holds a long or a double, Spark's types for
64-bit integers and floating-point numbers. The outcome, period, unit, and
cohort columns need numbers, as do any weights and covariates. Decimal columns
pass too, within limits that the :ref:`troubleshooting section
<spark-troubleshooting>` lists under their error messages.

If your table stores the unit identifier as text, the type check stops the fit
and asks you to convert it. Spark's ``xxhash64`` function hashes the text into
a 64-bit integer, the same number for the same text every time.

.. code-block:: python

   from pyspark.sql import functions as F

   # Casting the county number to text mimics an identifier stored as text.
   text = data.withColumn("county", F.col("countyreal").cast("string"))
   coded = text.withColumn("county_code", F.xxhash64("county"))

Passing the coded column as ``idname="county_code"`` gives the same estimates
as ``countyreal``. Only the bootstrap standard errors move slightly, because
the signs follow the identifiers.

.. admonition:: A shared code can merge two units
   :class: warning

   Two different texts can share a code by rare chance. Two units of an
   unbalanced panel that share one merge without an error when they have no
   period in common, belong to the same cohort, and, in a clustered fit, sit in
   the same cluster.

Unless you pass ``allow_unbalanced_panel=True``, a Spark fit balances the panel
just as an in-memory fit does. Counting the units that miss a period before the
fit tells you how much of your table it will keep.

.. code-block:: python

   # Count the counties observed in fewer years than the panel has.
   n_years = data.select("year").distinct().count()
   incomplete = data.groupBy("countyreal").count().filter(F.col("count") < F.lit(n_years))
   print(incomplete.count())

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         0

With no county missing a year, the fit keeps all 500. On your own table, it
would drop every unit that this query counts and any unit with a missing value
in a column it reads, unless you fit :ref:`an unbalanced panel
<spark-layouts>`.

.. _spark-layouts:

Repeated cross-sections and unbalanced panels
---------------------------------------------

Many tables that call for Spark don't follow the same units through every
period. When you tell ``att_gt`` how such a table was gathered, it applies the
repeated cross-section form of the estimator to the rows of each effect's two
periods. We'll make one table in each layout from the county data and fit it
both ways with ``spec``.

Repeated cross-sections
^^^^^^^^^^^^^^^^^^^^^^^

Surveys and logs of transactions often observe different people in each period
rather than following the same ones. Passing ``panel=False`` makes every row an
observation of its own. The standard errors then treat the rows as independent
unless you group them with ``clustervars``, such as by the state that sets the
policy you study. The county data's 2,500 rows can stand in for repeated
cross-sections, as if each year had sampled new counties. The fit below leaves
out the identifier because repeated cross-sections don't need one.

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

Each estimate and standard error matches its Spark counterpart once more. With
nothing to pair a county's years, the standard errors come out about three to
five times as large as the panel fit's.

Unbalanced panels
^^^^^^^^^^^^^^^^^

Panels of customers or administrative records often lose units over time as
accounts close or records lapse. Passing ``allow_unbalanced_panel=True`` keeps
those units rather than letting the default balancing drop them. A cluster
variable still has to stay fixed within each unit, since the standard errors
treat each unit, not each row, as independent. The :doc:`data guide <data>`
explains what keeping the incomplete units assumes. The filter below drops
about one county-year in five from the county data. A balanced fit of the 1,999
rows that remain
would keep only 153 of the 500 counties.

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

The columns match here as well. Across all twelve effects of either layout, the
two fits differ by less than :math:`10^{-15}`. Both layouts take longer than a
balanced panel of the same size, because each of their effects fits its own
models where a balanced fit shares models among effects that use the same
units.

.. admonition:: Timed on local Spark
   :class: note

   On an Apple M2 Pro laptop running local Spark 4.2, doubly robust fits of a
   generated panel with 20,000 units over ten periods took about 2.5 seconds
   balanced, 3.1 seconds unbalanced, and 5.4 seconds as repeated
   cross-sections, each the median of ten fits.

.. _spark-sample-check:

Checking a sample before the full fit
-------------------------------------

Your own table comes with no in-memory fit to compare against. You can make one
by fitting a sample of its units both on Spark and in memory. Filtering on a
hash of the identifier keeps or drops each unit with all of its rows, where
sampling single rows would leave almost every unit of a panel incomplete. For
repeated cross-sections, a hash of each row's values gives the sample instead.

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

The filter keeps 120 of the 500 counties with all five years of each. Both fits
of that sample agree on every estimate and standard error to about
:math:`10^{-16}`. A panel of millions of units needs a larger divisor to keep
the sample small enough to collect. Once the sample agrees, the same call on
the whole table gives you the full fit.

.. admonition:: Leave the bootstrap out of the check
   :class: important

   Because the two fits draw different bootstrap signs, their bootstrap
   standard errors differ by design and would fail the second assertion. Run
   the check with a specification that leaves out ``boot=True``, such as
   ``spec``.

Continuing the analysis
-----------------------

Everything that follows a fit works on a Spark fit's result as it does on an
in-memory one, from :func:`~moderndid.aggte` and :func:`~moderndid.to_df` to
the plots. Below, we average the effects into an event study and bound its
adoption-year effect with :func:`~moderndid.honest_did`. On Databricks, add its
``didhonest`` extra to the first cell's ``%pip install``, because restarting
Python later would clear the fitted result.

.. code-block:: python

   # Average the effects by event time and bound the adoption-year effect.
   event_study = did.aggte(result, type="dynamic")
   sensitivity = did.honest_did(event_study, event_time=0, sensitivity_type="smoothness")

The :ref:`sensitivity analysis example <example_honest_did>` shows how to read
the intervals that ``honest_did`` returns. The one difference you might run
into is that ``aggte`` reuses the bootstrap draws that the fit stored, since it
can't make new ones without unit-level values.

.. admonition:: Ask for enough bootstrap draws up front
   :class: tip

   Set ``biters`` in ``att_gt`` to the most draws any later aggregation will
   use. Asking ``aggte`` for more draws than the fit stored, for another seed,
   or for unit-level draws from a clustered fit raises a ValueError that says
   what to refit.

Tuning a large fit
------------------

How long a fit takes depends on the estimator, the bootstrap draws, and a few
Spark settings. Outcome regression is the quickest of the three estimators
because it estimates no propensity scores.

Of Spark's own settings, the one you're most likely to change is
``spark.sql.shuffle.partitions``. A panel fit begins with a shuffle, in which
Spark sends rows between executors until each unit's rows sit in the same
partition. The setting fixes how many partitions the shuffle makes, 200 by
default. Each partition then runs as a task with a fixed start-up cost. On one
machine, the tasks of 200 partitions can take longer than everything else a
small fit does.

.. code-block:: python

   # A few shuffle partitions per core suit a small panel on one machine.
   spark.conf.set("spark.sql.shuffle.partitions", "8")

On a laptop, that setting brings the county fit from about two seconds down to
under one. A cluster calls for the same few partitions per core, such as 192
for 64 cores.

The bootstrap draws are another cost you can control directly. Even with the
default ``boot=False``, the default ``cband=True`` draws ``biters`` random signs
for each unit or cluster and keeps their sums for the bands of later
aggregations. When no aggregation will need them, passing ``cband=False`` as
well skips the draws. A very large design may need fewer draws to stay under
the limits on what the driver collects, as the :ref:`troubleshooting section
<spark-troubleshooting>` explains.

For a look at how a Spark fit computes its estimates and draws without
collecting the units, the developer guide to :ref:`fits on shards
<dev-shards>` follows one from the inside.
