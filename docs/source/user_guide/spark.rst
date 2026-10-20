.. _spark:

Estimation on Spark
===================

When a panel sits in a Spark table, or holds more units than one machine can
keep in memory, the question is how to estimate group-time effects without
gathering its rows in one place. Collecting the table sends every row to the
driver, the Python process that runs your code. Even an estimator that read the
rows in place would fill the driver's memory if it kept each unit's influence
function values for the standard errors.

In this guide we pass a Spark DataFrame straight to :func:`~moderndid.att_gt`.
Spark's executors do all the work on the rows and send the driver only
summaries whose size doesn't depend on the number of units. You get back an
:class:`~moderndid.MPResult` with the estimates that an in-memory fit gives.
:func:`~moderndid.aggte`, :func:`~moderndid.honest_did`, the plots, and
:func:`~moderndid.to_df` all take that result as they take any other.

When an in-memory fit is enough
-------------------------------

If the data fits comfortably in memory, an in-memory fit is simpler and
usually faster, since every step on Spark pays for scheduling a job across the
cluster. A Spark DataFrame that small can be collected to the driver first.
Passing ``data.toArrow()`` to ``att_gt`` gives an in-memory fit that also
accepts the arguments a Spark fit rejects, such as ``panel=False`` and
``allow_unbalanced_panel=True``. Without a cluster, the ``n_jobs`` argument can
spread the comparisons over several threads, as the
:ref:`CPU parallelism guide <scaling>` shows.

Setting up Spark
----------------

You need Spark 4.0 or later, whether you run classic Spark, Spark Connect, or
Databricks. With an older PySpark, the fit stops before Spark runs a job and
names the version it found. On a laptop or a cluster you manage, the ``spark``
extra installs PySpark together with the Spark Connect client. Running Spark
on your own machine also needs Java 17 or later.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add "moderndid[spark]"

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install "moderndid[spark]"

Since the executors import ModernDiD to run each step of the fit, the Python
environment on every worker needs it too. You can install it on each worker or
ship a packed environment to the executors, as the PySpark guide to `Python
package management
<https://spark.apache.org/docs/latest/api/python/tutorial/python_packaging.html>`_
describes.

.. admonition:: Match versions on the driver and executors
   :class: important

   The driver and every executor need the same ModernDiD version and Python
   minor version, such as 3.13. If they differ, the fit stops at its first
   pass with an error that names the versions on both sides. If ``moderndid``
   fails to import on the executors, the message quotes the import error
   instead.

Because the first pass comes only after the shuffle that builds the unit table,
a missing or mismatched install shows up late on a large panel. Fitting a small
sample of units first checks a new cluster quickly.

Running on Databricks
^^^^^^^^^^^^^^^^^^^^^

Since we haven't yet run moderndid on Databricks, the steps in this section
follow Databricks' documentation. On a cluster, choose a runtime built on
Spark 4, such as Databricks Runtime 17.0 or later. On serverless compute,
moderndid needs `environment version
<https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/>`_
4 or later. In Databricks' list, versions 1 and 2 run Python 3.10 and 3.11,
below the 3.12 that moderndid needs. Version 3 comes with the Databricks
Connect 16.4 client of a Spark 3.5 runtime, below the Spark 4.0 that moderndid
needs.

On a cluster, install ModernDiD as a `cluster library
<https://docs.databricks.com/aws/en/libraries/cluster-libraries>`_ or with
``%pip`` as a `notebook-scoped library
<https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries>`_. On
serverless compute, which has no cluster libraries, use ``%pip``. Either kind
of library reaches the executors as well as the notebook. Pin the version, as
in ``%pip install moderndid==<version>``, to give every session the same one.
Since ``%pip`` doesn't restart Python, a notebook that imported moderndid
before the install keeps the old version on its driver. Put the ``%pip`` cell
first or follow it with ``dbutils.library.restartPython()``.

With Databricks Connect, the driver is the Python process on your own machine.
Install the same moderndid version there as on the compute. Your local Python
also needs the compute's minor version, as Databricks requires of `Python
functions that run on its compute
<https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/udf>`_.

.. admonition:: The ``spark`` extra conflicts with Databricks
   :class: warning

   Databricks compute provides its own PySpark, as does Databricks Connect on
   your machine. A second copy from the ``spark`` extra conflicts with either
   one and can keep the Spark session from starting.

Trying Spark on one machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you move to a cluster, you can run every call on this page in Spark's
local mode.

.. code-block:: python

   from pyspark.sql import SparkSession

   # Start Spark on this machine with two worker threads.
   spark = SparkSession.builder.master("local[2]").getOrCreate()

In a fresh Python process, ``SparkSession.builder.remote("local[2]").getOrCreate()``
starts a local Spark Connect server instead. That lets you check the code on
this page under Spark Connect before you point it at a running server with an
address such as ``sc://localhost:15002``.

.. admonition:: Point Spark at your Python
   :class: tip

   Local Spark starts its Python workers with the interpreter that
   ``PYSPARK_PYTHON`` names, or with ``python3`` from your path. If a local fit
   reports that the executors run another Python or can't import
   ``moderndid``, set ``PYSPARK_PYTHON`` to ``sys.executable`` before you
   create the session.

Preparing the data
------------------

The data takes the usual long format of ``att_gt``, one row for each unit in
each period. In the code below, ``SparkSession.builder.getOrCreate()`` returns
the session that is already running, such as a Databricks notebook's own
session or the local session from the setup above. We load the county data on
minimum wages into a Spark DataFrame. Since only a Spark DataFrame is fit on
the executors, convert a pandas-on-Spark DataFrame with its ``to_spark()``
method first.

.. code-block:: python

   import moderndid as did
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()

   # The county data stands in for a table that spark.read.table would load.
   data = spark.createDataFrame(did.load_mpdta().to_arrow())
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

Every column holds numbers, as ``att_gt`` requires of the outcome, period,
unit, and cohort columns. In a Spark fit, sampling weights and covariates must
hold numbers too. Because the fit checks these types against the schema, a
column of the wrong type stops it before Spark runs a single job. Units
observed more than once in a period stop the fit with an error. Rows with
missing values and units that miss a period drop out with a warning, just as
the :ref:`data guide <data-for-estimation>` describes.

A Spark fit also takes the decimal columns that warehouse tables often hold. It
reads decimal periods and cohorts as doubles and reads a decimal identifier or
cluster with no fraction digits and at most 18 digits as a 64-bit integer. Any
other decimal identifier or cluster stops the fit before Spark runs a job. Its
error asks you to cast the column to bigint or to code its values as integers
first.

As with an in-memory fit, a transformed covariate or a derived cluster
variable goes into the data as a column of its own before the fit. Spark's
column functions can add such a column without collecting any rows.

.. code-block:: python

   from pyspark.sql import functions as F

   # The leading digits of a county FIPS code are its state's code.
   data = data.withColumn("state", F.floor(F.col("countyreal") / 1000))

Passing ``clustervars=["state"]`` to ``att_gt`` would then cluster the standard
errors by state.

Unlike a cluster variable, the unit identifier must hold numbers. Spark's
``xxhash64`` function turns an identifier stored as text into a 64-bit integer,
as below for the county number cast to text.

.. code-block:: python

   # Code a text identifier as a 64-bit integer for idname.
   text = data.withColumn("county", F.col("countyreal").cast("string"))
   coded = text.withColumn("county_code", F.xxhash64("county"))

Passing ``idname="county"`` itself would stop the fit before Spark runs a job
with the in-memory error
``idname = 'county' is not numeric. Please convert it``. With
``idname="county_code"``, the fit gives the same estimates as one on
``countyreal``, though its bootstrap draws differ because they follow the codes.
Two identifiers that shared a code would repeat (unit, period) pairs and stop
the fit with an error rather than merge silently.

Fitting on the executors
------------------------

We reuse the specification of the
:ref:`staggered DiD example <example_staggered_did>` unchanged. It uses the
doubly robust estimator with log population as a covariate, never-treated
counties as the comparison group, and a universal base period. With 10,000
seeded bootstrap draws, every run gives the same simultaneous bands up to
rounding.

.. code-block:: python

   # Collect the arguments in one dictionary for the checks below to reuse.
   spec = dict(
       yname="lemp",
       tname="year",
       idname="countyreal",
       gname="first.treat",
       control_group="nevertreated",
       xformla="~lpop",
       est_method="dr",
       base_period="universal",
       anticipation=0,
       boot=True,
       biters=10000,
       cband=True,
       random_state=7,
   )

   # The executors fit every comparison where the rows live.
   result = did.att_gt(data, **spec)
   print(result)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         ==============================================================================
          Group-Time Average Treatment Effects
         ==============================================================================

         ┌───────┬──────┬──────────┬────────────┬──────────────────────────┐
         │ Group │ Time │ ATT(g,t) │ Std. Error │ [95% Simult. Conf. Band] │
         ├───────┼──────┼──────────┼────────────┼──────────────────────────┤
         │  2004 │ 2003 │   0.0000 │         NA │ NA                       │
         │  2004 │ 2004 │  -0.0145 │     0.0233 │ [-0.0752,  0.0462]       │
         │  2004 │ 2005 │  -0.0764 │     0.0301 │ [-0.1548,  0.0019]       │
         │  2004 │ 2006 │  -0.1404 │     0.0376 │ [-0.2383, -0.0426] *     │
         │  2004 │ 2007 │  -0.1069 │     0.0339 │ [-0.1950, -0.0188] *     │
         │  2006 │ 2003 │   0.0067 │     0.0309 │ [-0.0737,  0.0870]       │
         │  2006 │ 2004 │   0.0062 │     0.0191 │ [-0.0434,  0.0558]       │
         │  2006 │ 2005 │   0.0000 │         NA │ NA                       │
         │  2006 │ 2006 │   0.0010 │     0.0201 │ [-0.0512,  0.0532]       │
         │  2006 │ 2007 │  -0.0413 │     0.0200 │ [-0.0933,  0.0107]       │
         │  2007 │ 2003 │   0.0063 │     0.0249 │ [-0.0585,  0.0711]       │
         │  2007 │ 2004 │   0.0330 │     0.0215 │ [-0.0230,  0.0890]       │
         │  2007 │ 2005 │   0.0284 │     0.0187 │ [-0.0201,  0.0770]       │
         │  2007 │ 2006 │   0.0000 │         NA │ NA                       │
         │  2007 │ 2007 │  -0.0288 │     0.0170 │ [-0.0730,  0.0155]       │
         └───────┴──────┴──────────┴────────────┴──────────────────────────┘

         ------------------------------------------------------------------------------
          Signif. codes: '*' confidence band does not cover 0

          P-value for pre-test of parallel trends assumption:  0.2327

         ------------------------------------------------------------------------------
          Data Info
         ------------------------------------------------------------------------------
          Control Group:  Never Treated
          Anticipation Periods:  0

         ------------------------------------------------------------------------------
          Estimation Details
         ------------------------------------------------------------------------------
          Estimation Method:  Doubly Robust

         ------------------------------------------------------------------------------
          Inference
         ------------------------------------------------------------------------------
          Significance level: 0.05
          Bootstrap standard errors
         ==============================================================================
          Reference: Callaway and Sant'Anna (2021)

Every point estimate in the table and the pre-test p-value of 0.2327 match the
staggered DiD example to the printed digits. To check a fit on your own data,
you can fit a sample of its units both ways and compare the results. Sample
whole units rather than rows, because a unit that loses a period drops out of
both fits.

.. code-block:: python

   # Keep all the rows of about one county in twenty.
   sample = data.filter(F.xxhash64("countyreal") % 20 == 0)

Since the hash depends only on the identifier, the filter keeps or drops all of
a unit's rows together. On the county data, it keeps 22 of the 500 counties
with all five years each. A fit on so few counties warns that some cohorts are
small. The county panel itself is small enough for the check below to fit all
of it in memory.

.. code-block:: python

   import numpy as np

   # Fit the specification in memory for comparison.
   in_memory = did.att_gt(did.load_mpdta(), **spec)

   # Stop with an error unless the estimates and their analytic covariance agree.
   np.testing.assert_allclose(result.att_gt, in_memory.att_gt, rtol=1e-10, atol=1e-12)
   np.testing.assert_allclose(
       result.vcov_analytical, in_memory.vcov_analytical, rtol=1e-10, atol=1e-12
   )

   # Compare the critical values of the two simultaneous bands.
   print(round(result.critical_value, 4), round(in_memory.critical_value, 4))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         2.6014 2.6178

Passing both assertions means that the estimates and their analytic covariance
agree to rounding. The critical values of 2.6014 and 2.6178 differ because the
two fits use different bootstrap draws, as the section on
:ref:`bootstrap draws on Spark <spark-bootstrap>` explains.

What the result holds
---------------------

The result fills every field that the reports, plots, and aggregations read.
Unlike an in-memory fit, it leaves ``influence_func``, ``G``, and
``weights_ind`` as None, since each of them has one entry per unit. In their
place, ``influence_moments`` holds sums over units as an
:class:`~moderndid.InfluenceMoments`. Each unit adds one row that holds its
influence function value for every group-time effect, its sampling weight in
the column of its treated cohort, and a constant 1. The Gram matrix ``gram``
adds up the outer products of those rows. The bootstrap sums in
``bootstrap_sums`` add up the rows times each draw's random signs.

.. code-block:: python

   # Show the sums that replace the influence functions.
   moments = result.influence_moments
   print(result.influence_func, moments.gram.shape, moments.bootstrap_sums.shape)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         None (19, 19) (10000, 19)

The Gram matrix has a row for each of the 15 group-time effects in the table,
each of the 3 treated cohorts, and a constant. Because its size depends on the
effects rather than on the 500 counties, it stays this small however many
units the panel holds. The bootstrap sums have one row for each of the 10,000
draws. Since the result holds only NumPy arrays and plain Python values, you
can pickle it and aggregate it later in a session without Spark.

From these sums, :func:`~moderndid.aggte` computes the standard errors and
bands of an aggregation, such as the event study below.

.. code-block:: python

   # Average the effects by event time and show them as a table.
   event_study = did.aggte(result, type="dynamic")
   print(did.to_df(event_study))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (7, 6)
         ┌────────────┬───────────┬──────────┬───────────┬───────────┬──────────────────┐
         │ event_time ┆ att       ┆ se       ┆ ci_lower  ┆ ci_upper  ┆ treatment_status │
         │ ---        ┆ ---       ┆ ---      ┆ ---       ┆ ---       ┆ ---              │
         │ i64        ┆ f64       ┆ f64      ┆ f64       ┆ f64       ┆ str              │
         ╞════════════╪═══════════╪══════════╪═══════════╪═══════════╪══════════════════╡
         │ -4         ┆ 0.006296  ┆ 0.024915 ┆ -0.055751 ┆ 0.068344  ┆ Pre              │
         │ -3         ┆ 0.02686   ┆ 0.018227 ┆ -0.01853  ┆ 0.072251  ┆ Pre              │
         │ -2         ┆ 0.023244  ┆ 0.014801 ┆ -0.013616 ┆ 0.060104  ┆ Pre              │
         │ 0          ┆ -0.02106  ┆ 0.012012 ┆ -0.050973 ┆ 0.008853  ┆ Post             │
         │ 1          ┆ -0.053003 ┆ 0.016583 ┆ -0.094301 ┆ -0.011705 ┆ Post             │
         │ 2          ┆ -0.140448 ┆ 0.037629 ┆ -0.234156 ┆ -0.046741 ┆ Post             │
         │ 3          ┆ -0.106904 ┆ 0.033868 ┆ -0.191245 ┆ -0.022563 ┆ Post             │
         └────────────┴───────────┴──────────┴───────────┴───────────┴──────────────────┘

The event study also keeps its analytic covariance in ``vcov_analytical``. For
its sensitivity analysis, :func:`~moderndid.honest_did` reads that covariance
in place of the influence functions.

.. code-block:: python

   # Compute robust intervals for the adoption-year effect under smoothness bounds.
   sensitivity = did.honest_did(
       event_study,
       event_time=0,
       sensitivity_type="smoothness",
       m_vec=[0.0, 0.01, 0.02],
       method="FLCI",
   )
   print(sensitivity.robust_ci)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (3, 5)
         ┌───────────┬──────────┬────────┬─────────┬──────┐
         │ lb        ┆ ub       ┆ method ┆ delta   ┆ m    │
         │ ---       ┆ ---      ┆ ---    ┆ ---     ┆ ---  │
         │ f64       ┆ f64      ┆ str    ┆ str     ┆ f64  │
         ╞═══════════╪══════════╪════════╪═════════╪══════╡
         │ -0.050609 ┆ 0.0058   ┆ FLCI   ┆ DeltaSD ┆ 0.0  │
         │ -0.046257 ┆ 0.0352   ┆ FLCI   ┆ DeltaSD ┆ 0.01 │
         │ -0.052246 ┆ 0.053837 ┆ FLCI   ┆ DeltaSD ┆ 0.02 │
         └───────────┴──────────┴────────┴─────────┴──────┘

These intervals match those of the in-memory event study to every printed
digit, since both start from the same analytic covariance. The
:ref:`sensitivity analysis example <example_honest_did>` explains how to read
intervals like these.

The fit's own ``vcov_analytical`` gives standard errors for contrasts that
``aggte`` doesn't report. For fixed weights :math:`a` on the group-time
effects, such as the difference between two of them, the analytic standard
error is :math:`\sqrt{a'Va/n}` for :math:`V` the result's ``vcov_analytical``
and :math:`n` its ``n_units``. Since :math:`V` sums within clusters when the fit
clusters, that standard error is then cluster-robust.

Since the units stay on the executors, ``aggte`` can't make new bootstrap
draws from a Spark result and reuses the fit's draws instead. Asking it for
more draws than the fit stored, for another seed, or for draws by unit from a
clustered fit raises a ValueError that says what to refit.

.. admonition:: Ask for enough bootstrap draws up front
   :class: tip

   Set ``biters`` in ``att_gt`` to the largest number of draws any later
   aggregation will use. An aggregation with a smaller ``biters`` reads the
   fit's first draws. Because a fit with both ``boot=False`` and
   ``cband=False`` stores no draws at all, passing ``boot=True`` or
   ``cband=True`` to its aggregations raises an error.

.. _spark-bootstrap:

Bootstrap draws on Spark
------------------------

The multiplier bootstrap multiplies each unit's influence function by a random
sign in every draw, or each cluster's sum when the fit clusters. On Spark, each
sign depends only on the seed, the draw, and the value of its unit's or
cluster's identifier, never on which partition or executor holds the unit. A
fixed ``random_state`` therefore gives every unit the same signs on any
partitioning of the data, in classic Spark and Spark Connect alike. Since the
sums over units add up in another order when the partitions change, the
standard errors, the bands, and the Gram matrix agree to rounding rather than
bit for bit. Differences of that size stay well below the digits that the
reports print.

Because the county panel fits in one partition of the unit table, the check
below fits a generated panel of 40,000 units instead. Two settings of
``spark.sql.shuffle.partitions`` split its unit table differently.

.. code-block:: python

   # Generate a panel of 40,000 units, enough for a unit table of several partitions.
   generated = did.gen_did_scalable(
       40000, n_periods=3, n_cohorts=2, n_covariates=4, random_state=7
   )["data"]
   panel = spark.createDataFrame(generated.to_arrow())
   demo = dict(
       yname="y", tname="time", idname="id", gname="group", xformla="~cov1", boot=True, random_state=7
   )

   # Fit the panel under two shuffle settings that split the unit table differently.
   spark.conf.set("spark.sql.shuffle.partitions", "2")
   two = did.att_gt(panel, **demo)
   spark.conf.set("spark.sql.shuffle.partitions", "8")
   eight = did.att_gt(panel, **demo)

   # Stop with an error unless the standard errors agree.
   np.testing.assert_allclose(two.se_gt, eight.se_gt, rtol=1e-12)

   # Compare the critical values of the two fits.
   print(round(two.critical_value, 4), round(eight.critical_value, 4))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         2.238 2.238

The first setting splits the unit table into 2 partitions and the second into 5
that hold different sets of units. Both fits agree on every standard error to
rounding and print the same critical value of 2.238.

On the county data, an in-memory fit draws its signs in the order of its units.
With the same ``random_state=7``, it therefore gets a different set of signs
that is just as valid. Its bootstrap standard errors differ from those of the
Spark fit in the third decimal, such as 0.0225 against 0.0233 for the 2004
cohort in 2004.

Since both fits estimate the same bootstrap distribution, neither answer is
more accurate than the other. Their gap is the noise of a finite number of
draws and narrows as ``biters`` grows.

Because the signs follow the identifier's value, a unit identifier stored as
integers gives the same draws as one stored as floats or decimals that hold the
same numbers. A cluster variable may hold strings as well as numbers, although a
cluster named ``"17"`` draws other signs than one numbered ``17``. Without a
``random_state``, the fit draws a seed and stores it in
``influence_moments.random_state`` so that a later Spark fit can reproduce its
draws.

How a fit runs
--------------

The fit starts by building a unit table from the long data and leaves the data
itself where it is. A first aggregation by period and cohort counts the missing
values and finds the periods, cohorts, and weight totals. A single shuffle then
gathers each unit's rows into one row that holds its outcome, weight, and
covariates in every period. These two steps and the schema checks before them
reproduce every check and warning of the in-memory preprocessing. The fit
persists the unit table and releases it when it returns or when any error,
even an interrupt, stops it.

Each later step of the estimator runs as a pass over that table. A pass sends a
small function to every partition of the table. Each task applies it to its
units and returns statistics of a fixed size, such as counts, sums, maxima, and
the triangular factors of QR decompositions. An error inside a task reaches
you as PySpark's own exception with the worker's Python traceback in its
message. The passes of a fit come in four kinds and run in this order.

- A first pass collects the counts, weight sums, and QR factors that every
  fit starts from.
- One pass per iteration of the logit updates all the propensity score fits at
  once.
- Once every fit is final, a pass sums the parts of every effect and of its
  influence function.
- A last pass sums the outer products and bootstrap draws that stand in for
  the influence functions.

A doubly robust or inverse probability fit makes three passes plus one for
each iteration of its slowest logit. Each logit stops after 100 iterations at
most, as the in-memory binomial fit does. Since a logit whose propensity scores
separate the treated units from the comparison units can take a few dozen
iterations, a single such cell can add a few dozen passes to the fit. An
outcome regression fit needs no logit and makes three passes. The doubly
robust fit of the county data above makes nine passes. Spark runs the whole
fit, preprocessing included, as 15 jobs. With ``est_method="reg"`` and nothing
else changed, the fit takes three passes and nine jobs. The number of passes
doesn't grow with the number of units or group-time effects. The
:ref:`architecture guide <architecture-shards>` describes the kernels and
executors behind these passes.

Tuning a fit
------------

The shuffle that builds the unit table splits it into as many partitions as
``spark.sql.shuffle.partitions`` sets, 200 by default. On Databricks serverless
compute, the documented default is ``auto`` instead. Before the passes start,
the fit merges them into partitions of about a megabyte each. Each pass then
runs one task per megabyte of the table, up to the number that setting allows.

For a panel of modest size on one machine, the shuffle's 200 tasks can take
longer than all of the passes. Setting the value near the number of cores, as
below, shortens that step. On a cluster, a panel of many millions of units
needs a value large enough to give every core a few tasks per pass.

.. code-block:: python

   # Use fewer shuffle partitions for a small panel on one machine.
   spark.conf.set("spark.sql.shuffle.partitions", "8")

Persisting the unit table spares every pass from repeating the shuffle. When a
session refuses to persist it, as Databricks documents for `serverless compute
<https://docs.databricks.com/aws/en/compute/serverless/limitations>`_, the fit
tries a local checkpoint instead. If the session refuses the checkpoint too, a
warning says that each pass of the fit rebuilds the table from the data.

.. admonition:: Save an expensive query first
   :class: tip

   The fit reads its data twice, in the first aggregation and in the shuffle
   that builds the unit table. When the DataFrame comes from an expensive
   query, such as a join of large tables, write it to a table, or persist it if
   the session allows, before the fit.

In the last pass, each task returns its sums in one payload of about
:math:`8L(L + B)` bytes, where :math:`B` is ``biters`` and :math:`L` counts the
group-time effects, the treated cohorts, and a constant. With :math:`L = 19`
and 10,000 draws, the county fit above sends about 1.5 MB per task. A design
with 25 cohorts over 40 periods has an :math:`L` near 1,000 and would send
about 88 MB per task at 10,000 draws. When a pass has more than 16 tasks, a
second stage combines their payloads in groups before they reach the driver.
For :math:`P` tasks, each combining task holds a group of up to about
:math:`2\sqrt{P}` payloads at once. Because a pass of 16 tasks or fewer skips
that stage, the driver collects up to 16 payloads, or about :math:`\sqrt{P}` of
them once :math:`P` passes 256. Their total must stay under the 1 GB that
``spark.driver.maxResultSize`` allows by default. Sixteen payloads of the
larger design would come to about 1.4 GB. Under Spark Connect, each payload
must also fit in one message of at most 128 MiB by default. A design that large
needs fewer draws, or more memory and a larger result limit. On serverless
compute, whose documented settings include neither, fewer draws are the only
remedy.

.. admonition:: Skip draws you won't use
   :class: tip

   Even when ``boot=False``, the default ``cband=True`` makes the last pass
   draw ``biters`` random signs for every unit or cluster. If neither the fit
   nor its aggregations need bootstrap standard errors or a simultaneous band,
   passing both ``boot=False`` and ``cband=False`` skips those draws and makes
   a large panel fit faster.

A clustered fit partitions the unit table by cluster rather than by unit.
Every task then holds whole clusters and can sum within them itself. What
reaches the driver doesn't grow with the number of clusters. The cost is that
a pass never splits a cluster across tasks. On a panel large enough to fill
many partitions, clustering by state would leave no more tasks with work in
each pass than there are states. If one state had most of the units, every pass
would wait on its task.

What needs an in-memory fit
---------------------------

Spark takes balanced panels and the three built-in estimators of ``att_gt``.
Units missing from any period drop out with the warning an in-memory fit gives
under its default ``allow_unbalanced_panel=False``. Sampling weights, a cluster variable
besides ``idname``, both comparison groups, both base periods, and
anticipation all work on Spark. Each argument below raises a ValueError that
says what to pass instead.

- ``panel=False``, for a repeated cross-section.
- ``allow_unbalanced_panel=True``, for an unbalanced panel.
- A callable ``est_method``, in place of ``"dr"``, ``"ipw"``, or ``"reg"``.
- Any ``backend``, since the executors always compute with NumPy.
- ``n_jobs`` other than 1, since Spark already runs each pass in parallel.

Two error messages that a Spark fit shares with the in-memory checks still
suggest ``panel=False``. They appear when no unit is observed in more than one
period or when balancing the panel drops every unit. On Spark, either one
usually means that ``idname`` doesn't pick out the same unit in every period.
If the data is a repeated cross-section after all, collect it and fit it in
memory with ``panel=False``.

Among the estimators, only ``att_gt`` fits a Spark DataFrame where its rows
live. Other functions that take data, such as :func:`~moderndid.ddd`,
:func:`~moderndid.cont_did`, and :func:`~moderndid.core.panel.diagnose_panel`,
collect every row on the driver. Converting the DataFrame first with
``data.toArrow()`` works with PySpark 4.0 and later. With PySpark 3.5, which has
no ``toArrow()``, pass ``pyarrow.Table.from_pandas(data.toPandas())`` instead.
From PySpark 4.2 on, these functions also read a Spark DataFrame directly
through its Arrow stream.

The test suite runs fits like these on a local cluster with separate executor
processes and on a Docker cluster. The :ref:`testing guide <testing-spark>`
shows how to run both setups on your own machine.
