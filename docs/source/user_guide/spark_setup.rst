.. _spark-setup:

Setting up Spark
================

Before you can fit a Spark DataFrame, ModernDiD has to be installed both where
your code runs and where Spark's executors run. Your code and
:func:`~moderndid.att_gt` run in the driver, the Python process behind your
notebook or script. The executors are the processes that Spark runs on the
cluster's worker machines to work through your table's rows in parallel. Each
one imports ModernDiD for its share of the units and needs the driver's
ModernDiD version and Python minor version. Both need Python 3.12 or later and
PySpark 3.5 or later.

If a fit stops with an error, look for its message in the troubleshooting
section at the end. Once your setup works, the :ref:`Spark guide <spark>` takes
you through a fit.

Tested setups
-------------

Before you install anything, the table below shows whether a Spark fit has
already run on a setup like yours. Your code can reach Spark in a classic
session, where the driver controls Spark from its own machine, or through Spark
Connect, where it sends each query over the network to a server that may run
elsewhere. Continuous integration runs the Spark tests both ways on every pull
request, as the :ref:`testing guide <testing-spark>` shows.

.. list-table::
   :header-rows: 1
   :widths: 32 24 44

   * - Spark
     - Python and Java
     - Where it ran
   * - PySpark 4.2
     - Python 3.13 and Java 21
     - Continuous integration on a local cluster of two executors and on a
       cluster of Docker containers
   * - PySpark 3.5
     - Python 3.12 and Java 17
     - Continuous integration on a local cluster of two executors
   * - Databricks Runtime 16.4 LTS ML (Spark 3.5.2)
     - Python 3.12.3 and Java 17
     - Azure Databricks in October 2026, on classic compute in dedicated access
       mode

On Databricks, the Spark 4 runtimes (17.3 LTS and later), standard access
mode, serverless compute, Databricks Connect, and cluster libraries haven't run
a fit yet. Neither have Amazon EMR, Google Cloud Dataproc, and Spark on
Kubernetes, where the steps for a cluster you manage should carry over.

Databricks
----------

The runtime you choose decides which Python and PySpark your notebook gets.
Choose Databricks Runtime 16.4 LTS or a later supported runtime, because
earlier runtimes are past end of support or run Python 3.11 or older. An ML
runtime such as 16.4 LTS ML gets classic compute in dedicated access mode by
default, a cluster that you configure yourself and reserve for one user or
group. Most other runtimes get standard access mode and reach Spark through
Spark Connect.

Install ModernDiD as a `notebook-scoped library
<https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries>`_ in
the first cell and restart Python in the next, as Databricks recommends. On our
cluster, that install reached the executors as well as the notebook.

.. code-block:: python

   %pip install moderndid==<version>

.. code-block:: python

   dbutils.library.restartPython()

The restart is what loads the newer SciPy and statsmodels that the install
brings. Pin the release you want, 0.4.0 or later, in place of ``<version>``. To
install a commit from GitHub instead, put its hash after the repository's
address.

.. code-block:: python

   %pip install "moderndid @ git+https://github.com/jordandeklerk/moderndid.git@<commit>"

.. admonition:: Leave out the ``spark`` extra on Databricks
   :class: important

   Each Databricks runtime already provides the PySpark that the ``spark``
   extra would install a second time. Databricks says that installing PySpark
   in a serverless notebook stops the session.

The advice for serverless compute and Databricks Connect follows Databricks'
documentation until a fit runs there. A serverless notebook needs
`environment version
<https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/>`_
3 or later for its Python 3.12. With Databricks Connect, your own machine runs
your code and ``att_gt`` while the compute works on the rows. That machine
needs the compute's ModernDiD version and the Python 3.12 that Databricks
requires for `Python functions that run on its compute
<https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/udf>`_.

.. admonition:: Databricks Connect conflicts with PySpark
   :class: warning

   Databricks says that ``databricks-connect`` conflicts with a separate
   PySpark. Install ModernDiD on your machine without the ``spark`` extra and
   uninstall any PySpark before you install ``databricks-connect``.

A cluster you manage
--------------------

Every fit in the Spark guide runs unchanged on a cluster you manage once each
worker machine has ModernDiD in the Python that its executors run. You can
install it on each machine or ship the executors a packed Python environment,
an archive that holds ModernDiD and everything it needs, as PySpark's guide to
`Python package management
<https://spark.apache.org/docs/latest/api/python/tutorial/python_packaging.html>`_
describes. Pin PySpark to your cluster's Spark version rather than the latest
release that the ``spark`` extra installs. A pin to PySpark 3.5 needs Python
3.12, because its Spark Connect client requires NumPy 1 and NumPy 1 has no
wheels for Python 3.13 or later.

If the cluster runs a Spark Connect server, your session connects to its
address. The executors then have to match the ModernDiD and Python versions of
your own machine, where your code and ``att_gt`` run.

.. code-block:: python

   from pyspark.sql import SparkSession

   # Connect to the cluster's Spark Connect server.
   spark = SparkSession.builder.remote("sc://host:15002").getOrCreate()

A Spark 3.5 server that loads the Spark Connect package with ``--packages``
needs one more step, because its executors see only the jars, or Java
libraries, that they start with. The ``spark-connect`` jar therefore also has
to sit in each worker's Spark ``jars`` folder or on
``spark.executor.extraClassPath``.

.. admonition:: Test a new cluster on a sample
   :class: tip

   Because a missing or mismatched install shows up only after a panel fit has
   gathered each unit's rows over the network, a large panel makes a slow first
   test of a new cluster. The :ref:`sample check <spark-sample-check>` gathers
   only a sample's rows and finds the same problem sooner.

Your own machine
----------------

The quickest way to try a Spark fit is Spark's local mode, where every fit in
the Spark guide runs on one machine without a cluster. Outside Databricks, the
``spark`` extra installs the latest PySpark along with the Spark Connect client.
It keeps pandas below version 3 because PySpark doesn't yet fully support
pandas 3.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add "moderndid[spark]"

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install "moderndid[spark]"

Because Spark itself runs on Java, you'll also need Java 17 or 21 for Spark 4,
or Java 8, 11, or 17 for Spark 3.5. Java 17 is the safe choice because it runs
both. Under Java 21, Spark 3.5 fails at the first step that hands rows to
Python through Apache Arrow.

.. code-block:: python

   from pyspark.sql import SparkSession

   # Start Spark on this machine with two worker threads.
   spark = SparkSession.builder.master("local[2]").getOrCreate()

.. admonition:: Point Spark at your Python
   :class: tip

   Local Spark runs ModernDiD in Python workers, separate processes that it
   starts with the interpreter that ``PYSPARK_PYTHON`` names or with
   ``python3`` from your path. If a local fit fails in a worker because of
   another Python version or a failed import of ModernDiD, set
   ``PYSPARK_PYTHON`` to ``sys.executable`` before you create the session.

Notebooks in standard access mode, serverless compute, and Databricks Connect
all reach Spark through Spark Connect. If your code will run in one of them, a
local Spark Connect server lets you check it on that path first. On PySpark 4,
``SparkSession.builder.remote("local[2]").getOrCreate()`` alone starts one in
your Python process.
PySpark 3.5 needs the server's package from Maven Central, the public
repository of Java packages, and a fresh Python process, because a classic
session started earlier blocks the server even after you stop it.

.. code-block:: python

   import pyspark
   from pyspark.sql import SparkSession

   # On PySpark 3.5 only, add the Spark Connect server package of the same version.
   package = f"org.apache.spark:spark-connect_2.12:{pyspark.__version__}"
   builder = SparkSession.builder.config("spark.jars.packages", package)
   spark = builder.remote("local[2]").getOrCreate()

.. _spark-troubleshooting:

Troubleshooting
---------------

Each entry below starts from the message that a fit or a later aggregation
shows you and says what to change.

The executors can't run ModernDiD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An error that starts "The Spark executors failed to import moderndid with the
error" means that an executor's Python has no ModernDiD or can't import one of
its dependencies, the module that the quoted error names. The section above for
your setup shows how to install it where the executors run.

An error that starts "The Spark executors run moderndid" names the versions on
both sides. It usually means another ModernDiD version on the executors,
because a different Python trips PySpark's own check first. When
PySpark itself reports "Python in worker has different version", the executors
run another Python minor version. Set ``PYSPARK_PYTHON`` on your own machine,
as the tip above describes, or give a cluster one Python minor version
everywhere.

Java, Python, and PySpark versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If creating the session fails with "Java gateway process exited before sending
its port number", Spark most likely found no Java that it can run. Install
Java 17 and point ``JAVA_HOME`` at it before you create the session. A Java
error that says "sun.misc.Unsafe or java.nio.DirectByteBuffer.<init>(long,
int) not available" means that Spark 3.5 is running on Java 21. Point
``JAVA_HOME`` at Java 17 or move to Spark 4.

An error that ends in "No module named 'distutils'" means that PySpark 3.5 runs
on Python 3.12 or later, where that module is gone. Installing setuptools on
the driver and wherever the executors run brings it back.

.. code-block:: console

   python -m pip install setuptools

If the fit stops right away with "A fit of att_gt on a Spark DataFrame needs
pyspark 3.5 or later", upgrade the older PySpark that the message names. On
Databricks, an install that reports "Ignored the following versions that
require a different python version", or "requires a different Python" for an
install from GitHub, means that the runtime runs Python 3.11 or older. Moving
the compute to Runtime 16.4 LTS or a later supported runtime fixes it.

The fit refuses the data
^^^^^^^^^^^^^^^^^^^^^^^^

An error that says "idname = 'county' is not numeric. Please convert it" means
that the unit identifier holds text. :ref:`Preparing your table
<spark-prepare>` shows how ``xxhash64`` turns such an identifier into an
integer. Text in the outcome, period, cohort, sampling weights, or a covariate
stops the fit with a message that names that column. A cluster column of state
names needs no conversion, since the fit uses a cluster's values only as
labels.

When the message reads "idname='county' is a decimal(38,0) column" or "'state'
in clustervars is a decimal(38,0) column", the column's decimal type has
fraction digits or more than 18 digits. If it holds only whole numbers, casting
it to bigint, Spark's 64-bit integer type, lets it through.

.. code-block:: python

   from pyspark.sql import functions as F

   # A decimal identifier of whole numbers passes as a 64-bit integer.
   data = data.withColumn("county", F.col("county").cast("bigint"))

Two errors that a Spark fit inherits from the in-memory checks suggest
``panel=False``. One says "panel=True was specified, but no units appear in
multiple time periods". The other says "All observations dropped while
converting to balanced panel". If your table really holds repeated
cross-sections or a panel in which every unit misses some period, the
:ref:`Spark guide <spark-layouts>` shows how to fit it.

.. admonition:: These errors usually point to ``idname``
   :class: warning

   Either error usually means that ``idname`` doesn't pick out the same unit
   in every period. Fitting a panel as repeated cross-sections would change the
   estimator. Fix the identifier rather than switching to ``panel=False``.

Limits of the Spark session
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A warning that says "Spark refused to persist the table of units and to
checkpoint it", or the row table for repeated cross-sections, concerns the
table that a fit builds from your data before it estimates anything. Without a
way to keep that table on the executors, the fit rebuilds it at every step and
takes much longer on a large table. Compute that allows a persist, such as
classic compute in dedicated access mode, builds it only once.

If an error says "Total size of serialized results" and names
``spark.driver.maxResultSize``, the sums that the executors sent back in one
step outgrew Spark's cap on what the driver may collect at once. Those sums
grow with the number of bootstrap draws you ask for. Pass fewer ``biters`` or
raise the cap where Spark starts, because a running session can't change it.
Fewer draws are the only fix on serverless compute, since it doesn't let you
set the cap.

.. code-block:: python

   from pyspark.sql import SparkSession

   # Raise the driver's cap on collected results before the session starts.
   spark = SparkSession.builder.config("spark.driver.maxResultSize", "4g").getOrCreate()

Before Spark 4.1, Spark Connect also needs the sums from each of the fit's
parallel tasks to fit in one message of at most 128 MiB by default. From Spark
4.1 on, Spark Connect splits larger sums across several messages.

Aggregations that need new draws
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without unit-level values to draw from, :func:`~moderndid.aggte` can only reuse
the bootstrap draws that a Spark fit stored. An error that starts "boot=True
needs bootstrap draws" or "cband=True needs bootstrap draws" comes from a fit
with ``boot=False`` and ``cband=False``. Errors that start "biters=2000 is more
than the 1000 bootstrap draws", "random_state=8 is not the seed 7", or
"clustervars=[] asks for unit-level bootstrap draws" mean that ``aggte`` asked
for draws that the fit didn't make. Each message says what to pass instead and
how to refit for the draws you asked for.
