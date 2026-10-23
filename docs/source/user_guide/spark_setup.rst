.. _spark-setup:

Setting up Spark
================

Before you can fit a Spark DataFrame, ModernDiD has to be installed both where
your code runs and where Spark's executors run. Your code and
:func:`~moderndid.att_gt` run on the driver as they would in any Python
session. Each executor imports ModernDiD for its share of the units and needs
the same ModernDiD version and Python minor version as the driver. Both need
Python 3.12 or later and PySpark 3.5 or later. Wherever PySpark 3.5 runs on
Python 3.12 or later, it also needs setuptools for the ``distutils`` module
that Python 3.12 removed.

We'll start with the setups we've tested before walking through Databricks, a
cluster you manage, and your own machine in turn. If a fit stops with an error,
look for its message in the troubleshooting section at the end. Once your setup
works, the :ref:`Spark guide <spark>` takes you through a fit.

Tested setups
-------------

The setups in the table below are the ones where we know a Spark fit works.
Continuous integration runs the Spark tests on every pull request with each of
the two PySpark releases. Each release runs once in a classic session and once
through Spark Connect on a local cluster of two executors. PySpark 4.2 also
runs the tests through Spark Connect on a standalone cluster of two workers in
Docker. The :ref:`testing guide <testing-spark>` shows how to run the same
tests on your own machine.

.. list-table::
   :header-rows: 1
   :widths: 32 24 44

   * - Spark
     - Python and Java
     - Where it ran
   * - PySpark 4.2
     - Python 3.13 and Java 21
     - Continuous integration on a local cluster and on a standalone cluster
   * - PySpark 3.5
     - Python 3.12 and Java 17
     - Continuous integration on a local cluster
   * - Databricks Runtime 16.4 LTS ML (Spark 3.5.2)
     - Python 3.12.3 and Java 17
     - Azure Databricks in October 2026, on classic compute in dedicated access
       mode

We ran the Databricks fits by hand on a cluster of four Standard_E16ads_v5
workers with 64 cores in all. A notebook-scoped ``%pip`` install was enough to
give its executors ModernDiD as well as the notebook. The :ref:`Spark guide
<spark>` reports how long those fits took and how closely they matched their
in-memory counterparts and the effects planted in the data.

Several setups that you may well use haven't run a fit yet. On Databricks,
these are the Spark 4 runtimes (17.3 LTS and later), standard access mode,
serverless compute, Databricks Connect, and an install of ModernDiD as a
cluster library. No fit has run on Amazon EMR, Google Cloud Dataproc, or Spark
on Kubernetes either. The steps for a cluster you manage should carry over to
each of them.

Databricks
----------

The Databricks runtime you choose decides which Python and PySpark your
notebook gets. Since earlier runtimes are either past end of support or run
Python 3.11 or older, choose Databricks Runtime 16.4 LTS or a later supported
runtime. The runtime and its access mode also decide how your notebook reaches
Spark. Under the access mode's default Auto setting, an ML runtime such as
16.4 LTS ML gets dedicated access mode and the classic Spark session that our
run used. A notebook on most other runtimes gets standard access mode and
reaches Spark through Spark Connect instead of a classic session.

Databricks recommends installing a `notebook-scoped library
<https://docs.databricks.com/aws/en/libraries/notebooks-python-libraries>`_ with
``%pip`` in a notebook's first cell and restarting Python in the next.

.. code-block:: python

   %pip install moderndid==<version>

.. code-block:: python

   dbutils.library.restartPython()

Since ``%pip`` doesn't restart Python by itself, the restart in the second cell
is what makes the notebook load the newer SciPy and statsmodels that the
install brings. Following Databricks' advice to pin versions, write the release
you want in place of ``<version>``. If you'd rather install a commit from
GitHub, as our run did, put its hash after the repository's address.

.. code-block:: python

   %pip install "moderndid @ git+https://github.com/jordandeklerk/moderndid.git@<commit>"

Although no fit has run with one yet, a `cluster library
<https://docs.databricks.com/aws/en/libraries/cluster-libraries>`_ installs
ModernDiD for every notebook and job on a cluster.

.. admonition:: Leave out the ``spark`` extra on Databricks
   :class: important

   Each Databricks runtime already provides the PySpark that the ``spark``
   extra would install a second time. Databricks says that installing PySpark
   in a serverless notebook stops the session.

Because no fit has run on serverless compute or through Databricks Connect
yet, the advice for both follows Databricks' documentation. A serverless
notebook needs `environment version
<https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/>`_
3 or later for its Python 3.12. Since serverless compute refuses to persist
data, a fit there falls back to a local checkpoint or else reads and shuffles
the data again at every step.

With Databricks Connect, your own machine runs your code and ``att_gt`` while
the compute works on the rows. That machine therefore needs the same ModernDiD
version as the compute and the Python 3.12 that Databricks requires for
`Python functions that run on its compute
<https://docs.databricks.com/aws/en/dev-tools/databricks-connect/python/udf>`_.
Because Databricks says that ``databricks-connect`` conflicts with a separate
PySpark, install ModernDiD on that machine without the ``spark`` extra and
uninstall any PySpark before you install ``databricks-connect``.

A cluster you manage
--------------------

Getting ModernDiD to the executors of a cluster you manage is up to you. Every
fit in the Spark guide runs unchanged once ModernDiD is in each worker's
Python. You can install it on each node or ship a packed environment to the
executors. PySpark's guide to `Python package management
<https://spark.apache.org/docs/latest/api/python/tutorial/python_packaging.html>`_
walks through ``pyFiles``, conda-pack, venv-pack, and PEX. Wherever you install
ModernDiD, pin PySpark to your cluster's Spark version rather than the latest
release that the ``spark`` extra installs. Since the Spark Connect client of
PySpark 3.5 requires NumPy 1 and NumPy 1 has no wheels for Python 3.13 or
later, a pin to PySpark 3.5 with the ``spark`` extra needs Python 3.12.

If the cluster runs a Spark Connect server, you can pass its address, such as
``sc://host:15002``, to ``SparkSession.builder.remote``. The executors then
have to match the ModernDiD and Python versions on your own machine because
your code and ``att_gt`` run there. A failure on a worker reaches you as
PySpark's own exception with the worker's Python traceback in its message.

On Spark 3.5, the tasks of a Spark Connect session see only the executors'
classpath and the session's own jars. A Spark 3.5 server that loads the Spark
Connect package with ``--packages`` therefore also needs the ``spark-connect``
jar in each worker's Spark ``jars`` folder or on
``spark.executor.extraClassPath``.

.. admonition:: Test a new cluster on a sample
   :class: tip

   Because a missing or mismatched install shows up only after Spark has
   shuffled the data, a large panel makes a slow first test of a new cluster.
   The :ref:`sample check <spark-sample-check>` shuffles only a sample and
   finds the same problem sooner.

Your own machine
----------------

Spark's local mode runs every fit in the Spark guide on one machine without a
cluster. That makes it the quickest way to try a Spark fit. Outside Databricks,
the ``spark`` extra installs the latest PySpark along with the Spark Connect
client.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add "moderndid[spark]"

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install "moderndid[spark]"

Because Spark itself runs on Java, you'll also need Java 17 or 21 for Spark 4,
or Java 8, 11, or 17 for Spark 3.5. Java 17 is the safe choice because it runs
both Spark 3.5 and Spark 4. Under Java 21, a fit works on Spark 4 but fails on
Spark 3.5 at the first step that uses Spark's Arrow functions.

.. code-block:: python

   from pyspark.sql import SparkSession

   # Start Spark on this machine with two worker threads.
   spark = SparkSession.builder.master("local[2]").getOrCreate()

.. admonition:: Point Spark at your Python
   :class: tip

   Local Spark starts its Python workers with the interpreter that
   ``PYSPARK_PYTHON`` names, or with ``python3`` from your path when that's
   unset. A local Spark Connect server tries ``PYSPARK_DRIVER_PYTHON`` before
   it falls back to ``python3``. If a local fit fails in a Python worker
   because of another Python version, a failed import of ModernDiD, or a crash
   inside PySpark itself, set ``PYSPARK_PYTHON`` to ``sys.executable`` before
   you create the session.

Notebooks in standard access mode, serverless compute, and Databricks Connect
all reach Spark through Spark Connect. A local Spark Connect server lets you
check your code on that path. Since a classic session started earlier in the
same process blocks that server even after you stop it on PySpark 3.5, start
the server in a fresh Python process. On PySpark 4,
``SparkSession.builder.remote("local[2]").getOrCreate()`` alone starts the
server in that process. Because PySpark 3.5 doesn't bundle the server, a
session there has to add the server's package from Maven Central.

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

If the fit stops with an error that starts "The Spark executors failed to
import moderndid with the error", an executor's Python has no ModernDiD or
can't import one of its dependencies. The error it quotes names the module that
failed to import. The section above for your setup shows how to install
ModernDiD in the Python that the executors run.

An error that starts "The Spark executors run moderndid" names the ModernDiD
and Python versions on the executors and on the driver. Since PySpark catches a
different Python first, this error usually means that the executors run another
ModernDiD version. Installing one ModernDiD version on the driver and on every
executor clears it.

When PySpark reports "Python in worker has different version", the executors
run another Python minor version than the driver. On your own machine, set
``PYSPARK_PYTHON`` as the tip above describes. A cluster needs the same Python
minor version on the driver and on every worker.

Java, Python, and PySpark versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If creating the session fails with "Java gateway process exited before sending
its port number", Spark most likely found no Java that it can run. Install
Java 17 and point ``JAVA_HOME`` at it before you create the session.

A Java error that says "sun.misc.Unsafe or java.nio.DirectByteBuffer.<init>(long,
int) not available" means that Spark 3.5 is running on Java 21. Point
``JAVA_HOME`` at a Java 17 installation or move to Spark 4.

An error that ends in "No module named 'distutils'" comes from PySpark 3.5 on
Python 3.12 or later. Installing setuptools on the driver and wherever the
executors run brings the module back.

If the fit stops before any job with "A fit of att_gt on a Spark DataFrame
needs pyspark 3.5 or later", upgrade the older PySpark that the message names.
On Databricks, a ``%pip install`` that reports "Ignored the following versions
that require a different python version" means that the runtime runs Python
3.11 or older. An install from GitHub on such a runtime says "requires a
different Python" instead. Moving the compute to Runtime 16.4 LTS or a later
supported runtime fixes either one.

The fit refuses the data
^^^^^^^^^^^^^^^^^^^^^^^^

An error that says "idname = 'county' is not numeric. Please convert it" means
that the unit identifier holds text. :ref:`Preparing your table
<spark-prepare>` shows how ``xxhash64`` turns such an identifier into an
integer. Text in the outcome, period, cohort, sampling weights, or a covariate
stops the fit with a message that names that column. A cluster column of state
names needs no conversion because the fit uses a cluster variable's values only
as labels.

When the message reads "idname='county' is a decimal(38,0) column" or "'state'
in clustervars is a decimal(38,0) column", the identifier or cluster has a
decimal type with fraction digits or more than 18 digits. If the column holds
only whole numbers, casting it to bigint before the fit lets it through.

Two errors that a Spark fit inherits from the in-memory checks suggest
``panel=False`` even though a Spark fit refuses it. One says "panel=True was
specified, but no units appear in multiple time periods". The other says "All
observations dropped while converting to balanced panel".

.. admonition:: These errors usually point to ``idname``
   :class: warning

   On Spark, either error usually means that ``idname`` doesn't pick out the
   same unit in every period. Since fitting a panel as a repeated cross-section
   changes the estimator, fix the identifier rather than collecting the rows
   for an in-memory fit with ``panel=False``.

Limits of the Spark session
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A warning that says "Spark refused to persist the table of units and to
checkpoint it" means that the session allowed neither. Although the fit still
runs, it then gathers each unit's rows again from the data at every step and
takes much longer on a large panel. Compute that allows a persist, such as
classic compute in dedicated access mode, gathers them only once.

If an error says "Total size of serialized results" and names
``spark.driver.maxResultSize``, the executors' sums outgrew what the driver
accepts. Since a running session can't change ``spark.driver.maxResultSize``,
pass fewer ``biters`` or raise that limit where the driver starts. That's the
Spark configuration of classic compute on Databricks, the server's
configuration under Spark Connect, and the session builder on your own machine.
Because serverless compute doesn't let you set the limit at all, fewer draws are
the only fix there.

Before Spark 4.1, Spark Connect also needs each task's sums to fit in one
message of at most 128 MiB by default. From Spark 4.1 on, Spark Connect splits
larger sums across several messages.

Aggregations that need new draws
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After a Spark fit, :func:`~moderndid.aggte` can only reuse the bootstrap draws
that the fit stored. An error that starts "boot=True needs bootstrap draws" or
"cband=True needs bootstrap draws" comes from a fit with ``boot=False`` and
``cband=False``. Errors that start "biters=2000 is more than the 1000 bootstrap
draws", "random_state=8 is not the seed 7", or "clustervars=[] asks for
unit-level bootstrap draws" mean that the ``aggte`` call asked for draws that
the fit didn't make. Each message says what to pass to ``aggte`` instead and
how to refit for the draws you asked for.
