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

Your code can reach Spark in a classic session, where the driver controls Spark
from its own machine, or through Spark Connect, where it sends each query over
the network to a server that may run elsewhere. The sections below cover both
on Databricks, on a cluster you manage, and on your own machine. Once your
setup works, the :ref:`Spark guide <spark>` takes you through a fit.

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
the first cell and restart Python in the next, as Databricks recommends. On
classic compute in dedicated access mode, that install reaches the executors as
well as the notebook.

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
wheels for Python 3.13 or later. It also needs setuptools on the driver and
wherever the executors run, because PySpark 3.5 imports ``distutils``, a module
that Python 3.12 no longer includes.

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
