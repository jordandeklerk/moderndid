######################
Testing a contribution
######################

The tests in moderndid check that each estimator computes what its paper
defines. The mistakes that matter most raise no error and return plausible
numbers, such as an effect measured against the wrong comparison group or a
standard error from a misaligned influence function. A test can also pass for
the wrong reason, when its expected value came from the code it checks or when
it skipped because a package was missing.

This guide shows you how to write tests that catch those mistakes. We'll write
one for :func:`~moderndid.att_gt`, the estimator of group-time average
treatment effects, on the minimum wage panel of 500 US counties that
:func:`~moderndid.load_mpdta` loads. Later sections cover tolerances, reference
validation, optional packages, the tests that need a GPU or a Spark cluster,
and what each automated check runs.

Running the tests that cover a change
=====================================

A focused run tells you within seconds whether the behavior you're changing
still holds. Beyond the whole file that the :doc:`setup guide <guide>` runs,
you can name one test after its file and two colons or select the tests whose
names contain a keyword.

.. code-block:: bash

   pixi run -e dev pytest tests/did/test_att_gt.py::test_att_gt_estimation_methods -vv
   pixi run -e dev pytest tests/did/ -k "weights" -m "not slow" -vv

When your change touches what a slow test checks, run that test by name,
because ``-m "not slow"`` leaves out every test that carries the ``slow``
marker.

The ``tests-core`` and ``tests-full`` tasks of the ``dev`` environment run the
fast suite and the full suite across every core of your machine. A focused run
is usually all you need locally, since the pull request checks run the fast
suite on every supported Python version.

.. _testing-how-to-write:

Writing a regression test
=========================

A regression test pins down the behavior that your change fixes or adds. It
fails on the code from before your change and passes on the code after it.

The tests of ``moderndid/<family>/<module>.py`` live in
``tests/<family>/test_<module>.py``, even for a module in a subfolder of its
family. A test module opens with a docstring such as
``"""Tests for group-time average treatment effects."""`` and its imports.
Write each test as a plain function whose name says the behavior it checks.
Keep tests out of classes and leave out comments and docstrings.

Data that several tests share comes from a fixture, a function that pytest
calls to build an input for each test that names it as an argument. Fixtures,
each with a one-line docstring, go in the ``conftest.py`` of their folder, the
file that pytest reads before the tests beside it. Look for one that already
builds what you need, such as the fixture for the county data below.

.. code-block:: python

   import pytest

   from moderndid import load_mpdta


   @pytest.fixture
   def mpdta_data():
       """Minimum wage panel of 500 counties from 2003 to 2007."""
       return load_mpdta()

The test below checks that :func:`~moderndid.att_gt` gives the same estimates
when the rows of the county data arrive in a random order. Because
preprocessing sorts the rows before any estimation, both fits must give
identical estimates and standard errors. Log population as a covariate puts
the outcome regression and the propensity score of the default doubly robust
estimator through the check.

.. code-block:: python

   import numpy as np

   from moderndid import att_gt


   def test_att_gt_gives_the_same_estimates_for_shuffled_rows(mpdta_data):
       spec = {
           "yname": "lemp",
           "tname": "year",
           "idname": "countyreal",
           "gname": "first.treat",
           "xformla": "~ lpop",
       }
       shuffled = mpdta_data.sample(fraction=1.0, shuffle=True, seed=42)

       expected = att_gt(mpdta_data, **spec)
       actual = att_gt(shuffled, **spec)

       assert not shuffled.equals(mpdta_data)
       np.testing.assert_array_equal(actual.groups, expected.groups)
       np.testing.assert_array_equal(actual.times, expected.times)
       np.testing.assert_array_equal(actual.att_gt, expected.att_gt)
       np.testing.assert_array_equal(actual.se_gt, expected.se_gt)

The first assertion makes sure that the shuffle moved the rows, because a
shuffle that returned the original order would let the test pass without
checking anything. Use ``pytest.mark.parametrize`` when several estimation
methods or input types share one expectation. Different expectations get
separate tests, because a failure should name the behavior that changed.

.. _testing-numerical-tolerances:

Choosing numerical tolerances
=============================

A tolerance says how far a result may stray from its expected value before the
test fails. Two runs that must agree exactly, such as the shuffled and original
fits above, compare with ``np.testing.assert_array_equal``. A value computed
another way compares with ``np.testing.assert_allclose`` and an explicit
``rtol`` and ``atol``, because the two calculations add their terms in
different orders. For a deterministic calculation, set the tolerance to what
the code achieves, usually ``rtol=1e-12`` or ``1e-10``.

Never paste moderndid's own output into a test as the expected value, because
such a test only checks that the output hasn't changed. Rebuild the value from
pieces that the result exposes, compute it by hand on a small case, or check a
property that the method implies. In ``tests/did/test_att_gt.py``, for example,
``test_att_gt_without_the_bootstrap_sums_the_influence_function_within_clusters``
sums each unit's influence function within its state with ``np.bincount`` and
compares the standard errors it gives at ``rtol=1e-12``.

.. admonition:: A wider tolerance hides the mistake
   :class: warning

   When a comparison fails by more than rounding, trace the sample, the
   weights, and the inference settings before you touch the tolerance.
   Loosening it until the test passes hides the mistake the test exists to
   catch. The :ref:`debugging guide <debugging>` shows how to find where two
   results part.

To make every run see the same numbers, build test data with a ``gen_*``
simulator from ``moderndid.core.data`` at ``random_state=42`` or with
:func:`~moderndid.load_mpdta`. Anything else random comes from a seeded
``np.random.default_rng`` rather than the global ``np.random`` functions.
Every call that draws a bootstrap takes an explicit ``random_state``. That
includes an ``att_gt`` fit whose result reaches :func:`~moderndid.aggte`,
because the aggregation inherits the fit's seed and draws for its uniform
bands under the default ``cband=True``.

.. _testing-validation:

Running reference validation
============================

The tests in ``tests/validation/`` compare moderndid's estimates and standard
errors with those of independent implementations of the same methods on the
same data. The ``validation`` Pixi environment provides those implementations
on Linux and macOS. Its ``setup-r`` task installs the reference packages that
the environment lacks and updates outdated ones. One of them compiles from
source and needs a Rust toolchain.

.. code-block:: bash

   pixi install -e validation
   pixi run -e validation setup-r
   pixi run -e validation did

The ``did`` task runs ``tests/validation/test_r_did.py`` for the staggered
adoption estimator. Tasks named ``drdid``, ``didcont``, ``didtriple``,
``didinter``, ``didhonest``, and ``npiv`` run the files of the other
estimators. The ``all`` task runs all of them, along with the extended TWFE
file that has no task of its own. Run that file, or a single test from any
file, through ``pixi run -e validation pytest``. The dynamic balancing tests
run only when the ``DYNBALANCING_PATH`` environment variable names a checkout
of their reference implementation.

.. admonition:: Read the skip reasons
   :class: important

   A reference package that failed to install skips its comparisons rather
   than failing them. Add ``-ra`` to the command and read the skip reasons
   before you treat a passing validation run as evidence that the estimates
   agree.

The :ref:`worked comparisons <r-comparisons>` run most of these estimators
side by side on application data.

Checking errors, warnings, and optional packages
================================================

Tests check the messages of errors and warnings as well as their types, since
users read them. Match an error with ``pytest.raises(ValueError, match=...)``
on a distinctive part of its message. Wrap that part in ``re.escape`` when it
holds brackets or other regular expression syntax. A bare
``pytest.raises(Exception)`` would also pass on an ``AttributeError`` from a
missing export.

When a warning is the behavior you're testing, assert it with
``pytest.warns(UserWarning, match=...)``. A test about something else names
the one warning it expects in a ``pytest.mark.filterwarnings`` marker, because
a broader filter could hide a new warning from the calculation.

.. admonition:: Prove that a guard needs no silencing
   :class: tip

   Library code guards a calculation rather than silencing NumPy's
   floating-point warnings with ``np.errstate``. The ``forbid_errstate``
   fixture in ``tests/conftest.py`` turns any call to ``np.errstate`` into a
   failure for the test that takes it.

A test that needs an optional package imports it through ``importorskip`` from
``tests.helpers`` at the top of its module. Without the package, that call
skips every test in the module. A folder whose fixtures need the package makes
the call in its ``conftest.py`` instead. A run of the whole suite then skips
the folder. Naming that folder on the command line stops pytest with an error,
because pytest imports a ``conftest.py`` named this way before it can skip
anything. In the development environment, which has no PySpark, the 39 tests
in ``tests/spark/test_preprocess.py`` shrink to a single line of output.

.. code-block:: bash

   pixi run -e dev pytest tests/spark/test_preprocess.py -ra -q

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         =========================== short test summary info ============================
         SKIPPED [1] tests/spark/test_preprocess.py:17: could not import 'pyspark': No module named 'pyspark'
         1 skipped in 0.03s

Setting the ``MODERNDID_REQUIRE_ALL_DEPS`` environment variable turns that
skip into a ``ModuleNotFoundError`` at collection. Set it wherever the tests
should find every package they need, as the Spark jobs on pull requests do. A
limit of the environment rather than a missing package, such as a pandas too
new for a conversion test, gets ``pytest.mark.skipif`` with a reason that names
the limit.

.. _testing-gpu:

Testing on a GPU
================

Several estimators in moderndid can run their array work on an NVIDIA GPU
through CuPy, a library that offers NumPy's interface on the GPU. We write each
GPU test to skip on the many machines without a GPU and to run in full wherever
CuPy reaches one.

Writing a GPU test
------------------

The ``gpu`` marker, registered in ``pyproject.toml``, flags a test that needs
a GPU. Unless CuPy is installed and ``cupy.is_available()`` finds a CUDA
device, ``tests/conftest.py`` skips each marked test with the reason
``needs CuPy and a CUDA GPU``. Neither the development environment nor the
pull request checks install CuPy. Test modules therefore never import it and
take it from a fixture instead.

A check against an independent reference, such as statsmodels or a closed
form, takes the ``xp`` fixture. The fixture runs the test once under each
backend and hands it the array module of the active one. The NumPy case runs
on every machine, since only the CuPy case carries the marker. The test below
checks the weighted least squares fit of ``cupy_wls``, the regression routine
of the GPU backend, against statsmodels. It also asserts that the coefficients
come back as arrays of the active backend before ``to_numpy`` moves them to the
host. A fuller version with three designs sits in
``tests/core/test_gpu_regression.py``.

.. code-block:: python

   import numpy as np
   import statsmodels.api as sm

   from moderndid.cupy.backend import to_numpy
   from moderndid.cupy.regression import cupy_wls


   def test_cupy_wls_matches_statsmodels(xp):
       rng = np.random.default_rng(42)
       design = np.column_stack([np.ones(200), rng.normal(size=200)])
       outcome = design @ np.array([1.0, 2.0]) + rng.normal(size=200)
       weights = rng.uniform(0.5, 2.0, size=200)

       beta, _ = cupy_wls(xp.asarray(outcome), xp.asarray(design), xp.asarray(weights))
       expected = sm.WLS(outcome, design, weights=weights).fit().params

       assert isinstance(beta, xp.ndarray)
       np.testing.assert_allclose(to_numpy(beta), expected, rtol=1e-10)

A test that compares a GPU run with the CPU run of the same call, as most GPU
tests of estimators and bootstraps do, carries the ``gpu`` marker alone. It
runs the call under the default NumPy backend and again inside
``use_backend("cupy")``, the context that moves moderndid to the GPU, or with
``backend="cupy"`` where the function takes that argument. Such a test takes
CuPy from the ``cp`` fixture, because the ``xp`` fixture would only add a case
that compares NumPy with itself.

GPU results differ from the CPU's in their last bits, because the GPU's
libraries round differently from LAPACK and NumPy. Every tolerance in a GPU
test leaves room for that difference. Inputs also stay clear of thresholds the
code applies, such as the cutoff that decides whether a design is singular.
CuPy's random generator draws different numbers from NumPy's for the same
seed. A bootstrap test on the GPU therefore compares summaries of the draws,
such as their support, their mean, or a standard error at a loose tolerance. A
weighted bootstrap stays on the CPU under every backend and can still match the
NumPy run exactly when a test calls it directly.

Running GPU tests
-----------------

The ``gpu`` Pixi environment, defined only for ``linux-64``, adds CuPy to the
development environment for machines with an NVIDIA GPU. A run in which every
GPU test skipped proves nothing. The command below passes ``-ra`` to list each
skipped test with its reason.

.. code-block:: bash

   pixi run -e gpu pytest -m gpu -ra

Without a GPU of your own, you can run the same tests on a T4 GPU through
Modal, a service that runs Python functions on cloud GPUs and bills your
account for the time. Run the commands below from the root of your checkout.

.. code-block:: bash

   uv tool install modal
   modal setup
   modal run .github/scripts/gpu_tests.py

The script builds its environment from ``uv.lock`` and mounts ``moderndid/``,
``tests/``, and ``pyproject.toml`` from your checkout. It checks that CuPy can
see the GPU before it runs ``pytest -m gpu``. A broken CUDA setup then fails
instead of passing with every test skipped.

.. admonition:: Run GPU changes on a GPU
   :class: important

   A change to the CuPy backend or to an estimator's GPU route needs a run on
   a GPU, your own or Modal's, before it merges, as no pull request check runs
   the GPU tests.

.. _testing-spark:

Testing on Spark
================

A fit of :func:`~moderndid.att_gt` on a Spark DataFrame never gathers its rows
on one machine. In each of several passes, every block of the data, called a
shard, sums its contribution where it lives. The driver adds up those sums, as
the guide to :ref:`fits on shards <dev-shards>` shows. As the engines behind
those passes never import PySpark, the ordinary suite runs them through
``LocalExecutor``, an executor that holds the shards in memory. The parity
tests in ``tests/did/test_shards.py`` and ``tests/did/test_row_shards.py``
compare those engines with the in-memory fit.

.. admonition:: Keep the shard fixtures in step with spark_att_gt
   :class: important

   The ``att_gt_on_shards`` fixture in ``tests/conftest.py`` and the
   ``att_gt_on_row_shards`` fixture in ``tests/did/conftest.py`` run a fit
   through ``LocalExecutor``. Both build ``estimation_params`` and
   ``cluster_counts`` by hand the way ``spark_att_gt``, the function that
   ``att_gt`` calls for a Spark DataFrame, builds them. A change to either one
   in ``spark_att_gt`` needs the same change in both fixtures.

The tests in ``tests/spark/`` run the whole path on a Spark session. They
compare :func:`~moderndid.att_gt`, :func:`~moderndid.aggte`, and
:func:`~moderndid.honest_did` on a Spark DataFrame with their in-memory
results and confirm that the bootstrap draws don't depend on how the rows fall
into partitions. Others cover the tables that a Spark fit builds and the
executor that runs its passes.

Writing a Spark test
--------------------

A parity test fits the same Polars data in memory and on Spark, as the first
test in ``tests/spark/test_att_gt.py`` does for each estimation method,
comparison group, and base period. The ``to_spark`` fixture turns the data into
a Spark DataFrame through Arrow. The ``record_warnings`` fixture returns each
fit along with the warnings it raised. Once the warnings match,
``assert_same_fit`` compares the cells, estimates, and standard errors of the
two fits.

.. code-block:: python

   @pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
   @pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
   @pytest.mark.parametrize("base_period", ["varying", "universal"])
   def test_att_gt_on_spark_matches_att_gt(
       mpdta_data, to_spark, record_warnings, assert_same_fit, est_method, control_group, base_period
   ):
       spec = {
           "yname": "lemp",
           "tname": "year",
           "idname": "countyreal",
           "gname": "first.treat",
           "xformla": "~lpop",
           "est_method": est_method,
           "control_group": control_group,
           "base_period": base_period,
           "random_state": 7,
       }
       frame = to_spark(mpdta_data)

       memory, memory_warnings = record_warnings(att_gt, mpdta_data, **spec)
       distributed, spark_warnings = record_warnings(att_gt, frame, **spec)

       assert spark_warnings == memory_warnings
       assert_same_fit(distributed, memory, 1e-12)

PySpark 3.5 can neither build a DataFrame from a pyarrow Table nor collect one
with ``toArrow()``. A test calls the ``to_spark`` or ``to_arrow`` fixture for
those steps rather than the PySpark method. ``record_warnings`` also leaves out
two warnings that PySpark raises on its own, about unclosed sockets and its
deprecated ``distutils`` version check.

Before its first pass, a Spark fit lays the data out as a table with one row
per unit, from ``build_unit_table``, or one row per observation, from
``build_row_table``. With both builders merging partitions until each holds
about a megabyte, most test tables have one partition. To reach several,
repartition the table that the ``unit_table_of`` or ``row_table_of`` fixture
returns, as ``tests/spark/test_executor.py`` does. A ``SparkExecutor`` built
with ``whole_clusters=True`` needs that repartition on the cluster column, as
the section on :ref:`tasks of a Spark pass <dev-spark-tasks>` explains. A panel
as large as
the ``large_clustered_panel`` fixture gets several partitions from
``build_unit_table`` itself.

A bootstrap test needs reference values that the engine didn't compute,
because a test that compared the engine only with itself would pass a mistake
that both sides share. The tests in ``tests/did/test_shards.py`` apply the
``keyed_rademacher`` signs that the engine draws for each unit or cluster to
the in-memory influence function, summed within clusters for a clustered fit.
``_mboot_from_draws`` turns those draws into standard errors and a critical
value that the engine's must match at ``1e-12``. The Spark tests compare their
bootstrap with a fit through ``LocalExecutor`` because these reference tests
already pin that fit.

.. admonition:: Multipliers on shards
   :class: note

   Each multiplier on shards depends only on the seed, the draw, and the key of
   its unit or cluster, as :ref:`random draws on shards <dev-shard-draws>`
   explains. An in-memory bootstrap draws its signs in the order of the units
   instead and can't serve as the reference.

Running the Spark tests
-----------------------

On Linux or macOS, the ``spark`` Pixi environment runs the Spark tests with
PySpark 4.2 and Java 21 on Python 3.13. By default the session fixture starts a
classic session on ``local-cluster[2,1,1024]``, a cluster of two executors with
one core and 1024 MB each. Those executors run in Java processes of their own,
as they would on a real cluster.

Every module in ``tests/spark/`` carries the ``spark`` marker and skips itself
when PySpark is missing. Unlike a folder whose ``conftest.py`` calls
``importorskip`` at the top, ``tests/spark/conftest.py`` imports PySpark only
when it's installed. Naming the folder on the command line without PySpark
therefore skips its tests rather than stopping the run. The session fixture
also needs Java on ``JAVA_HOME`` or ``PATH`` and skips without it, unless the
tests connect to a running server at an ``sc://`` address.

Spark Connect sends a client's DataFrame operations to a separate Spark server,
the way notebooks on Databricks serverless compute reach Spark. Setting the
``MODERNDID_SPARK_REMOTE`` environment variable switches the session fixture to
Spark Connect. The value ``local-cluster[2,1,1024]`` starts a local Spark
Connect server over the same kind of cluster. An address such as
``sc://localhost:15002`` connects to a server that's already running. Each mode
needs a pytest run of its own, since a classic session and a Spark Connect
session can't share a process.

.. code-block:: bash

   pixi run -e spark pytest tests/spark -ra
   MODERNDID_SPARK_REMOTE="local-cluster[2,1,1024]" pixi run -e spark pytest tests/spark -ra

Tox builds an isolated virtual environment for each named setup in
``tox.ini``. Its ``spark35`` environment runs the same tests on PySpark 3.5 with
numpy below 2, pandas below 2.2, and pyarrow below 16, close to the packages of
Databricks Runtime 16.4 LTS. It also runs the conversion tests for pandas older
than 2.2 in ``tests/core/test_dataframe.py``, the one place where they don't
skip. The environment uses Python 3.12, where numpy below 2 still has wheels,
and needs three things that the Pixi runs above don't.

- Java 17 on ``JAVA_HOME``, because Spark 3.5 can't run its Arrow functions on
  Java 21.
- The tox-uv plugin that lets Tox install with uv, because pip rejects the
  extra of ``pyspark[connect]`` in the constraint file that
  ``constrain_package_deps = true`` writes.
- Network access on the first run through Spark Connect, when Spark downloads
  the server from Maven Central, the public repository of Java packages, into
  ``~/.ivy2``. A custom ``spark.jars.ivy`` breaks those runs, because the
  session fixture hands the executors the server's jar from ``~/.ivy2/jars``.

.. code-block:: bash

   uvx --with tox-uv tox -e spark35
   MODERNDID_SPARK_REMOTE="local-cluster[2,1,1024]" uvx --with tox-uv tox -e spark35

Testing on a Docker cluster
---------------------------

The executors of a local cluster share the Python environment of your test
session. The Docker cluster in ``tests/spark/cluster/`` runs them from an image
of their own, as a deployed cluster would. Its compose file starts a master,
two workers, and a Spark Connect server on Python 3.13, the minor version of
the ``spark`` Pixi environment, because Spark Connect runs the client's Python
functions on the workers. The ``--build`` flag below rebuilds the image with
the code you're testing, since the image installs moderndid from your checkout.

.. code-block:: bash

   docker compose -f tests/spark/cluster/compose.yaml up -d --build --wait
   MODERNDID_SPARK_REMOTE=sc://localhost:15002 pixi run -e spark pytest tests/spark -ra
   docker compose -f tests/spark/cluster/compose.yaml down

On macOS, the compose commands need a container runtime such as Colima. Start
it with ``colima start --cpu 4 --memory 8`` before the first command and stop
it with ``colima stop`` after the last.

Understanding automated checks
==============================

Every push to a pull request starts the jobs in ``.github/workflows/test.yml``
on GitHub Actions. Each job runs a Tox environment from ``tox.ini``, as the
table below shows.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Job
     - What it runs
   * - ``test``
     - The fast suite in ``core-coverage``, on Python 3.12, 3.13, and 3.14 on
       Ubuntu and Windows
   * - ``check``
     - The pre-commit hooks over every tracked file, as ``pixi run lint`` runs
       them, and a build of the package
   * - ``polars-floor``
     - The fast suite on the oldest Polars that ``pyproject.toml`` allows
   * - ``minimal``
     - The fast suite with no extras in ``minimal`` and with only the ``plots``
       extra in ``minimal-plots``
   * - ``spark``
     - The Spark tests on a local cluster, classic and through Spark Connect,
       in ``spark`` with PySpark 4.2 and in ``spark35`` with PySpark 3.5
   * - ``spark-cluster``
     - The Spark tests on the Docker cluster through Spark Connect, in
       ``spark``
   * - ``coverage``
     - The full suite with coverage in ``full-coverage``, on pushes to
       ``main`` only

To reproduce a failed job, run its environment on your machine with
``uvx --with tox-uv tox -e`` and the environment's name. Only the ``minimal``
environments catch an import of a package that a smaller install lacks,
because every other job that runs the fast suite installs all the extras.

.. admonition:: Move the Polars pin with the floor
   :class: important

   Raising the Polars floor in ``pyproject.toml`` means changing the
   ``polars==1.38.*`` pin that the ``polars-floor`` environment installs from
   ``tox.ini``.

The ``spark35`` entries of the ``spark`` job test the 3.5 floor of the
``spark`` extra. A call to a PySpark API that 3.5 lacks fails there unless the
code keeps a 3.5 path, as the :ref:`rules for Spark code <dev-spark-rules>`
require.

Three workflows run on Sundays and can also be started by hand from the
Actions tab. The full test suite starts at 02:00 UTC and runs every test, the
slow ones included, on the Python versions and operating systems of the
``test`` job. Another job of that workflow runs every example notebook with the
package installed from the checkout. The nightly workflow starts at 03:00 UTC
and tests the fast suite against nightly and prerelease builds of the
scientific Python packages. The GPU tests workflow starts at 04:00 UTC and runs
the Modal script with a token that only the ``main`` branch may use. CodeQL,
GitHub's code scanner, checks the Python code on pull requests, on pushes to
``main``, and once a week.

List the focused checks you ran in the pull request. A reviewer can then tell a
tested behavior from one that still needs a check in another environment.
:doc:`reviewing` explains how we weigh those checks alongside the method and
the code.
