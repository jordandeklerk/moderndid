######################
Testing a contribution
######################

The test you write should catch the problem that brought you to the code.
An incorrect comparison group, a misaligned influence
function, and a confidence interval built from the wrong variance each need a
check of the affected behavior. We'll begin with the smallest case that exposes
that problem and add numerical evidence where the calculation needs it.

Choosing what to run
====================

In the development environment from :doc:`guide`, you can select the affected
file, one function, or a group of tests that share the behavior you're changing.
We'll use existing tests for group-time effects below to show the selection patterns
you can apply to your own work.

.. code-block:: bash

   pixi run -e dev pytest tests/did/test_att_gt.py -m "not slow" -vv
   pixi run -e dev pytest \
       tests/did/test_att_gt.py::test_att_gt_estimation_methods -vv
   pixi run -e dev pytest tests/did/ -k "weights" -m "not slow" -vv

Tests that take longer, such as some bootstrap and numerical validation
checks, carry the ``slow`` marker. If your change affects that behavior, run the
relevant slow test explicitly rather than assuming the faster tests cover it.
The ``tests-core`` and ``tests-full`` Pixi tasks select the fast suite and the
suite without marker filtering respectively. Both cover much more of the
package than a focused development check.

.. _testing-how-to-write:

Writing a regression test
=========================

Place a test near the behavior it checks in ``tests/<module>/``. Use a
standalone test function whose name describes the result you expect and put
reusable data or setup in that module's ``conftest.py``. The fixture below
gives the county employment tests access to the same dataset while keeping data
loading separate from the calculation under test.

.. code-block:: python

   import pytest

   from moderndid import load_mpdta


   @pytest.fixture
   def mpdta_data():
       return load_mpdta()

A bug fix needs an assertion that fails before the fix and passes after it. New
calculations can be checked against a hand-computed small case, a property
implied by the method, or an independently obtained reference result. Merely
checking that the function returns an object won't catch a plausible estimate
computed for the wrong sample.

The test below checks that reordering the rows preserves group-time effects
when the county and year identifiers remain unchanged. We turn off bootstrap
inference and simultaneous bands so the comparison checks row handling without
introducing a random draw.

.. code-block:: python

   import numpy as np

   from moderndid import att_gt


   def test_att_gt_row_order(mpdta_data):
       spec = {
           "yname": "lemp",
           "tname": "year",
           "idname": "countyreal",
           "gname": "first.treat",
           "est_method": "reg",
           "boot": False,
           "cband": False,
       }
       expected = att_gt(data=mpdta_data, **spec)
       shuffled = mpdta_data.sample(fraction=1, shuffle=True, seed=42)
       actual = att_gt(data=shuffled, **spec)

       np.testing.assert_array_equal(actual.groups, expected.groups)
       np.testing.assert_array_equal(actual.times, expected.times)
       np.testing.assert_allclose(
           actual.att_gt, expected.att_gt, rtol=1e-10, atol=1e-12
       )

When several methods or input forms should meet the same expectation,
``pytest.mark.parametrize`` lets you express that relationship in one test.
Different expectations belong in separate tests so you can tell which behavior
changed when one fails. Keeping imports at the top of the file and fixtures in
``conftest.py`` leaves the test body free to show its input, call, and
assertion.

Checking numerical results
==========================

.. _testing-numerical-tolerances:

Choosing numerical tolerances
-----------------------------

Use the absolute tolerance to control differences near zero and the relative
tolerance to scale with the reference value's magnitude. Set both explicitly in
``np.testing.assert_allclose`` and choose them for the quantity you're
checking. A deterministic difference of means usually permits a tighter
comparison than a standard error from an iterative fit.

For a bootstrap or simulation, use an explicit ``random_state`` when the
function accepts it and a seeded ``np.random.default_rng`` for generated data.
Although reproducibility helps diagnose a discrepancy, identical seeds do not
guarantee identical draws across different algorithms.

.. admonition:: A wider tolerance is not a fix
   :class: warning

   A comparison of independent bootstrap runs needs a tolerance supported by
   their sampling variation, not a wider bound chosen only after the test
   fails.

If a discrepancy is larger than expected, trace the sample selection,
normalization, weight convention, and inference settings before changing the
tolerance. For help locating the source of a numerical discrepancy, the
:ref:`debugging guide <debugging>` follows the calculation through the
implementation.

Running reference validation
----------------------------

The tests in ``tests/validation/`` compare estimates and inference with
independent reference calculations on the same inputs. On Linux and macOS,
their environment provides the reference tools and the optional packages these
comparisons need, such as scikit-learn for dynamic balancing and pyfixest for
extended TWFE. Since the setup task compiles some dependencies from source, it
needs a working Rust toolchain and can take longer than installing the ordinary
development environment.

.. code-block:: bash

   pixi install -e validation
   pixi run -e validation setup-r
   pixi run -e validation did

The ``did`` task selects ``tests/validation/test_r_did.py`` to check the
staggered adoption estimator. Other available tasks include ``drdid``,
``didcont``, ``didtriple``, ``didinter``, ``didhonest``, and ``npiv``. You can
select an individual test through ``pixi run -e validation pytest`` when the
whole estimator's validation file is unnecessary. The extended TWFE tests have
no task of their own and run from ``tests/validation/test_r_etwfe.py`` through
the same ``pytest`` command. The dynamic balancing tests run only when
``DYNBALANCING_PATH`` names a checkout of its reference implementation. The
setup script checks installed versions and can update outdated dependencies
on later runs as well as installing missing ones.

.. admonition:: Read validation skips
   :class: important

   A missing reference dependency can skip a numerical comparison rather
   than fail it. Check the skip reasons before treating a completed
   validation run as evidence that your estimates agree.

The :ref:`worked comparisons <r-comparisons>` cover most of these estimators
on application data. Each comparison sets the
tolerances of its own assertions in its code. The
:ref:`numerical checks <benchmarks-numerical-checks>` in the benchmarking guide
give the tolerances of the timing runner.

Handling warnings and optional dependencies
===========================================

When a warning is part of the behavior you're changing, assert it with
``pytest.warns`` and a message match. If a test is about another behavior and a
particular warning is expected, use a narrowly matched
``pytest.mark.filterwarnings`` marker. A broad warning filter can hide new
problems in the calculation you're trying to check.

Tests that require an optional dependency can use
``tests.helpers.importorskip`` at module level. Since a missing dependency
normally skips that file, keep those tests separate from checks that should run
with the base package. GPU tests are the exception because they reach CuPy
through :ref:`fixtures of their own <testing-gpu>`. When the fixtures in a
folder need an extra, its ``conftest.py`` calls ``importorskip`` at the top. A
run without that extra then skips the whole folder. Passing that folder to
pytest by name stops the run with an error instead, because pytest loads its
``conftest.py`` before collection starts.

Setting the ``MODERNDID_REQUIRE_ALL_DEPS`` environment variable makes
``importorskip`` raise the import error instead of skipping. Because a skipped
file reports no failure, set it for tests whose packages your environment
should provide, such as the ``didml`` tests in the development environment.

.. code-block:: bash

   MODERNDID_REQUIRE_ALL_DEPS=1 pixi run -e dev pytest tests/didml -vv

Use ``pytest.mark.skipif`` for a known environment limitation and explain it in
the reason.

.. admonition:: Make expected failures strict
   :class: tip

   An expected failure should point to an unresolved issue and use
   ``strict=True``. An unexpected pass then calls attention to a marker that
   may no longer belong there.

.. _testing-gpu:

Testing on a GPU
================

Although most machines that run the test suite have no GPU, the CuPy backend
and the estimators' GPU routes need tests that run on a real one. We write each
GPU test once so that it skips on those machines and runs on your own NVIDIA
GPU or on a cloud GPU from Modal.

Writing a GPU test
------------------

``pyproject.toml`` registers the ``gpu`` marker for tests that need a GPU.
Unless CuPy is installed and ``cupy.is_available()`` finds a CUDA GPU,
``tests/conftest.py`` skips each marked test and reports the reason
``needs CuPy and a CUDA GPU``. Neither the development environment nor the pull
request checks install CuPy. A test module therefore never imports it and takes
it from a fixture instead.

A check against an independent reference, such as statsmodels, NumPy's linear
algebra, or a closed form, takes the ``xp`` fixture. With only its CuPy case
marked, a test that takes the fixture runs under NumPy on every machine and
under CuPy only where CuPy can reach a GPU. The fixture activates the backend
for the whole test and yields its array module.

The test below builds its inputs with ``xp.asarray`` and compares the
coefficients that ``cupy_wls`` fits with those from statsmodels. Because the
function should return arrays of the active backend, the test asserts the type
of its coefficients before ``to_numpy`` from ``moderndid.cupy.backend`` moves
them to the host for the comparison.

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

       beta, _ = cupy_wls(
           xp.asarray(outcome), xp.asarray(design), xp.asarray(weights)
       )
       expected = sm.WLS(outcome, design, weights=weights).fit().params

       assert isinstance(beta, xp.ndarray)
       np.testing.assert_allclose(to_numpy(beta), expected, rtol=1e-10)

A test that compares a GPU run with the CPU run of the same call carries the
marker alone. It runs the call under the default NumPy backend and again inside
``use_backend("cupy")``, or with ``backend="cupy"`` where the function takes
that argument. Estimators, bootstraps, and the warnings a fit raises are
usually checked this way. The ``cp`` fixture supplies CuPy for device inputs
and type checks. Giving such a test the ``xp`` fixture would only add a NumPy
case that compares NumPy with itself.

Allowing for rounding and random draws
--------------------------------------

Results on a GPU differ from the CPU's in their last bits because cuBLAS,
cuSOLVER, and CuPy's reductions round differently from LAPACK and NumPy. Every
tolerance in a GPU test should leave room for that difference. Inputs also
need a clear margin from any threshold the code applies, such as the cutoff
that decides whether a design counts as singular.

Since CuPy's random generator draws different numbers from NumPy's for the
same seed, a bootstrap on the GPU can't match the CPU draw for draw. Its test
compares the distribution of the draws through summaries such as their
support, their mean, or a standard error checked with a loose tolerance. A
step that runs on the CPU by design, such as a weighted bootstrap, can still
match the NumPy run exactly when a test calls it directly.

Running GPU tests
-----------------

On Linux with an NVIDIA GPU, you can run the marked tests in the ``gpu`` Pixi
environment. It adds CuPy to the development environment and exists only for
``linux-64``. Because a run in which every GPU test skipped proves nothing about
the GPU code, the command below passes ``-ra`` to list each skipped test with
its reason.

.. code-block:: bash

   pixi run -e gpu pytest -m gpu -ra

Without a GPU of your own, you can run the same tests on a T4 GPU through
Modal. The commands below install its command line client, sign you in, and
start the run from the repository root.

.. code-block:: bash

   uv tool install modal
   modal setup
   modal run .github/scripts/gpu_tests.py

Modal bills the run's GPU time to your own account. The script builds the test
environment from ``uv.lock`` and mounts ``moderndid/``, ``tests/``, and
``pyproject.toml`` from your checkout. Before running ``pytest -m gpu``, it
checks that CuPy can see the GPU and fails when CuPy can't.

.. _testing-spark:

Testing on Spark
================

A fit on a Spark DataFrame has to reproduce the in-memory result on a cluster
whose tasks run in other processes. Since the shard engine never imports
PySpark, we test it in the ordinary suite. Its parity tests in
``tests/did/test_shards.py`` run it through ``LocalExecutor``, as the
:ref:`architecture guide <architecture-shards>` describes.

The tests in ``tests/spark/`` cover the whole path on Spark. They compare the
results of ``att_gt``, ``aggte``, and ``honest_did`` on a Spark DataFrame with
the in-memory results and confirm that the bootstrap draws don't change with
the partitioning. Other tests check the unit table that ``build_unit_table``
builds, the executor that runs each pass, and whether a Spark fit raises the
same errors and warnings as an in-memory fit.

Writing a Spark test
--------------------

Every test module in ``tests/spark/`` carries the ``spark`` marker that
``pyproject.toml`` registers and skips itself when PySpark is missing. Unlike a
folder whose ``conftest.py`` calls ``importorskip`` at the top,
``tests/spark/conftest.py`` imports PySpark only when it is installed. Passing
the folder to pytest by name without PySpark therefore skips its tests instead
of stopping the run. The session fixture in the folder's ``conftest.py`` also
needs a Java runtime on ``JAVA_HOME`` or ``PATH`` and skips without one, unless
the tests connect to a running server at an ``sc://`` address.

A parity test fits the same Polars data in memory and on Spark. The
``to_spark`` fixture turns the data into a Spark DataFrame through Arrow. Since
PySpark 3.5 can neither build a DataFrame from a pyarrow Table nor collect a
DataFrame with ``toArrow()``, a test that needs either step calls ``to_spark``
or the ``to_arrow`` fixture rather than the PySpark method.
``record_warnings`` returns each fit with the warnings it raised. Once the test
has checked that both fits raised the same warnings, ``assert_same_fit``
compares their results at a tolerance such as 1e-12.

.. admonition:: Keep the fixture in step with ``spark_att_gt``
   :class: important

   The ``att_gt_on_shards`` fixture in ``tests/conftest.py`` runs a fit
   through ``LocalExecutor``. Since it builds ``estimation_params`` and
   ``cluster_counts`` by hand the way ``spark_att_gt`` does, a change to how
   ``spark_att_gt`` builds either one needs the same change in the fixture.

Most Spark tests fit a unit table of one partition, since ``build_unit_table``
merges its partitions to about a megabyte each and a test panel rarely fills
more than one. To reach several partitions, repartition the table that the
``unit_table_of`` fixture returns, as ``test_executor.py`` does. Only a
``SparkExecutor`` built with ``whole_clusters=True`` needs that repartition to
hash on the cluster. A panel whose unit table passes a megabyte, such as
``large_clustered_panel``, gets more than one partition from
``build_unit_table`` itself. The eight shuffle partitions of the session
fixture change how long a fit takes rather than what it estimates.

A bootstrap test needs reference values that the engine didn't compute. Since
a fit on shards draws different signs from an in-memory fit, its bootstrap
standard errors don't match the in-memory ones. The tests in
``tests/did/test_shards.py`` instead multiply ``keyed_rademacher`` signs on
``canonical_keys`` of the unit or cluster identifiers by the in-memory
influence function, summed within clusters for a clustered fit. They finish
those draws with ``_mboot_from_draws`` and compare the engine's standard errors
and critical value with the result at 1e-12. A test that compares the engine
only with itself would pass a mistake that both sides share. The Spark tests
compare their bootstrap with a fit through ``LocalExecutor`` because these
reference tests already pin that fit.

Running the Spark tests
-----------------------

On Linux or macOS, you can run the Spark tests in the ``spark`` Pixi
environment. It provides PySpark 4.2 and Java 21 on Python 3.13. By default the
session fixture starts a classic ``local-cluster[2,1,1024]`` session. Since its
two executors have JVMs of their own, the tasks of every pass run outside the
driver's JVM as they would on a cluster.

Setting the ``MODERNDID_SPARK_REMOTE`` environment variable switches the
fixture to Spark Connect. The value ``local-cluster[2,1,1024]`` starts a local
Spark Connect server over the same kind of cluster. An address such as
``sc://localhost:15002`` connects to a server that is already running. Each
mode needs a pytest run of its own because a classic session and a Spark
Connect session can't share one process.

.. code-block:: bash

   pixi run -e spark pytest tests/spark -ra
   MODERNDID_SPARK_REMOTE="local-cluster[2,1,1024]" pixi run -e spark pytest tests/spark -ra

As in the GPU command, ``-ra`` lists any skipped test with its reason.

The ``spark35`` Tox environment runs the same tests on PySpark 3.5 with numpy
below 2, pandas below 2.2, and pyarrow below 16, as on Databricks Runtime 16.4
LTS. Since the tests of older pandas in ``tests/core/test_dataframe.py`` skip on
any pandas with the Arrow PyCapsule interface, the environment runs that file
too. It needs three things that the Pixi runs above don't.

- Java 17 on ``JAVA_HOME``, because Spark 3.5 can't run its Arrow functions on
  Java 21.
- The tox-uv plugin, because ``constrain_package_deps = true`` puts the pins
  into a constraint file and pip rejects the extra of ``pyspark[connect]``
  there.
- Network access on the first run through Spark Connect, when Spark downloads
  the server's package from Maven Central into ``~/.ivy2``.

The environment runs on Python 3.12, the Python of Databricks Runtime 16.4 LTS.
The commands below run it in a classic session and through Spark Connect.

.. code-block:: bash

   uvx --with tox-uv tox -e spark35
   MODERNDID_SPARK_REMOTE="local-cluster[2,1,1024]" uvx --with tox-uv tox -e spark35

Since PySpark 3.5 doesn't bundle the Spark Connect server, the session fixture
adds its package through ``spark.jars.packages`` on 3.5. A ``local[2]`` server
like the one in the :ref:`Spark setup guide <spark-setup>` needs nothing more, because its
executor shares the driver's JVM. On the executors of a ``local-cluster``
server, Spark 3.5 runs the tasks of each Connect session with a class loader
that sees only the executor's classpath and the session's own jars. The
fixture therefore also puts the jar from Ivy's default folder,
``~/.ivy2/jars``, on ``spark.executor.extraClassPath``. A custom
``spark.jars.ivy`` would move the jar out of that folder and break those runs.

Testing on a Docker cluster
---------------------------

The executors of a local cluster share the Python environment of your test
session. The Docker cluster in ``tests/spark/cluster/`` runs them in an image
of their own, as a deployed cluster would. Its compose file starts a master,
two workers, and a Spark Connect server. The image runs Python 3.13 like the
``spark`` Pixi environment, since the executors need the same Python minor
version as the test session. Because the image installs moderndid from your
checkout, ``--build`` rebuilds it with the code you're testing.

.. code-block:: bash

   docker compose -f tests/spark/cluster/compose.yaml up -d --build --wait
   MODERNDID_SPARK_REMOTE=sc://localhost:15002 pixi run -e spark pytest tests/spark -ra
   docker compose -f tests/spark/cluster/compose.yaml down

The ``--wait`` flag makes the first command return once the cluster's health
checks pass. On macOS, the compose commands need a container runtime such as
Colima. Start it with ``colima start --cpu 4 --memory 8`` before the first
command and stop it with ``colima stop`` after the last.

Understanding automated checks
==============================

``.github/workflows/test.yml`` defines pull request checks and a matrix of
Python 3.12, 3.13, and 3.14 on Ubuntu and Windows. Each matrix job runs ``tox``
with ``TEST_SUITE=core`` to select the fast suite. The ``[gh-actions]`` sections
of ``tox.ini`` map that value and the job's Python version to the
``core-coverage`` environment.

Outside the matrix, ``tox -e check`` gets a job of its own on Ubuntu with
Python 3.12. That environment runs the pre-commit hooks over every tracked
file, as ``pixi run lint`` does, before it builds the package. The
``polars-floor`` job runs ``tox -e polars-floor`` on Python 3.12 to test the
fast suite against the oldest Polars that ``pyproject.toml`` allows. Raising
that floor means changing the ``polars==1.38.*`` pin in ``tox.ini`` too. A
separate job runs ``full-coverage`` on Python 3.14 for pushes to ``main``.

Because the other jobs install every extra, two more jobs on Python 3.14 run the
fast suite with smaller installs. The ``minimal`` environment installs the
package with its test tools and no extras. The ``minimal-plots`` environment
adds the ``plots`` extra to that install. In both, every test that needs a
missing extra skips through ``importorskip``.

The ``spark`` and ``spark-cluster`` jobs run the Spark tests on Ubuntu. The
``spark`` job runs them on a local cluster, once in a classic session and once
through Spark Connect, on each of two PySpark lines. Its PySpark 4.2 entries
run ``tox -e spark`` on Python 3.13 with Java 21. Its PySpark 3.5 entries run
``tox -e spark35`` on Python 3.12 with Java 17. The ``spark-cluster`` job
starts the Docker cluster from ``tests/spark/cluster/``, runs ``tox -e spark``
against ``sc://localhost:15002``, and stops the cluster even when a test fails.
Since both tox environments set ``MODERNDID_REQUIRE_ALL_DEPS``, a test that
would skip for lack of PySpark or Java fails instead.

The ``spark`` environment installs the newest PySpark, 4.2 at the time of
writing. The Docker cluster runs the 4.2.0 that its image pins. Because the
``spark35`` entries test the 3.5 floor of the ``spark`` extra, a call to a
PySpark API that 3.5 lacks fails there unless the code keeps a 3.5 path, as
the rules for Spark code in the :ref:`architecture guide <architecture-shards>`
require. The ``record_warnings`` fixture ignores two warnings that come from
PySpark rather than from a fit. Classic sessions of PySpark 3.5 through 4.1
leave the socket of each collect for the garbage collector to close.
PySpark 3.5 also checks the versions of pandas and pyarrow with the deprecated
version classes of ``distutils``.

The weekly full-suite workflow runs on Sunday at 02:00 UTC and can also be
started manually. An upstream dependency workflow runs at 03:00 UTC on Sunday
despite its ``nightly`` name. It selects ``tox -e nightly`` to test against
nightly and prerelease builds of scientific Python dependencies. You can run the
same command locally wherever Tox is installed. CodeQL scans Python on pull
requests, pushes to ``main``, and its weekly schedule.

The GPU tests workflow in ``.github/workflows/gpu.yml`` runs the Modal script
from a GitHub runner on Sunday at 04:00 UTC and can also be started manually.
It reads the Modal token from the ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``
secrets of a GitHub environment named ``gpu``. That environment's deployment
branch rule allows only ``main`` to use them. A maintainer also sets a monthly
budget for the Modal workspace in Modal's settings. Because pull requests don't
run this workflow, a reviewer runs ``modal run`` on any branch that changes GPU
code.

A failed job gives the command and environment you need to reproduce it.
Include that evidence and your focused local checks in the pull request so a
reviewer can distinguish a tested behavior from one that still needs an
environment-specific check. :doc:`reviewing` explains how we read those checks
alongside the method and implementation.
