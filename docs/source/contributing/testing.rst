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
through :ref:`fixtures of their own <testing-gpu>`.

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
