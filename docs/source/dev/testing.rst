==================
Testing ModernDiD
==================

How to run the test suite
=========================

The recommended way to run the test suite is via ``pixi``, which manages isolated
environments and ensures consistent results across different machines. Pixi
handles dependencies automatically, allowing you to test the library with
different combinations of optional dependencies from the same development
environment.

To run the fast test suite (recommended for development):

.. code-block:: bash

   pixi run -e dev tests-core

To run the full test suite (all tests including slow ones):

.. code-block:: bash

   pixi run -e dev tests-full

To run the Dask distributed tests:

.. code-block:: bash

   pixi run -e dev tests-dask

To run the Spark distributed tests:

.. code-block:: bash

   pixi run -e dev tests-spark

To run the R validation tests (requires R with ``did``, ``DRDID``, and related
packages installed via the ``validation`` environment):

.. code-block:: bash

   pixi run -e validation tests-validation

To run style checks:

.. code-block:: bash

   pixi run -e check lint

.. _testing-how-to-write:

How to write tests
==================

Consistent test conventions make the codebase easier to navigate and help
contributors understand what to expect when reading or writing tests. The
patterns here have evolved from practical experience with the test suite.

Imports and optional dependencies
---------------------------------

ModernDiD supports several optional dependencies, and tests need to handle cases
where these dependencies may not be installed. Use the ``importorskip`` helper
function from `tests/helpers.py <https://github.com/jordandeklerk/moderndid/tree/main/tests/helpers.py>`__ for any import outside of the Python standard
library plus NumPy:

.. code-block:: python

   import numpy as np

   from tests.helpers import importorskip

   pd = importorskip("pandas")

   # in the code use pd.DataFrame, pd.Series as usual

When ``importorskip`` encounters a missing dependency, it skips all tests in that
file. Because of this behavior, you should organize tests so that core functionality
tests live in their own files with no optional dependency imports, while tests that
require optional dependencies go in separate files.

Test structure and style
------------------------

Each test file should have a module-level docstring describing what it tests.
Individual test functions should not have docstrings because the function name
itself should clearly communicate the test's purpose. A well-named test function
like ``test_drdid_panel_with_weights`` is more useful than a generic name with
a docstring explanation:

.. code-block:: python

   """Tests for the DRDID panel estimator."""

   def test_drdid_panel_with_weights():
       # No docstring needed, the function name explains the test
       ...

Prefer standalone test functions over test classes. Classes add organizational
overhead without much benefit for most tests. Use classes only when you have a
logical group of related tests that benefit from the organizational clarity, such
as testing different aspects of the same component:

.. code-block:: python

   class TestValidators:
       def test_column_validator(self):
           ...

       def test_treatment_validator(self):
           ...

Fixtures belong in conftest.py
------------------------------

All pytest fixtures should be defined in ``conftest.py`` files, never in test
files themselves. This keeps test files focused on test logic and makes fixtures
discoverable and reusable. Each submodule has its own
``tests/<submodule>/conftest.py`` for fixtures specific to that module (e.g.,
`tests/did/conftest.py <https://github.com/jordandeklerk/moderndid/tree/main/tests/did/conftest.py>`__):

.. code-block:: python

   # tests/did/conftest.py

   @pytest.fixture(scope="module")
   def mpdta_data():
       return load_mpdta()

   @pytest.fixture
   def rng():
       return np.random.default_rng(42)

Parameterization
----------------

When you need to test the same logic with different inputs, use
``pytest.mark.parametrize`` instead of writing multiple similar test functions.
Parameterization reduces code duplication and makes it clear that the tests are
variations of the same scenario. It also makes adding new test cases trivial:

.. code-block:: python

   @pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
   def test_att_gt_estimation_methods(est_method, mpdta_data):
       result = att_gt(data=mpdta_data, est_method=est_method, ...)
       assert result.att is not None

   @pytest.mark.parametrize(
       "value,expected",
       [
           (1e-20, 0.0),
           (0.5, 0.5),
       ],
   )
   def test_round_eps(value, expected):
       assert round_eps(value) == expected

Random number generation
------------------------

Reproducibility is essential for debugging test failures. NumPy recommends using
the `Generator <https://numpy.org/doc/stable/reference/random/generator.html>`_
interface rather than the legacy ``np.random`` functions. Always use
``np.random.default_rng()`` with an explicit seed so that tests produce the same
results every time:

.. code-block:: python

   def test_drdid_panel_basic():
       rng = np.random.default_rng(42)

       d = rng.binomial(1, 0.5, n_units)
       y0 = rng.standard_normal(n_units)
       ...

Marking slow tests
------------------

Tests that take significant time to run should be marked as slow so developers
can skip them during rapid iteration. This is particularly important for tests
that involve bootstrap inference or validation against R implementations, which
can take several minutes or more.

Use module-level marking when all tests in a file are slow:

.. code-block:: python

   import pytest

   pytestmark = pytest.mark.slow

   def test_expensive_r_validation():
       ...

For individual slow tests in an otherwise fast file, use the decorator directly:

.. code-block:: python

   @pytest.mark.slow
   def test_bootstrap_with_many_iterations():
       ...

Run tests excluding slow ones with ``pytest -m "not slow"``.

Warning handling
----------------

Some code paths intentionally raise warnings, and tests need to handle these
appropriately. When testing code that raises expected warnings, suppress them
with ``pytest.mark.filterwarnings`` to keep test output clean. This prevents
expected warnings from cluttering the test output while still allowing unexpected
warnings to surface:

.. code-block:: python

   @pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
   def test_single_treated_unit():
       ...

When you need to verify that code correctly raises a warning, use ``pytest.warns``
to assert that the expected warning appears:

.. code-block:: python

   def test_asymmetric_matrix_warning():
       with pytest.warns(UserWarning, match="Matrix sigma not exactly symmetric"):
           validate_symmetric_psd(asymmetric_matrix)

Skipping and expected failures
------------------------------

When a test cannot run under certain conditions (missing dependency, wrong
platform, known bug), use pytest markers to handle it gracefully rather than
letting it fail with a confusing error.

To skip a test conditionally based on the environment:

.. code-block:: python

   import sys

   @pytest.mark.skipif(sys.platform == "win32", reason="R validation not supported on Windows")
   def test_r_validation():
       ...

To mark a test as a known failure that you expect to be fixed later:

.. code-block:: python

   @pytest.mark.xfail(reason="known precision issue with small sample sizes, see #42")
   def test_small_sample_bootstrap():
       ...

An ``xfail`` test that unexpectedly passes will be reported as ``XPASS``,
alerting you that the underlying issue may have been resolved and the marker
can be removed.

Running specific tests
-----------------------

During development you rarely need to run the full suite. Pytest provides
several ways to narrow down what runs.

.. code-block:: bash

   # Run a single test file
   pytest tests/did/test_att_gt.py -vv

   # Run a single test function
   pytest tests/did/test_att_gt.py::test_basic_panel -vv

   # Run tests matching a keyword expression
   pytest tests/did/ -k "bootstrap and not slow" -vv

   # Run tests for a specific estimator module
   pytest tests/didcont/ -vv

.. _testing-numerical-tolerances:

Numerical tolerances
--------------------

Floating-point arithmetic means that numerical results rarely match exactly.
When comparing results, choose tolerances appropriate to what you're testing.
Deterministic calculations should match to high precision, while stochastic
methods like bootstrap naturally have more variation.

.. code-block:: python

   # High precision for deterministic calculations
   np.testing.assert_allclose(py_result, r_result, rtol=1e-5, atol=1e-6)

   # More lenient for standard errors
   np.testing.assert_allclose(py_se, r_se, rtol=1e-3, atol=1e-4)

   # Even more lenient for bootstrap/Monte Carlo results
   assert 0.7 < se_ratio < 1.3

When using ``np.testing.assert_allclose``, prefer passing both ``rtol`` and
``atol`` explicitly rather than relying on defaults. The defaults
(``rtol=1e-7``, ``atol=0``) are often too strict for econometric
computations.

Testing with different backends
-------------------------------

Some tests need to verify behavior across multiple backends (NumPy vs CuPy,
local vs Dask). Use ``importorskip`` to gate backend-specific tests so they
are skipped gracefully when the backend is not available.

.. code-block:: python

   from tests.helpers import importorskip

   cp = importorskip("cupy")

   def test_gpu_att_gt():
       from moderndid.cupy.backend import use_backend

       with use_backend("cupy"):
           result = att_gt(data=df, boot=False)
       assert result.att_gt is not None

Organize backend-specific tests in separate files (e.g., ``test_att_gt_gpu.py``)
so that a missing optional dependency skips the entire file rather than
producing scattered skips throughout the main test file.
