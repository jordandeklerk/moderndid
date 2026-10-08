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

The ``slow`` marker identifies tests that take longer, including some bootstrap
and numerical validation checks. If your change affects that behavior, run the
relevant slow test explicitly rather than assuming the faster tests cover it.
The ``tests-core`` and ``tests-full`` Pixi tasks select the fast suite and the
suite without marker filtering respectively; they cover much more than a
focused development check.

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
guarantee identical draws across different algorithms. A comparison of
independent bootstrap runs needs a tolerance supported by their sampling
variation, not a wider bound chosen only after the test fails.

If a discrepancy is larger than expected, trace the sample selection,
normalization, weight convention, and inference settings before changing the
tolerance. For help locating the source of a numerical discrepancy, the
:ref:`debugging guide <debugging>` follows the calculation through the
implementation.

Running reference validation
----------------------------

The tests in ``tests/validation/`` compare estimates and inference with
independent reference calculations on the same inputs. Their environment has
additional tools and supports Linux and macOS. Since the setup task compiles
some dependencies from source, it needs a working Rust toolchain and can take
longer than installing the ordinary development environment.

.. code-block:: bash

   pixi install -e validation
   pixi run -e validation setup-r
   pixi run -e validation did

The ``did`` task selects ``tests/validation/test_r_did.py`` to check the
staggered adoption estimator. Other available tasks include ``drdid``,
``didcont``, ``didtriple``, ``didinter``, ``didhonest``, and ``npiv``. You can
select an individual test through ``pixi run -e validation pytest`` when the
whole estimator's validation file is unnecessary. The setup script checks
installed versions and can update outdated dependencies on later runs as well
as installing missing ones.

.. admonition:: Read validation skips
   :class: important

   A missing reference dependency can skip a numerical comparison rather
   than fail it. Check the skip reasons before treating a completed
   validation run as evidence that your estimates agree.

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
with the base package. GPU tests also need a usable device and runtime;
importing the backend alone does not establish that a computation can run.

Use ``pytest.mark.skipif`` for a known environment limitation and explain it in
the reason. An expected failure should point to an unresolved issue and use
``strict=True`` so an unexpected pass calls attention to a marker that may no
longer belong there.

Understanding automated checks
==============================

``.github/workflows/test.yml`` defines pull request checks and a matrix of
Python 3.12 and 3.13 on Ubuntu and Windows. The environments selected by its
Tox invocation depend on ``tox.ini`` and the interpreter mappings defined
there. A separate job runs ``full-coverage`` on Python 3.14 for pushes to
``main``.

The weekly full-suite workflow runs on Sunday at 02:00 UTC and can also be
started manually. An upstream dependency workflow runs at 03:00 UTC on Sunday
despite its ``nightly`` name. It selects ``tox -e nightly`` to try prerelease
scientific Python dependencies; the command is available locally wherever Tox
is installed. CodeQL scans Python on pull requests, pushes to ``main``, and its
weekly schedule.

A failed job gives the command and environment you need to reproduce it.
Include that evidence and your focused local checks in the pull request so a
reviewer can distinguish a tested behavior from one that still needs an
environment-specific check. :doc:`reviewing` explains how we read those checks
alongside the method and implementation.
