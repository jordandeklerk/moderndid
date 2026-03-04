.. _contributing:

########################
Contributing to ModernDiD
########################

Welcome to ModernDiD! We appreciate your interest in contributing to the project.
Whether you're fixing a bug, adding a new feature, improving documentation, or
helping with code review, your contributions are valuable.

If you have questions or run into issues, feel free to open an issue on
`GitHub <https://github.com/jordandeklerk/moderndid>`__.

Development process
===================

Here's a summary of the contribution workflow:

1. **Set up your environment**

   We use `pixi <https://pixi.sh/>`__ to manage development environments. Pixi
   handles Python, conda, and PyPI dependencies in a single lockfile so every
   contributor gets an identical setup.

   * `Install pixi <https://pixi.sh/latest/#installation>`__ if you don't have
     it already

   * Fork the repository on GitHub, then clone your fork::

      git clone https://github.com/your-username/moderndid.git
      cd moderndid

   * Add the upstream repository::

      git remote add upstream https://github.com/jordandeklerk/moderndid.git

   * Install the dev environment (this creates an isolated environment with all
     dependencies)::

      pixi install -e dev

   **Alternative: pip-based setup.** If you prefer not to use pixi, you can set
   up a virtual environment manually::

      python -m venv .venv && source .venv/bin/activate
      uv pip install -e ".[all,test,dev]"

2. **Develop your contribution**

   * Create a branch for your work. Use a descriptive name that reflects
     what you're working on::

      git checkout -b fix-bootstrap-standard-errors

   * Make your changes, writing tests for any new functionality

   * Commit locally as you progress using clear, descriptive commit messages

3. **Validate your changes**

   * Run the test suite to make sure your changes don't break anything::

      pixi run -e dev tests-core

   * Run the pre-commit hooks to check style::

      pixi run lint

   * If you've modified documentation, build and review it::

      pixi run docs

4. **Submit your contribution**

   * Push your changes to your fork::

      git push origin fix-bootstrap-standard-errors

   * Open a pull request on GitHub. Provide a clear title and description
     explaining what your changes do and why

5. **Review process**

   * Reviewers will provide feedback on your pull request. This is a
     collaborative process, and we review all contributions with the goal
     of improving the project together

   * Update your PR by making changes locally, committing, and pushing
     to the same branch. The PR will update automatically

   * CI tests must pass before your PR can be merged

Guidelines
==========

All code changes should include tests that verify the new behavior. See
:ref:`how to write tests <testing-how-to-write>` for details on conventions
including fixtures, parameterization, and
:ref:`numerical tolerances <testing-numerical-tolerances>`.

Public functions and classes should be documented with docstrings following the
NumPy docstring standard. This ensures consistency across the codebase and
enables automatic API documentation generation.

If you're adding a new estimator, follow the established architecture patterns
described in :doc:`architecture`. That document covers the preprocessing pipeline,
result object design, and the consistent API conventions that make ModernDiD
predictable for users.

All changes require review and approval before merging. If you don't receive
feedback within a week, feel free to ping the reviewers on the pull request.

Stylistic guidelines
====================

We follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__ style conventions.
Run ``pixi run lint`` to verify your code before submitting a pull request.

For imports, use the standard conventions of ``import numpy as np`` and
``import polars as pl``. Keep imports organized with standard library imports
first, followed by third-party packages, and then local imports.

Prefer clear, descriptive names over brevity. Code is read more often than it
is written, and a few extra characters in a variable name can save significant
time for future readers.

Code quality tools
==================

We use `ruff <https://docs.astral.sh/ruff/>`__ for linting and formatting. Ruff
is built in Rust and is very fast. It replaces flake8, pylint, black, and isort
in a single tool:

* **Linting**: Pyflakes, Pycodestyle, pydocstyle, bugbear, and more
* **Formatting**: Consistent code style (replaces black)
* **Import sorting**: Organized imports (replaces isort)

Ruff is configured in ``pyproject.toml``. To check your code manually::

   ruff check moderndid tests     # Lint
   ruff format moderndid tests    # Format

To auto-fix issues::

   ruff check --fix moderndid tests

Pre-commit hooks
----------------

We use `prek <https://github.com/j178/prek>`__ to manage pre-commit hooks. Prek
is built in Rust and is very fast. Hooks run automatically before each commit
to catch issues early.

To install the hooks after cloning the repository::

   prek install

The hooks will then run automatically on ``git commit``. To run all hooks
manually on all files::

   prek run --all-files

If you need to bypass hooks temporarily (not recommended)::

   git commit --no-verify

Test coverage
=============

Pull requests that modify code should include tests that sufficiently cover the new functionality.
Tests should aim to address edge cases and realistic scenarios. We aim for high test coverage
across the codebase.

Run the test suite locally before pushing::

   pixi run -e dev tests-core      # Fast test suite (recommended during development)
   pixi run -e dev tests-full      # Full test suite including slow tests

To run distributed test suites::

   pixi run -e dev tests-dask      # Dask distributed tests
   pixi run -e dev tests-spark     # Spark distributed tests

R validation tests
==================

ModernDiD includes a validation suite that compares Python estimates against
the original R packages (``did``, ``DRDID``, ``contdid``, ``triplediff``,
``HonestDiD``, ``DIDmultiplegtDYN``). These tests live in
`validation <https://github.com/jordandeklerk/moderndid/tree/main/tests/validation>`__ and run inside the ``validation`` pixi environment,
which is supported on **Linux and macOS** only (``linux-64``, ``osx-arm64``,
``osx-64``). Windows is not supported because several R dependencies
(``r-base``, ``r-did``, ``r-drdid``) lack reliable conda-forge Windows
builds.

Prerequisites
^^^^^^^^^^^^^

The validation environment requires a **Rust toolchain** (``cargo``,
``rustc``) to compile the R ``polars`` package from source. Install Rust
via `rustup <https://rustup.rs/>`__ if you don't have it already::

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

R itself and the R packages that are on conda-forge (``did``, ``DRDID``,
``jsonlite``) are installed automatically by pixi when you first use the
validation environment.

One-time setup
^^^^^^^^^^^^^^

Before running validation tests for the first time, install the CRAN-only
R packages::

   pixi run -e validation setup-r

This runs `setup.sh <https://github.com/jordandeklerk/moderndid/tree/main/scripts/setup.sh>`__, which installs ``contdid``,
``triplediff``, ``HonestDiD``, ``DIDmultiplegtDYN``, ``Rglpk``, and
``polars`` from CRAN and r-universe. The first run compiles everything from
source and can take a few minutes (most of that is the Rust build for
``polars``). Subsequent runs finish in seconds because the script only
installs packages that are missing.

Running tests
^^^^^^^^^^^^^

Run validation tests for individual estimators::

   pixi run -e validation did          # Staggered DiD
   pixi run -e validation drdid        # Doubly robust DiD
   pixi run -e validation didcont      # Continuous treatment
   pixi run -e validation didtriple    # Triple differences
   pixi run -e validation didinter     # Intertemporal treatment
   pixi run -e validation didhonest    # Sensitivity analysis

Or run the full validation suite::

   pixi run -e validation all

Each test file calls the corresponding R package via ``subprocess``, runs
the same estimation on the same data in both R and Python, and asserts that
the results match within numerical tolerance. Tests that depend on an R
package that failed to install are automatically skipped.


Building documentation
======================

Documentation is built using Sphinx and lives in the `docs <https://github.com/jordandeklerk/moderndid/tree/main/docs>`__ directory. The
documentation includes API references generated from docstrings, user guides,
and example notebooks.

To build and preview the documentation locally::

   pixi run docs

The built documentation will be available in ``docs/_build/``. Open
``index.html`` in a browser to review your changes before submitting.

Continuous integration
======================

Every pull request and push to ``main`` triggers automated checks via GitHub
Actions. Understanding what each workflow does helps you diagnose failures
quickly.

Primary test suite
-------------------

The ``test.yml`` workflow runs on every pull request and on pushes to ``main``
(excluding changes under ``docs/``). It has four jobs.

- The ``test`` job runs the core test suite (excluding slow and distributed
  tests) across Python 3.11, 3.12, and 3.13. This is the most common job to
  check when your PR fails.
- The ``dask`` job runs the Dask distributed tests on Python 3.12 and 3.13
  with a 120-second timeout per test.
- The ``spark`` job runs the Spark distributed tests on Python 3.12 and 3.13.
  It also provisions Java 17, which Spark requires.
- The ``coverage`` job runs the full test suite (including slow tests) on
  ``main`` only. It does not run on PRs.

All jobs upload coverage reports to Codecov.

Weekly full test suite
-----------------------

The ``test-full.yml`` workflow runs every Sunday at 02:00 UTC and can be
triggered manually. It exercises the full test suite including slow tests that
are skipped in normal CI. Check this workflow if a release candidate fails
tests that passed in regular CI.

Package publishing
-------------------

The ``publish.yml`` workflow triggers when a ``v*`` tag is pushed. It builds
the wheel and source distribution with build provenance attestation, then
publishes to PyPI via Trusted Publishing (OIDC). The publish step requires
maintainer approval through the ``publish`` GitHub environment. See
:doc:`releasing` for the full release process.

Post-release changelog
-----------------------

The ``post-release.yml`` workflow runs when a GitHub Release is published. It
regenerates ``CHANGELOG.md`` from all releases using
``changelog-from-release`` and opens a PR with the updated file.

Nightly upstream testing
-------------------------

The ``nightly.yml`` workflow runs every Sunday at 03:00 UTC (one hour after the
full suite) and can be triggered manually. It installs nightly wheels of numpy,
scipy, polars, pyarrow, and statsmodels from the
`scientific-python-nightly-wheels <https://anaconda.org/scientific-python-nightly-wheels>`__
index and runs the core test suite against them.

Failures here are expected and informational. They flag upcoming breaking
changes in upstream packages before those changes are released. This workflow
does not block PRs.

To run the same check locally::

   tox -e nightly

Security scanning
------------------

The ``codeql.yml`` workflow runs CodeQL static analysis for Python on pushes
to ``main``, pull requests against ``main``, and weekly on Monday at midnight
UTC.

Diagnosing CI failures
-----------------------

When CI fails on your PR, start by clicking through to the failing job in the
GitHub Actions tab. The most common causes are

- Test failures in the ``test`` job. The output shows which test failed and
  why. Run the same test locally with ``pixi run -e dev tests-core`` to
  reproduce.
- Lint failures from ruff or mypy. Run ``pixi run lint`` locally to see the
  same errors.
- Timeout failures in Dask or Spark jobs (120-second limit). These usually
  indicate a test that hangs or does excessive computation on the driver.
- Platform differences. CI runs on Ubuntu while you may develop on macOS.
  Floating-point behavior can differ slightly between platforms. See
  :doc:`debugging` for guidance on numerical tolerances.

Registering new public API
==========================

ModernDiD uses a lazy-loading import system in `__init__.py <https://github.com/jordandeklerk/moderndid/tree/main/moderndid/__init__.py>`__ so
that ``import moderndid`` is fast even though the package has many optional
dependencies. When you add a new public function, class, or module, you need
to register it in this system.

The lazy loader resolves names through three dictionaries checked in order.

``_lazy_imports``
   Maps names to their source module for functions and classes that are always
   available (no optional dependencies). For example,
   ``"att_gt": "moderndid.did.att_gt"`` means that ``moderndid.att_gt`` will
   import ``att_gt`` from ``moderndid.did.att_gt`` on first access.

``_optional_imports``
   Maps names to a ``(module_path, extra_name)`` tuple for items that require
   an optional dependency. If the dependency is not installed, accessing the
   name raises an ``ImportError`` with a helpful message telling the user
   which extra to install. For example,
   ``"cont_did": ("moderndid.didcont.cont_did", "didcont")`` means the user
   sees ``uv pip install 'moderndid[didcont]'`` in the error.

``_submodules``
   A set of submodule names that can be accessed as ``moderndid.<submodule>``.
   When accessed, the full submodule is imported.

To register a new always-available function, add an entry to ``_lazy_imports``
and add the name to ``__all__``. For a new optional function, add it to
``_optional_imports`` with the correct extra name and add it to ``__all__``.

.. note::

   If a function name shadows a submodule name (as ``drdid`` the function
   shadows ``drdid`` the submodule), the function must be imported eagerly
   at the top of ``__init__.py`` rather than through the lazy loader. See
   the existing ``from moderndid.drdid.drdid import drdid`` line for this
   pattern.

Dependency management
=====================

Version constraints
-------------------

Core dependencies are pinned with minimum versions in ``pyproject.toml``
(e.g., ``numpy>=1.22.0``, ``polars>=1.0.0``). These minimums represent the
oldest versions we test against and support. When bumping a minimum version,
ensure the full CI matrix still passes since all Python versions in the matrix
use the same dependency floor.

Optional dependencies are grouped under extras in ``pyproject.toml`` and
mirrored as pixi features in ``pixi.toml``. The ``all`` extra includes
everything except GPU support.

Adding a new dependency
------------------------

Before adding a dependency, consider whether it is truly necessary. Each new
dependency increases installation complexity and potential for version
conflicts.

If the dependency is needed for only one estimator or feature, make it an
optional extra rather than a core dependency. Follow the existing pattern
in ``pyproject.toml`` to define a new optional group, then register the
affected functions in ``_optional_imports`` in ``moderndid/__init__.py`` so
users get a clear error message when the dependency is missing. Add the
dependency to the appropriate pixi feature in ``pixi.toml`` and the
corresponding tox testenv in ``tox.ini``.

Python version support
-----------------------

ModernDiD supports Python 3.11 and above (``requires-python = ">=3.11"``).
CI tests against 3.11, 3.12, and 3.13. Do not use language features that
require a Python version above 3.11 (e.g., ``type`` statement from 3.12)
without gating them behind a version check.
