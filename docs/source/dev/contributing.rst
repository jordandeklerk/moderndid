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
:doc:`testing` for details on how to write tests that follow our conventions,
including guidance on fixtures, parameterization, and numerical tolerances.

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

Building documentation
======================

Documentation is built using Sphinx and lives in the ``docs/`` directory. The
documentation includes API references generated from docstrings, user guides,
and example notebooks.

To build and preview the documentation locally::

   pixi run docs

The built documentation will be available in ``docs/_build/``. Open
``index.html`` in a browser to review your changes before submitting.
