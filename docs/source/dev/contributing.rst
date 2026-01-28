.. _contributing:

########################
Contributing to ModernDiD
########################

Welcome to ModernDiD! We appreciate your interest in contributing to the project.
Whether you're fixing a bug, adding a new feature, improving documentation, or
helping with code review, your contributions are valuable.

This guide walks you through the contribution process. If you have questions or
run into issues, feel free to open an issue on
`GitHub <https://github.com/jordandeklerk/moderndid>`__.

Development process
===================

Here's a summary of the contribution workflow:

1. **Set up your environment**

   * Set up a Python development environment using
     `venv <https://docs.python.org/3/library/venv.html>`__,
     `virtualenv <https://virtualenv.pypa.io/>`__, or
     `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

   * Install tox for managing test environments::

      uv pip install tox

   * Fork the repository on GitHub, then clone your fork::

      git clone https://github.com/your-username/moderndid.git
      cd moderndid

   * Add the upstream repository::

      git remote add upstream https://github.com/jordandeklerk/moderndid.git

   * Install the package in development mode::

      uv pip install -e .

2. **Develop your contribution**

   * Create a branch for your work. Use a descriptive name that reflects
     what you're working on::

      git checkout -b fix-bootstrap-standard-errors

   * Make your changes, writing tests for any new functionality

   * Commit locally as you progress using clear, descriptive commit messages

3. **Validate your changes**

   * Run the test suite to make sure your changes don't break anything::

      tox -e core

   * Check that your code follows the project's style guidelines::

      tox -e check

   * If you've modified documentation, build and review it::

      tox -e docs

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
Run ``tox -e check`` to verify your code before submitting a pull request.

For imports, use the standard conventions of ``import numpy as np`` and
``import polars as pl``. Keep imports organized with standard library imports
first, followed by third-party packages, and then local imports.

Prefer clear, descriptive names over brevity. Code is read more often than it
is written, and a few extra characters in a variable name can save significant
time for future readers.

Test coverage
=============

Pull requests that modify code should include tests that sufficiently cover the new functionality.
Tests should aim to address edge cases and realistic scenarios. We aim for high test coverage
across the codebase.

Run the test suite locally before pushing::

   tox -e core      # Fast test suite (recommended during development)
   tox -e full      # Full test suite including slow tests

To run tests with coverage reporting::

   tox -e core-coverage
   tox -e full-coverage

Building documentation
======================

Documentation is built using Sphinx and lives in the ``docs/`` directory. The
documentation includes API references generated from docstrings, user guides,
and example notebooks.

To build and preview the documentation locally::

   tox -e docs

The built documentation will be available in ``.tox/docs/docs_out/``. Open
``index.html`` in a browser to review your changes before submitting.

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   self
   architecture
   testing
