.. _contributing:

=======================
Contributing guidelines
=======================

Before contributing
===================

Welcome to moderndid! Before contributing to the project, please ensure you understand our contribution process.

Contributing code
=================

1. Set up a Python development environment
   (advice: use `venv <https://docs.python.org/3/library/venv.html>`__,
   `virtualenv <https://virtualenv.pypa.io/>`__, or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__)
2. Install tox: ``python -m pip install tox``
3. Clone the repository: ``git clone https://github.com/YOUR_USERNAME/moderndid.git``
4. Start a new branch off main: ``git switch -c new-branch main``
5. Make your code changes
6. Check that your code follows the style guidelines of the project: ``tox -e check``
7. Build the documentation: ``tox -e docs``
8. Run the tests: ``tox -e core`` (fast test suite) or ``tox -e full`` (all tests)
9. Commit, push, and open a pull request!

Development environment
=======================

We use `pyproject.toml <../pyproject.toml>`__ to manage project dependencies and configuration. You can install the package in development mode with:

.. code-block:: bash

   pip install -e .

Tests
=====

Tests are located in the ``tests/`` directory and can be run with tox:

.. code-block:: bash

   tox -e core      # Fast test suite (recommended for development)
   tox -e full      # Full test suite (all tests including slow ones)

To run tests with coverage reporting:

.. code-block:: bash

   tox -e core-coverage
   tox -e full-coverage

Documentation
=============

Documentation is built using Sphinx and is located in the ``docs/`` directory. Build the documentation with:

.. code-block:: bash

   tox -e docs

Code style
==========

We follow the PEP 8 style guide. You can check your code with:

.. code-block:: bash

   tox -e check

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   self
   testing
