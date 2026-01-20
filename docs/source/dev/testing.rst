==================
Testing moderndid
==================

How to run the test suite
=========================

The recommended way to run the test suite is to do it via ``tox``.
Tox manages the environment, its env variables and the command to run
to allow testing the library with different combinations of optional dependencies
from the same development env.

To run the fast test suite (recommended for development):

.. code-block:: bash

   tox -e core

To run the full test suite (all tests including slow ones):

.. code-block:: bash

   tox -e full

To run tests with coverage reporting:

.. code-block:: bash

   tox -e core-coverage
   tox -e full-coverage

To run style checks:

.. code-block:: bash

   tox -e check

How to write tests
==================

Use the ``importorskip`` helper function from ``tests/helpers.py`` for any import outside of
the Python standard library plus NumPy. For example:

.. code-block:: python

   import numpy as np

   from .helpers import importorskip

   pd = importorskip("pandas")

   #... in the code use pd.DataFrame, pd.Series as usual

As ``importorskip`` will skip all tests in that file, tests should be divided into
files with tests of the core functionality always being in their own file
with no optional dependencies import, and tests that require optional dependencies
in a separate file.

About moderndid testing
=========================

The test suite is structured to ensure both the core functionality and the optional
dependencies integrations work as expected.

The ``importorskip`` helper function from ``tests/helpers.py`` is used when importing
optional dependencies so that tests are skipped if a dependency is not available.
In addition, the env variable ``moderndid_REQUIRE_ALL_DEPS`` can be set to disable this behavior
and ensure uninstalled dependencies raise an error.

When using ``tox -e full`` all optional dependencies are installed,
and ``moderndid_REQUIRE_ALL_DEPS`` is set to ensure all tests in the test suite run.
However, ``tox -e core`` only installs the core dependencies and doesn't set the env variable,
which ensures that the minimal install is viable and works as expected.

On GitHub Actions, the full test suite is run for all supported Python versions
and the core test suite for one Python version.
The test configuration is defined by the combination of ``tox.ini`` and ``.github/workflows/test.yml``.
