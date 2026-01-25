============
Installation
============

Installing moderndid
--------------------

The only prerequisite for installing moderndid is Python 3.10 or later.

From PyPI
^^^^^^^^^

The recommended way to install moderndid is from PyPI using
`uv <https://docs.astral.sh/uv/>`_ or pip:

.. code-block:: console

    uv pip install moderndid

Or with pip:

.. code-block:: console

    pip install moderndid

From source
^^^^^^^^^^^

To install the latest development version from GitHub:

.. code-block:: console

    uv pip install git+https://github.com/jordandeklerk/moderndid.git

Or with pip:

.. code-block:: console

    pip install git+https://github.com/jordandeklerk/moderndid.git

.. note::

    moderndid is not yet available on conda-forge. We recommend using uv or pip
    for installation.

Verifying the installation
--------------------------

To verify that moderndid is installed correctly:

.. code-block:: console

    python -c "import moderndid; print(moderndid.__version__)"

Development
-----------

To install moderndid for development:

.. code-block:: console

    git clone https://github.com/jordandeklerk/moderndid.git
    cd moderndid
    uv pip install -e ".[dev,test]"

This installs moderndid in editable mode along with development and test
dependencies.
