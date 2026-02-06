============
Installation
============

Installing moderndid
--------------------

The only prerequisite for installing moderndid is Python 3.11 or later.

From PyPI
^^^^^^^^^

The base installation includes core DiD estimators (``did``, ``drdid``, ``didinter``, ``didtriple``).

.. code-block:: console

    uv pip install moderndid

For full functionality including all estimators, plotting, and performance optimizations, install with the ``all`` extra.

.. code-block:: console

    uv pip install moderndid[all]

Or install just the base with pip.

.. code-block:: console

    pip install moderndid

Optional Extras
^^^^^^^^^^^^^^^

Extras are additive. They add functionality to the base install, so you always
get the core estimators plus whatever extras you specify.

- **didcont** - Base + continuous treatment DiD (``cont_did``)
- **didhonest** - Base + sensitivity analysis (``honest_did``)
- **plots** - Base + visualization (``plot_gt``, ``plot_event_study``, ...)
- **numba** - Base + faster bootstrap inference
- **all** - Everything

.. code-block:: console

    uv pip install moderndid[didcont]     # Base estimators + cont_did
    uv pip install moderndid[numba]       # Base estimators with faster bootstrap
    uv pip install moderndid[plots,numba] # Combine multiple extras

From source
^^^^^^^^^^^

To install the latest development version from GitHub, use the following.

.. code-block:: console

    uv pip install git+https://github.com/jordandeklerk/moderndid.git

Or with pip.

.. code-block:: console

    pip install git+https://github.com/jordandeklerk/moderndid.git

.. tip::

    We recommend ``uv pip install moderndid[all]`` for full functionality.
    The ``numba`` extra provides significant performance gains for bootstrap inference and the
    ``plots`` extra provides customizable, batteries-included plotting out of the box.
    Install minimal extras only if you have specific dependency constraints.

Verifying the installation
--------------------------

To verify that moderndid is installed correctly, run the following.

.. code-block:: console

    python -c "import moderndid; print(moderndid.__version__)"

Development
-----------

To install moderndid for development, clone the repository and install in editable mode.

.. code-block:: console

    git clone https://github.com/jordandeklerk/moderndid.git
    cd moderndid
    uv pip install -e ".[all,dev,test]"

This installs moderndid in editable mode along with all optional dependencies,
development tools, and test dependencies.
