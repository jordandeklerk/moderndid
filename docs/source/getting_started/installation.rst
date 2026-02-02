============
Installation
============

Installing moderndid
--------------------

The only prerequisite for installing moderndid is Python 3.11 or later.

From PyPI
^^^^^^^^^

The base installation (~80MB) includes core DiD estimators (``did``, ``drdid``, ``didinter``, ``didtriple``):

.. code-block:: console

    uv pip install moderndid

For full functionality including all estimators, plotting, and performance optimizations:

.. code-block:: console

    uv pip install moderndid[all]

Or install just the base with pip:

.. code-block:: console

    pip install moderndid

Optional Extras
^^^^^^^^^^^^^^^

Extras are additive. They add functionality to the base install, so you always
get the core estimators plus whatever extras you specify.

.. list-table:: Available Extras
   :header-rows: 1
   :widths: 15 45 20

   * - Extra
     - What you get
     - Additional dependencies
   * - ``didcont``
     - Base + continuous treatment DiD (``cont_did``)
     - formulaic
   * - ``didhonest``
     - Base + sensitivity analysis (``honest_did``)
     - cvxpy, sympy
   * - ``plots``
     - Base + visualization (``plot_gt``, ``plot_event_study``, ...)
     - plotnine
   * - ``numba``
     - Base + faster bootstrap inference
     - numba
   * - ``all``
     - Everything
     - all of the above

.. code-block:: console

    uv pip install moderndid[didcont]     # Base estimators + cont_did
    uv pip install moderndid[numba]       # Base estimators with faster bootstrap
    uv pip install moderndid[plots,numba] # Combine multiple extras

.. tip::

    **Recommended:** ``uv pip install moderndid[all]`` for full functionality.
    The ``numba`` extra provides significant speedups for bootstrap inference.
    Install minimal extras only if you have specific dependency constraints.

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
    uv pip install -e ".[all,dev,test]"

This installs moderndid in editable mode along with all optional dependencies,
development tools, and test dependencies.
