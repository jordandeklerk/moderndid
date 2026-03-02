============
Installation
============

Installing moderndid
--------------------

The only prerequisite for installing moderndid is Python 3.11 or later.

From PyPI
^^^^^^^^^

The base installation includes core DiD estimators
(:func:`~moderndid.att_gt`, :func:`~moderndid.drdid`,
:func:`~moderndid.did_multiplegt`, :func:`~moderndid.ddd`).

.. code-block:: console

    uv pip install moderndid

For full functionality including all estimators, plotting, and performance
optimizations, install with the ``all`` extra.

.. code-block:: console

    uv pip install "moderndid[all]"

Or install just the base with pip.

.. code-block:: console

    pip install moderndid

Optional extras
^^^^^^^^^^^^^^^

Some estimators and features require additional dependencies that are not
installed by default. Extras are additive and build on the base install, so
you always get the core estimators plus whatever extras you specify.

- **didcont** -- Continuous treatment DiD (:func:`~moderndid.cont_did`)
- **didhonest** -- Sensitivity analysis (:func:`~moderndid.honest_did`)
- **plots** -- Visualization (``plot_gt``, ``plot_event_study``, ...)
- **numba** -- Faster bootstrap inference
- **gpu** -- GPU-accelerated estimation (requires CUDA)
- **dask** -- Distributed estimation on Dask clusters
- **spark** -- Distributed estimation on Spark clusters
- **all** -- Everything except ``gpu``, which requires specific infrastructure

.. code-block:: console

    uv pip install "moderndid[all]"             # All extras except gpu
    uv pip install "moderndid[didcont,plots]"   # Combine specific extras
    uv pip install "moderndid[gpu,spark]"       # GPU + distributed

.. tip::

    We recommend ``uv pip install "moderndid[all]"`` for full functionality.
    The ``numba`` extra provides significant performance gains for bootstrap
    inference and the ``plots`` extra provides customizable, batteries-included
    plotting out of the box. Install minimal extras only if you have specific
    dependency constraints.

.. warning::

    When a package manager like ``uv`` or ``pip`` cannot resolve a dependency
    required by an extra, it may silently fall back to an older version of
    moderndid where that extra does not exist, rather than raising an error.

    The ``gpu`` extra is the most likely to trigger this, since it depends on
    ``cupy-cuda12x`` (Linux and Windows only) and ``rmm-cu12`` (Linux only),
    both of which require NVIDIA CUDA. If you see a warning like
    ``The package moderndid==0.0.3 does not have an extra named 'gpu'``, this
    is what happened. To use the ``gpu`` extra, install on a machine with
    NVIDIA CUDA drivers, or pin the version to get a clear error instead of a
    silent downgrade:

    .. code-block:: console

        uv pip install "moderndid[gpu]>=0.1.0"

From source
^^^^^^^^^^^

To install the latest development version from GitHub.

.. code-block:: console

    uv pip install "moderndid[all] @ git+https://github.com/jordandeklerk/moderndid.git"

Or with pip.

.. code-block:: console

    pip install "moderndid @ git+https://github.com/jordandeklerk/moderndid.git"

Verifying the installation
--------------------------

To verify that moderndid is installed correctly, run the following.

.. code-block:: console

    python -c "import moderndid; print(moderndid.__version__)"

Development
-----------

To install moderndid for development, clone the repository and install in
editable mode.

.. code-block:: console

    git clone https://github.com/jordandeklerk/moderndid.git
    cd moderndid
    uv pip install -e ".[all,dev,test]"

This installs moderndid in editable mode along with all optional
dependencies, development tools, and test dependencies.
