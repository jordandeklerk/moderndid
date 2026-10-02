============
Installation
============

Installing **ModernDiD**
------------------------

The only prerequisite for installing **ModernDiD** is Python 3.12 or later.

From PyPI
^^^^^^^^^

The base installation includes core DiD estimators
(:func:`~moderndid.att_gt`, :func:`~moderndid.drdid`,
:func:`~moderndid.did_multiplegt`, :func:`~moderndid.ddd`). In a project managed
by `uv <https://docs.astral.sh/uv/>`_, add it as a dependency.

.. code-block:: console

    uv add moderndid

With pip, install it into your environment instead.

.. code-block:: console

    pip install moderndid

For full functionality including all estimators, plotting, and performance
optimizations, add the ``all`` extra.

.. code-block:: console

    uv add "moderndid[all]"

Optional extras
^^^^^^^^^^^^^^^

Some estimators and features require additional dependencies that are not
installed by default. Extras are additive and build on the base install, so
you always get the core estimators plus whatever extras you specify.

- **diddynamic** -- Dynamic covariate balancing DiD (:func:`~moderndid.dyn_balancing`)
- **didcont** -- Continuous treatment DiD (:func:`~moderndid.cont_did`)
- **didhonest** -- Sensitivity analysis (:func:`~moderndid.honest_did`)
- **etwfe** -- Extended TWFE (:func:`~moderndid.etwfe`)
- **plots** -- Visualization (``plot_gt``, ``plot_event_study``, ...)
- **numba** -- Faster bootstrap inference
- **gpu** -- GPU-accelerated estimation (requires CUDA)
- **all** -- Everything except ``gpu``, which requires an NVIDIA GPU

.. code-block:: console

    uv add "moderndid[all]"             # All extras except gpu
    uv add "moderndid[didcont,plots]"   # Combine specific extras
    uv add "moderndid[gpu]"             # GPU acceleration

.. tip::

    We recommend ``uv add "moderndid[all]"`` for full functionality.
    The ``numba`` extra provides significant performance gains for bootstrap
    inference, and the ``plots`` extra adds customizable plotting functions for
    every estimator's results. Install minimal extras only if you have specific
    dependency constraints.

From source
^^^^^^^^^^^

To install the latest development version from GitHub.

.. code-block:: console

    uv add "moderndid[all] @ git+https://github.com/jordandeklerk/moderndid.git"

Or with pip.

.. code-block:: console

    pip install "moderndid @ git+https://github.com/jordandeklerk/moderndid.git"

Verifying the installation
--------------------------

To verify that **ModernDiD** is installed correctly, run the following.

.. code-block:: console

    python -c "import moderndid; print(moderndid.__version__)"

Development
-----------

To install **ModernDiD** for development, clone the repository and install in
editable mode.

.. code-block:: console

    git clone https://github.com/jordandeklerk/moderndid.git
    cd moderndid
    uv pip install -e ".[all,dev,test]"

This installs **ModernDiD** in editable mode along with all optional
dependencies, development tools, and test dependencies.

Troubleshooting
---------------

Checking which extras are available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After installing, verify that the extras you need are actually available.

.. code-block:: python

    from moderndid.core.numba_utils import HAS_NUMBA
    from moderndid.cupy.backend import HAS_CUPY

    print("numba:", HAS_NUMBA)
    print("cupy:", HAS_CUPY)

If an extra is missing, **ModernDiD** raises an ``ImportError`` with the
install command when you first call a function that needs it.

.. code-block:: python

    >>> moderndid.cont_did(...)
    ImportError: 'cont_did' requires extra dependencies. Install them with uv add 'moderndid[didcont]' or pip install 'moderndid[didcont]'

Silent version downgrades
^^^^^^^^^^^^^^^^^^^^^^^^^^

When a package manager cannot resolve a dependency required by an extra, it
may silently install an older version of **ModernDiD** where that extra does not
exist. The ``gpu`` extra is the most common trigger, but this can also happen
with ``etwfe`` or ``didhonest`` if their dependencies conflict with your
environment.

Check your installed version against what you expected.

.. code-block:: console

    python -c "import moderndid; print(moderndid.__version__)"

If the version is older than expected, pin a version floor to get a clear
error instead of a silent downgrade.

.. code-block:: console

    uv add "moderndid[gpu]>=0.1.0"

GPU extra
^^^^^^^^^^

The ``gpu`` extra depends on ``cupy-cuda12x`` and ``rmm-cu12``, both of which
require NVIDIA hardware and drivers.

- ``cupy-cuda12x`` has no macOS wheels. ``rmm-cu12`` is Linux-only.
- Installing on a machine without CUDA drivers will silently fall back to an
  older **ModernDiD** version (see above).
- Having multiple CuPy packages installed (``cupy``, ``cupy-cuda11x``,
  ``cupy-cuda12x``) causes import conflicts. Only one should be present.

Verify your CUDA setup and check for conflicting CuPy packages.

.. code-block:: console

    nvidia-smi                  # confirm CUDA driver version
    pip list | grep -i cupy     # check for conflicting packages

If you have multiple CuPy packages, remove the extras before installing.

.. code-block:: console

    pip uninstall cupy cupy-cuda11x cupy-cuda12x -y
    pip install "moderndid[gpu]"


Sensitivity analysis extra
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``didhonest`` extra depends on ``cvxpy[ECOS]``, which compiles C
extensions during installation. If a C compiler is not available, the
install fails with a long build error.

On macOS, install the Xcode command-line tools.

.. code-block:: console

    xcode-select --install

On Windows, install the
`Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
and select "Desktop development with C++" during setup.

On Ubuntu/Debian.

.. code-block:: console

    sudo apt install build-essential
