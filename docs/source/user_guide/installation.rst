============
Installation
============

ModernDiD requires Python 3.12 or later. For the :doc:`quickstart`, we use the
``plots`` extra so you can both estimate treatment effects and draw the event
study. You can add the dependencies for other estimators when you need them.

Install in your environment
---------------------------

Choose the command that matches how you manage your Python environment. If
you use `uv <https://docs.astral.sh/uv/guides/projects/>`_, run ``uv add`` from
your project directory. With pip, use the Python interpreter that will run
your analysis so the package is installed in that environment.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add "moderndid[plots]"

      Run your analysis through ``uv run python`` to use the project's
      environment, or select that environment as your notebook's kernel.

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install "moderndid[plots]"

      If you use a notebook, select the same Python environment as its kernel
      before importing the package.

You can install ``moderndid`` without any extras if you don't need plots or
the optional estimators. The base installation includes staggered adoption,
two-period DiD, triple differences, intertemporal treatment effects,
nonparametric IV, and the panel utilities.

Choose optional dependencies
----------------------------

Extras add dependencies to the base installation. If you expect to work
through several treatment designs, ``moderndid[all]`` includes every
estimator extra, plotting, and Numba. GPU dependencies are installed
separately because they require a compatible NVIDIA setup.

.. tab-set::

   .. tab-item:: Estimators

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Extra
           - Use it for
         * - ``didcont``
           - Continuous treatment effects with :func:`~moderndid.cont_did`.
         * - ``diddynamic``
           - Dynamic covariate balancing with :func:`~moderndid.diddynamic.dyn_balancing`.
         * - ``didhonest``
           - Sensitivity analysis with :func:`~moderndid.honest_did`.
         * - ``didml``
           - DiD with machine learning nuisance models using :func:`~moderndid.didml`.
         * - ``etwfe``
           - Extended two-way fixed effects with :func:`~moderndid.etwfe`.

   .. tab-item:: Plots and computation

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Extra
           - Use it for
         * - ``plots``
           - The package's plotnine figures and plotting themes.
         * - ``numba``
           - Compiled CPU routines used by bootstrap and aggregation calculations.
         * - ``gpu``
           - CuPy computation on supported NVIDIA hardware.
         * - ``all``
           - All estimator, plotting, and Numba dependencies, without GPU dependencies.

For example, these commands install continuous DiD and plotting together.
Replace the extras inside the brackets to match your analysis.

.. code-block:: console

   uv add "moderndid[didcont,plots]"

.. code-block:: console

   python -m pip install "moderndid[didcont,plots]"

Check the installation
-----------------------

Before fitting a model, confirm that your analysis environment can import the
package and load one of its datasets.

.. code-block:: python

   import moderndid as did

   print(did.__version__)
   print(did.load_mpdta().shape)

The dataset has 2,500 rows and six columns. If this works in a terminal but
fails in a notebook, check which interpreter the notebook uses.

.. code-block:: python

   import sys

   print(sys.executable)

Install a missing extra into that interpreter's environment before trying
the estimator again. Optional functions load dependencies when you access
or call them, depending on the estimator. A successful import of ModernDiD
therefore does not check every dependency an optional fit needs.

Resolve an installation problem
--------------------------------

A dependency conflict can cause an installer to select an older package
release. Check ``did.__version__`` if the installed functions or extras don't
match these pages. Retain the installer's error message when asking for help
so the failed dependency can be identified. Pinning the release you intend
to use makes a conflict explicit rather
than allowing a different release to satisfy the request.

.. admonition:: Install GPU dependencies separately
   :class: important

   The ``gpu`` extra uses ``cupy-cuda12x`` and ``rmm-cu12``. It needs a
   compatible NVIDIA CUDA environment and cannot provide GPU acceleration on
   macOS. Follow the :doc:`GPU guide <gpu>` for the hardware and installation
   checks before adding this extra.

If a dependency has to build from source, the installer may require a C or
C++ compiler. This can affect the optimization packages used by sensitivity
analysis. Read the first failed dependency in the build log before installing
compiler tools, since a wheel may be available for another supported Python
version.

Use the development version
---------------------------

To try the current repository version in an analysis environment, install
directly from GitHub. Since that version can change between installs, record
the commit you use if you need to reproduce the analysis later.

.. code-block:: console

   python -m pip install "moderndid[plots] @ git+https://github.com/jordandeklerk/moderndid.git"

If you're planning to change the package itself, the
:doc:`contributor setup <../contributing/guide>` covers the development
environment. Once the installation is working, the :doc:`quickstart` uses
the county data you just loaded to estimate your first event study.
