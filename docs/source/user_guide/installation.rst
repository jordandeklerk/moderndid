.. _installation:

============
Installation
============

ModernDiD requires Python 3.12 or later in the environment that runs your
analysis. To follow the :doc:`first analysis <quickstart>`, we'll install the
``plots`` extra so you can estimate treatment effects and draw the event
study. The dependencies for other estimators can wait until you need those
methods in your analysis.

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
      environment. If you work in a notebook, select that environment as its
      kernel instead.

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

Each extra listed below adds the dependencies of an optional estimator,
plotting, or faster computation to the base installation.

.. admonition:: Install GPU dependencies separately
   :class: important

   Since the ``gpu`` extra needs an NVIDIA GPU and cannot provide acceleration
   on macOS, follow the :doc:`GPU guide <gpu>` for the hardware and
   installation checks before adding it.

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
         * - ``gpu``
           - CuPy computation on supported NVIDIA hardware.
         * - ``spark``
           - Spark computation of :func:`~moderndid.att_gt` with PySpark 4.0 or later.
         * - ``all``
           - All estimator and plotting dependencies, without GPU or Spark dependencies.

For example, either of the commands below installs continuous DiD and
plotting together. Replace the extras inside the brackets to match your
analysis.

.. code-block:: console

   uv add "moderndid[didcont,plots]"

.. code-block:: console

   python -m pip install "moderndid[didcont,plots]"

Besides the ``spark`` extra, running Spark on your own machine needs Java 17 or
later. On Databricks, leave out the ``spark`` extra, since the runtime already
provides PySpark. The :doc:`Spark guide <spark>` says which Databricks runtimes
to choose and shows how to install ``moderndid`` on the executors of a
Databricks cluster or one you manage.

Check the installation
-----------------------

Before fitting a model, confirm that your analysis environment can import the
package and load one of its datasets.

.. code-block:: python

   import moderndid as did

   print(did.__version__)
   print(did.load_mpdta().shape)

You should see a shape of 2,500 rows and six columns, one row for each county
and year. If this code works in a terminal
but fails in a notebook, the notebook may be using a different Python
environment. You can check its interpreter by running the following cell
inside the notebook.

.. code-block:: python

   import sys

   print(sys.executable)

If an estimator needs an extra you haven't installed, add it to the
environment identified by that interpreter before trying the fit again.
Importing ModernDiD alone does not check every dependency a fit may need
because optional functions load their dependencies when you access or call
them.

Resolve an installation problem
--------------------------------

A dependency conflict can cause an installer to select an older package
release. Check ``did.__version__`` if the installed functions or extras don't
match these pages. Retain the installer's error message when asking for help
so the failed dependency can be identified. Pinning the release you intend
to use makes a conflict explicit rather
than allowing a different release to satisfy the request.

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
environment. Once the installation is working, the
:doc:`first analysis <quickstart>` uses the county data you just loaded to
estimate your first event study.
