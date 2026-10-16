.. _r-comparisons:

##################
Comparisons with R
##################

This section checks ModernDiD's results against the R packages that implement
the same methods. Each comparison below fits one estimator in both languages on
the same application data and sets the two results side by side. Where the
numbers differ, the comparison traces each gap to its source.

A gap can come from an R package that departs from the paper it implements,
from two defensible definitions of one quantity, or from numerical noise in
bootstrap draws, solvers, simulated quantiles, or numerical derivatives. Since
agreement in one specification doesn't carry over to every option, each
comparison states the settings it checks.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Comparison
     - Description
   * - :doc:`Staggered DiD <r_staggered>`
     - Compares ``att_gt`` and ``aggte`` with ``did``. Every cohort-period
       effect, summary, and standard error agrees.
   * - :doc:`Two-period DiD <r_two_period>`
     - Compares ``drdid``, ``ipwdid``, and ``ordid`` with ``DRDID``. All three
       estimators agree on the ATT and its standard error.
   * - :doc:`Triple differences <r_triple>`
     - Compares ``ddd`` and ``agg_ddd`` with ``triplediff``. Every
       post-treatment effect and summary agrees. Two pre-treatment effects
       differ by up to 1.7e-5.
   * - :doc:`Continuous treatment <r_continuous>`
     - Compares ``cont_did`` with ``contdid`` and ``ptetools``. The overall ATT
       and ACRT agree. The dose curves and every standard error differ.
   * - :doc:`Intertemporal treatment <r_intertemporal>`
     - Compares ``did_multiplegt`` with ``DIDmultiplegtDYN``. Both horizons,
       the placebo, and their clustered standard errors agree.
   * - :doc:`Sensitivity analysis <r_honest>`
     - Compares ``honest_did`` with ``HonestDiD``. The original interval, the
       conditional test, and the FLCI hybrid agree. Two methods that rely on
       simulation differ slightly.
   * - :doc:`Extended TWFE <r_etwfe>`
     - Compares ``etwfe`` and ``emfx`` with ``etwfe``. Every coefficient and
       aggregated estimate agrees to within 1e-14. Aggregated standard errors
       differ by up to 2e-8.
   * - :doc:`Nonparametric IV <r_npiv>`
     - Compares ``npiv`` with ``npiv``. The chosen dimension and everything
       that doesn't rest on bootstrap draws agree. The critical values of the
       bands differ slightly.

.. toctree::
   :maxdepth: 1
   :hidden:

   Staggered DiD <r_staggered>
   Two-period DiD <r_two_period>
   Triple differences <r_triple>
   Continuous treatment <r_continuous>
   Intertemporal treatment <r_intertemporal>
   Sensitivity analysis <r_honest>
   Extended TWFE <r_etwfe>
   Nonparametric IV <r_npiv>

.. _r-comparisons-running:

Running the comparisons
=======================

Since the documentation build shows each notebook's saved outputs without
rerunning it, you can read every comparison without installing R. The
notebooks run R inside the Python session through ``rpy2``. Their saved
outputs came from R 4.3.3 in the ``validation`` environment and from the
repository's ``.venv``, a Python 3.12 environment with ModernDiD and its
``all`` extra installed in editable mode alongside ``rpy2``, ``nbclient``,
``ipykernel``, and ``pandas``.

.. tab-set::

   .. tab-item:: Validation environment

      .. code-block:: console

         pixi run -e validation setup-r
         pixi run --frozen -e validation .venv/bin/python docs/run_r_examples.py

      Because pixi puts the validation environment's R first on the path,
      ``rpy2`` loads the R release that pixi locks for your platform. The
      setup task installs each notebook's reference packages and updates
      older copies to the current CRAN release. A fresh setup can therefore
      record newer releases than the saved outputs did.

   .. tab-item:: Locked docs-r environment

      For a setup that pixi locks, the optional ``docs-r`` environment
      provides Python 3.12, R 4.5, ``rpy2`` 3.6, and the notebook execution
      dependencies.

      .. code-block:: console

         pixi install --frozen -e docs-r
         pixi run -e docs-r setup-r
         pixi run -e docs-r python docs/run_r_examples.py

      Its ``setup-r`` task installs the R reference packages that conda-forge
      lacks or carries in an older release. Because those packages come from
      CRAN and R-universe, their versions can change independently of
      ``pixi.lock``.

To rerun only some of the notebooks, name them after the script, as in
``docs/run_r_examples.py r_triple r_continuous``. Since the script saves a
notebook only after every cell succeeds, a failed cell leaves that notebook's
previous outputs in place. Before you read a difference after a rerun as a
change in the code, compare the versions table at the end of the notebook.

.. admonition:: The HonestDiD version
   :class: note

   In the validation environment on Apple Silicon, where pixi locks R 4.3.3,
   the setup task installs ``HonestDiD`` 0.2.7 from the CRAN archive because
   the current release needs R 4.4. A rerun there prints the same intervals as the saved ``r_honest``
   output from 0.2.6.

Estimators without a comparison
===============================

Dynamic covariate balancing has no worked comparison yet, although the
:doc:`testing guide <../contributing/testing>` explains how to run its R
reference tests in ``tests/validation/``.

Machine learning DiD (:func:`~moderndid.didml` and
:func:`~moderndid.aggte_didml`) has neither a worked comparison nor an R
reference test. Since ModernDiD departs from the authors' ``MLDID`` code in
four places to follow the paper of Hatamyar, Kreif, Rocha, and Huber (2023)
and its sources, the two would not agree.

- Each :math:`ATT(g,t)` averages the predicted conditional effects over cohort
  :math:`g` in period :math:`t`, as the paper and the authors' own simulation
  do. ``MLDID`` averages them over every row of the cell, both periods and
  both groups.
- As in the program of Lu, Nie, and Wager (2019), the balancing weights sum to
  zero over the post-treatment rows and over the cohort's rows. They balance
  the treated cell against its own covariate mean. ``MLDID`` leaves both sums
  free and balances against the full-sample mean.
- Standard errors come from a unit bootstrap that refits every cell, or from
  influence functions with ``boot=False``. ``MLDID`` treats each unit's two
  periods as independent observations and sums share-weighted cell standard
  errors across event times.
- The event study weights each cohort by its share of the units in the cohorts
  that reach that event time, as the paper's equation 19 does. ``MLDID`` gives
  every cohort the same weight in a balanced panel.

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
