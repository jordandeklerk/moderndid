.. module:: moderndid

.. _api:

#############
API Reference
#############

:Release: |version|

Start with the treatment design that matches your analysis to find the
estimator, its arguments, and the result objects it returns. Once you have a
fit, the sensitivity and results pages explain the functions you can use to
examine it. The data pages help you prepare the sample, while the building
blocks and GPU reference cover the routines used within an estimation.

If you're choosing a method, the :ref:`User Guide <user-guide>` connects those
choices to worked analyses. The :ref:`Background <background>` section
develops the effects each method targets and the assumptions that identify
them.

Estimators
==========

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`Staggered DiD <multiperiod>`
     - Group-time effects under staggered adoption and their aggregation into event studies and overall effects.
   * - :doc:`Two-period DiD <drdid>`
     - Two-period estimators with doubly robust, inverse probability weighting, and outcome regression methods.
   * - :doc:`Triple differences <didtriple>`
     - Triple differences for policies that reach only an eligible part of each treated group.
   * - :doc:`Continuous treatment <didcont>`
     - Continuous treatment doses and the dose-response functions they trace out.
   * - :doc:`Intertemporal treatment <didinter>`
     - Treatments that switch on and off or change in intensity over time.
   * - :doc:`Dynamic covariate balancing <diddynamic>`
     - Average outcomes under one treatment history against another when treatment varies over time.
   * - :doc:`Machine learning DiD <didml>`
     - Group-time and individual conditional effects estimated with cross-fitted machine learning models.
   * - :doc:`Extended TWFE <etwfe>`
     - Extended two-way fixed effects regressions and the marginal effects that aggregate them.
   * - :doc:`Nonparametric IV <npiv>`
     - Nonparametric instrumental variables estimation with uniform confidence bands.

.. toctree::
   :hidden:
   :maxdepth: 2

   Staggered DiD <multiperiod>
   Two-period DiD <drdid>
   Triple differences <didtriple>
   Continuous treatment <didcont>
   Intertemporal treatment <didinter>
   Dynamic covariate balancing <diddynamic>
   Machine learning DiD <didml>
   Extended TWFE <etwfe>
   Nonparametric IV <npiv>

Robustness
==========

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`Sensitivity analysis <honestdid>`
     - Confidence sets for event-study estimates that stay valid when parallel trends fail within limits you choose.

.. toctree::
   :hidden:
   :maxdepth: 2

   Sensitivity analysis <honestdid>

Data
====

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`panel`
     - Diagnostics, validation, and reshaping for panel data before estimation.
   * - :doc:`data`
     - Bundled datasets and the simulators that generate data for each design.

.. toctree::
   :hidden:
   :maxdepth: 2

   panel
   data

Results
=======

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`plotting`
     - Plots of estimates, event studies, dose-response curves, and sensitivity analyses.
   * - :doc:`results`
     - Convert fitted results into Polars DataFrames for inspection, reporting, or further analysis.

.. toctree::
   :hidden:
   :maxdepth: 2

   plotting
   results

Building blocks
===============

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`propensity`
     - The propensity score and weighting steps inside the two-period estimators.
   * - :doc:`bootstrap`
     - Weighted and multiplier bootstrap inference for the estimators.

.. toctree::
   :hidden:
   :maxdepth: 2

   propensity
   bootstrap

Scaling
=======

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`backend`
     - Switches between NumPy on the CPU and CuPy on an NVIDIA GPU for the estimators that support it.

.. toctree::
   :hidden:
   :maxdepth: 2

   backend

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
