.. module:: moderndid

.. _api:

#############
API Reference
#############

:Release: |version|

This reference documents every public function and class in moderndid. The
methods behind them are explained in the :ref:`Background <background>` section.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Reference
     - Description
   * - :doc:`multiperiod`
     - Group-time effects under staggered adoption and their aggregation into event studies and overall effects.
   * - :doc:`drdid`
     - Two-period estimators with doubly robust, inverse probability weighting, and outcome regression methods.
   * - :doc:`didtriple`
     - Triple differences for policies that reach only an eligible part of each treated group.
   * - :doc:`didcont`
     - Continuous treatment doses and the dose-response functions they trace out.
   * - :doc:`didinter`
     - Treatments that switch on and off or change in intensity over time.
   * - :doc:`diddynamic`
     - Dynamic covariate balancing for treatment histories that vary over time.
   * - :doc:`didml`
     - Group-time and individual conditional effects estimated with cross-fitted machine learning models.
   * - :doc:`etwfe`
     - Extended two-way fixed effects regressions and the marginal effects that aggregate them.
   * - :doc:`honestdid`
     - Sensitivity analysis for departures from parallel trends.
   * - :doc:`npiv`
     - Nonparametric instrumental variables estimation with uniform confidence bands.
   * - :doc:`panel`
     - Diagnostics, validation, and reshaping for panel data before estimation.
   * - :doc:`propensity`
     - Propensity score estimators behind the doubly robust and weighting methods.
   * - :doc:`bootstrap`
     - Weighted and multiplier bootstrap inference for the estimators.
   * - :doc:`plotting`
     - Plots of estimates, event studies, and sensitivity analyses.
   * - :doc:`results`
     - Result objects and the converter that turns any of them into a polars DataFrame.
   * - :doc:`data`
     - Bundled datasets and the simulators that generate data for each design.

.. toctree::
   :caption: Core estimators
   :hidden:
   :maxdepth: 2

   multiperiod
   drdid

.. toctree::
   :caption: Extensions
   :hidden:
   :maxdepth: 2

   didtriple
   didcont
   didinter
   diddynamic
   didml
   etwfe
   honestdid

.. toctree::
   :caption: Nonparametric IV
   :hidden:
   :maxdepth: 2

   npiv

.. toctree::
   :caption: Utilities
   :hidden:
   :maxdepth: 2

   panel
   propensity
   bootstrap
   plotting
   results
   data

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
