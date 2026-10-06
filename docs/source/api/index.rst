.. module:: moderndid

.. _api:

#############
API Reference
#############

:Release: |version|

This reference groups every public function and class in ModernDiD by the part
of an analysis it serves. In the first group, each
treatment design gets a page that opens with the functions you call and closes
with the result objects they return. The groups after it cover sensitivity
analysis, data, results, the lower-level routines the estimators call, and
running on a GPU. The :ref:`Background <background>` section explains the
methods behind the estimators.

Estimators
==========

.. list-table::
   :class: section-index-table
   :widths: 35 65

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
     - Average outcomes under one treatment history against another when treatment varies over time.
   * - :doc:`didml`
     - Group-time and individual conditional effects estimated with cross-fitted machine learning models.
   * - :doc:`etwfe`
     - Extended two-way fixed effects regressions and the marginal effects that aggregate them.
   * - :doc:`npiv`
     - Nonparametric instrumental variables estimation with uniform confidence bands.

.. toctree::
   :hidden:
   :maxdepth: 2

   multiperiod
   drdid
   didtriple
   didcont
   didinter
   diddynamic
   didml
   etwfe
   npiv

Sensitivity analysis
====================

.. list-table::
   :class: section-index-table
   :widths: 35 65

   * - :doc:`honestdid`
     - Confidence sets for event-study estimates that stay valid when parallel trends fail within limits you choose.

.. toctree::
   :hidden:
   :maxdepth: 2

   honestdid

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
     - The function that turns a result into a polars DataFrame and the converters behind it.

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
