.. module:: moderndid

.. _api:

#############
API Reference
#############

:Release: |version|

This is the API reference for **ModernDiD**; details on the underlying
methodology are found in :ref:`background`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Reference
     - Description
   * - :doc:`multiperiod` and :doc:`drdid`
     - Core estimators for staggered adoption and two-period designs.
   * - :doc:`didtriple`, :doc:`didcont`, and :doc:`didinter`
     - Triple differences, continuous doses, and treatments that change over time.
   * - :doc:`diddynamic`, :doc:`didml`, and :doc:`etwfe`
     - Covariate balancing, machine learning, and extended two-way fixed effects.
   * - :doc:`honestdid`
     - Sensitivity analysis for departures from parallel trends.
   * - :doc:`npiv`
     - Nonparametric instrumental variables estimation and inference.
   * - :doc:`panel`, :doc:`propensity`, and :doc:`bootstrap`
     - Data preparation, propensity scores, and bootstrap inference.
   * - :doc:`plotting`, :doc:`results`, and :doc:`data`
     - Visualizations, result objects, and bundled datasets.

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
   honestdid
   etwfe

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
