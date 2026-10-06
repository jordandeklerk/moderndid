.. _background:

##########
Background
##########

Each guide develops the assumptions, identification arguments, and inference
results behind a ModernDiD estimator. We connect those results to the choices
you make in the package so you can see what an estimate measures and what
its interpretation requires. The :doc:`examples <../examples/index>` show those
choices in complete analyses.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - :doc:`did`
     - Group-time effects and aggregation for staggered treatment adoption.
   * - :doc:`drdid`
     - Doubly robust estimation for two-period designs.
   * - :doc:`tripledid`
     - Identification and inference with a third comparison group.
   * - :doc:`didcont`
     - Treatment effects for continuous treatment doses.
   * - :doc:`didinter`
     - Dynamic effects under intertemporal treatment changes.
   * - :doc:`diddynamic`
     - Covariate balancing for dynamic treatment effects.
   * - :doc:`didhonest`
     - Sensitivity analysis under restrictions on parallel-trends violations.
   * - :doc:`etwfe`
     - Extended two-way fixed effects for heterogeneous treatment effects.
   * - :doc:`npiv`
     - Nonparametric instrumental variables and uniform inference.

.. toctree::
   :maxdepth: 2
   :hidden:

   did
   drdid
   tripledid
   didcont
   didinter
   diddynamic
   didhonest
   etwfe
   npiv

Acknowledgements
================

The :doc:`acknowledgements <../acknowledgements>` credit the researchers and
software authors whose work these estimators build on. Each background guide
links to the source papers for the method it develops.

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
