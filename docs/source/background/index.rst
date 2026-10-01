.. _background:

##########
Background
##########

This section provides theoretical background on the difference-in-differences
methodologies implemented in **ModernDiD**; for practical usage see the
:ref:`User Guide <user-guide>`.

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

The **ModernDiD** package implements various difference-in-differences methodologies from
the econometric literature. We acknowledge the original authors of these methods and the
authors of the R packages that inspired this implementation. See the
:doc:`acknowledgements <../acknowledgements>` for the full list.

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
