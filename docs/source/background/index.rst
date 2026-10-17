.. _background:

##########
Background
##########

Each method guide develops the assumptions, identification arguments, and
inference results behind a ModernDiD estimator. We connect those results to the
choices you make in the package so you can see what an estimate measures and
what its interpretation requires. The :doc:`examples <../examples/index>` show
those choices in complete analyses of real data.

Many of the method guides build their standard errors, aggregations, and
simultaneous bands from influence functions without stopping to introduce
them. Since it supplies that introduction, the
:doc:`influence functions <influence_functions>` guide comes first in the table
below.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Guide
     - Description
   * - :doc:`Influence functions <influence_functions>`
     - Per-unit error contributions behind the guides' standard errors and bands.
   * - :doc:`Staggered DiD <did>`
     - Group-time effects and aggregation for staggered treatment adoption.
   * - :doc:`Two-period DiD <drdid>`
     - Doubly robust estimation for two-period designs.
   * - :doc:`Triple differences <tripledid>`
     - Identification and inference with a third comparison group.
   * - :doc:`Continuous treatment <didcont>`
     - Level effects of each dose, the extra assumptions that dose slopes need,
       and bands over the dose curve.
   * - :doc:`Intertemporal treatment <didinter>`
     - Effects of a treatment path relative to the initial treatment level and
       their normalization by exposure.
   * - :doc:`Dynamic covariate balancing <diddynamic>`
     - Identification of a full treatment history's effect and the sequential
       weights that estimate it.
   * - :doc:`Sensitivity analysis <didhonest>`
     - Identified sets and confidence sets under bounds on violations of
       parallel trends.
   * - :doc:`Extended TWFE <etwfe>`
     - Cohort-period effects from a saturated regression, their averages, and
       nonlinear outcome models.
   * - :doc:`Nonparametric IV <npiv>`
     - Sieve estimation of a structural function, data-driven dimension choice,
       and uniform bands.

.. toctree::
   :maxdepth: 2
   :hidden:

   Influence functions <influence_functions>
   Staggered DiD <did>
   Two-period DiD <drdid>
   Triple differences <tripledid>
   Continuous treatment <didcont>
   Intertemporal treatment <didinter>
   Dynamic covariate balancing <diddynamic>
   Sensitivity analysis <didhonest>
   Extended TWFE <etwfe>
   Nonparametric IV <npiv>

Acknowledgements
================

The :doc:`acknowledgements <../acknowledgements>` credit the researchers and
software authors whose work these estimators build on. Each background guide
links to the source papers for the method it develops.

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
