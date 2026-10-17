.. _examples:

========
Examples
========

Each example works through one research design on real data, from the
question it asks to the estimates, plots, and checks behind its answer. If
you're not sure which design matches your study, the
:doc:`estimator guide <../user_guide/estimator_overview>` explains when to
use each method.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 35 65

   * - Example
     - Description
   * - :doc:`Staggered DiD <../user_guide/example_staggered_did>`
     - Estimate group-time effects and aggregate them into an event study and an overall effect.
   * - :doc:`Triple differences <../user_guide/example_triple_did>`
     - Add a third comparison to a difference-in-differences design.
   * - :doc:`Intertemporal treatment <../user_guide/example_inter_did>`
     - Study treatments that change over time and their dynamic effects.
   * - :doc:`Continuous treatment <../user_guide/example_cont_did>`
     - Estimate effects when treatment varies in dose rather than just status.
   * - :doc:`Fracking paper replication <../user_guide/example_cont_did_replication>`
     - Reproduce the published event studies and pooled dose curves for county employment.
   * - :doc:`Dynamic covariate balancing <../user_guide/example_dyn_balancing>`
     - Compare treatment histories by balancing past outcomes and covariates.
   * - :doc:`Sensitivity analysis <../user_guide/example_honest_did>`
     - Assess how departures from parallel trends affect your conclusions.
   * - :doc:`Extended TWFE <../user_guide/example_etwfe>`
     - Fit extended two-way fixed effects and compare event-study estimates.
   * - :doc:`Nonparametric IV <../user_guide/example_npiv>`
     - Estimate nonparametric functions with instrumental variables.

.. toctree::
   :hidden:
   :maxdepth: 1

   Staggered DiD <../user_guide/example_staggered_did>
   Triple differences <../user_guide/example_triple_did>
   Intertemporal treatment <../user_guide/example_inter_did>
   Continuous treatment <../user_guide/example_cont_did>
   Fracking paper replication <../user_guide/example_cont_did_replication>
   Dynamic covariate balancing <../user_guide/example_dyn_balancing>
   Sensitivity analysis <../user_guide/example_honest_did>
   Extended TWFE <../user_guide/example_etwfe>
   Nonparametric IV <../user_guide/example_npiv>

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
