========
Examples
========

Each example works through a treatment design using data, estimates, and
plots. Choose the design that matches your study. The
:doc:`estimator overview <../user_guide/estimator_overview>` explains when to
use each method.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Example
     - Description
   * - :doc:`Staggered adoption <../user_guide/example_staggered_did>`
     - Estimate group-time effects and aggregate them into an event study.
   * - :doc:`Triple differences <../user_guide/example_triple_did>`
     - Add a third comparison to a difference-in-differences design.
   * - :doc:`Intertemporal treatment <../user_guide/example_inter_did>`
     - Study treatments that change over time and their dynamic effects.
   * - :doc:`Continuous treatment <../user_guide/example_cont_did>`
     - Estimate effects when treatment varies in dose rather than just status.
   * - :doc:`Dynamic covariate balancing <../user_guide/example_dyn_balancing>`
     - Balance covariates and outcomes for dynamic treatment effects.
   * - :doc:`Sensitivity analysis <../user_guide/example_honest_did>`
     - Assess how departures from parallel trends affect your conclusions.
   * - :doc:`Extended TWFE <../user_guide/example_etwfe>`
     - Fit extended two-way fixed effects and compare event-study estimates.
   * - :doc:`Nonparametric IV <../user_guide/example_npiv>`
     - Estimate nonparametric functions with instrumental variables.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../user_guide/example_staggered_did
   ../user_guide/example_triple_did
   ../user_guide/example_inter_did
   ../user_guide/example_cont_did
   ../user_guide/example_dyn_balancing
   ../user_guide/example_honest_did
   ../user_guide/example_etwfe
   ../user_guide/example_npiv

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
