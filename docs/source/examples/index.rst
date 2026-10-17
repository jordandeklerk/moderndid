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
     - ``att_gt`` and ``aggte`` estimate how staggered state minimum wage
       increases changed teen employment across US counties.
   * - :doc:`Triple differences <../user_guide/example_triple_did>`
     - ``ddd`` and ``agg_ddd`` estimate how weather insurance changed the share
       of savings that tobacco farmers kept in checking accounts.
   * - :doc:`Intertemporal treatment <../user_guide/example_inter_did>`
     - ``did_multiplegt`` estimates how lifting state banking restrictions in
       steps changed county mortgage lending over the following years.
   * - :doc:`Continuous treatment <../user_guide/example_cont_did>`
     - ``cont_did`` estimates how county employment changed after fracking
       potential became publicly known and how the change varies with
       geological prospectivity.
   * - :doc:`Fracking paper replication <../user_guide/example_cont_did_replication>`
     - ``att_gt``, ``aggte``, and ``BSpline`` reproduce the published event
       studies and dose curves of county employment after fracking potential
       became publicly known.
   * - :doc:`Dynamic covariate balancing <../user_guide/example_dyn_balancing>`
     - ``dyn_balancing`` estimates how a stretch of democracy changed GDP per
       capita when countries can move into and out of democracy.
   * - :doc:`Sensitivity analysis <../user_guide/example_honest_did>`
     - ``honest_did`` finds how far trends could depart from parallel before the
       evidence that states raised insurance coverage by expanding Medicaid
       disappears.
   * - :doc:`Extended TWFE <../user_guide/example_etwfe>`
     - ``etwfe`` and ``emfx`` estimate how state minimum wage increases changed
       county teen employment in one regression with an effect for each cohort
       and year.
   * - :doc:`Nonparametric IV <../user_guide/example_npiv>`
     - ``npiv`` uses earnings as an instrument to estimate how the share of
       British household budgets spent on food changes with total spending.

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
