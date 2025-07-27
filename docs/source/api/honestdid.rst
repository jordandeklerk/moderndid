.. _honestdid:

Honest DiD Sensitivity Analysis
===============================

Placeholder for Honest DiD sensitivity analysis.

Theoretical Background
----------------------

.. note::
   Theoretical background for Rambachan and Roth (2023) sensitivity analysis methodology to be added.

Functions
---------

.. currentmodule:: didpy

Main
~~~~~

.. autosummary::
   :toctree: generated/

   honest_did

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   construct_original_cs
   create_sensitivity_results_rm
   create_sensitivity_results_sm

Bounds and Constraints
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   compute_delta_sd_lowerbound_m
   compute_delta_sd_upperbound_m
   create_monotonicity_constraint_matrix
   create_pre_period_constraint_matrix
   create_second_difference_matrix
   create_sign_constraint_matrix

Fixed Length Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   affine_variance
   compute_flci
   folded_normal_quantile
   maximize_bias
   minimize_variance

ARP Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   compute_arp_ci
   compute_arp_nuisance_ci
   compute_least_favorable_cv
   compute_vlo_vup_dual
   lp_conditional_test
   test_delta_lp

Delta Methods - Relative Magnitudes (RM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   compute_conditional_cs_rm
   compute_conditional_cs_rmb
   compute_conditional_cs_rmm
   compute_identified_set_rm
   compute_identified_set_rmb
   compute_identified_set_rmm

Delta Methods - Second Differences (SD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   compute_conditional_cs_sd
   compute_conditional_cs_sdb
   compute_conditional_cs_sdm
   compute_identified_set_sd
   compute_identified_set_sdb
   compute_identified_set_sdm

Delta Methods - Combined (SDRM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   compute_conditional_cs_sdrm
   compute_conditional_cs_sdrmb
   compute_conditional_cs_sdrmm
   compute_identified_set_sdrm
   compute_identified_set_sdrmb
   compute_identified_set_sdrmm

Plotting
~~~~~~~~

.. autosummary::
   :toctree: generated/

   event_study_plot
   plot_sensitivity_rm
   plot_sensitivity_sm
