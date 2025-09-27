.. _api-honestdid:

Honest DiD
==========

The Honest DiD module provides sensitivity analysis tools for potential violations
of the parallel trends assumption, following `Rambachan and Roth (2023) <https://arxiv.org/abs/2203.04511>`_.
This allows researchers to assess the robustness of their DiD estimates to
various forms of pre-trend violations.

Main Functions
--------------

.. currentmodule:: moderndid

High-Level Interface
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   honest_did
   construct_original_cs

Sensitivity Analysis
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   create_sensitivity_results_rm
   create_sensitivity_results_sm

Confidence Intervals
--------------------

ARP Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_arp_ci
   compute_arp_nuisance_ci
   compute_least_favorable_cv
   compute_vlo_vup_dual
   lp_conditional_test
   test_in_identified_set
   test_in_identified_set_flci_hybrid
   test_in_identified_set_lf_hybrid

Fixed-Length Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_flci
   folded_normal_quantile
   maximize_bias
   minimize_variance

Restriction Types
-----------------

Relative Magnitude Restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_rm
   compute_identified_set_rm
   compute_conditional_cs_rmb
   compute_identified_set_rmb
   compute_conditional_cs_rmm
   compute_identified_set_rmm

Smoothness Restrictions
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_sd
   compute_identified_set_sd
   compute_conditional_cs_sdb
   compute_identified_set_sdb
   compute_conditional_cs_sdm
   compute_identified_set_sdm

Combined Restrictions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_sdrm
   compute_identified_set_sdrm
   compute_conditional_cs_sdrmb
   compute_identified_set_sdrmb
   compute_conditional_cs_sdrmm
   compute_identified_set_sdrmm
