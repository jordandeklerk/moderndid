.. module:: moderndid

.. _api:

API Reference
=============

This is the class and function reference of **ModernDiD**, and the following content
is generated automatically from the code documentation strings.
Please refer to the `full user guide <https://moderndid.readthedocs.io>`_ for
further details, as this low-level documentation may not be enough to give
full guidelines on their use.

.. note::
   The following modules are considered stable and are supported by the package. There are
   additional modules under development that are not yet supported and will be added in future releases.

.. _api-multiperiod:

Multi-Period Staggered DiD
--------------------------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/multiperiod/
   :nosignatures:

   att_gt
   aggte

.. _api-drdid-estimators:

Doubly Robust DiD
-----------------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid
   drdid_imp_panel
   drdid_panel
   drdid_imp_local_rc
   drdid_imp_rc
   drdid_rc
   drdid_trad_rc

.. _api-ipw-estimators:

IPW DiD Estimators
^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/ipw/
   :nosignatures:

   ipwdid
   ipw_did_panel
   std_ipw_did_panel
   ipw_did_rc
   std_ipw_did_rc

.. _api-or-estimators:

Outcome Regression Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/or/
   :nosignatures:

   ordid
   twfe_did_panel
   twfe_did_rc
   reg_did_panel
   reg_did_rc

.. _api-honestdid:

Honest DiD
----------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   honest_did
   construct_original_cs
   create_sensitivity_results_rm
   create_sensitivity_results_sm

.. _api-honestdid-arp-ci:

ARP Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/ci/
   :nosignatures:

   compute_arp_ci
   compute_arp_nuisance_ci
   compute_least_favorable_cv
   compute_vlo_vup_dual
   lp_conditional_test
   test_in_identified_set
   test_in_identified_set_flci_hybrid
   test_in_identified_set_lf_hybrid

.. _api-honestdid-ci:

Fixed-Length Confidence Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/ci/
   :nosignatures:

   compute_flci
   folded_normal_quantile
   maximize_bias
   minimize_variance

.. _api-honestdid-rm:

Relative Magnitude Restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/rm/
   :nosignatures:

   compute_conditional_cs_rm
   compute_identified_set_rm
   compute_conditional_cs_rmb
   compute_identified_set_rmb
   compute_conditional_cs_rmm
   compute_identified_set_rmm

.. _api-honestdid-sd:

Smoothness Restrictions
^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/sd/
   :nosignatures:

   compute_conditional_cs_sd
   compute_identified_set_sd
   compute_conditional_cs_sdb
   compute_identified_set_sdb
   compute_conditional_cs_sdm
   compute_identified_set_sdm

.. _api-honestdid-combined:

Combined Restrictions
^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/honestdid/combined/
   :nosignatures:

   compute_conditional_cs_sdrm
   compute_identified_set_sdrm
   compute_conditional_cs_sdrmb
   compute_identified_set_sdrmb
   compute_conditional_cs_sdrmm
   compute_identified_set_sdrmm

.. _api-didcont:

Continuous Treatment DiD
------------------------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   npiv
   npiv_est
   npiv_choose_j
   npiv_j
   npiv_jhat_max
   compute_cck_ucb
   compute_ucb
   prodspline

.. _api-propensity:

Propensity Score Functions
---------------------------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   calculate_pscore_ipt
   aipw_did_panel
   aipw_did_rc_imp1
   aipw_did_rc_imp2
   ipw_rc

.. _api-bootstrap:

Bootstrap Functions
-------------------
.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   mboot
   mboot_did
   mboot_twfep_did
   wboot_dr_tr_panel
   wboot_drdid_imp_panel
   wboot_drdid_rc1
   wboot_drdid_rc2
   wboot_drdid_ipt_rc1
   wboot_drdid_ipt_rc2
   wboot_ipw_panel
   wboot_ipw_rc
   wboot_reg_panel
   wboot_reg_rc
   wboot_std_ipw_panel
   wboot_std_ipw_rc
   wboot_twfe_panel
   wboot_twfe_rc
