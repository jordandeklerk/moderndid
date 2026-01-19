.. _api-bootstrap:

Bootstrap Functions
===================

The bootstrap module provides functions for computing bootstrap standard errors
and confidence intervals for all DiD estimators in the package. Both weighted
bootstrap and multiplier bootstrap methods are supported.

Multiplier Bootstrap
---------------------

.. currentmodule:: moderndid

Main Functions
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   mboot
   mboot_did
   mboot_twfep_did

Weighted Bootstrap
------------------

Panel Data Bootstrap
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   wboot_drdid_imp_panel
   wboot_dr_tr_panel
   wboot_ipw_panel
   wboot_std_ipw_panel
   wboot_reg_panel
   wboot_twfe_panel

Repeated Cross-Section Bootstrap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   wboot_drdid_rc1
   wboot_drdid_rc2
   wboot_drdid_ipt_rc1
   wboot_drdid_ipt_rc2
   wboot_ipw_rc
   wboot_std_ipw_rc
   wboot_reg_rc
   wboot_twfe_rc
