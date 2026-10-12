.. _api-bootstrap:

Bootstrap
=========

The staggered, two-period, and triple differences estimators call these
functions when you ask for bootstrapped inference. The multiplier bootstrap
reweights each unit's influence function instead of reestimating the model.
The weighted bootstrap draws random weights for the units and reestimates the
model on every draw.

.. currentmodule:: moderndid

Multiplier bootstrap
--------------------

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   mboot
   mboot_did
   mboot_twfep_did
   mboot_ddd

Weighted bootstrap
------------------

Panel data
^^^^^^^^^^

.. autosummary::
   :toctree: generated/bootstrap/
   :nosignatures:

   wboot_drdid_imp_panel
   wboot_dr_tr_panel
   wboot_ipw_panel
   wboot_std_ipw_panel
   wboot_reg_panel
   wboot_twfe_panel
   wboot_ddd

Repeated cross-sections
^^^^^^^^^^^^^^^^^^^^^^^

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
