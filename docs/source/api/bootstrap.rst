.. _bootstrap:

Bootstrap Methods
=================

Placeholder for background.

Theoretical Background
----------------------

.. note::
   Theoretical background for bootstrap inference methods in DiD to be added.

Functions
---------

.. currentmodule:: didpy

Panel Data Bootstrap
~~~~~~~~~~~~~~~~~~~~

Bootstrap methods specifically designed for panel data structures:

.. autosummary::
   :toctree: generated/

   wboot_dr_tr_panel
   wboot_drdid_imp_panel
   wboot_ipw_panel
   wboot_reg_panel
   wboot_std_ipw_panel
   wboot_twfe_panel

Repeated Cross-Section Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bootstrap methods for repeated cross-sectional data:

.. autosummary::
   :toctree: generated/

   wboot_drdid_ipt_rc1
   wboot_drdid_ipt_rc2
   wboot_drdid_rc1
   wboot_drdid_rc2
   wboot_ipw_rc
   wboot_reg_rc
   wboot_std_ipw_rc
   wboot_twfe_rc

Multiplier Bootstrap
~~~~~~~~~~~~~~~~~~~~

Multiplier bootstrap methods for inference based on influence functions:

.. autosummary::
   :toctree: generated/

   mboot_did
   mboot_twfep_did
