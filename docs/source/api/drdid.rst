.. _api-drdid:

Doubly Robust DiD
=================

The drdid module provides all two-period difference-in-differences estimators,
including doubly robust, inverse propensity weighted, and outcome regression methods.
These estimators follow the frameworks from `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_
and related literature.

.. currentmodule:: moderndid

Main Functions
--------------

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid
   ipwdid
   ordid

Panel Data Estimators
---------------------

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid_imp_panel
   drdid_panel
   ipw_did_panel
   std_ipw_did_panel
   reg_did_panel
   twfe_did_panel

Repeated Cross-Section Estimators
---------------------------------

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid_imp_rc
   drdid_imp_local_rc
   drdid_rc
   drdid_trad_rc
   ipw_did_rc
   std_ipw_did_rc
   reg_did_rc
   twfe_did_rc
