.. _api-drdid:

Two-period DiD
==============

With two periods, :func:`~moderndid.drdid`, :func:`~moderndid.ipwdid`, and
:func:`~moderndid.ordid` estimate the average treatment effect on the treated
by doubly robust, inverse probability weighting, and outcome regression
methods. Below them, one set of lower-level estimators takes panel data as
arrays and another takes repeated cross-sections. The
:ref:`background page <background-drdid>` derives the doubly robust estimators
of `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid
   ipwdid
   ordid

Panel data estimators
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

Repeated cross-section estimators
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
