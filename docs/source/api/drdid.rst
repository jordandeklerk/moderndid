.. _api-drdid:

Two-period DiD
==============

With an outcome measured before and after treatment, you can estimate the
average effect on treated units using :func:`~moderndid.drdid`,
:func:`~moderndid.ipwdid`, or :func:`~moderndid.ordid`. These functions use
doubly robust estimation, inverse probability weighting, and outcome
regression, respectively. Choose the sampling design as well as the method:
a panel follows the same units through both periods, while repeated
cross-sections observe different samples.

The lower-level routines below accept arrays prepared for one of those
designs. The :ref:`background page <background-drdid>` explains why their
calculations differ and derives the doubly robust estimators of
`Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.

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
