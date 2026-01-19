.. _api-propensity:

Propensity Score Functions
===========================

The propensity score module provides core functions for calculating propensity
scores and implementing augmented inverse probability weighting (AIPW) estimators.
These functions are used internally by the main DiD estimators but are also
available for direct use.

Main Functions
--------------

.. currentmodule:: moderndid

Propensity Score Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   calculate_pscore_ipt

AIPW Estimators
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   aipw_did_panel
   aipw_did_rc_imp1
   aipw_did_rc_imp2

IPW Functions
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   ipw_rc
