.. _api-propensity:

Propensity Score Functions
===========================

The propensity score module provides core functions for calculating propensity
scores via inverse probability tilting (IPT), augmented inverse
probability weighting (AIPW), and inverse probability weighting (IPW).

.. currentmodule:: moderndid

Inverse Probability Tilting
---------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   calculate_pscore_ipt

Augmented Inverse Probability Weighting
---------------------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   aipw_did_panel
   aipw_did_rc_imp1
   aipw_did_rc_imp2

Inverse Probability Weighting
------------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   ipw_rc
