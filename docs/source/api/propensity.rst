.. _api-propensity:

Propensity scores
=================

The :doc:`two-period DiD estimators <drdid>` use propensity scores to weight
the comparison observations when estimating the average effect on treated
units. :func:`~moderndid.calculate_pscore_ipt` fits the score by inverse
probability tilting. The remaining routines use fitted scores to construct
weighting estimates, with outcome adjustment in the augmented estimators.

The :ref:`two-period background <background-drdid>` explains how the weights
and outcome model enter the comparison and what double robustness requires.

.. currentmodule:: moderndid

Inverse probability tilting
---------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   calculate_pscore_ipt

Augmented inverse probability weighting
---------------------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   aipw_did_panel
   aipw_did_rc_imp1
   aipw_did_rc_imp2

Inverse probability weighting
-----------------------------

.. autosummary::
   :toctree: generated/propensity/
   :nosignatures:

   ipw_rc
