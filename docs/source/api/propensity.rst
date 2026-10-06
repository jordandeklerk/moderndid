.. _api-propensity:

Propensity scores
=================

The doubly robust and weighting estimators on the
:doc:`two-period DiD page <drdid>` build on these functions.
Among them, :func:`~moderndid.calculate_pscore_ipt` fits the propensity score
itself by inverse probability tilting. The other functions turn fitted propensity scores into
weighted estimates of the average treatment effect on the treated.

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
