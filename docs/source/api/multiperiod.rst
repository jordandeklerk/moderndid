.. _api-multiperiod:

Staggered DiD
=============

Following `Callaway and Sant'Anna (2021) <https://www.sciencedirect.com/science/article/pii/S0304407620303948>`_,
:func:`~moderndid.att_gt` estimates an average treatment effect on the treated
for each cohort in each period when units adopt treatment at different times.
:func:`~moderndid.aggte` combines those group-time effects into an event study,
a summary by cohort or calendar period, or one overall effect. See the
:ref:`staggered DiD example <example_staggered_did>` for a full analysis of the
minimum wage data and the :ref:`background page <background-did>` for the
assumptions behind the estimator.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/multiperiod/
   :nosignatures:

   att_gt
   aggte

Result objects
--------------

.. autosummary::
   :toctree: generated/multiperiod/
   :nosignatures:

   MPResult
   AGGTEResult
   ATTgtResult
   ComputeATTgtResult
   MPPretestResult
