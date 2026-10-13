.. _api-multiperiod:

Staggered DiD
=============

When units adopt treatment at different times, :func:`~moderndid.att_gt`
estimates an average treatment effect for each adoption cohort in each
period. Keeping those effects separate lets you examine how the response
differs across cohorts and lengths of exposure. To answer a question about
an average across those comparisons, :func:`~moderndid.aggte` combines them
into an event study, a summary by cohort or calendar period, or one overall
effect.

The :ref:`staggered DiD example <example_staggered_did>` follows this workflow
with the minimum wage data. The :ref:`background page <background-did>`
develops the identifying assumptions and aggregation weights from
`Callaway and Sant'Anna (2021) <https://www.sciencedirect.com/science/article/pii/S0304407620303948>`_.

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
