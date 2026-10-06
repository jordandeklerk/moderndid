.. _api-didtriple:

Triple DiD
==========

Following `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_,
:func:`~moderndid.ddd` estimates treatment effects when a unit is treated only
if its group enables the policy and it belongs to the eligible part of the
population. Unlike standard DiD, the design tolerates parallel trends violations
that are specific to a group or to the eligible part.
:func:`~moderndid.agg_ddd` aggregates the group-time effects of a staggered
design into an event study, summaries by group or calendar period, or one
overall effect. See the :ref:`triple differences example <example_triple_did>`
for a full analysis of the agricultural insurance data and the
:ref:`background page <background-tripledid>` for the identification argument.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd
   agg_ddd

Two-period estimators
---------------------

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd_panel
   ddd_rc

Multi-period estimators
-----------------------

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd_mp
   ddd_mp_rc

Result objects
--------------

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   DDDMultiPeriodResult
   DDDMultiPeriodRCResult
   DDDPanelResult
   DDDRCResult
   DDDAggResult
