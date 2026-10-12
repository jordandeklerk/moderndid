.. _api-didtriple:

Triple DiD
==========

When a policy reaches only eligible units within adopting groups,
:func:`~moderndid.ddd` uses the additional eligibility comparison to estimate
the effect on treated units. This design allows untreated trends to differ
by group or eligibility when those differences cancel in the triple
comparison. For staggered adoption, :func:`~moderndid.agg_ddd` combines the
group-time effects into an event study, summaries by group or calendar
period, or one overall effect.

The :ref:`triple differences example <example_triple_did>` works through the
agricultural insurance data. The :ref:`background page <background-tripledid>`
explains the comparison and the identifying restrictions developed by
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_.

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
