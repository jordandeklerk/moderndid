.. _api-didtriple:

Triple DiD
==========

The triple DiD (DDD) module provides estimators for settings where
units must satisfy two criteria to be treated: belonging to a group that enables treatment
and being in an eligible partition of the population. This design allows for both
group-specific and partition-specific violations of parallel trends, relaxing the
assumptions required by standard DiD. The implementation follows
`Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_.

Main Functions
--------------

.. currentmodule:: moderndid

High-Level Wrapper
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd

Two-Period Estimators
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd_panel

Multi-Period Estimators
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   ddd_mp
   agg_ddd

Data Generation
---------------

.. autosummary::
   :toctree: generated/didtriple/
   :nosignatures:

   gen_dgp_2periods
   gen_dgp_mult_periods
   generate_simple_ddd_data
