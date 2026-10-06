.. _api-diddynamic:

Dynamic covariate balancing
===========================

Following `Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_,
:func:`~moderndid.diddynamic.dyn_balancing` compares average outcomes under two
treatment histories when treatment can switch on and off over time. Since
treatment in each period may depend on past outcomes and treatments, its
weights balance covariates period by period. See the
:ref:`dynamic covariate balancing example <example_dyn_balancing>` for a full
analysis of the democracy and economic growth data and the
:ref:`background page <background-diddynamic>` for the estimator.

.. currentmodule:: moderndid.diddynamic

Main functions
--------------

.. autosummary::
   :toctree: generated/diddynamic/
   :nosignatures:

   dyn_balancing

Result objects
--------------

.. autosummary::
   :toctree: generated/diddynamic/
   :nosignatures:

   DynBalancingResult
   DynBalancingHistoryResult
   DynBalancingHetResult
