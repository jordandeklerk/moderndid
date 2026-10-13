.. _api-diddynamic:

Dynamic covariate balancing
===========================

:func:`~moderndid.diddynamic.dyn_balancing` compares average outcomes under two
treatment histories. Because treatment in each period may depend on past
outcomes and treatments, the method balances covariates period by period
rather than making a single adjustment at baseline.

The :ref:`dynamic covariate balancing example <example_dyn_balancing>` uses
the democracy and economic growth data to choose histories and interpret
the resulting contrast. The :ref:`background page <background-diddynamic>`
develops the sequential comparisons and assumptions in
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_.

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
