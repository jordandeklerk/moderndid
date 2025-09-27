.. _api-multiperiod:

Multi-Period Staggered DiD
===========================

The multi-period DiD module provides functions for analyzing treatment effects
with staggered treatment timing, following the methodology of
`Callaway and Sant'Anna (2021) <https://www.sciencedirect.com/science/article/pii/S0304407620303948>`_.

This module handles the complexities of difference-in-differences analysis when
treatment is adopted at different times across units, avoiding the bias that
can arise from two-way fixed effects regressions in such settings.

Main Functions
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/multiperiod/
   :nosignatures:

   att_gt
   aggte
