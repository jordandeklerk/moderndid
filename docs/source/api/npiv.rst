.. _api-npiv:

Nonparametric Instrumental Variables
=====================================

The NPIV module provides nonparametric instrumental variables estimation
with uniform confidence bands. The implementation follows
`Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_.

Core NPIV Functions
-------------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   npiv
   npiv_est
   npiv_choose_j
   npiv_j
   npiv_jhat_max

Uniform Confidence Bounds
-------------------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   compute_cck_ucb
   compute_ucb

Spline Functions
----------------

Product Splines
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   prodspline

Result Objects
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   NPIVResult
