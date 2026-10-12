.. _api-npiv:

Nonparametric IV
================

:func:`~moderndid.npiv` estimates a nonparametric instrumental variables model
with B-spline sieves and constructs uniform confidence bands for the fitted
function. The lower-level routines let you follow the calculation from a
chosen spline space through dimension selection and band construction.

The :ref:`nonparametric IV example <example_npiv>` uses the Engel household
expenditure data to illustrate the fit and its interpretation. The
:ref:`background page <background-npiv>` explains the identification and
inference conditions behind the procedure in
`Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   npiv

Sieve estimation
----------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   npiv_est

Dimension selection
-------------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   npiv_choose_j
   npiv_j
   npiv_jhat_max

Uniform confidence bands
------------------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   compute_ucb
   compute_cck_ucb

Spline basis
------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   prodspline

Result objects
--------------

.. autosummary::
   :toctree: generated/npiv/
   :nosignatures:

   NPIVResult
