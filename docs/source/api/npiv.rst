.. _api-npiv:

Nonparametric IV
================

Following `Chen, Christensen, and Kankanala (2024) <https://arxiv.org/abs/2107.11869>`_,
:func:`~moderndid.npiv` estimates a nonparametric instrumental variables model
with B-spline sieves and builds uniform confidence bands around the estimated
function. The lower-level functions below run the sieve estimation, choose the
sieve dimension from the data, and compute the bands. See the
:ref:`nonparametric IV example <example_npiv>` for a full analysis of the Engel
household expenditure data and the :ref:`background page <background-npiv>`
for the estimator and its bands.

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
