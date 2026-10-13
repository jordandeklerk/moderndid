.. _api-didml:

Machine learning DiD
====================

:func:`~moderndid.didml` estimates group-time average treatment effects in a
staggered design along with conditional effects for treated units. Following
`Hatamyar, Kreif, Rocha, and Huber (2023) <https://arxiv.org/abs/2310.11962>`_,
it fits the nuisance functions with cross-fitted machine learning models and
combines them in a doubly robust score.

To examine how effects change with exposure, :func:`~moderndid.aggte_didml`
and :func:`~moderndid.dynamic_cates` summarize the group-time and unit-level
results by event time. The heterogeneity functions below help you investigate
how conditional effects vary with covariates.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   didml
   aggte_didml
   dynamic_cates

Heterogeneity analysis
----------------------

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   het_prep
   blp_eventtimes
   clan_glhtest
   clan_ttest

Doubly robust score
-------------------

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   lnw_did
   amle_weights

Nuisance models
---------------

.. currentmodule:: moderndid.didml.nuisance

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   fit_rlearner
   fit_causal_forest
   fit_delta

Result objects
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   DIDMLResult
   DIDMLAggResult
   BLPResult
   CLANResult
