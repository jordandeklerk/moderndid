.. _api-didml:

Machine learning DiD
====================

Following `Hatamyar, Kreif, Rocha, and Huber (2023) <https://arxiv.org/abs/2310.11962>`_,
:func:`~moderndid.didml` estimates group-time average treatment effects in a
staggered design along with each treated unit's conditional effect. It fits
the nuisance functions with cross-fitted machine learning models and combines
them in a doubly robust score. :func:`~moderndid.aggte_didml` and
:func:`~moderndid.dynamic_cates` summarize the group-time and unit-level results
by event time. The
heterogeneity functions below test how the conditional effects vary with
covariates.

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
