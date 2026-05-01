.. _api-didml:

Machine Learning DiD
====================

The machine learning DiD module estimates group-time average treatment effects
on the treated and individual conditional treatment effects (CATTs) in
staggered adoption designs using cross-fitted ML nuisance models. Treatment
effect heterogeneity is recovered through a doubly-robust orthogonal score
plus an augmented minimax-linear weighting scheme. The implementation follows
`Hatamyar, Kreif, Rocha, and Huber (2023) <https://arxiv.org/abs/2310.11962>`_.

Doubly-Robust Score
-------------------

.. currentmodule:: moderndid.didml

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   lnw_did
   amle_weights

Nuisance Backends
-----------------

.. currentmodule:: moderndid.didml.nuisance

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   fit_rlearner
   fit_causal_forest
   fit_delta

Result Objects
--------------

.. currentmodule:: moderndid.didml

.. autosummary::
   :toctree: generated/didml/
   :nosignatures:

   DIDMLResult
   DIDMLAggResult
   BLPResult
   CLANResult
