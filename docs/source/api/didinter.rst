.. _api-didinter:

Intertemporal DiD
=================

The intertemporal treatment effects module provides methods for difference-in-differences
estimation with non-binary, non-absorbing (time-varying) treatments where lagged
treatments may affect the outcome. The implementation follows
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_.

Main Functions
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didinter/
   :nosignatures:

   did_multiplegt

Inference
---------

.. currentmodule:: moderndid.didinter.variance

.. autosummary::
   :toctree: generated/didinter/
   :nosignatures:

   compute_clustered_variance
   compute_joint_test

Result Objects
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didinter/
   :nosignatures:

   DIDInterResult
   DIDInterEffectsResult
   DIDInterPlacebosResult
   DIDInterATEResult
