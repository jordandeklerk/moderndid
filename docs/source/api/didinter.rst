.. _api-didinter:

Intertemporal DiD
=================

When treatment can switch on and off or change in intensity, a unit's past
exposure may still affect its current outcome. :func:`~moderndid.did_multiplegt`
estimates effects in this setting using the comparisons developed by
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_.
The :ref:`background page <background-didinter>` explains which treatment
changes contribute to each effect and how to interpret its normalization.

The :ref:`intertemporal treatment example <example_inter_did>` follows the
banking deregulation data through estimation, placebo comparisons, and
inference.

.. currentmodule:: moderndid

Main functions
--------------

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

Result objects
--------------

.. currentmodule:: moderndid.didinter.container

.. autosummary::
   :toctree: generated/didinter/
   :nosignatures:

   DIDInterResult
   EffectsResult
   PlacebosResult
   ATEResult
   HeterogeneityResult
