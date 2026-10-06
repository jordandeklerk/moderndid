.. _api-didinter:

Intertemporal DiD
=================

Following `de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_,
:func:`~moderndid.did_multiplegt` estimates treatment effects when the
treatment can take many values or switch on and off and when past treatments
may still affect the outcome. See the
:ref:`intertemporal treatment example <example_inter_did>` for a full analysis
of the banking deregulation data and the
:ref:`background page <background-didinter>` for the effects it estimates.

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
