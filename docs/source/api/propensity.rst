.. _propensity:

Propensity Score Methods
========================

Placeholder for background.

Theoretical Background
----------------------

.. note::
   Theoretical background for propensity score methods in DiD to be added.

Functions
---------

.. currentmodule:: didpy

Core Propensity Score Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   calculate_pscore_ipt

AIPW Estimators
~~~~~~~~~~~~~~~

Augmented inverse propensity weighted estimators combine propensity score weighting
with outcome regression for improved efficiency:

.. autosummary::
   :toctree: generated/

   aipw_did_panel
   aipw_did_rc_imp1
   aipw_did_rc_imp2

IPW Estimators
~~~~~~~~~~~~~~

Core inverse propensity weighted estimators:

.. autosummary::
   :toctree: generated/

   ipw_rc
