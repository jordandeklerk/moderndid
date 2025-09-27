.. _api-drdid:

Two-Period DiD Estimators
==========================

The drdid module provides all two-period difference-in-differences estimators,
including doubly robust, inverse propensity weighted, and outcome regression methods.
These estimators follow the frameworks from `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_
and related literature.

Doubly Robust DiD
------------------

.. currentmodule:: moderndid

High-Level Wrapper
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid

Panel Data Estimators
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid_imp_panel
   drdid_panel

Repeated Cross-Section Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   drdid_imp_local_rc
   drdid_imp_rc
   drdid_rc
   drdid_trad_rc

IPW DiD Estimators
------------------

High-Level Wrapper
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   ipwdid

Panel Data Estimators
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   ipw_did_panel
   std_ipw_did_panel

Repeated Cross-Section Estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   ipw_did_rc
   std_ipw_did_rc

Outcome Regression Estimators
------------------------------

High-Level Wrapper
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   ordid

Two-Way Fixed Effects
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   twfe_did_panel
   twfe_did_rc

Regression DiD
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/drdid/
   :nosignatures:

   reg_did_panel
   reg_did_rc
