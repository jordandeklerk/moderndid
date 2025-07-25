=============
API Reference
=============

This reference manual details functions, modules, and objects
included in didpy, describing what they are and what they do.
For learning how to use didpy, see the :doc:`../tutorial/index`.

.. _api-overview:

Python API
==========

didpy adheres to the following naming convention:

* Public modules and functions should have an all lowercase name, potentially
  with underscores separating words.
* Private modules should be prefixed with an underscore, ``_like_this``.

.. _api-importing:

Importing from didpy
====================

The recommended way to use didpy functions is the following:

.. code-block:: python

   import didpy
   result = didpy.drdid_imp_panel(...)

Main estimation functions
=========================

.. autosummary::
   :toctree: generated/

   didpy.att_gt
   didpy.aggte
   didpy.drdid
   didpy.ipwdid
   didpy.ordid

Doubly-robust DiD estimators
============================

Panel data
----------

.. autosummary::
   :toctree: generated/

   didpy.drdid_imp_panel
   didpy.drdid_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   didpy.drdid_imp_local_rc
   didpy.drdid_imp_rc
   didpy.drdid_rc
   didpy.drdid_trad_rc

Inverse propensity weighted DiD estimators
==========================================

Panel data
----------

.. autosummary::
   :toctree: generated/

   didpy.ipw_did_panel
   didpy.std_ipw_did_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   didpy.ipw_did_rc
   didpy.std_ipw_did_rc

Outcome regression DiD estimators
=================================

Panel data
----------

.. autosummary::
   :toctree: generated/

   didpy.reg_did_panel
   didpy.twfe_did_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   didpy.reg_did_rc
   didpy.twfe_did_rc

Propensity score utilities
==========================

Core propensity score estimation
--------------------------------

.. autosummary::
   :toctree: generated/

   didpy.calculate_pscore_ipt

AIPW estimators
---------------

.. autosummary::
   :toctree: generated/

   didpy.aipw_did_panel
   didpy.aipw_did_rc_imp1
   didpy.aipw_did_rc_imp2

IPW estimators
--------------

.. autosummary::
   :toctree: generated/

   didpy.ipw_rc

Bootstrap inference
===================

Panel data bootstrap
--------------------

.. autosummary::
   :toctree: generated/

   didpy.wboot_drdid_imp_panel
   didpy.wboot_dr_tr_panel
   didpy.wboot_ipw_panel
   didpy.wboot_std_ipw_panel
   didpy.wboot_reg_panel
   didpy.wboot_twfe_panel

Repeated cross-section bootstrap
--------------------------------

.. autosummary::
   :toctree: generated/

   didpy.wboot_drdid_rc1
   didpy.wboot_drdid_rc2
   didpy.wboot_drdid_ipt_rc1
   didpy.wboot_drdid_ipt_rc2
   didpy.wboot_ipw_rc
   didpy.wboot_std_ipw_rc
   didpy.wboot_reg_rc
   didpy.wboot_twfe_rc

Multiplier bootstrap
--------------------

.. autosummary::
   :toctree: generated/

   didpy.mboot_did
   didpy.mboot_twfep_did

Supporting functions
====================

Weighted OLS
------------

.. autosummary::
   :toctree: generated/

   didpy.wols_panel
   didpy.wols_rc
