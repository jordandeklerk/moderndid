=============
API Reference
=============

This reference manual details functions, modules, and objects
included in pyDiD, describing what they are and what they do.
For learning how to use pyDiD, see the :doc:`../tutorial/index`.

.. _api-overview:

Python API
==========

pyDiD adheres to the following naming convention:

* Public modules and functions should have an all lowercase name, potentially
  with underscores separating words.
* Private modules should be prefixed with an underscore, ``_like_this``.

.. _api-importing:

Importing from pyDiD
====================

The recommended way to use pyDiD functions is the following:

.. code-block:: python

   import pydid
   result = pydid.drdid_imp_panel(...)

Main estimation functions
=========================

.. autosummary::
   :toctree: generated/

   pydid.att_gt
   pydid.aggte
   pydid.drdid
   pydid.ipwdid
   pydid.ordid

Doubly-robust DiD estimators
============================

Panel data
----------

.. autosummary::
   :toctree: generated/

   pydid.drdid_imp_panel
   pydid.drdid_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   pydid.drdid_imp_local_rc
   pydid.drdid_imp_rc
   pydid.drdid_rc
   pydid.drdid_trad_rc

Inverse propensity weighted DiD estimators
==========================================

Panel data
----------

.. autosummary::
   :toctree: generated/

   pydid.ipw_did_panel
   pydid.std_ipw_did_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   pydid.ipw_did_rc
   pydid.std_ipw_did_rc

Outcome regression DiD estimators
=================================

Panel data
----------

.. autosummary::
   :toctree: generated/

   pydid.reg_did_panel
   pydid.twfe_did_panel

Repeated cross-sections
-----------------------

.. autosummary::
   :toctree: generated/

   pydid.reg_did_rc
   pydid.twfe_did_rc

Propensity score utilities
==========================

Core propensity score estimation
--------------------------------

.. autosummary::
   :toctree: generated/

   pydid.calculate_pscore_ipt

AIPW estimators
---------------

.. autosummary::
   :toctree: generated/

   pydid.aipw_did_panel
   pydid.aipw_did_rc_imp1
   pydid.aipw_did_rc_imp2

IPW estimators
--------------

.. autosummary::
   :toctree: generated/

   pydid.ipw_rc

Bootstrap inference
===================

Panel data bootstrap
--------------------

.. autosummary::
   :toctree: generated/

   pydid.wboot_drdid_imp_panel
   pydid.wboot_dr_tr_panel
   pydid.wboot_ipw_panel
   pydid.wboot_std_ipw_panel
   pydid.wboot_reg_panel
   pydid.wboot_twfe_panel

Repeated cross-section bootstrap
--------------------------------

.. autosummary::
   :toctree: generated/

   pydid.wboot_drdid_rc1
   pydid.wboot_drdid_rc2
   pydid.wboot_drdid_ipt_rc1
   pydid.wboot_drdid_ipt_rc2
   pydid.wboot_ipw_rc
   pydid.wboot_std_ipw_rc
   pydid.wboot_reg_rc
   pydid.wboot_twfe_rc

Multiplier bootstrap
--------------------

.. autosummary::
   :toctree: generated/

   pydid.mboot_did
   pydid.mboot_twfep_did

Supporting functions
====================

Weighted OLS
------------

.. autosummary::
   :toctree: generated/

   pydid.wols_panel
   pydid.wols_rc
