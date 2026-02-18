.. _api-didcont:

Continuous Treatment DiD
========================

The continuous treatment DiD module provides methods for handling continuous
treatment variables in difference-in-differences settings. This includes
nonparametric instrumental variables (NPIV) estimation and spline-based methods.
The implementation follows `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://psantanna.com/files/CGBS.pdf>`_.

Main Functions
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   cont_did

Non-Parametric Instrumental Variables
--------------------------------------

Core NPIV Functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   npiv
   npiv_est
   npiv_choose_j
   npiv_j
   npiv_jhat_max

Uniform Confidence Bounds
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   compute_cck_ucb
   compute_ucb

Spline Functions
----------------

Product Splines
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   prodspline

B-Spline Basis
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   BSpline

Result Objects
--------------

.. currentmodule:: moderndid.didcont

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   NPIVResult
   PTEAggteResult
   GroupTimeATTResult
   DoseResult
