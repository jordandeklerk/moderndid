.. _api-didcont:

Continuous Treatment DiD
========================

The continuous treatment DiD module provides methods for handling continuous
treatment variables in difference-in-differences settings. This includes
spline-based methods for dose-response estimation.
The implementation follows `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://psantanna.com/files/CGBS.pdf>`_.

Main Functions
--------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   cont_did

Spline Functions
----------------

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

   PTEResult
   PTEAggteResult
   GroupTimeATTResult
   DoseResult
