.. _api-didcont:

Continuous treatment DiD
========================

Following `Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://psantanna.com/files/CGBS.pdf>`_,
:func:`~moderndid.cont_did` estimates how the effect of a treatment changes
with its dose. It returns dose-response functions for the average treatment
effect and the average causal response, or an event study by time since
treatment. See the :ref:`continuous treatment example <example_cont_did>` for a
full analysis of simulated data and the :ref:`background page <background-didcont>`
for the parameters the estimator targets.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   cont_did

Spline basis
------------

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   BSpline

Result objects
--------------

.. currentmodule:: moderndid.didcont

.. autosummary::
   :toctree: generated/didcont/
   :nosignatures:

   PTEResult
   PTEAggteResult
   GroupTimeATTResult
   DoseResult
