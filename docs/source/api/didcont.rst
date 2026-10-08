.. _api-didcont:

Continuous treatment DiD
========================

The December 2025 paper by `Callaway, Goodman-Bacon, and Sant'Anna
<https://psantanna.com/files/CGBS_v4.pdf>`_ develops the identification
framework behind :func:`~moderndid.cont_did`. The function returns fitted
level effects and their dose derivatives, or an event study by time since
treatment. A causal interpretation of those derivatives requires stronger
restrictions than ordinary parallel trends. The :ref:`background page
<background-didcont>` explains the target populations, identifying assumptions,
and estimation paths supported by the package.

The :ref:`continuous treatment example <example_cont_did>` uses county
employment data to fit dose curves and an event study with the public
estimator. The :ref:`fracking replication <example_cont_did_replication>`
reproduces the figures from the authors' separate 2024 event-study paper
and explains how its pooling procedure differs.

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
