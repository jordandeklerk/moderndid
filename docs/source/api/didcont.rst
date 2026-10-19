.. _api-didcont:

Continuous treatment
====================

When units receive different amounts of treatment, :func:`~moderndid.cont_did`
fits level effects across doses and their derivatives, or an event study by
time since treatment. The December 2025 paper by
`Callaway, Goodman-Bacon, and Sant'Anna
<https://psantanna.com/files/CGBS_v4.pdf>`_ develops the identification
framework. The :ref:`background page <background-didcont>` explains whose
effects each curve describes and which estimation paths the package supports.

.. admonition:: A fitted slope can include selection
   :class: warning

   Under ordinary parallel trends, moving along the dose curve changes both
   the dose and the population receiving it. Interpreting its derivative as
   a causal response requires stronger restrictions that remove this
   selection term.

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
