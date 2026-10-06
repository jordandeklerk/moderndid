.. _api-etwfe:

Extended TWFE
=============

Following `Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_ and
`Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_,
:func:`~moderndid.etwfe` fits a two-way fixed effects regression with a
separate treatment effect for each cohort and period. :func:`~moderndid.emfx`
aggregates those cell-level effects into one overall effect, an event study, or
summaries by cohort or calendar period. See the
:ref:`extended TWFE example <example_etwfe>` for a full analysis of the minimum
wage data and the :ref:`background page <background-etwfe>` for the regression
and its assumptions.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/etwfe/
   :nosignatures:

   etwfe
   emfx

Result objects
--------------

.. autosummary::
   :toctree: generated/etwfe/
   :nosignatures:

   EtwfeResult
   EmfxResult
