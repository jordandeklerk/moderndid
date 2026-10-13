.. _api-etwfe:

Extended TWFE
=============

:func:`~moderndid.etwfe` fits a two-way fixed effects regression that allows
the treatment effect to differ across cohorts and periods. Once the model
is fitted, :func:`~moderndid.emfx` averages the cell-level effects to answer
your question about the policy: an overall effect, an event study, or a
summary by cohort or calendar period.

The :ref:`extended TWFE example <example_etwfe>` follows those choices with
the minimum wage data. The :ref:`background page <background-etwfe>` explains
the regression and assumptions developed by
`Wooldridge (2023) <https://doi.org/10.1093/ectj/utad016>`_ and
`Wooldridge (2025) <https://doi.org/10.1007/s00181-025-02807-z>`_.

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
