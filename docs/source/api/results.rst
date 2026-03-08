.. _api-results:

Result Extraction
=================

The :func:`~moderndid.to_df` function converts any estimator result object
into a polars DataFrame. It auto-detects the result type, so there is one
function to remember regardless of which estimator produced the result.

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/results/
   :nosignatures:

   to_df

Individual Converters
---------------------

The named converters are the underlying implementations that
:func:`~moderndid.to_df` dispatches to. They are available for direct use
when you need explicit control over the conversion.

.. currentmodule:: moderndid.core.converters

.. autosummary::
   :toctree: generated/results/
   :nosignatures:

   aggteresult_to_polars
   mpresult_to_polars
   dddaggresult_to_polars
   dddmpresult_to_polars
   doseresult_to_polars
   pteresult_to_polars
   honestdid_to_polars
   didinterresult_to_polars
