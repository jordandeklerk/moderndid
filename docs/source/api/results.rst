.. _api-results:

Result extraction
=================

:func:`~moderndid.to_df` converts a ModernDiD result into a polars DataFrame by
detecting the result's type and calling the matching converter below. The
:ref:`plotting guide <plotting-extracting-data>` shows an event study converted
this way and the columns it holds.

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/results/
   :nosignatures:

   to_df

Converters
----------

Since each converter handles one result type, you can call it directly when you
know what your result is.

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
   heterogeneityresult_to_polars
   emfxresult_to_polars
   dynbalancingresult_to_polars
   dynbalancinghistoryresult_to_polars
   dynbalancinghetresult_to_polars
   dynbalancingcoefs_to_polars
