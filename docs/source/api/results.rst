.. _api-results:

Result extraction
=================

To inspect fitted effects or use them in a report or another calculation,
pass a supported result to :func:`~moderndid.to_df`. It selects the matching
converter and returns a Polars DataFrame with columns appropriate to that result.
The :doc:`results guide <../user_guide/results>` explains how to read those
fields, and the :ref:`plotting guide <plotting-extracting-data>` shows the
conversion for an event study.

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/results/
   :nosignatures:

   to_df

Converters
----------

Each converter handles one result type. You can call it directly when
working with that type, or use ``to_df`` to select it for you.

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
