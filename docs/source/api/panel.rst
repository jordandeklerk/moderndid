.. _api-panel:

Panel utilities
===============

Before estimating an effect, inspect whether the panel follows each unit
through the periods your comparison needs. The diagnostic and validation
functions below help you locate missing periods, duplicates, and changes
within units. The transformation functions reshape or complete the panel
once you've decided how to handle those observations.

The utilities accept supported Arrow-compatible DataFrames, and functions
that return a transformed DataFrame preserve its input type. The
:ref:`panel utilities guide <panel-utilities>` follows an incomplete panel
through those decisions.

.. currentmodule:: moderndid.core.panel

Diagnostics
-----------

.. autosummary::
   :toctree: generated/panel/
   :nosignatures:

   diagnose_panel
   PanelDiagnostics

Validation
----------

.. autosummary::
   :toctree: generated/panel/
   :nosignatures:

   is_balanced_panel
   has_gaps
   scan_gaps
   are_varying

Transformation
--------------

.. autosummary::
   :toctree: generated/panel/
   :nosignatures:

   make_balanced_panel
   fill_panel_gaps
   complete_data
   deduplicate_panel
   get_first_difference
   get_group
   assign_rc_ids
   panel_to_wide
   wide_to_panel
