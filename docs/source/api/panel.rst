.. _api-panel:

Panel Utilities
===============

The panel module provides diagnostic, validation, and transformation tools
for preparing panel data before estimation. All functions accept any
Arrow-compatible DataFrame and return the same format.

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
