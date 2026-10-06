.. _api-panel:

Panel utilities
===============

These functions check and reshape a panel before you estimate. Every function
accepts any Arrow-compatible DataFrame and hands back data in the same type you
passed in. The :ref:`panel utilities guide <panel-utilities>` uses them to
prepare a real panel.

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
