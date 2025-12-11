.. _api-plotting:

Plotting
========

The plotting module provides a unified interface for visualizing difference-in-differences
results, sensitivity analyses, and event studies. All plotting functions return a
:class:`PlotCollection` object that enables method chaining and post-plot customization.

The module supports flexible theming through :class:`PlotTheme` objects, with built-in
themes including ``"default"``, ``"minimal"``, ``"publication"``, and ``"colorful"``.

DiD Result Plots
----------------

High-level function for plotting treatment effect estimates from DiD analyses.
Automatically selects the appropriate visualization based on the aggregation type
(group-time ATT, event study, and continuous treatment DiD).

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_did

Continuous Treatment Plots
--------------------------

Functions for visualizing dose-response relationships from continuous treatment DiD.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_cont_did

Sensitivity Analysis Plots
--------------------------

Functions for visualizing HonestDiD sensitivity analysis results.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_sensitivity_sm
   plot_sensitivity_rm
   plot_sensitivity_event_study

Theming
-------

Classes for customizing plot appearance.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   PlotTheme
