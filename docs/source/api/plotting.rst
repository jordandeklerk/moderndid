.. _api-plotting:

Plotting
========

The plotting module provides a unified interface for visualizing difference-in-differences
results, sensitivity analyses, and event studies. All plotting functions return
plotnine ``ggplot`` objects that can be further customized using standard plotnine syntax.

DiD Result Plots
----------------

High-level function for plotting treatment effect estimates from DiD analyses.
Automatically selects the appropriate visualization based on the aggregation type
(group-time ATT, event study, and continuous treatment DiD).

.. currentmodule:: moderndid.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_did
   plot_att_gt
   plot_event_study

Continuous Treatment Plots
--------------------------

Functions for visualizing dose-response relationships from continuous treatment DiD.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_cont_did
   plot_dose_response

Sensitivity Analysis Plots
--------------------------

Functions for visualizing HonestDiD sensitivity analysis results.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_sensitivity
