.. _api-plotting:

Plotting
========

The plotting module provides a unified interface for visualizing difference-in-differences
results, sensitivity analyses, and event studies. All plotting functions return
plotnine ``ggplot`` objects that can be further customized using standard plotnine syntax.

DiD Result Plots
----------------

High-level functions for plotting treatment effect estimates from DiD analyses.
Automatically selects the appropriate visualization based on the result type
(group-time ATT, event study, and aggregated effects).

.. currentmodule:: moderndid.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_gt
   plot_event_study
   plot_agg

Continuous Treatment Plots
--------------------------

Functions for visualizing dose-response relationships from continuous treatment DiD.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_dose_response

Intertemporal DiD Plots
-----------------------

Functions for visualizing dynamic treatment effects from the intertemporal DiD estimator.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_multiplegt

Sensitivity Analysis Plots
--------------------------

Functions for visualizing HonestDiD sensitivity analysis results.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_sensitivity
