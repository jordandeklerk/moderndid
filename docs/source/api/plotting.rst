.. _api-plotting:

Plotting
========

.. note::

   Plotting functionality requires matplotlib, which is an optional dependency.
   Install it with: ``pip install moderndid[plots]``

The plotting module provides a unified interface for visualizing difference-in-differences
results, sensitivity analyses, and event studies. All plotting functions return a
:class:`PlotCollection` object that enables method chaining and post-plot customization.

The module supports flexible theming through :class:`PlotTheme` objects, with built-in
themes including ``"default"``, ``"minimal"``, ``"publication"``, and ``"colorful"``.

Importing Plotting Functions
----------------------------

Since plotting is optional, functions must be imported from their submodules:

.. code-block:: python

   from moderndid.did.plots import plot_att_gt, plot_event_study, plot_did
   from moderndid.didcont.plots import plot_cont_did
   from moderndid.didhonest.plots import plot_sensitivity_sm, plot_sensitivity_rm
   from moderndid.plots import PlotCollection, PlotTheme

DiD Result Plots
----------------

High-level function for plotting treatment effect estimates from DiD analyses.
Automatically selects the appropriate visualization based on the aggregation type
(group-time ATT, event study, and continuous treatment DiD).

.. currentmodule:: moderndid.did.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_did
   plot_att_gt
   plot_event_study

Continuous Treatment Plots
--------------------------

Functions for visualizing dose-response relationships from continuous treatment DiD.

.. currentmodule:: moderndid.didcont.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_cont_did

Sensitivity Analysis Plots
--------------------------

Functions for visualizing HonestDiD sensitivity analysis results.

.. currentmodule:: moderndid.didhonest.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_sensitivity_sm
   plot_sensitivity_rm
   plot_sensitivity_event_study

Theming
-------

Classes for customizing plot appearance.

.. currentmodule:: moderndid.plots

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   PlotTheme
