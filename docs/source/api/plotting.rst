.. _api-plotting:

Plotting
========

Each function here takes a result and returns a plotnine ``ggplot`` that you
can extend with layers, scales, and themes. The :ref:`plotting guide <plotting>`
walks through the plots and how to customize them.

.. currentmodule:: moderndid.plots

Treatment effect plots
----------------------

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_gt
   plot_event_study
   plot_agg

Continuous treatment plots
--------------------------

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_dose_response

Intertemporal DiD plots
-----------------------

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_multiplegt

Dynamic covariate balancing plots
---------------------------------

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_dyn_balancing
   plot_dyn_balancing_history
   plot_dyn_balancing_het
   plot_dyn_balancing_coefs

Sensitivity analysis plots
--------------------------

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   plot_sensitivity

Themes
------

Each theme replaces the default gray panels and grid lines when you add it to a
plot with ``+``.

.. autosummary::
   :toctree: generated/plotting/
   :nosignatures:

   theme_moderndid
   theme_publication
   theme_minimal

Plot data
---------

To get the numbers behind a plot as a polars DataFrame, pass its result to
:func:`~moderndid.to_df`. The :ref:`result extraction reference <api-results>`
lists the converter for each result type.
