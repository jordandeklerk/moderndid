.. _plotting:

***************************
Plotting and Visualization
***************************

Every ModernDiD estimator produces a result object that can be passed directly
to a built-in plot function. The visualization layer is built on
`plotnine <https://plotnine.org/>`_, a Python implementation of R's ggplot2.
If you have used ggplot2 in R, the syntax will feel familiar. If you are new
to the grammar of graphics, the essential idea is that plots are built by
composing layers (data mappings, aesthetics, geometric shapes, scales, and
themes) into a single object.

All plot functions return a plotnine ``ggplot`` object. This means you get a
useful default plot immediately, but you can also add layers, swap themes,
and modify any visual element using the full plotnine API.


A first plot
============

The quickest path from data to visualization is three function calls. Load
data, estimate, and plot. Here the minimum wage dataset from
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
provides the input.

.. code-block:: python

    import moderndid as did

    data = did.load_mpdta()

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
    )

    event_study = did.aggte(result, type="dynamic")
    did.plot_event_study(event_study)

.. image:: /_static/images/plot_guide_event_study.png
   :alt: Basic event study plot
   :width: 100%

Built-in plot functions
=======================

ModernDiD provides six plot functions, each designed for a specific type of
treatment effect estimate. All share a common interface with parameters for
confidence intervals, reference lines, axis labels, and titles.


Event studies
-------------

Event study plots are the most common visualization in applied DiD work.
They show treatment effects aligned relative to the period of treatment
adoption, making it easy to assess pre-trends and the dynamic path of
effects after treatment.

The ``ref_period`` parameter adds a vertical dotted line marking the
reference period (typically the last pre-treatment period). When
``ref_period`` is specified, the connecting line between estimates is
suppressed so each point stands on its own.

.. code-block:: python

    did.plot_event_study(event_study, ref_period=-1)

.. image:: /_static/images/plot_guide_event_study_ref.png
   :alt: Event study with reference period line
   :width: 100%

``plot_event_study`` accepts results from :func:`~moderndid.aggte` (with
``type="dynamic"``), :func:`~moderndid.agg_ddd` (with
``type="eventstudy"``), and continuous treatment event studies. See the
:ref:`Staggered DiD <example_staggered_did>`,
:ref:`Triple DiD <example_triple_did>`, and
:ref:`Continuous Treatment <example_cont_did>` walkthroughs for worked
examples with each estimator.


Group-time estimates
--------------------

When you want to see every group-time ATT before aggregation,
``plot_gt`` creates a faceted plot with one panel per treatment cohort.
Each panel shows point estimates and confidence intervals across time
periods, with color distinguishing pre-treatment from post-treatment
estimates. This function also accepts triple DiD results from
:func:`~moderndid.ddd`.

.. code-block:: python

    did.plot_gt(result, ncol=3)

.. image:: /_static/images/plot_guide_gt.png
   :alt: Group-time treatment effects
   :width: 100%

The ``ncol`` parameter controls the number of columns in the facet grid.
Setting ``ncol=3`` arranges the three cohort panels side by side, while the
default ``ncol=1`` stacks them vertically. The ``title`` parameter controls
the prefix in each panel label (default ``"Group"``).


Group and calendar aggregations
-------------------------------

The ``plot_agg`` function visualizes treatment effects aggregated by
treatment cohort or by calendar time. These aggregations provide different
perspectives on the same underlying group-time estimates.

.. code-block:: python

    group_agg = did.aggte(result, type="group")
    did.plot_agg(group_agg)

.. image:: /_static/images/plot_guide_group_agg.png
   :alt: Treatment effects by group
   :width: 100%

Group aggregation averages post-treatment effects within each cohort,
revealing heterogeneity across early and late adopters. Calendar
aggregation averages across all cohorts that are treated at each calendar
time, showing how the overall effect evolves in absolute time.

.. code-block:: python

    calendar_agg = did.aggte(result, type="calendar")
    did.plot_agg(calendar_agg)

.. image:: /_static/images/plot_guide_calendar_agg.png
   :alt: Treatment effects by calendar time
   :width: 100%


Dose-response curves
--------------------

For settings with continuous treatment intensity, ``plot_dose_response``
displays the estimated treatment effect as a function of the dose level.
A shaded ribbon shows the confidence band. The ``effect_type`` parameter
switches between the average treatment effect on treated (``"att"``) and
the average causal response on treated (``"acrt"``), which captures the
marginal effect at each dose.

.. code-block:: python

    did.plot_dose_response(dose_result, effect_type="att")
    did.plot_dose_response(dose_result, effect_type="acrt")

See the :ref:`Continuous Treatment walkthrough <example_cont_did>` for a
complete example with simulated data.


Sensitivity analysis
--------------------

``plot_sensitivity`` displays confidence intervals from
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_
across a grid of sensitivity parameter values. Each method appears in a
different color, allowing direct comparison of how robust the original
finding is to violations of parallel trends.

.. code-block:: python

    did.plot_sensitivity(honest_result)

See the :ref:`Sensitivity Analysis walkthrough <example_honest_did>` for
complete examples with both relative magnitudes and smoothness
restrictions.


Intertemporal effects
---------------------

``plot_multiplegt`` visualizes treatment effects from the
`de Chaisemartin and D'Haultfoeuille (2024) <https://doi.org/10.1162/rest_a_01414>`_
estimator, which handles non-binary and non-absorbing treatments. The plot
shows placebo horizons (pre-treatment) and effect horizons (post-treatment)
around a vertical reference line at horizon zero.

.. code-block:: python

    did.plot_multiplegt(inter_result)

See the :ref:`Intertemporal Treatment walkthrough <example_inter_did>` for
a complete example using the Favara and Imbs banking deregulation data.


Common parameters
=================

All plot functions share a set of parameters that control the most
frequently adjusted visual elements.

``show_ci``
    Toggle confidence intervals on or off. Set ``show_ci=False`` to display
    only point estimates without error bars or ribbons. This can be useful
    when overlaying estimates from multiple specifications.

``ref_line``
    Y-value for the horizontal reference line. The default is 0, which
    places a dashed line at zero to help readers assess whether effects are
    statistically or economically meaningful. Set ``ref_line=None`` to
    remove it entirely.

``ref_period``
    Available only in ``plot_event_study``. Adds a vertical dotted line at
    the specified event time, typically set to ``-1`` to mark the last
    pre-treatment period.

``xlab``, ``ylab``, ``title``
    Custom axis labels and plot title. Each function supplies sensible
    defaults (``"Event Time"`` for event studies, ``"Treatment Cohort"`` for
    group aggregations), but custom labels are common in applied work.

.. code-block:: python

    did.plot_event_study(
        event_study,
        ref_period=-1,
        xlab="Years Relative to Minimum Wage Increase",
        ylab="ATT (Log Teen Employment)",
        title="Dynamic Treatment Effects",
    )

.. image:: /_static/images/plot_guide_labels.png
   :alt: Event study with custom labels
   :width: 100%


Themes
======

ModernDiD ships three built-in themes that control the overall appearance of
plots. All are applied by adding them to a plot object with the ``+``
operator.

``theme_moderndid``
    The default theme. White background, visible axis lines, no grid.
    Suitable for exploratory work and presentations.

``theme_publication``
    Designed for academic papers. Smaller font sizes, panel border instead
    of axis lines, legend at the bottom, and a default figure size of 6 by
    4 inches at 300 DPI.

``theme_minimal``
    Reduced visual elements with lighter axis lines and no legend or strip
    background fills. Suitable for dashboards and slide decks.

.. code-block:: python

    from moderndid.plots import theme_publication

    p = did.plot_event_study(event_study, ref_period=-1)
    p + theme_publication()

.. image:: /_static/images/plot_guide_publication.png
   :alt: Event study with publication theme
   :width: 100%

Since themes are composable, you can start from any built-in theme and
override individual elements. For example, to use the publication theme
but move the legend to the right.

.. code-block:: python

    from plotnine import theme

    p + theme_publication() + theme(legend_position="right")


Color palette
=============

The built-in color assignments are stored in the ``COLORS`` dictionary.

.. code-block:: python

    from moderndid.plots import COLORS
    print(COLORS)

.. code-block:: text

    {
        "pre_treatment": "#1a3a5c",
        "post_treatment": "#c0392b",
        "line": "#3a3a3a",
        "ci_fill": "#bfbfbf",
        "reference": "gray",
        "original": "#d4a017",
        "flci": "#1a3a5c",
        "conditional": "#2ecc71",
        "c_f": "#9b59b6",
        "c_lf": "#c0392b",
    }

Pre-treatment estimates use a dark navy (``#1a3a5c``) and post-treatment
estimates use red (``#c0392b``). Dose-response curves use a dark slate
line (``#2c3e50``) with a gray ribbon (``#95a5a6``).

To override colors on a specific plot, use plotnine's ``scale_color_manual``.

.. code-block:: python

    from plotnine import scale_color_manual

    p = did.plot_event_study(event_study, ref_period=-1)
    p + scale_color_manual(
        values={"Pre": "#e67e22", "Post": "#252525"},
        limits=["Pre", "Post"],
        name="Period",
    )

.. image:: /_static/images/plot_guide_grayscale.png
   :alt: Event study with grayscale colors
   :width: 100%

This is useful when you want to match an existing color scheme or
distinguish estimator phases more clearly.


Saving plots
============

The ``save`` method on any ``ggplot`` object writes the figure to disk.
The format is inferred from the file extension.

.. code-block:: python

    p = did.plot_event_study(event_study, ref_period=-1)

    # PNG for slides or web
    p.save("figure1.png", dpi=200, width=8, height=5)

    # PDF for LaTeX documents
    p.save("figure1.pdf", width=8, height=5)

    # SVG for scalable web graphics
    p.save("figure1.svg", width=8, height=5)

The ``width`` and ``height`` parameters are in inches. You can combine a
theme with specific export settings to match whatever format you need.

.. code-block:: python

    from moderndid.plots import theme_publication

    p = did.plot_event_study(event_study, ref_period=-1) + theme_publication()
    p.save("figure1.pdf", width=6, height=4, dpi=300)


Building custom plots
=====================

Every plot function converts its result object into a polars DataFrame
internally. You can call these converters directly to extract the
underlying data, then build any visualization you want with plotnine or
another plotting library.


Extracting plot data
--------------------

Each result type has its own converter function that returns a polars
DataFrame ready for plotting.

.. code-block:: python

    from moderndid.plots import aggteresult_to_polars

    df = aggteresult_to_polars(event_study)
    print(df)

.. code-block:: text

    shape: (7, 6)
    ┌────────────┬───────────┬──────────┬───────────┬───────────┬──────────────────┐
    │ event_time ┆ att       ┆ se       ┆ ci_lower  ┆ ci_upper  ┆ treatment_status │
    │ ---        ┆ ---       ┆ ---      ┆ ---       ┆ ---       ┆ ---              │
    │ f64        ┆ f64       ┆ f64      ┆ f64       ┆ f64       ┆ str              │
    ╞════════════╪═══════════╪══════════╪═══════════╪═══════════╪══════════════════╡
    │ -3.0       ┆ 0.030507  ┆ 0.015034 ┆ -0.010777 ┆ 0.071791  ┆ Pre              │
    │ -2.0       ┆ -0.000563 ┆ 0.013292 ┆ -0.037064 ┆ 0.035937  ┆ Pre              │
    │ -1.0       ┆ -0.024459 ┆ 0.014236 ┆ -0.063554 ┆ 0.014636  ┆ Pre              │
    │ 0.0        ┆ -0.019932 ┆ 0.011826 ┆ -0.052408 ┆ 0.012545  ┆ Post             │
    │ 1.0        ┆ -0.050957 ┆ 0.016893 ┆ -0.097349 ┆ -0.004566 ┆ Post             │
    │ 2.0        ┆ -0.137259 ┆ 0.036436 ┆ -0.237315 ┆ -0.037202 ┆ Post             │
    │ 3.0        ┆ -0.100811 ┆ 0.034359 ┆ -0.195166 ┆ -0.006457 ┆ Post             │
    └────────────┴───────────┴──────────┴───────────┴───────────┴──────────────────┘

The DataFrame contains one row per displayed estimate with columns for the
x-axis value, point estimate, standard error, confidence bounds, and a
treatment status label used for coloring. Reference period rows (where the
standard error is ``NaN``) are automatically filtered out.

The available converters are:

- :func:`~moderndid.plots.aggteresult_to_polars` for staggered DiD event studies
- :func:`~moderndid.plots.mpresult_to_polars` for staggered DiD group-time estimates
- :func:`~moderndid.plots.dddaggresult_to_polars` for triple DiD event studies
- :func:`~moderndid.plots.dddmpresult_to_polars` for triple DiD group-time estimates
- :func:`~moderndid.plots.doseresult_to_polars` for continuous treatment dose-response
- :func:`~moderndid.plots.pteresult_to_polars` for continuous treatment event studies
- :func:`~moderndid.plots.honestdid_to_polars` for sensitivity analysis
- :func:`~moderndid.plots.didinterresult_to_polars` for intertemporal effects


Advanced customization
----------------------

A common task is overlaying estimates from different estimators on the
same figure. The code below compares the
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
estimator against a standard two-way fixed effects (TWFE) event study
estimated with `pyfixest <https://github.com/py-econometrics/pyfixest>`_.

The TWFE specification regresses log teen employment on event-time
indicators (with event time -1 as the omitted category), log population,
and county and year fixed effects. We cluster standard errors at the county
level to match the level of treatment assignment.

.. code-block:: python

    import numpy as np
    import pyfixest as pf

    pdf = data.to_pandas()
    pdf["rel_time"] = np.where(
        pdf["first.treat"] > 0,
        pdf["year"] - pdf["first.treat"],
        -99,
    )

    fit = pf.feols(
        "lemp ~ i(rel_time, ref=-1) + lpop | countyreal + year",
        data=pdf,
        vcov={"CRV1": "countyreal"},
    )

To build the comparison figure, we extract the ModernDiD event study
estimates with ``aggteresult_to_polars`` and the TWFE coefficients from
pyfixest, then combine them into a single DataFrame with an ``estimator``
column.

.. code-block:: python

    import polars as pl
    import pandas as pd
    from moderndid.plots import aggteresult_to_polars

    # ModernDiD estimates
    es_df = aggteresult_to_polars(event_study)
    mdid_pd = es_df.select([
        pl.col("event_time").cast(pl.Int64),
        "att", "ci_lower", "ci_upper",
    ]).to_pandas()
    mdid_pd["estimator"] = "CS (2021)"

    # TWFE estimates (event time -1 is the omitted reference, fixed at 0)
    coefs = fit.coef()
    ci = fit.confint()
    mask = coefs.index.str.contains("rel_time")
    event_times = sorted(set(pdf["rel_time"]) - {-99, -1})
    twfe_pd = pd.DataFrame({
        "event_time": event_times,
        "att": coefs[mask].values,
        "ci_lower": ci.loc[mask, "2.5%"].values,
        "ci_upper": ci.loc[mask, "97.5%"].values,
        "estimator": "TWFE",
    })
    ref_row = pd.DataFrame({
        "event_time": [-1], "att": [0.0],
        "ci_lower": [np.nan], "ci_upper": [np.nan],
        "estimator": ["TWFE"],
    })
    twfe_pd = pd.concat([twfe_pd, ref_row], ignore_index=True)

    # Combine and filter to common event times
    plot_df = pd.concat([
        mdid_pd[mdid_pd["event_time"].between(-3, 3)],
        twfe_pd[twfe_pd["event_time"].between(-3, 3)],
    ], ignore_index=True)

With both sets of estimates in a single DataFrame, building a
publication-quality comparison figure is straightforward.

``position_dodge`` offsets the two estimators horizontally so their
confidence intervals do not overlap. ``scale_color_manual`` and
``scale_shape_manual`` assign distinct colors and marker shapes to each
estimator. ``theme_gray`` applies the classic R ggplot2 look with a gray
background and white gridlines. The ``labs`` function adds a subtitle for
methodological context and a multiline caption with estimation details.

.. code-block:: python

    from plotnine import (
        aes, element_text, geom_errorbar, geom_hline, geom_point,
        geom_vline, ggplot, labs, position_dodge, scale_color_manual,
        scale_shape_manual, scale_x_continuous, theme, theme_gray,
    )

    caption = """\
    TWFE estimates an event-study regression with county and year fixed effects.
    CS (2021) uses the Callaway and Sant'Anna estimator, which avoids negative
    weighting under heterogeneous treatment effects across cohorts.\
    """

    dodge = position_dodge(width=0.25)

    p = (
        ggplot(plot_df, aes(
            x="event_time", y="att",
            color="estimator", shape="estimator",
        ))
        + geom_hline(yintercept=0, color="black", size=0.4)
        + geom_vline(
            xintercept=-1, linetype="dashed",
            color="gray", size=0.4,
        )
        + geom_errorbar(
            aes(ymin="ci_lower", ymax="ci_upper"),
            width=0.15, size=0.6, position=dodge, na_rm=True,
        )
        + geom_point(size=3, position=dodge)
        + scale_color_manual(
            values={"TWFE": "#1a3a5c", "CS (2021)": "#c0392b"},
        )
        + scale_shape_manual(
            values={"TWFE": "o", "CS (2021)": "^"},
        )
        + scale_x_continuous(breaks=list(range(-3, 4)))
        + labs(
            x="Event Time",
            y="ATT (Log Employment)",
            title="Minimum Wage Effects on Teen Employment",
            subtitle="Comparing TWFE and heterogeneity-robust estimators"
                     " on staggered adoption data",
            caption=caption,
            color="", shape="",
        )
        + theme_gray()
        + theme(
            legend_position="bottom",
            plot_caption=element_text(
                ha="left",
                margin={"t": 1, "units": "lines"},
                linespacing=1.25,
            ),
        )
    )

.. image:: /_static/images/plot_guide_twfe_comparison.png
   :alt: Comparison of CS (2021) and TWFE event study estimates
   :width: 100%

Next steps
==========

- `plotnine documentation <https://plotnine.org/>`_ for the full grammar of
  graphics API, including geoms, scales, facets, and coordinate systems.
- :ref:`Staggered DiD <example_staggered_did>`,
  :ref:`Triple DiD <example_triple_did>`, and
  :ref:`Continuous Treatment <example_cont_did>`
  :ref:`Intertemporal Treatment <example_inter_did>`
  :ref:`Sensitivity Analysis <example_honest_did>` for worked examples
  with plotting integrated into the analysis workflow.
