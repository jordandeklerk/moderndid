---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(plotting)=

# Plotting treatment effects

An event-study plot lets your reader follow the estimated response after
adoption without searching through each row of a report. Its labels and
confidence bands need to describe the estimates you computed. ModernDiD's
plot functions read those estimates directly from result objects so you can
adjust the presentation without rebuilding the results.

We use the minimum wage data to make an event study whose labels and reference
lines describe the employment comparison. After preparing that figure for
export, we put two specifications on the same axis to see their estimates
together. Every plotting function returns a [plotnine](https://plotnine.org/)
`ggplot` object that you can customize by adding a theme or another layer
with `+`.

If your existing installation does not include the optional plotting
dependencies, add them with the `plots` extra before running these examples.

```bash
pip install "moderndid[plots]"
```

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Starting with an event study

The {ref}`staggered DiD example <example_staggered_did>` studies whether
state minimum wage increases reduced teen employment. We use the same
county data here and adjust for population measured before the increases.
The universal base period measures each estimate against the year before
adoption, so the event-study reference line belongs at event time minus one.

Without `clustervars`, this bootstrap treats counties as independent units.
The bands below therefore differ from those in the
{doc}`results guide <results>`. To use state-clustered bands in these plots,
fit the specification from that guide before calling the plotting functions.

```{code-cell} ipython3
import moderndid as did

data = did.load_mpdta()
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla="~lpop",
    est_method="dr",
    control_group="nevertreated",
    base_period="universal",
    boot=True,
    cband=True,
    biters=1000,
    random_state=7,
)
result = did.att_gt(data, **spec)
event_study = did.aggte(result, type="dynamic", random_state=7)
print(event_study)
```

The report gives the estimates and simultaneous bands that the plot will
display. Before reading the post-adoption path, check the estimates before
adoption for evidence of different trends. At longer event times, check which
cohorts are still observed; the minimum wage example explains why the points
two and three years after adoption describe only the earliest cohort.

```{code-cell} ipython3
p = (
    did.plot_event_study(
        event_study,
        ref_period=-1,
        xlab="Years relative to the minimum wage increase",
        ylab="Effect on log teen employment",
        title="Minimum wage effects on teen employment",
    )
    + did.theme_moderndid()
)
p
```

The navy points mark estimates before adoption and the red points mark
effects after adoption. The horizontal dashed line marks zero and the vertical
line marks the omitted base year. Because `lemp` is a log outcome, the
vertical axis reports changes in log employment rather than numbers of jobs.

:::{admonition} Preserve the inference you estimated
:class: important

The plot reads its confidence bounds from the result object. Changing
`show_ci` only changes whether those bounds are drawn; pointwise intervals
and simultaneous bands must be chosen during estimation or aggregation.
:::

## Choosing a plot for your result

The plot's axis follows the effect you have chosen to report. For the
unaggregated fit, each cohort has its own effects in calendar time. An
aggregation instead puts event times, cohorts, or calendar periods on that
axis. The functions below accept those different results; the
{doc}`results guide <results>` explains how to choose an aggregation for your
question.

```{eval-rst}
.. tab-set::

   .. tab-item:: Adoption and dose

      .. list-table::
         :header-rows: 1
         :widths: 30 35 35

         * - Function
           - What it shows
           - Result to pass
         * - :func:`~moderndid.plots.plot_event_study`
           - Effects by time relative to adoption
           - Dynamic :func:`~moderndid.aggte` results, event-study
             :func:`~moderndid.agg_ddd` results, event-aggregated
             :func:`~moderndid.emfx` results, or a :func:`~moderndid.cont_did`
             result with an event study
         * - :func:`~moderndid.plots.plot_gt`
           - Group-time effects in separate cohort panels
           - :func:`~moderndid.att_gt` or multi-period :func:`~moderndid.ddd` results
         * - :func:`~moderndid.plots.plot_agg`
           - Effects by cohort or calendar period
           - Group or calendar :func:`~moderndid.aggte` results, or group or
             calendar :func:`~moderndid.agg_ddd` results
         * - :func:`~moderndid.plots.plot_dose_response`
           - Level effects or fitted slopes over the dose grid
           - A dose-aggregated :func:`~moderndid.cont_did` result

   .. tab-item:: Treatment histories

      .. list-table::
         :header-rows: 1
         :widths: 30 35 35

         * - Function
           - What it shows
           - Result to pass
         * - :func:`~moderndid.plots.plot_multiplegt`
           - Effects and placebos by horizon after a treatment change
           - :func:`~moderndid.did_multiplegt` results
         * - :func:`~moderndid.plots.plot_dyn_balancing`
           - An average effect or potential-outcome estimates for two histories
           - A single :func:`~moderndid.diddynamic.dyn_balancing` result
         * - :func:`~moderndid.plots.plot_dyn_balancing_history`
           - Estimates across treatment-history lengths
           - :func:`~moderndid.diddynamic.dyn_balancing` results from ``histories_length``
         * - :func:`~moderndid.plots.plot_dyn_balancing_het`
           - Estimates across final periods
           - :func:`~moderndid.diddynamic.dyn_balancing` results from ``final_periods``
         * - :func:`~moderndid.plots.plot_dyn_balancing_coefs`
           - Selected covariates and their fitted LASSO coefficients
           - :func:`~moderndid.diddynamic.dyn_balancing` results with stored coefficients

   .. tab-item:: Sensitivity

      .. list-table::
         :header-rows: 1
         :widths: 30 35 35

         * - Function
           - What it shows
           - Result to pass
         * - :func:`~moderndid.plots.plot_sensitivity`
           - Confidence intervals as the sensitivity restriction changes
           - :func:`~moderndid.honest_did` results
```

For example, the unaggregated minimum wage estimates can be plotted one
cohort at a time without refitting the model.

```{code-cell} ipython3
from plotnine import theme

cohort_plot = (
    did.plot_gt(result)
    + did.theme_moderndid()
    + theme(figure_size=(12, 8))
)
cohort_plot
```

With each cohort in its own panel, you can compare their estimated responses
over calendar time. The taller figure leaves enough room to read the
intervals and axis labels in all of those panels.


## Changing labels, references, and themes

Most effect plots accept `show_ci`, `ref_line`, `xlab`, `ylab`, and
`title`. Set `ref_line=None` to remove the horizontal reference line, or
`show_ci=False` to draw estimates without their intervals. The sensitivity
and coefficient-selection plots have their own arguments, so consult the
linked function before adapting a call to a different kind of result.

For an event study, `ref_period` controls the vertical reference line.
If your estimates use a varying base period instead of one omitted base year,
`ref_period=None` removes that line and joins the estimates with a dotted
line. Moving the reference line never changes how the effects were normalized.

A theme controls the fonts, axes, and background of a figure. We use
{func}`~moderndid.plots.theme_moderndid` throughout the examples for white panels,
visible axes, and no grid lines. {func}`~moderndid.plots.theme_publication` uses
smaller text and a panel border for print, while
{func}`~moderndid.plots.theme_minimal` removes strip and legend background fills.

```{code-cell} ipython3
publication_plot = (
    did.plot_event_study(event_study, ref_period=-1)
    + did.theme_publication()
    + theme(figure_size=(12, 5), dpi=100)
)
publication_plot
```

The final `theme` overrides the publication theme's default six-by-four-inch
size and 300 dpi resolution for display in the documentation. You can make
similar overrides for legend placement or text size without replacing the
rest of the theme.

The default navy and red colors are stored in `moderndid.plots.COLORS`.
For an event study, plotnine's `scale_color_manual` lets you substitute
colors for the `Pre` and `Post` labels used in the plot data.

```{code-cell} ipython3
from plotnine import scale_color_manual

recolored = p + scale_color_manual(
    values={"Pre": "#e67e22", "Post": "#252525"},
    limits=["Pre", "Post"],
    name="Treatment status",
)
recolored
```

The estimates before adoption now appear in orange and the effects after
adoption in charcoal. Both points and error bars change color because their
layers share the treatment-status mapping. Dose-response curves use a fixed line and
ribbon color, so a manual treatment-status scale does not recolor that plot.
For a different dose palette, you can build a plot from its data as described
below.


## Saving a figure

Once the labels and layout describe your analysis, the plot object's `save`
method writes the figure to disk. The extension selects the output format,
and the dimensions are in inches unless you specify other units.

```python
p.save("minimum_wage.png", width=12, height=5, dpi=100)
p.save("minimum_wage.pdf", width=6, height=4)
p.save("minimum_wage.svg", width=12, height=5)
```

PNG is useful for a slide or web page. For a paper, PDF and SVG keep lines
and text scalable when the figure is resized. Check the exported figure at
its intended size, since labels that
look readable in a notebook may become too small in a journal column.


(plotting-extracting-data)=

## Building a comparison from plot data

A built-in plot draws one result at a time. To compare specifications in a
single figure, {func}`~moderndid.to_df` gives you the underlying estimates,
standard errors, and confidence bounds as a Polars DataFrame. The supported
result types are listed in its API reference; column names depend on the
kind of effect being represented.

```{code-cell} ipython3
plot_data = did.to_df(event_study)
print(plot_data)
```

For this event study, the columns are `event_time`, `att`, `se`,
`ci_lower`, `ci_upper`, and `treatment_status`. Normalized base-period
rows with undefined standard errors are left out, since they are reference
values rather than estimated effects. The confidence bounds retain the
result's critical values, including simultaneous bands when requested.
For a dose result, `did.to_df(dose_result, effect_type="acrt")` instead
extracts the fitted slope curve; the default `effect_type="att"` extracts
level effects. Interpreting the slope as a causal response requires additional
restrictions, as the {ref}`continuous treatment background <background-didcont>`
explains.

To see both specifications on the same axis, we add an event study that also
uses counties whose states have not yet increased their minimum wage. Adding
later adopters to the comparison group changes the required parallel trends
assumption. The {ref}`staggered DiD example <example_staggered_did>` examines
how much that choice changes the employment estimates.

```{code-cell} ipython3
import polars as pl

alternative = did.att_gt(data, **(spec | {"control_group": "notyettreated"}))
alternative_event_study = did.aggte(alternative, type="dynamic", random_state=7)
comparison = pl.concat(
    [
        did.to_df(event_study).with_columns(pl.lit("Never treated").alias("controls")),
        did.to_df(alternative_event_study).with_columns(
            pl.lit("Not yet treated").alias("controls")
        ),
    ]
)
```

The `controls` column now identifies which specification produced each
estimate and its bounds. A small horizontal offset gives each estimate its
own space so you can compare both sets of intervals at the same event time.

```{code-cell} ipython3
:tags: [hide-input]

from plotnine import (
    aes,
    geom_errorbar,
    geom_hline,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_shape_manual,
)

dodge = position_dodge(width=0.25)
comparison_plot = (
    ggplot(comparison, aes("event_time", "att", color="controls", shape="controls"))
    + geom_hline(yintercept=0, linetype="dashed", color="gray")
    + geom_vline(xintercept=-1, linetype="dashed", color="gray")
    + geom_errorbar(
        aes(ymin="ci_lower", ymax="ci_upper"),
        width=0.15,
        position=dodge,
    )
    + geom_point(size=3, position=dodge)
    + scale_color_manual(values={"Never treated": "#1a3a5c", "Not yet treated": "#c0392b"})
    + scale_shape_manual(values={"Never treated": "o", "Not yet treated": "^"})
    + labs(
        x="Years relative to the minimum wage increase",
        y="Effect on log teen employment",
        color="Comparison group",
        shape="Comparison group",
    )
    + did.theme_moderndid()
    + theme(figure_size=(12, 5), legend_position="bottom")
)
comparison_plot
```

The navy circles and red triangles compare the same log-employment outcome
under the two comparison-group choices. Overlapping intervals alone do not
test whether the estimates differ, since both specifications use the same
county data and their estimation errors are related.


(plotting_tutorials)=

## Figures in the worked examples

The examples put these figures back into the questions their data answer.
The {ref}`staggered adoption example <example_staggered_did>` reads cohort
plots alongside an event study. The
{ref}`continuous treatment example <example_cont_did>` plots fitted dose
curves and employment effects over time in the fracking data. For other designs, see the
{ref}`triple difference <example_triple_did>`,
{ref}`intertemporal treatment <example_inter_did>`, and
{ref}`dynamic balancing <example_dyn_balancing>` examples.
The {ref}`sensitivity example <example_honest_did>` shows how to read intervals
under departures from parallel trends. To present a figure alongside estimates
and a description of the sample, the
{ref}`publication tables <publication_tables>` guide takes these result
objects through the corresponding table formats.
