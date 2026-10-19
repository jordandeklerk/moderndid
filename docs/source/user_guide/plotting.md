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

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Starting with an event study

The plots below draw the minimum wage fit of the
{ref}`first analysis <quickstart>`. With a universal base period, each estimate
is measured against the year before adoption. The event-study reference line
therefore belongs at event time minus one.

```{code-cell} ipython3
import moderndid as did

# Load the county panel and set the specification that every plot below starts from.
data = did.load_mpdta()
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    base_period="universal",
    anticipation=0,
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)

# Fit the cohort effects and line them up by years since adoption.
result = did.att_gt(data, **spec)
event_study = did.aggte(result, type="dynamic")
print(event_study)
```

The report gives the estimates and simultaneous bands that the plot will
display.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Event study of minimum wage effects on log teen employment. Navy estimates sit near zero at event times −4 to −2 before a dashed line that marks the base year at −1. Red effects fall below zero from event time 0 to 3.
---
# Label the axes in the employment question's terms and mark the base year.
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
line marks the omitted base year.

:::{admonition} Preserve the inference you estimated
:class: important

Because the plot reads its confidence bounds from the result object, changing
`show_ci` only changes whether those bounds are drawn. You choose between
pointwise intervals and simultaneous bands when you estimate or aggregate.
:::

## Choosing a plot for your result

The plot's axis follows the effect you have chosen to report. For the
unaggregated fit, each cohort has its own effects in calendar time. An
aggregation instead puts event times, cohorts, or calendar periods on that
axis. Each function below draws one or more of those kinds of result. The
{ref}`results guide <working-with-results>` helps you choose an aggregation
for your question.

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
---
mystnb:
  image:
    alt: Group-time effects from 2003 to 2007 in three stacked panels for the 2004, 2006, and 2007 cohorts. Navy estimates come before each cohort's adoption and red effects after it.
---
from plotnine import theme

# One panel per cohort on a taller canvas so the three panels stay readable.
cohort_plot = did.plot_gt(result) + did.theme_moderndid() + theme(figure_size=(12, 8))
cohort_plot
```

(plotting_tutorials)=

Four of the worked examples draw other plot types on their own data. The
{ref}`continuous treatment <example_cont_did>`,
{ref}`intertemporal treatment <example_inter_did>`,
{ref}`dynamic balancing <example_dyn_balancing>`, and
{ref}`sensitivity analysis <example_honest_did>` examples plot dose curves,
effects after a treatment change, estimates across history lengths, and
intervals under departures from parallel trends.

## Changing labels, references, and themes

Most effect plots accept `show_ci`, `ref_line`, `xlab`, `ylab`, and
`title`. Set `ref_line=None` to remove the horizontal reference line, or
`show_ci=False` to draw estimates without their intervals. If you adapt a call
to a different kind of result, consult the linked function first, since the
sensitivity and coefficient-selection plots have their own arguments.

For an event study, `ref_period` controls the vertical reference line.
If your estimates use a varying base period instead of one omitted base year,
`ref_period=None` removes that line and joins the estimates with a dotted
line. Moving the reference line never changes how the effects were normalized.

A theme controls the fonts, axes, and background of a figure. We use
{func}`~moderndid.plots.theme_moderndid` throughout the examples for white panels,
visible axes, and no grid lines. {func}`~moderndid.plots.theme_publication` uses
smaller text and a panel border for print. For a lighter look,
{func}`~moderndid.plots.theme_minimal` removes the strip and legend background
fills.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The minimum wage event study drawn in the publication theme's smaller text and bordered panel.
---
# Resize the print theme from its six-by-four-inch default to fit the page.
publication_plot = (
    did.plot_event_study(event_study, ref_period=-1)
    + did.theme_publication()
    + theme(figure_size=(12, 5), dpi=100)
)
publication_plot
```

The default navy and red colors are stored in `moderndid.plots.COLORS`.
For an event study, plotnine's `scale_color_manual` lets you substitute
colors for the `Pre` and `Post` labels used in the plot data.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The minimum wage event study redrawn with orange estimates before adoption and charcoal effects after adoption.
---
from plotnine import scale_color_manual

# Replace the treatment-status colors of the first event study plot.
recolored = p + scale_color_manual(
    values={"Pre": "#e67e22", "Post": "#252525"},
    limits=["Pre", "Post"],
    name="Treatment status",
)
recolored
```

The estimates before adoption now appear in orange and the effects after
adoption in charcoal. Both points and error bars change color because their
layers share the treatment-status mapping. Dose-response curves keep their
fixed line and ribbon color under a manual treatment-status scale. For a
different dose palette, you can build a plot from its data as described below.


## Saving a figure

Once the labels and layout describe your analysis, the plot object's `save`
method writes the figure to disk. It infers the format from the file extension
and reads `width` and `height` in inches unless you set other `units`.

```python
p.save("minimum_wage.png", width=12, height=5, dpi=100)
p.save("minimum_wage.pdf", width=6, height=4)
p.save("minimum_wage.svg", width=12, height=5)
```

While PNG suits a slide or web page, PDF and SVG keep lines and text scalable
when you resize the figure for a paper.

:::{admonition} Check the figure at its printed size
:class: tip

Labels that look readable in a notebook may become too small in a journal
column.
:::


(plotting-extracting-data)=

## Building a comparison from plot data

To compare specifications in a single figure, you need more than a built-in
plot because each one draws a single result. {func}`~moderndid.to_df` gives you
the underlying estimates, standard errors, and confidence bounds as a Polars
DataFrame. The result types that `to_df` supports are listed in its API
reference. The column names depend on the kind of effect a result represents.

```{code-cell} ipython3
# The estimates, standard errors, and bounds behind the event study plot.
plot_data = did.to_df(event_study)
plot_data
```

For a dose result, `did.to_df(dose_result, effect_type="acrt")` extracts
the fitted slope curve instead of the level effects that the default
`effect_type="att"` returns. Interpreting the slope as a causal response
requires additional restrictions, as the
{ref}`continuous treatment background <background-didcont>` explains.

To see both specifications on the same axis, we add an event study that also
uses counties whose states have not yet increased their minimum wage. Adding
later adopters to the comparison group changes the required parallel trends
assumption. The {ref}`staggered DiD example <example_staggered_did>` examines
how much that choice changes the employment estimates.

```{code-cell} ipython3
# Refit with counties not yet treated in the comparison group.
alternative = did.att_gt(data, **(spec | {"control_group": "notyettreated"}))
alternative_event_study = did.aggte(alternative, type="dynamic")
print(alternative_event_study)
```

Rounded to two decimals, every estimate in this report equals its counterpart
in the never-treated event study above.

Since plotnine maps a column of the data to colors and shapes, the two event
studies need to sit in one long table whose `controls` column names the
specification behind each row.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Minimum wage event studies for the never-treated and not-yet-treated comparison groups side by side at each estimated event time from −4 to 3. The navy circles and red triangles nearly coincide and their intervals overlap.
---
import polars as pl
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

# Stack both event studies with a column that names their comparison group.
comparison = pl.concat(
    [
        did.to_df(event_study).with_columns(pl.lit("Never treated").alias("controls")),
        did.to_df(alternative_event_study).with_columns(
            pl.lit("Not yet treated").alias("controls")
        ),
    ]
)

# Dodge the two specifications so their intervals sit side by side at each event time.
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
    + theme(legend_position="bottom")
)
comparison_plot
```

The dodge sets the navy circle of the never-treated comparison beside the red
triangle of the not-yet-treated one at every event time. Each pair lets you
compare two estimates of the same effect and their intervals without either
one hiding the other.

:::{admonition} Overlap is not a test of difference
:class: warning

Overlapping intervals alone do not test whether the estimates differ, since
both specifications use the same county data and their estimation errors are
related.
:::

For a paper that reports both specifications, the
{ref}`publication tables guide <publication_tables>` sets them in table columns
beside the weighting and regression estimators.
