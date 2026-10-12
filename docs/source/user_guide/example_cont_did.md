---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
language_info:
  name: python
---

(example_cont_did)=

# Continuous difference-in-differences

Fracking's economic potential depends on geological conditions that vary even
within a shale formation. We're going to study how county employment changed
after a formation's fracking potential became publicly known and how the size
of that change depends on geological prospectivity. The data combine adoption
years between 2001 and 2012 with a county score that remains fixed throughout
the analysis.

{func}`~moderndid.cont_did` lets you examine these changes as a curve over
prospectivity scores or as an event study over years since adoption. A
regression of employment on the dose interacted with a post-adoption indicator
would compress the variation across scores and over time into a single
coefficient. Here you will fit both views of the employment effects, read their
uncertainty, and check how the dose curve changes with the comparison group and
spline specification.

The data come from [Bartik, Currie, Greenstone, and Knittel
(2019)](https://doi.org/10.1257/app.20170487) and the fracking application in
[Callaway, Goodman-Bacon, and Sant'Anna's 2024 event-study
paper](https://doi.org/10.1257/pandp.20241047). The
{ref}`companion replication <example_cont_did_replication>` reproduces that
paper's figures and explains its separate pooling procedure. Our
{ref}`background page <background-didcont>` develops identification using the
[December 2025 main paper](https://psantanna.com/files/CGBS_v4.pdf). That
paper's own empirical application concerns Medicare reimbursement rather than
fracking.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Counties, scores, and adoption years

Each county's score rates how favorable its geology is for fracking rather
than how many wells were eventually drilled there. Because the pace of drilling
can depend on local economic conditions, a count of wells would mix exposure
with the investment that exposure attracted. The outcome is total county
employment because it records job changes in every industry rather than only in
oil and gas.

We use {func}`~moderndid.load_fracking` to load 10,050 observations from
402 counties, each observed annually from 1990 through 2014. The loader omits
2015 because all employment outcomes are missing and removes counties with an
incomplete remaining panel. Because the outcome `y` already contains log total
employment, you can use it directly without another transformation.

```{code-cell} ipython3
import moderndid as did
import numpy as np
import polars as pl

# The loader retains the source outcome in log employment.
data = did.load_fracking()
data.head()
```

The identifier `i` is the county FIPS code and `t` is the calendar year. Among
the 329 positive-dose counties, `d` records prospectivity scores from 0.20 to
9.34. For these counties, `G` records the adoption year of their formation. A
coded-zero county has `G=0` and supplies an untreated comparison throughout the
panel. The source formation date remains available in `G_original`, alongside
the shale basin identifier `shale_basin1`.

The [original research design](https://www.aeaweb.org/articles/materials/11519)
dates each adoption year to the first public news of successful fracking in the
county's formation. To match the annual outcomes, news that arrived after June
counts toward the following year. The dates therefore record when a formation's
economic prospects changed rather than when the first well was drilled in each
county.

:::{admonition} Coded zeros include missing scores
:class: warning

When the data were prepared, missing prospectivity scores were set to zero
without a flag that marks the imputed ones. Because the 73 coded-zero counties
mix observed zeros with that coding decision, their place in the comparison
group does not show that they lacked geological exposure.
:::

## Set up the dose analysis

The first analysis asks how the employment contrast varies over the recorded
scores. We fit a separate curve for each adoption cohort and calendar year
before averaging those curves. The specification below keeps the outcome,
comparison group, spline, and inference choices together so you can reuse them
when changing one part of the analysis.

```{code-cell} ipython3
spec = dict(
    # Relate log employment to fixed exposure and formation adoption.
    yname="y",
    tname="t",
    idname="i",
    gname="G",
    dname="d",
    xformla="~1",
    treatment_type="continuous",
    allow_unbalanced_panel=False,
    weightsname=None,
    # Compare each cohort with counties still untreated in that year.
    control_group="notyettreated",
    base_period="universal",
    anticipation=0,
    # Fit a modest curve within every cohort's observed score range.
    target_parameter="level",
    aggregation="dose",
    dose_est_method="parametric",
    degree=3,
    num_knots=0,
    dvals=np.linspace(3.0, 4.6, 65),
    # Use county-level multipliers and simultaneous bands over the grid.
    alpha=0.05,
    boot_type="multiplier",
    biters=999,
    cband=True,
    clustervars=None,
    random_state=20240103,
)
```

### Keep exposure separate from adoption

Because prospectivity is already positive before fracking becomes viable,
`gname="G"` is essential here. Inferring adoption from the first positive score
would place every positive-dose county in the first observed year and leave
no untreated baseline. Since `cont_did` currently supports only balanced panels
without covariates or sampling weights, the spec sets
`allow_unbalanced_panel=False`, `xformla="~1"`, and `weightsname=None` to match
the prepared county panel.

### Compare with counties still untreated

A not-yet-treated comparison uses the coded-zero counties together with
positive-dose counties whose formation has not adopted by the later of the
outcome year and the cohort's reference year. The identifying assumption is
that, without adoption, each treated cohort's mean employment change at a given
score would match the untreated comparison's mean change. This assumption
concerns untreated trends rather than equal employment levels. The
{ref}`continuous treatment background <background-didcont>` states parallel
trends formally for staggered adoption. The
[comparison-group check](#use-only-coded-zero-comparisons) below drops the
future adopters and keeps only the coded-zero counties.

The universal base period anchors every comparison in the year just before the
cohort adopts. Setting `anticipation=0` assumes the employment response does
not begin before the recorded adoption year. If it could begin earlier, raising
`anticipation` moves the baseline back by the same number of years. The
staggered example's [anticipation check](example_staggered_did.md#anticipation)
shows that change for a binary treatment.

:::{admonition} Base period and the dose curve
:class: note

Since post-adoption comparisons start from the year before adoption under
either base period, the dose curves come out the same with
`base_period="varying"`. That setting would change only the pre-adoption
estimates in the event study below. The staggered example's
[base-period check](example_staggered_did.md#the-base-period) shows the same
pattern for a binary treatment.
:::

### Fit within the observed score ranges

A cubic spline with no interior knots gives a cubic polynomial in each
cohort-period comparison. We start with this specification because some
cohorts contain too few distinct scores to support a more flexible curve.

The 2006 cohort contains nine counties with only six distinct scores. Since
a cubic spline with three interior knots requires seven coefficients, that
cohort cannot identify such a flexible curve. With `num_knots=3`, `cont_did`
would warn and leave the 2006 cohort out of the dose curves and their
derivative summary. Its counties would still count toward the overall ATT in
the report below.

Across the eight adoption cohorts, the observed score ranges overlap from 2.99
to 4.67. Our evaluation grid from 3.0 to 4.6 stays inside that overlap.
Because `dvals` controls where the curves are evaluated rather than which
counties enter estimation, all 329 positive-dose counties remain in the
analysis. Staying inside the ranges does not settle how many observations sit
near each grid score or how well a cubic approximates the curve. The
[spline check](#allow-one-interior-knot) below tries one interior knot to
examine that approximation.

:::{admonition} Compare scores within one play
:class: important

The [original appendix](https://www.aeaweb.org/articles/materials/11519)
builds prospectivity separately for each shale play without putting the plays
on a common scale. Although the pooled panel illustrates the estimator, a
substantive comparison of geological intensity across plays requires a
justified common scale or an analysis within one play.
:::

### Measure uncertainty across the curve

The multiplier bootstrap gives each county its own random weight in every draw
and therefore treats counties as independent sampling units. With
`cband=True`, each plotted curve has a simultaneous 95 percent band over its
65 evaluation points rather than a separate pointwise interval at each score.
Each band covers its own curve rather than providing joint coverage of both
the level and slope curves. In contrast to those curve bands, the scalar
overall summaries retain pointwise intervals for their respective targets.

The 999 draws and fixed seed make the reported results reproducible. We do not
cluster by formation because `cont_did` currently does not support that
inference. Since counties within a formation may share shocks, these
county-level bands should not be read as accounting for formation-level
dependence. The [interval check](#read-pointwise-intervals) below shows the
narrower claim made by pointwise intervals under the same sampling assumptions.

The fit now uses the specification we have just worked through.

```{code-cell} ipython3
# Fit the dose curves and their overall summaries.
result = did.cont_did(data, **spec)
print(result)
```

## Read the effects across scores

The overall ATT is 0.0424 log points with a 95 percent interval from 0.0240
to 0.0609. It averages post-adoption effects over each cohort's observed
treated doses and years before weighting cohorts by their treated-county
shares. It is therefore a summary of adoption effects at the doses these
counties actually have, rather than the effect of giving every county one
common score.

The derivative summary labeled ACRT describes how the estimated level curve
changes with the score. Interpreting it as the causal effect of increasing
exposure requires stronger assumptions than those identifying level effects. Before
reading that derivative, we can examine the level curve itself.

### Compare the fitted level contrasts

{func}`~moderndid.plots.plot_dose_response` draws the level curve stored in
`result` along with its simultaneous band. At each score, `cont_did` averages
the cohort-period level curves using cohort shares that stay fixed across scores
and equal weights across each cohort's observed post-adoption years. These
weights describe a different target from averaging only counties whose recorded
score equals that value. The
{ref}`aggregation discussion <background-didcont-aggregation>` explains the distinction
from the paper's dose-dependent timing weights.

```{code-cell} ipython3
# Draw the level curve with its simultaneous band.
(
    did.plot_dose_response(
        result,
        effect_type="att",
        xlab="Geological prospectivity score",
        ylab="Effect on log employment",
        title="Employment contrasts across prospectivity scores",
    )
    + did.theme_moderndid()
)
```

The dark line rises toward the upper part of the displayed range. The table
below uses {func}`~moderndid.to_df` to read the estimates and band endpoints at
scores 3.0, 4.0, and 4.5.

```{code-cell} ipython3
# Read the level curve and its band at three scores.
did.to_df(result).filter(pl.col("dose").is_in([3.0, 4.0, 4.5]))
```

At score 4.0, the fitted contrast of 0.0361 log points corresponds to about
3.7 percent higher employment using `100 * np.expm1(estimate)`. This is the
proportional contrast implied by the log estimate rather than a separately
estimated effect on mean employment in levels. The simultaneous band runs
from 0.0103 to 0.0619 log points there. With this seed, its lower edge stays
above zero at every score on the plot. At score 3.0 it clears zero by only
0.0017 log points. Across 30 other bootstrap seeds, the band stays above zero
from score 3.1 up every time. Since 7 of those seeds put its lower edge below
zero at score 3.0, the band does not establish a positive effect at the lowest
scores.

### Distinguish the slope from a causal response

The slope curve is available in the same dose result even though the call
requested `target_parameter="level"`. Changing `effect_type` selects the
stored derivative rather than estimating another model. Its vertical axis is
log employment per additional unit of the prospectivity score.

```{code-cell} ipython3
# Draw the derivative stored in the same result.
(
    did.plot_dose_response(
        result,
        effect_type="acrt",
        xlab="Geological prospectivity score",
        ylab="Fitted slope in log points per score unit",
        title="Derivative of the fitted employment curve",
    )
    + did.theme_moderndid()
)
```

With the same seed, the slope's simultaneous band includes zero at every
displayed score. Six of the 30 other seeds lift its lower edge just above zero
somewhere between 4.3 and 4.6. The overall ACRT of 0.0442 in the report above
pools the slope over every treated county's observed score into one number.
That single target has an interval from 0.0253 to 0.0630 that excludes zero.

Because a slope comparison involves different counties at different doses,
the curve can vary with their gains from treatment as well as with exposure.
Ordinary parallel trends identifies their level effects without separating
those two sources of variation. Stronger restrictions must rule out that
selection before you interpret the derivative as a causal response. We use this
plot only to describe the fitted curve. The
{ref}`continuous treatment background <background-didcont>` gives the
assumptions for that stronger interpretation.

## Follow employment after adoption

Since averaging over post-adoption years conceals whether the employment
contrast develops gradually, we now change the aggregation to an event study and
follow the first five years after adoption. With `target_parameter="level"`,
this path averages adoption effects at counties' observed doses rather than
fitting a spline at each event time. The degree, knots, and evaluation grid
therefore do not determine these level estimates.

This call retains the outcome and comparison choices in `spec` while displaying
event times from eleven years before adoption through four years after it.
We request pointwise intervals to make the event-study inference comparable
in form to the 2024 paper's figures.

```{code-cell} ipython3
# Reuse the specification for an event study with pointwise intervals.
event_result = did.cont_did(
    data,
    **dict(spec, aggregation="eventstudy", cband=False),
    min_e=-11,
    max_e=4,
)
print(event_result)
```

The event-study report averages the five post-adoption estimates into an
overall ATT of 0.0227 log points. That differs from the dose report's 0.0424
because the event summary gives equal weight to these five event times. The
dose summary instead covers each cohort's full observed post-adoption history.
Neither number is a slope or an effect at one chosen score.

```{code-cell} ipython3
# Plot the estimates before and after adoption with their pointwise intervals.
(
    did.plot_event_study(
        event_result,
        xlab="Years since formation adoption",
        ylab="Effect on log employment",
        title="County employment before and after fracking adoption",
    )
    + did.theme_moderndid()
)
```

The red post-adoption points rise over time to 0.0474 log points in year four,
or about 4.9 percent higher employment. The blue pre-adoption points have
intervals that include zero at every displayed horizon. Event time minus one is
the normalized reference rather than an estimated effect with its own interval.

:::{admonition} Pre-adoption estimates do not prove parallel trends
:class: warning

The pre-adoption comparisons give you evidence about untreated trends without
establishing parallel trends or the stronger restrictions for causal dose
responses.
:::

Cohort composition also changes with the horizon because the 2012 adopters
cannot contribute to years three and four in a panel ending in 2014. The
{ref}`companion replication <example_cont_did_replication>` separates low- and
high-dose counties to reproduce Figure 1 and checks this pooled path against
appendix Figure B.1. Although the bootstrap intervals here need not match the
historical ribbons, `cont_did` reproduces the Figure B.1 point estimates up to
the precision of the extracted plotting paths.

## Check the choices behind the dose curve

The fitted dose pattern depends on choices that the event-study plot cannot
settle. We return to the main dose specification and change the comparison
group, spline, and interval construction one at a time. Each check shows the
contrast at score 4.0 so you can compare the same point across fits.

### Use only coded-zero comparisons

Positive-dose counties awaiting adoption might differ from coded-zero counties
in their untreated employment changes. Restricting the comparison group to
`"nevertreated"` removes those future adopters from the controls. Agreement
between the resulting curves is informative about this comparison choice
without establishing either version of parallel trends.

```{code-cell} ipython3
# Compare each cohort with the coded-zero counties alone.
never_treated = did.cont_did(data, **dict(spec, control_group="nevertreated"))
print("Overall ATT:", round(never_treated.overall_att, 4))
did.to_df(never_treated).filter(pl.col("dose") == 4.0)
```

Using coded-zero controls alone lowers the contrast at score 4.0 from 0.0361
to 0.0331 log points and the overall ATT from 0.0424 to 0.0394.

### Allow one interior knot

A knot lets the cubic spline bend more flexibly around the median positive
score. Since the small cohorts limit that flexibility, we try one interior knot
rather than the three supported by the pooled paper replication. This changes
the dose curves while retaining the original comparison group and inference
settings.

```{code-cell} ipython3
# Place one interior knot at the median positive score.
one_knot = did.cont_did(data, **dict(spec, num_knots=1))
print("Overall ATT:", round(one_knot.overall_att, 4))
did.to_df(one_knot).filter(pl.col("dose") == 4.0)
```

Adding a knot moves the contrast at score 4.0 from 0.0361 to 0.0384 log points
without changing the overall ATT of 0.0424. Since the overall adoption effect
averages observed outcome-change contrasts rather than fitted values on our
evaluation grid, its stability does not establish that the dose curve is
insensitive to smoothing.

### Read pointwise intervals

If you want an interval for a prespecified score, `cband=False` replaces the
simultaneous critical value with a pointwise one. The estimates and standard
errors remain the same because only the coverage claim changes.

:::{admonition} Pointwise intervals assume a prespecified score
:class: warning

Selecting an apparently significant score after viewing the whole curve would
call for the simultaneous band used in the main analysis.
:::

```{code-cell} ipython3
# Replace the simultaneous critical value with a pointwise one.
pointwise = did.cont_did(data, **dict(spec, cband=False))
did.to_df(pointwise).filter(pl.col("dose") == 4.0)
```

The pointwise interval from 0.0132 to 0.0590 log points is narrower than the
main band from 0.0103 to 0.0619 because its coverage claim concerns a single
score rather than the whole curve. For the assumptions that separate an
adoption effect from a response to greater exposure and for the package's
aggregation weights, continue with the
{ref}`continuous treatment background <background-didcont>`. To reproduce the
paper's window-averaged curves alongside its event studies, the
{ref}`fracking replication <example_cont_did_replication>` works through the
published calculations and their checks.
