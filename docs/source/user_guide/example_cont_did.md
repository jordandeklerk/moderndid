---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_cont_did)=

# Continuous difference-in-differences

Fracking's economic potential depends on geological conditions that vary even
within a shale formation. We're going to study how county employment changed after a
formation's fracking potential became publicly known and how those changes
vary with geological prospectivity. The data combine adoption years between
2001 and 2012 with a county score that remains fixed throughout the analysis.

{func}`~moderndid.cont_did` lets you examine these changes as a curve over
prospectivity scores or as an event study over years since adoption. A single
coefficient from a regression of employment on dose interacted with a
post-adoption indicator would obscure that distinction. Here you will fit both views of the
employment effects, read their uncertainty, and check how the dose curve
changes with the comparison group and spline specification.

The data come from [Bartik, Currie, Greenstone, and Knittel
(2019)](https://doi.org/10.1257/app.20170487) and the fracking application in
[Callaway, Goodman-Bacon, and Sant'Anna's 2024 event-study
paper](https://doi.org/10.1257/pandp.20241047). The
{ref}`companion replication <example_cont_did_replication>` reproduces that
paper's figures and explains its separate pooling procedure. Our
{ref}`background page <background-didcont>` develops identification using the
[December 2025 main paper](https://psantanna.com/files/CGBS_v4.pdf), whose
empirical application concerns Medicare reimbursement rather than fracking.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Employment and geological exposure

The geological score measures the potential for fracking rather than the
number of wells a county eventually drilled. Since drilling can respond to
local economic conditions, the geological score helps distinguish potential
exposure from subsequent investment. Total employment captures changes across
industries rather than only employment in oil and gas.

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
the 329 positive-dose counties, `d` records prospectivity on a scale from 0.20
to 9.34. For these counties, `G` records the adoption year of their formation. A coded-zero county has `G=0` and supplies an
untreated comparison throughout the panel. The source formation date remains
available in `G_original`, alongside the shale basin identifier `shale_basin1`.

The [original research design](https://www.aeaweb.org/articles/materials/11519)
dates adoption to when successful fracking became publicly known within a
formation. If an announcement falls after June, adoption enters the following year. Consequently,
these dates describe a change in a formation's economic prospects rather than
the first well drilled in each county.

:::{admonition} Zero includes imputed scores
:class: warning

The data preparation recodes missing prospectivity scores to zero without
retaining an imputation flag. Since the 73 coded-zero counties reflect
that decision as well as observed zeros, their comparison-group status
does not independently establish an absence of geological exposure.
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
    alp=0.05,
    boot=True,
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
no untreated baseline. The prepared data matches the function's current support for balanced panels
without covariates or sampling weights through `xformla="~1"` and
`weightsname=None`.

### Compare counties before they adopt

A not-yet-treated comparison uses the coded-zero counties and positive-dose
counties whose formation remains untreated in both the outcome year and the
cohort's reference year. The identifying
assumption is that, without adoption, each treated cohort's mean employment
change at a given score would match the untreated comparison's mean change.
This assumption concerns untreated trends rather than equal employment levels.

The universal base period measures changes relative to the year before a
cohort adopts. Setting `anticipation=0` assumes the employment response does
not begin before the recorded announcement year. If a response begins earlier,
you need to change that assumption and the baseline accordingly. The
{ref}`background discussion <background-didcont>` explains those comparisons;
the [comparison-group check](#use-only-coded-zero-comparisons) below restricts
the analysis to never-treated counties.

### Fit within the observed score ranges

A cubic spline with no interior knots gives a cubic polynomial in each
cohort-period comparison. We start with this specification because some
cohorts contain too few distinct scores to support a more flexible curve.

The 2006 cohort contains nine counties with only six distinct scores. Since
a cubic spline with three interior knots requires seven coefficients, that
cohort cannot identify such a flexible curve. Across the eight adoption
cohorts, the observed score ranges overlap from 2.99 to 4.67. Our evaluation
grid from 3.0 to 4.6 stays inside that overlap.
`dvals` controls where the curves are evaluated rather than which counties
enter estimation; all 329 positive-dose counties remain in the analysis.
Being inside the ranges still leaves the density of nearby observations and
the polynomial approximation to assess. The
[spline check](#allow-one-interior-knot) below tries one interior knot to examine
that approximation.

:::{admonition} Scores depend on the formation
:class: important

The [original appendix](https://www.aeaweb.org/articles/materials/11519)
constructs prospectivity separately for each shale play without calibrating
scores on a common scale. Although the full panel illustrates the estimator,
a substantive comparison of geological intensity requires a justified common
scale or an analysis within one play.
:::

### Measure uncertainty across the curve

The multiplier bootstrap treats counties as independent sampling units. With
`cband=True`, each plotted curve has a simultaneous 95 percent band over its
65 evaluation points rather than a separate pointwise interval at each score.
Each band covers its own curve rather than providing joint coverage of both
the level and slope curves. In contrast to those curve bands, the scalar
overall summaries retain pointwise intervals for their respective targets.

The 999 draws and fixed seed make the reported results reproducible. We do not
cluster by formation because `cont_did` currently does not support that
inference. Because counties within a formation may share shocks, these county-level
bands should not be read as accounting for formation-level dependence. The
[interval check](#read-pointwise-intervals) below shows the narrower claim
made by pointwise intervals under the same sampling assumptions.

The fit now uses the specification we have just worked through.

```{code-cell} ipython3
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

{func}`~moderndid.plots.plot_dose_response` draws the curve stored in `result`. At
each score, the package averages the cohort-period level curves using cohort
shares that stay fixed across scores and equal weights across each cohort's
observed post-adoption years. These weights describe a different target from
averaging only counties whose recorded score equals that value. The
{ref}`aggregation discussion <background-didcont>` explains the distinction
from the paper's dose-dependent timing weights.

```{code-cell} ipython3
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

The dark line rises toward the upper part of the displayed range. You can
read the estimates and their band endpoints directly with
{func}`~moderndid.to_df`, as the selected scores below show.

```{code-cell} ipython3
did.to_df(result).filter(pl.col("dose").is_in([3.0, 4.0, 4.5]))
```

At score 4.0, the fitted contrast of 0.0361 log points corresponds to about
3.7 percent higher employment using `100 * np.expm1(estimate)`. This is the
proportional contrast implied by the log estimate rather than a separately
estimated effect on mean employment in levels. The simultaneous band runs
from 0.0103 to 0.0619 log points there. Since its lower edge stays above zero
at every score on the plot, the band supports a positive effect throughout the
displayed range. At score 3.0 that edge clears zero by only 0.0017 log points.

### Distinguish the slope from a causal response

The slope curve is available in the same dose result even though the call
requested `target_parameter="level"`. Changing `effect_type` selects the
stored derivative rather than estimating another model. Its vertical axis is
log employment per additional unit of the prospectivity score.

```{code-cell} ipython3
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

Because a slope comparison involves different counties at different doses,
the curve can vary with their gains from treatment as well as with exposure.
Ordinary parallel trends identifies their level effects without separating
those two sources of variation. Stronger restrictions must rule out that selection before
you interpret the derivative as a causal response. We use this plot to describe
the fitted curve; the {ref}`continuous treatment background
<background-didcont>` gives the assumptions for that stronger interpretation.

## Follow employment after adoption

Averaging over post-adoption years conceals whether the employment contrast
develops gradually, a feature you can examine through the event study. We now change the aggregation to an
event study and follow the first five years after adoption. With
`target_parameter="level"`, this path averages adoption effects at counties'
observed doses rather than fitting a spline at each event time. The degree,
knots, and evaluation grid therefore do not determine these level estimates.

This call retains the outcome and comparison choices in `spec` while displaying
event times from eleven years before adoption through four years after it.
We request pointwise intervals to make the event-study inference comparable
in form to the 2024 paper's figures.

```{code-cell} ipython3
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
because the event summary gives equal weight to these five event times,
whereas the dose summary includes each cohort's full observed post-adoption
history. Neither number is a slope or an effect at one chosen score.

```{code-cell} ipython3
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
intervals that include zero at every displayed horizon. Those comparisons give you evidence
about untreated trends without establishing parallel trends or the stronger
restrictions for causal dose responses. Event time minus one is the normalized
reference rather than an estimated effect with its own interval.

Cohort composition also changes with the horizon because the 2012 adopters
cannot contribute to years three and four in a panel ending in 2014. The
{ref}`companion replication <example_cont_did_replication>` separates low-
and high-dose counties to reproduce Figure 1 and checks this pooled path
against Figure B1. The public estimator reproduces their point estimates up
to the precision of the extracted plotting paths; its bootstrap intervals
need not match the historical ribbons.

## Check the choices behind the dose curve

The fitted dose pattern depends on choices that the event-study plot cannot
settle. We return to the main dose specification and change the spline,
comparison group, and interval construction one at a time. Each check shows
the contrast at score 4.0 so you can compare the same point across fits.

### Allow one interior knot

A knot lets the cubic spline bend more flexibly around the median positive
score. Since the small cohorts limit that flexibility, we try one interior knot
rather than the three supported by the pooled paper replication. This changes the dose curves while retaining the original
comparison group and inference settings.

```{code-cell} ipython3
one_knot = did.cont_did(data, **dict(spec, num_knots=1))
did.to_df(one_knot).filter(pl.col("dose") == 4.0)
```

### Use only coded-zero comparisons

Positive-dose counties awaiting adoption might differ from coded-zero counties
in their untreated employment changes. Restricting the comparison group to
`"nevertreated"` removes those future adopters from the controls. Agreement
between the resulting curves is informative about this comparison choice
without establishing either version of parallel trends.

```{code-cell} ipython3
never_treated = did.cont_did(data, **dict(spec, control_group="nevertreated"))
did.to_df(never_treated).filter(pl.col("dose") == 4.0)
```

### Read pointwise intervals

If you want an interval for a prespecified score, `cband=False` replaces the
simultaneous critical value with a pointwise one. The estimates and standard
errors remain the same because only the coverage claim changes. Selecting an
apparently significant score after viewing the whole curve would call for
the simultaneous band used in the main analysis.

```{code-cell} ipython3
pointwise = did.cont_did(data, **dict(spec, cband=False))
did.to_df(pointwise).filter(pl.col("dose") == 4.0)
```

Adding a knot moves the score-4 contrast from 0.0361 to 0.0384 log points
without changing the overall ATT of 0.0424. Since the overall adoption effect
averages observed outcome-change contrasts rather than fitted values on our
evaluation grid, its stability does not establish that the dose curve is
insensitive to smoothing. Using coded-zero controls alone gives a somewhat
smaller contrast of 0.0331 log points at score 4.0 and an overall ATT of 0.0394.
The pointwise interval from 0.0132 to 0.0590 log points is narrower than the
main band from 0.0103 to 0.0619 because its coverage claim concerns a single
score rather than the whole curve.

The distinction between an adoption effect and a response to greater exposure
remains central to interpreting these results. For the assumptions behind
that distinction and the package's aggregation weights, continue with the
{ref}`continuous treatment background <background-didcont>`. To reproduce
the paper's window-averaged curves alongside its event studies, the
{ref}`fracking replication <example_cont_did_replication>` works through the
published calculations and their checks.
