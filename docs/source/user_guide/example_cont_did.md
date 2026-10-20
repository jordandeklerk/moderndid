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

{func}`~moderndid.cont_did` implements the estimator of [Callaway,
Goodman-Bacon, and Sant'Anna (2025)](https://psantanna.com/files/CGBS_v4.pdf)
and lets you examine these changes as a curve over prospectivity scores or as an
event study over years since adoption. A regression of employment on the dose
interacted with a post-adoption indicator would compress the variation across
scores and over time into a single coefficient. Here you will fit both views of
the employment effects, read their uncertainty, and check how the dose curve
changes with the comparison group and spline specification.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (8.7, 3.6)
options.dpi = 100
```

## Counties, scores, and adoption years

The data come from [Bartik, Currie, Greenstone, and Knittel
(2019)](https://doi.org/10.1257/app.20170487) as prepared for the fracking
application in [Callaway, Goodman-Bacon, and Sant'Anna
(2024)](https://doi.org/10.1257/pandp.20241047). Each county's score rates how
favorable its geology is for fracking. Since the pace of drilling can depend on
local economic conditions, a count of the wells eventually drilled there would
mix that exposure with the investment it attracted. The outcome is total county
employment because it records job changes across the local economy and not only
in oil and gas.

{func}`~moderndid.load_fracking` loads 10,050 observations from 402 counties,
each observed annually from 1990 through 2014. The loader omits 2015 because all
employment outcomes are missing and removes counties with an incomplete
remaining panel. Because the outcome `y` already contains log total employment,
you can use it directly without another transformation.

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
economic prospects changed, not when the first well was drilled in each county.

:::{admonition} Coded zeros include missing scores
:class: warning

When the data were prepared, missing prospectivity scores were set to zero
without a flag that marks the imputed ones. Because the 73 coded-zero counties
mix observed zeros with that coding decision, their place in the comparison
group does not show that they lacked geological exposure.
:::

## The effect of adoption at a given score

For counties whose formation adopted in year $g$ and whose score is $d$, the
level effect in year $t$ compares their log employment with what it would have
been had their formation not adopted. The average causal response on the
treated is the slope of that effect in the score for those same counties,

$$
\begin{aligned}
ATT(g,t,d\mid g,d)
   &=\mathbb{E}[Y_t(g,d)-Y_t(0)\mid G=g,D=d],\\
ACRT(g,t,d\mid g,d)
   &=\left.\frac{\partial ATT(g,t,l\mid g,d)}{\partial l}\right|_{l=d}.
\end{aligned}
$$

`cont_did` fits a curve in the score for every cohort and year after adoption.
The dose curve averages those curves with weights that give each cohort its
share of the treated counties, split evenly over its years after adoption. The
overall ATT and ACRT average each cohort and year's effect and slope over the
scores its treated counties have. Both describe adoption at the scores counties
actually have rather than the effect of giving every county one common score.
The {ref}`continuous treatment background <background-didcont>` derives these
targets and the assumptions that identify them.

## Set up the dose analysis

Each choice behind the dose curves sits in the specification below, grouped by
the question it answers.

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
concerns untreated trends and leaves employment levels free to differ. The
{ref}`continuous treatment background <background-didcont>` states parallel
trends formally for staggered adoption. The
[comparison-group check](#use-only-coded-zero-comparisons) below drops the
future adopters and keeps only the coded-zero counties.

As in the {ref}`staggered example <example_staggered_did>`,
`base_period="universal"` and `anticipation=0` measure each comparison from the
year before the cohort adopts and assume employment doesn't respond any earlier.

### Fit within the observed score ranges

A cubic spline with no interior knots gives a cubic polynomial in each
cohort-period comparison. We start with this specification because some
cohorts contain too few distinct scores to support a more flexible curve.

The 2006 cohort contains nine counties with only six distinct scores. Since
a cubic spline with three interior knots requires seven coefficients, that
cohort cannot identify such a flexible curve.

:::{admonition} Three knots and the 2006 cohort
:class: note

With `num_knots=3`, `cont_did` warns and leaves the 2006 cohort out of the dose
curves and their derivative summary. Its counties still count toward the overall
ATT in the report below.
:::

Across the eight adoption cohorts, the observed score ranges overlap from 2.99
to 4.67. Our evaluation grid from 3.0 to 4.6 stays inside that overlap. Since
`dvals` only sets where the curves are evaluated, all 329 positive-dose counties
remain in the estimation. Staying inside the ranges does not settle how many
observations sit near each grid score or how well a cubic approximates the
curve. The [spline check](#allow-one-interior-knot) below tries one interior
knot to examine that approximation.

:::{admonition} Compare scores within one play
:class: important

The [original appendix](https://www.aeaweb.org/articles/materials/11519)
builds prospectivity separately for each shale play without putting the plays
on a common scale. Although the pooled panel illustrates the estimator, a
substantive comparison of geological intensity across plays requires a
justified common scale or an analysis within one play.
:::

### Measure uncertainty across the curve

Since the multiplier bootstrap gives each county its own random weight in every
draw, it treats counties as independent sampling units. With `cband=True`, each
plotted curve has a simultaneous 95 percent band that covers all 65 evaluation
points at once. The overall ATT and ACRT keep pointwise intervals because each
is a single number.

The 999 draws and fixed seed make the reported results reproducible. The
{ref}`bootstrap background <background-did-bootstrap>` explains how the
simultaneous critical value is built. Since `cont_did` can't cluster by
formation, these county-level bands don't account for shocks that counties in
the same formation may share.

:::{admonition} Pointwise intervals assume a prespecified score
:class: warning

Setting `cband=False` gives a narrower pointwise interval that holds only for a
score chosen before looking at the curve. Selecting an apparently significant
score after viewing the whole curve calls for the simultaneous band.
:::

The fit now uses the specification we have just worked through.

```{code-cell} ipython3
# Fit the dose curves and their overall summaries.
result = did.cont_did(data, **spec)
print(result)
```

## Read the effects across scores

The overall ATT is 0.0424 log points with a 95 percent interval from 0.0240
to 0.0609. Before reading the ACRT, which summarizes the curve's slope, you can
see how the level effect itself differs across scores.

### Compare the fitted level contrasts

{func}`~moderndid.plots.plot_dose_response` draws the level curve stored in
`result` with its simultaneous band. Because the cohort weights stay the same at
every score, the curve holds the mix of adoption years fixed. The counties that
have a given score may come from a different mix, as the
{ref}`aggregation discussion <background-didcont-aggregation>` explains.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Employment effects rising from about 0.030 log points at score 3.0 to 0.057 at score 4.6 inside a simultaneous band that stays above zero
---
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
3.7 percent higher employment using `100 * np.expm1(estimate)`. The simultaneous
band runs from 0.0103 to 0.0619 log points there. With this seed, its lower edge
stays above zero at every score on the plot. At score 3.0 it clears zero by only
0.0017 log points. Across 30 other bootstrap seeds, the band stays above zero
from score 3.1 up every time. Since 7 of those seeds put its lower edge below
zero at score 3.0, the band does not establish a positive effect at the lowest
scores.

### Distinguish the slope from a causal response

The slope curve is available in the same dose result even though the call
requested `target_parameter="level"`. Setting `effect_type="acrt"` draws that
stored derivative without estimating another model. Its vertical axis is log
employment per additional unit of the prospectivity score.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: The fitted slope of the employment effect, above zero from score 3.0 to 4.6, with a simultaneous band that includes zero at every score
---
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

On average across the treated counties, the employment effect grows with
prospectivity. The overall ACRT in the report above pools the slope over every
treated county's observed score into one number. Its estimate of 0.0442 log
points per score unit has an interval from 0.0253 to 0.0630 that excludes zero.

The data can't place that growth at any particular score on the plot, though.
Although the fitted slope stays above zero across the plot, its simultaneous
band includes zero at every displayed score with the same seed. Six of the 30
other seeds lift its lower edge just above zero somewhere between 4.3 and 4.6.

:::{admonition} The slope compares different counties
:class: warning

Counties with higher scores may also gain more from adoption at any given
score. Under ordinary parallel trends, the slope mixes those differences with
the response to greater exposure. Only the stronger restrictions in the
{ref}`continuous treatment background <background-didcont>` make it a causal
response.
:::

## Follow employment after adoption

Since averaging over post-adoption years conceals whether the employment
contrast develops gradually, we now change the aggregation to an event study and
follow the first five years after adoption. With `target_parameter="level"`,
each event time averages binary adoption effects over the counties' observed
doses. The degree, knots, and evaluation grid therefore do not determine these
level estimates. Pointwise intervals make the event-study inference comparable
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

The report's overall ATT of 0.0227 log points averages the five post-adoption
estimates with equal weights. The dose report's 0.0424 instead covers each
cohort's full observed history after adoption.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: County employment before and after adoption, with pre-adoption estimates near zero and post-adoption estimates rising to 0.047 log points in year four
---
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
the reference year, fixed at zero without an interval of its own.

:::{admonition} Pre-adoption estimates do not prove parallel trends
:class: warning

Intervals that include zero before adoption can't establish parallel trends, as
the {ref}`sensitivity analysis example <example_honest_did>` explains. They
can't test the stronger restrictions behind a causal slope at all, since no
county's outcomes under adoption are observed before it adopts.
:::

The last two event times cover fewer cohorts because the 2012 adopters can't
contribute to years three and four of a panel that ends in 2014. The
{ref}`fracking replication <example_cont_did_replication>` separates low- and
high-dose counties to reproduce the paper's Figure 1 and checks this pooled path
against its appendix Figure B.1. Although the bootstrap intervals here need not
match the paper's ribbons, `cont_did` reproduces the Figure B.1 point estimates
up to the precision of the extracted plotting paths.

## Check the choices behind the dose curve

The fitted dose pattern depends on choices that the event-study plot cannot
settle.

### Use only coded-zero comparisons

Positive-dose counties awaiting adoption might differ from coded-zero counties
in their untreated employment changes. Restricting the comparison group to
`"nevertreated"` removes those future adopters from the controls.

```{code-cell} ipython3
# Compare each cohort with the coded-zero counties alone.
never_treated = did.cont_did(data, **dict(spec, control_group="nevertreated"))
print("Overall ATT:", round(never_treated.overall_att, 4))
did.to_df(never_treated).filter(pl.col("dose") == 4.0)
```

Using coded-zero controls alone lowers the contrast at score 4.0 from 0.0361
to 0.0331 log points and the overall ATT from 0.0424 to 0.0394.

### Allow one interior knot

A knot at the median positive score lets the cubic spline bend there. Since the
2006 cohort's six distinct scores can't support the three knots of the pooled
paper replication, the check adds one.

```{code-cell} ipython3
# Place one interior knot at the median positive score.
one_knot = did.cont_did(data, **dict(spec, num_knots=1))
print("Overall ATT:", round(one_knot.overall_att, 4))
did.to_df(one_knot).filter(pl.col("dose") == 4.0)
```

Adding a knot moves the contrast at score 4.0 from 0.0361 to 0.0384 log points
and leaves the overall ATT at 0.0424. Since the overall ATT averages observed
outcome changes and never uses the fitted curve, its stability says nothing
about how sensitive the curve is to smoothing.

Neither check moves the contrast at score 4.0 by more than 0.0030 log points, a
small shift beside the band from 0.0103 to 0.0619 at that score. Only the
comparison group moves the overall ATT, by the same 0.0030 log points. The
{ref}`fracking replication <example_cont_did_replication>` reproduces the
paper's event studies for low- and high-dose counties and its window-averaged
dose curves from the same data.
