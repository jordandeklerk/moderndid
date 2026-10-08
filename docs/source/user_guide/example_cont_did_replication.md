---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_cont_did_replication)=

# Fracking paper replication

The {ref}`continuous treatment example <example_cont_did>` shows how to
analyze county employment with {func}`~moderndid.cont_did`. Here we return to
the same fracking data to reproduce the figures in [Callaway, Goodman-Bacon,
and Sant'Anna's *Event Studies with a Continuous
Treatment*](https://doi.org/10.1257/pandp.20241047). The paper asks how employment
changes after adoption and how those changes vary with geological
prospectivity, a score that stays fixed for each county.

Figure 1 compares low- and high-dose event studies before Figure 2 turns to
short- and long-run dose curves. The paper estimates those curves by averaging
adjusted outcome changes within each county before fitting a spline. Since
`cont_did` fits cohort-period curves before aggregating them, reproducing the
published curves requires the window-specific calculations below. The final
section checks these estimates and the pooled appendix figures against values
extracted from the published plotting paths.

The data come from [Bartik, Currie, Greenstone, and Knittel
(2019)](https://doi.org/10.1257/app.20170487) through the
[continuous-treatment replication files](https://doi.org/10.3886/E201785V1).
This is the 2024 companion paper's application rather than the Medicare
application in the [December 2025 main
paper](https://psantanna.com/files/CGBS_v4.pdf). Our
{ref}`background page <background-didcont>` uses the latter paper to explain
identification of level effects and the stronger restrictions needed for
causal responses to a larger dose.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## County employment and geological exposure

Prospectivity measures the geological conditions that make fracking attractive,
rather than how many wells a county eventually drilled. Since actual drilling
can respond to local economic conditions, the geological score helps distinguish
potential exposure from that subsequent investment. Total county employment
captures changes across the local economy rather than only employment in the
oil and gas industry.

{func}`~moderndid.load_fracking` loads a balanced panel of 402 counties observed
every year from 1990 through 2014. After dropping 2015 because all its
employment outcomes are missing, the preparation removes missing outcomes and
counties without the complete remaining panel. Each row identifies a county
with `i` and a year with `t`. Because the outcome `y` is already the log of
total employment, the estimation below uses it directly.

```{code-cell} ipython3

import moderndid as did
import numpy as np
import polars as pl
from scipy.stats import norm

# Keep the source outcome in log employment throughout the analysis.
data = did.load_fracking()
data.head()
```

The score `d` gives each county's fixed dose and `G` records the adoption year
used by the estimator. The `G_original` column preserves the source formation
date even when a county's zero dose changes its estimation `G` to zero. The
remaining column, `shale_basin1`, identifies the shale basin. The [original research
design](https://www.aeaweb.org/articles/materials/11519) dates adoption to when
successful fracking became publicly known within a formation. Announcements
after June enter the following year. That date represents a change in a
formation's economic prospects, rather than the first well drilled in every
county. A zero score places a county in the untreated group and gives it
`G=0` for estimation. The [original appendix](https://www.aeaweb.org/articles/materials/11519)
constructs prospectivity separately within shale plays without calibrating
the scores on a common scale. The pooled curves below reproduce the paper's
presentation rather than establishing comparable physical units of exposure
across plays.

:::{admonition} Zero includes imputed scores
:class: warning

The preparation recodes missing prospectivity scores to zero without retaining
a flag that distinguishes them from observed zeros. The untreated group
therefore depends on that coding decision, rather than an independently
observed absence of geological exposure in every county.
:::

Before splitting the treated counties, we count each county once and find the
median among the positive scores. The split uses the stored median rather than
a rounded `3.95` cutoff so ties stay in the correct group. This leaves 329
positive-dose counties and 73 coded-zero counties. A score at or below the median of 3.95 puts a treated
county in the low-dose group; a score above it puts the county in the high-dose
group.

```{code-cell} ipython3

# Count each county once and preserve the stored precision of the median.
units = data.filter(pl.col("t") == data["t"].min()).sort("i")
positive = units.filter(pl.col("d") > 0)
median_d = positive["d"].median()
low_data = data.filter((pl.col("d") <= median_d))
high_data = data.filter((pl.col("d") > median_d) | (pl.col("d") == 0))
```

There are 177 low-dose and 152 high-dose counties because several counties share
the median score. The smallest positive score of 0.20 describes the observed
support rather than an additional trimming rule. Each group keeps
the same 73 coded-zero counties for comparisons. Of the treated counties,
159 in the low-dose sample and 148 in the high-dose sample can be followed
through year four. Counties in the other
positive-dose group are excluded from its estimation sample.

## Employment over time in low- and high-dose counties

The first comparison asks whether employment responded differently in counties
that were more or less suited to fracking. Splitting at the median gives you
two employment histories without forcing the effect to be linear in the score.
Within each group, the starting point is the average effect of adoption at a
given dose,

$$
ATT(g,t,d)
= E[Y_t(g,d)-Y_t(0)\mid G=g,D=d].
$$

Here, `g` is the adoption year, `t` is the outcome year, and `d` is the
prospectivity score. The contrast describes adoption at dose `d` against
remaining untreated for the counties that actually received that dose. The
event study averages these effects within each dose group at a common number
of years since adoption.

That interpretation requires adoption not to affect employment beforehand and
untreated employment trends to agree across the relevant dose and timing
groups, neither of which follows from geological variation alone. Since low-
and high-dose counties are different populations, neither curve describes the
effect of assigning a higher prospectivity score to the same county. The {ref}`continuous-treatment background <background-didcont>`
develops this distinction between an effect at a dose and a causal response
to increasing the dose.

```{code-cell} ipython3

event_spec = dict(
    # Map county employment, calendar year, county, and adoption timing.
    yname="y",
    tname="t",
    idname="i",
    gname="G",
    # Compare untreated counties without adding covariates or sampling weights.
    control_group="notyettreated",
    est_method="reg",
    # Measure every change from the year before adoption.
    base_period="universal",
    anticipation=0,
    # Extract influence functions for the pointwise bootstrap below.
    boot=False,
    cband=False,
)

bootstrap_spec = dict(biters=1000, random_state=20240103)
```

### Compare counties before they adopt

The comparison group contains counties that remain untreated in both the
outcome year and the cohort's reference year, including the coded-zero counties
that remain untreated throughout the panel. Within either dose group, a
positive-dose county can serve as a comparison only if it belongs to that same
group and has not yet adopted at either date.
{func}`~moderndid.att_gt` estimates each cohort's employment effect against
these untreated counties using the regression method without covariates.

### Keep the year before adoption as the reference

The universal base period measures each cohort's employment changes relative
to the year before its adoption. The event study therefore sets event time
minus one to zero by construction. Keeping zero anticipation follows the
paper's specification and makes the interpretation depend on employment not
responding before the recorded adoption year.

### Follow the paper's event window

{func}`~moderndid.aggte` averages the cohort effects at each event time from
11 years before adoption through four years afterward. The aggregation weights
cohorts by their numbers of treated counties and uses every cohort observed at
the requested event time. It does not impose a balanced event window across
all the plotted years.

All 177 low-dose and 152 high-dose counties contribute at every event time
from minus 11 through two. Since the 2012 cohort cannot supply years three and
four before the panel ends in 2014, those estimates cover 159 low-dose and 148
high-dose counties. A change along either curve can therefore reflect a change
in the contributing counties as well as a change in their employment effects.

County influence functions describe how each county's observation contributes
to sampling variation in an estimated effect. The helper below gives those
contributions independent Rademacher multipliers of minus one or plus one in
1,000 draws using the fixed seed `20240103`. For each effect, it divides the
draws' interquartile range by the standard normal interquartile range to
estimate its standard error. A normal critical value then gives the pointwise
intervals from those bootstrap standard errors.

```{code-cell} ipython3

def paper_event_study(sample, biters, random_state):
    # Retain county influence functions without drawing the default bootstrap weights.
    group_time = did.att_gt(sample, **event_spec)
    result = did.aggte(
        group_time, type="dynamic", min_e=-11, max_e=4, boot=False, cband=False
    )

    # Give each county an independent equal-probability minus-one/plus-one multiplier.
    influence = np.column_stack([result.influence_func, result.influence_func_overall])
    n = influence.shape[0]
    rng = np.random.default_rng(random_state)
    weights = rng.choice([-1.0, 1.0], size=(biters, n))
    draws = weights @ influence / n
    quartiles = np.quantile(draws, [0.25, 0.75], axis=0, method="inverted_cdf")
    normal_iqr = norm.ppf(0.75) - norm.ppf(0.25)
    standard_errors = (quartiles[1] - quartiles[0]) / normal_iqr
    standard_errors[standard_errors < 1e-12] = np.nan

    # Report the inference actually computed, including the overall standard error.
    params = dict(
        result.estimation_params, bootstrap=True, uniform_bands=False,
        biters=biters, random_state=random_state,
    )
    call_info = dict(
        result.call_info, multiplier="rademacher", biters=biters, random_state=random_state
    )
    return result._replace(
        se_by_event=standard_errors[:-1],
        overall_se=float(standard_errors[-1]),
        critical_values=np.full(len(result.event_times), norm.ppf(0.975)),
        estimation_params=params,
        call_info=call_info,
    )


high_event = paper_event_study(high_data, **bootstrap_spec)
low_event = paper_event_study(low_data, **bootstrap_spec)
print("HIGH DOSE")
print(high_event)
print("LOW DOSE")
print(low_event)
```

Figure 1 uses orange for the low-dose counties and blue for the high-dose
counties. Each ribbon shows uncertainty around an individual estimate and lets
you read whether the employment contrast at a particular year is distinguishable
from zero. Although the point estimates reproduce the published curves to the
resolution of the paper's figure, the recomputed intervals do not exactly
reproduce the historical ribbons.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {"image": {"alt": "Low- and high-dose county employment event studies, with larger high-dose estimates after adoption and shaded pointwise intervals."}}

from plotnine import (
    aes, geom_hline, geom_line, geom_ribbon, geom_vline, ggplot, labs,
    scale_color_manual, scale_fill_manual, scale_x_continuous, scale_y_continuous,
)


def event_frame(result, label):
    # Draw the normalized reference as zero rather than as an estimated effect.
    se = np.where(result.event_times == -1, 0.0, result.se_by_event)
    return pl.DataFrame({
        "x": result.event_times,
        "estimate": result.att_by_event,
        "lower": result.att_by_event - norm.ppf(0.975) * se,
        "upper": result.att_by_event + norm.ppf(0.975) * se,
        "series": [label] * len(se),
    })


def comparison_plot(frame, colors, xlabel, event=False):
    plot = (
        ggplot(frame.to_pandas(), aes("x", "estimate", color="series", fill="series"))
        + geom_hline(yintercept=0, linetype="dotted", color="#555555")
        + geom_ribbon(aes(ymin="lower", ymax="upper"), alpha=0.15, color=None)
        + geom_line(size=1.1)
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + scale_y_continuous(breaks=np.arange(-0.04, 0.121, 0.02))
        + labs(x=xlabel, y="Average difference in log employment", color="", fill="")
        + did.theme_moderndid()
    )
    if event:
        plot += geom_vline(xintercept=-1, linetype="dotted", color="#777777")
        plot += scale_x_continuous(breaks=range(-11, 5))
    else:
        plot += scale_x_continuous(breaks=np.arange(2, 5.6, 0.5))
    return plot


figure1_data = pl.concat([
    event_frame(high_event, "High dose"), event_frame(low_event, "Low dose")
])
figure1 = comparison_plot(
    figure1_data, {"High dose": "#00008B", "Low dose": "#E69F00"},
    "Years relative to adoption", event=True,
)
figure1
```

At four years after adoption, the low-dose estimate is 0.0235 log points and
the high-dose estimate is 0.0719 log points. Converting those log contrasts
gives about 2.4 percent and 7.5 percent, respectively. These are averages of
county log employment effects relative to their estimated untreated paths,
rather than percentage increases in the total number of jobs across each
sample. The larger blue estimate describes a larger employment gain among
high-dose counties; comparing the two estimates alone does not test whether
the groups' effects differ statistically.

:::{admonition} Read intervals one estimate at a time
:class: important

These 95 percent pointwise intervals do not cover the whole curve with
95 percent confidence. The joint pre-treatment Wald test is unavailable
because its estimated covariance matrix is singular. Pre-treatment intervals
that include zero therefore provide neither a joint acceptance of parallel
trends nor proof of the identifying assumption.
:::

The pre-adoption estimates show no pronounced divergence in employment before
the recorded onset. Their uncertainty still allows some differences in those
earlier trends. Because the inference treats counties as independent sampling
units, it does not account for shocks shared by counties in the same formation.

## Short- and long-run effects across prospectivity scores

Figure 2 keeps the score continuous to show the variation that the median split
hides within each dose group. Averaging across years after adoption gives a
short-run curve for the adoption year through year two and a long-run curve for
years three and four.

Each point on either curve is the estimated effect for counties at that
prospectivity score, averaged over the corresponding years. It still compares
those counties with their untreated employment path. A difference across
scores can reflect differences between the counties that received them,
rather than a causal effect of raising a county's score.

:::{admonition} Match the paper's aggregation
:class: tip

The paper pools counties' adjusted employment changes before fitting one
spline for each time window. With `aggregation="dose"`,
{func}`~moderndid.cont_did` instead fits cohort-period curves and aggregates
them. The code below constructs the paper's pooled outcome explicitly and
uses the public {class}`~moderndid.BSpline` helper. Reproducing this calculation
therefore requires more than changing a `cont_did` argument.
:::

```{code-cell} ipython3

dose_spec = dict(
    # Hold the county population fixed within each event-time window.
    short_window=(0, 2),
    long_window=(3, 4),
    # Fit one pooled cubic with knots at the treated-dose quartiles.
    degree=3,
    knot_probabilities=(0.25, 0.5, 0.75),
    # Fit all positive doses and display only observed doses inside this range.
    display_range=(2, 6),
    # Use the paper's normal critical value for pointwise spline intervals.
    critical_value=1.96,
)
```

### Average each county over the requested years

For a county that adopts in year `g`, the calculation starts with its change
in log employment from `g-1` to each year in the selected window. From that
change it subtracts the average change among the eligible coded-zero counties
over the same calendar years. After averaging these adjusted changes, a county
observed for three short-run years contributes only one row to the spline fit.

The event studies above use not-yet-treated counties within each dose group.
The dose-response calculation follows the preparation used to reproduce the
paper's plotted curves and uses only coded-zero counties for its adjustment.
This restriction is narrower than the comparison group described in the
paper's text. The code uses this narrower comparison to reproduce the plotted
calculation rather than labeling those means as not-yet-treated comparisons.

The window filter retains only counties whose formation's original adoption
year allows the entire window to be observed by 2014. Keeping that original
date matters even for coded-zero counties whose estimation `G` is zero. The
short-run sample contains 329 treated counties and 73 comparison counties.
The long-run sample contains 307 treated counties and 60 comparison counties,
since the 2012 cohort removes 22 positive-dose counties and 13 coded-zero
counties under this filter.

Since each county's dose and adoption dates stay fixed, one value per county
is enough to prepare each window. The code keeps these values in the same
county order as the employment matrix so every adjusted change uses the
corresponding county's outcome, dose, and dates.

```{code-cell} ipython3

# Align the balanced outcomes and the fixed county characteristics.
years = np.sort(data["t"].unique().to_numpy())
outcomes = data.pivot(index="i", on="t", values="y").sort("i").select(
    [str(year) for year in years]
).to_numpy()
groups = units["G"].to_numpy()
original_groups = units["G_original"].to_numpy()
doses = units["d"].to_numpy()


def prepare_window(start, end):
    # Preserve the original formation-date restriction even for coded-zero counties.
    controls = (doses == 0) & (original_groups + end <= years[-1])
    prepared = []
    for g in np.unique(groups[groups > 0]):
        if g + end > years[-1]:
            continue
        treated = groups == g
        base = outcomes[:, g - 1 - years[0]]
        adjusted = []
        for event in range(start, end + 1):
            change = outcomes[:, g + event - years[0]] - base
            adjusted.append(change[treated] - change[controls].mean())
        prepared.append(pl.DataFrame({
            "i": units["i"].to_numpy()[treated],
            "d": doses[treated],
            "adjusted_change": np.mean(adjusted, axis=0),
        }))
    return pl.concat(prepared).sort("i"), int(controls.sum())


short_sample, short_controls = prepare_window(*dose_spec["short_window"])
long_sample, long_controls = prepare_window(*dose_spec["long_window"])
```

Within each window, every retained county contributes in every averaged year.
The comparison between the short- and long-run curves still changes the
population because the short-run window includes the 2012 cohort. It therefore
combines differences in exposure length with differences in eligible counties,
even though composition stays fixed inside each window.

### Let the spline describe changes across doses

The pooled regression approximates each window's adjusted employment changes
with a cubic spline. Its three internal knots of 3.35, 3.95, and 4.34 sit at the
treated-dose quartiles in both samples. These knots allow
the relationship to bend within the observed score distribution without
requiring one linear employment response over the whole range.

The fit uses all eligible positive doses and displays the predictions at
observed scores strictly between two and six as the paper does. That plotting
restriction keeps the figure focused on the central range; it does not remove
other positive-dose counties from estimation. Each fitted curve remains a
model-based approximation, rather than a separate nonparametric estimate at
every plotted score.

```{code-cell} ipython3

def fit_window(sample):
    # Use the complete spline basis without adding another regression intercept.
    dose = sample["d"].to_numpy()
    response = sample["adjusted_change"].to_numpy()
    knots = np.quantile(dose, dose_spec["knot_probabilities"])
    boundaries = [dose.min(), dose.max()]
    basis_spec = dict(
        internal_knots=knots, boundary_knots=boundaries, degree=dose_spec["degree"]
    )
    basis = did.BSpline(dose, **basis_spec).basis()
    beta = np.linalg.lstsq(basis, response, rcond=None)[0]

    # Form the treated-county residual influence function used for the plotted intervals.
    n = len(dose)
    residual = response - basis @ beta
    bread = np.linalg.pinv(basis.T @ basis / n)
    influence = (residual[:, None] * basis) @ bread
    values = np.unique(dose)
    prediction_basis = did.BSpline(values, **basis_spec).basis()
    estimate = prediction_basis @ beta
    curve_influence = influence @ prediction_basis.T
    se = np.sqrt(np.mean(curve_influence ** 2, axis=0) / n)
    curve = pl.DataFrame({
        "x": values, "estimate": estimate, "se": se,
        "lower": estimate - dose_spec["critical_value"] * se,
        "upper": estimate + dose_spec["critical_value"] * se,
    })
    return {"curve": curve, "knots": knots, "coefficients": beta, "counties": n}


short_fit = fit_window(short_sample)
long_fit = fit_window(long_sample)
```

The pointwise intervals for these dose curves come from the residual variation
in the pooled spline regression. They condition on the comparison counties'
estimated mean changes and do not add a separate uncertainty contribution for
estimating those means. Following this calculation reproduces the plotted
dose-response results; its intervals should not be read as the simultaneous
bands returned by `cont_did`.

Figure 2 uses orange for the short-run effects and blue for the long-run
effects. Before comparing their heights, keep in
mind that the change in color now marks a time window rather than a low- or
high-dose group.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {"image": {"alt": "Short- and long-run county employment effects across prospectivity scores, with the long-run curve rising, dipping, and reaching its largest estimates near five."}}

def displayed_curve(fit, label):
    lower, upper = dose_spec["display_range"]
    return fit["curve"].filter(
        (pl.col("x") > lower) & (pl.col("x") < upper)
    ).with_columns(pl.lit(label).alias("series"))


figure2_data = pl.concat([
    displayed_curve(short_fit, "Short run"), displayed_curve(long_fit, "Long run")
])
figure2 = comparison_plot(
    figure2_data, {"Short run": "#E69F00", "Long run": "#00008B"},
    "County prospectivity score",
)
figure2
```

The long-run curve rises toward a score of three, dips around the middle
of the displayed range, and reaches its largest values near five. Its estimates
exceed the short-run curve throughout the displayed range and show larger
late employment gains beyond the counties with the very highest scores.

At a score of 5.00, the short-run estimate is 0.0125 log points and its
pointwise interval runs from -0.0007 to 0.0257. The long-run estimate of
0.0733 log points corresponds to about 7.6 percent and has an interval from
0.0388 to 0.1079 log points. This contrast describes a larger estimated employment
gain several years after adoption; a formal test of the difference would
also need the covariance between the two estimates.

Because prospectivity scores are constructed differently across formations,
the same increase in the recorded index need not represent the same increase
in physical resources everywhere. The shape of the blue curve describes
differences in estimated effects among counties at their recorded scores
rather than the response to an additional well or an additional unit of
geological resources in the same county.

## What the pooled figures leave out

The appendix averages over all positive doses for its overall event study and
over the adoption year through year four for its overall dose curve. Each
gives a shorter account of the employment gains at the cost of hiding
differences across doses or exposure lengths.

### Average the event study across positive doses

Appendix Figure B.1 uses all 329 positive-dose counties and the 73 coded-zero
counties in one event-study fit. It keeps the same not-yet-treated comparison
group, reference period, event window, and pointwise inference as Figure 1.
At years three and four, the estimate again covers the 307 treated counties
observed for that long.

```{code-cell} ipython3

all_event = paper_event_study(data, **bootstrap_spec)
print(all_event)
```

The green curve below summarizes employment changes for all positive-dose
counties at their observed doses. It answers how adoption affected that pooled
population at each exposure length, rather than how the effect varies across
prospectivity scores.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {"image": {"alt": "Pooled county employment event study across all positive doses, with larger post-adoption estimates in the later years and a shaded pointwise interval."}}

figure_b1 = comparison_plot(
    event_frame(all_event, "All positive doses"), {"All positive doses": "#009E73"},
    "Years relative to adoption", event=True,
)
figure_b1
```

Four years after adoption, the pooled estimate is 0.0474 log points,
equivalent to about 4.9 percent. This reading averages the effects at the
counties' actual scores rather than imposing a common dose.

The pooled post-adoption estimates sit between the low- and high-dose profiles
and conceal how much larger the later high-dose contrast is. Since this fit
also uses a broader not-yet-treated comparison population than either dose
group's fit, its estimates are not just a weighted average of the two plotted
profiles. The later points still cover fewer cohorts and the pointwise
intervals still treat counties as independent sampling units.

### Average the dose curve over the first five years

Appendix Figure B.3 uses the same adjusted-outcome construction as Figure 2
but averages event times zero through four. That window contains five annual
observations per treated county, including the adoption year. Its 307 treated
counties and 60 coded-zero comparison counties match the long-run sample
because both windows require observation through year four.

```{code-cell} ipython3

pooled_sample, pooled_controls = prepare_window(0, 4)
pooled_fit = fit_window(pooled_sample)
```

The green curve gives an effect at each score averaged over those five years.
It therefore brings the smaller early effects and larger later effects into
one contrast, at the cost of hiding when the employment changes emerged.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {"image": {"alt": "County employment effects across prospectivity scores averaged over event times zero through four, with a green curve and shaded pointwise interval."}}

figure_b3 = comparison_plot(
    displayed_curve(pooled_fit, "Years 0–4"), {"Years 0–4": "#009E73"},
    "County prospectivity score",
)
figure_b3
```

At a score of 5.00, the five-year average of 0.0393 log points corresponds to
about 4.0 percent and has a pointwise interval from 0.0177 to 0.0609 log points.
Because its sample matches the long-run sample, this estimate combines two
later annual effects with three earlier effects for the same treated counties.

### Check the calculations against the published figures

The [reference values](../_static/fracking_paper_reference.csv) below come from
the vector paths in the [author's paper and appendix](https://psantanna.com/files/CGBS_AEAPP.pdf),
rather than from exact tabulated estimates. Figure 2 also appears as appendix
Figure B.4 at a larger scale; those paths supply its reference values. The
comparison checks every plotted point and both interval endpoints. Its small
dose-curve differences reflect graphical rounding. Recomputed event-study
intervals retain larger differences from the historical bootstrap ribbons.

```{code-cell} ipython3
:tags: [hide-input]

# Match each published plotting location to its original observed score or event time.
reference = pl.read_csv("../_static/fracking_paper_reference.csv")
checks = [
    ("Figure 1", "high", event_frame(high_event, "High dose")),
    ("Figure 1", "low", event_frame(low_event, "Low dose")),
    ("Figure 2", "short", displayed_curve(short_fit, "Short run")),
    ("Figure 2", "long", displayed_curve(long_fit, "Long run")),
    ("Figure B1", "all", event_frame(all_event, "All doses")),
    ("Figure B3", "pooled", displayed_curve(pooled_fit, "Years 0–4")),
]
print(f"{'Figure':<11} {'Series':<8} {'Points':>7} {'Max estimate gap':>18} "
      f"{'Max interval gap':>18}")
for figure, series, calculated in checks:
    published = reference.filter((pl.col("figure") == figure) & (pl.col("series") == series))
    nearest = np.abs(
        published["x"].to_numpy()[:, None] - calculated["x"].to_numpy()
    ).argmin(axis=1)
    actual = calculated[nearest]
    estimate_gap = np.max(np.abs(
        actual["estimate"].to_numpy() - published["estimate"].to_numpy()
    ))
    interval_gap = max(
        np.max(np.abs(actual["lower"].to_numpy() - published["lower"].to_numpy())),
        np.max(np.abs(actual["upper"].to_numpy() - published["upper"].to_numpy())),
    )
    assert estimate_gap < 1.5e-5
    if figure in {"Figure 2", "Figure B3"}:
        assert interval_gap < 1.5e-5
    print(f"{figure:<11} {series:<8} {published.height:>7} {estimate_gap:>18.6f} "
          f"{interval_gap:>18.6f}")
```

The checks recover the published point estimates and dose-curve intervals
within the figures' graphical precision. The larger event-study interval
differences persist despite the agreement in point estimates and the fixed
seed used to make this page reproducible.

The employment gains are larger in higher-prospectivity counties and in the
later years after adoption. Interpreting those contrasts as effects of fracking
requires the assumptions about untreated employment trends and the recorded
adoption dates. The {ref}`continuous-treatment background <background-didcont>` develops
the identification and aggregation results behind these choices; the
{ref}`staggered adoption example <example_staggered_did>` shows how the same
cohort effects are used when treatment is binary.
