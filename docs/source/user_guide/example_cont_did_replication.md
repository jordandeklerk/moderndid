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

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (8.7, 3.6)
options.dpi = 100
```

## County employment and geological exposure

The {ref}`continuous treatment example <example_cont_did>` describes the
counties, prospectivity scores, adoption dates, and coded-zero comparison
counties in the paper's [replication data](https://doi.org/10.3886/E201785V1).
Figure 1 splits the 329 positive-dose counties at their median score of 3.95.
Counties at or below the median form the low-dose group and the rest form the
high-dose group. The split uses the stored median instead of a rounded `3.95`
cutoff so that ties land in the right group.

```{code-cell} ipython3
import moderndid as did
import numpy as np
import polars as pl
from scipy.stats import norm

# Keep the source outcome in log employment throughout the analysis.
data = did.load_fracking()

# Count each county once and preserve the stored precision of the median.
units = data.filter(pl.col("t") == data["t"].min()).sort("i")
positive = units.filter(pl.col("d") > 0)
median_d = positive["d"].median()
low_data = data.filter(pl.col("d") <= median_d)
high_data = data.filter((pl.col("d") > median_d) | (pl.col("d") == 0))
print("Median positive score:", median_d)
```

Because several counties share the median score, the groups hold 177 low-dose
and 152 high-dose counties. Each group keeps the same 73 coded-zero counties for
comparisons and leaves the other group's counties out of its estimation sample.

## Employment over time in low- and high-dose counties

The first comparison asks whether employment responded differently in counties
that were more or less suited to fracking. Splitting at the median gives you
two employment histories without forcing the effect to be linear in the score.
Within each group, the starting point is the average effect of adoption at a
given dose,

$$
ATT(g,t,d\mid g,d)
= \mathbb{E}[Y_t(g,d)-Y_t(0)\mid G=g,D=d].
$$

Here, $g$ is the adoption year, $t$ is the outcome year, and $d$ is the
prospectivity score. The contrast describes adoption at dose $d$ against
remaining untreated for the counties that actually received that dose. The
event study averages these effects within each dose group at a common number
of years since adoption.

That interpretation requires adoption not to affect employment beforehand and
untreated employment trends to agree across the relevant dose and timing
groups. Neither requirement follows from the geological variation in
prospectivity alone.

:::{admonition} Dose groups and causal responses
:class: note

Since low- and high-dose counties are different populations, neither curve
describes the effect of assigning a higher prospectivity score to the same
county. The {ref}`continuous-treatment background <background-didcont>`
develops this distinction between an effect at a dose and a causal response to
increasing the dose.
:::

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
```

### Compare counties before they adopt

The comparison group contains counties that remain untreated in both the
outcome year and the cohort's reference year. Since the coded-zero counties
remain untreated throughout the panel, every comparison includes them. Within
either dose group, a positive-dose county can serve as a comparison only if it
belongs to that same group and has not yet adopted at either date.
{func}`~moderndid.att_gt` estimates each cohort's employment effect against
these untreated counties using the regression method without covariates.

### Keep the year before adoption as the reference

As in the paper and the {ref}`staggered example <example_staggered_did>`,
`base_period="universal"` and `anticipation=0` measure each cohort's employment
changes from the year before it adopts and fix event time minus one at zero.

### Follow the paper's event window

{func}`~moderndid.aggte` averages the cohort effects at each event time from
11 years before adoption through four years afterward. It weights cohorts by
their numbers of treated counties and uses every cohort observed at the
requested event time without imposing a balanced window. All 177 low-dose and
152 high-dose counties contribute at every event time from minus 11 through two.

:::{admonition} Later event times drop the 2012 cohort
:class: warning

Since the 2012 cohort cannot supply years three and four before the panel ends
in 2014, those estimates cover 159 low-dose and 148 high-dose counties. A
change along either curve can therefore reflect a change in the contributing
counties as well as a change in their employment effects.
:::

```{code-cell} ipython3
# Estimate each cohort's effects in both dose groups without the default bootstrap.
high_group_time = did.att_gt(high_data, **event_spec)
low_group_time = did.att_gt(low_data, **event_spec)

# Average the cohort effects at each event time in the paper's window.
dynamic_spec = dict(type="dynamic", min_e=-11, max_e=4, boot=False, cband=False)
high_dynamic = did.aggte(high_group_time, **dynamic_spec)
low_dynamic = did.aggte(low_group_time, **dynamic_spec)

# Read the estimates four years after adoption, the last event time in the window.
print("High dose:", round(float(high_dynamic.att_by_event[-1]), 4))
print("Low dose:", round(float(low_dynamic.att_by_event[-1]), 4))
```

County influence functions describe how each county's observation contributes
to sampling variation in an estimated effect. The event studies below give
those contributions independent Rademacher multipliers of minus one or plus one
in 1,000 draws using the fixed seed `20240103`. Each standard error is the
draws' interquartile range divided by the standard normal interquartile range.
A normal critical value then turns those standard errors into pointwise
intervals.

```{code-cell} ipython3
:tags: [hide-input]

def paper_inference(result, biters, random_state):
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
        result.estimation_params, bootstrap=True, biters=biters, random_state=random_state
    )
    call_info = dict(
        result.call_info, multiplier="rademacher", biters=biters, random_state=random_state
    )
    return result._replace(
        se_by_event=standard_errors[:-1],
        overall_se=float(standard_errors[-1]),
        estimation_params=params,
        call_info=call_info,
    )
```

```{code-cell} ipython3
:tags: [hide-input, scroll-output]

# Give every event study on this page the same number of draws and seed.
bootstrap_spec = dict(biters=1000, random_state=20240103)
high_event = paper_inference(high_dynamic, **bootstrap_spec)
low_event = paper_inference(low_dynamic, **bootstrap_spec)
print("HIGH DOSE")
print(high_event)
print("LOW DOSE")
print(low_event)
```

Figure 1 uses orange for the low-dose counties and blue for the high-dose
counties. Although the point estimates reproduce the published curves to the
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
        ggplot(frame, aes("x", "estimate", color="series", fill="series"))
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
gives about 2.4 percent and 7.5 percent, respectively. The larger blue estimate
describes a larger employment gain among high-dose counties. Comparing the two
estimates alone does not test whether the groups' effects differ statistically.

:::{admonition} Read intervals one estimate at a time
:class: important

These 95 percent pointwise intervals do not cover the whole curve with
95 percent confidence. The joint pre-treatment Wald test is unavailable
because its estimated covariance matrix is singular. As the
{ref}`sensitivity analysis example <example_honest_did>` explains, intervals
that include zero before adoption wouldn't establish parallel trends anyway.
:::

The pre-adoption estimates show no pronounced divergence in employment before
the recorded onset. Because the inference treats counties as independent sampling
units, it does not account for shocks shared by counties in the same formation.

## Short- and long-run effects across prospectivity scores

Figure 2 keeps the score continuous to show the variation that the median split
hides within each dose group. Averaging across years after adoption gives a
short-run curve for the adoption year through year two and a long-run curve for
years three and four.

Each point on either curve is the estimated effect for counties at that
prospectivity score, averaged over the corresponding years. Like the dose
groups above, the curve compares different counties at different scores.

:::{admonition} Match the paper's aggregation
:class: tip

When no {func}`~moderndid.cont_did` argument matches a published calculation,
the code below shows how to rebuild it with the public
{class}`~moderndid.BSpline` class that `cont_did` uses for its own splines.
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

For a county that adopts in year $g$, the calculation starts with its change
in log employment from $g-1$ to each year in the selected window. From that
change it subtracts the average change among the eligible coded-zero counties
over the same calendar years. After averaging these adjusted changes, a county
observed for three short-run years contributes only one row to the spline fit.

:::{admonition} The comparison behind the dose curves
:class: note

The event studies above use not-yet-treated counties within each dose group.
The dose-response calculation instead adjusts with coded-zero counties alone
because that preparation reproduces the paper's plotted curves. This
comparison is narrower than the one the paper's text describes.
:::

The window filter retains only counties whose formation's original adoption
year allows the entire window to be observed by 2014. Keeping that original
date in `G_original` matters even for coded-zero counties, whose estimation `G`
is zero. The short-run sample contains 329 treated counties and 73 comparison
counties. The long-run sample contains 307 treated counties and 60 comparison
counties, since the 2012 cohort removes 22 positive-dose counties and 13
coded-zero counties under this filter.

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
    return pl.concat(prepared).sort("i")


short_sample = prepare_window(*dose_spec["short_window"])
long_sample = prepare_window(*dose_spec["long_window"])
short_sample.head()
```

Within each window, every retained county contributes in every averaged year.
Because the short-run window includes the 2012 cohort, a comparison of the two
curves mixes differences in exposure length with differences in the counties.

### Let the spline describe changes across doses

The pooled regression approximates each window's adjusted employment changes
with a cubic spline. Its three internal knots of 3.35, 3.95, and 4.34 sit at the
treated-dose quartiles in both samples. These knots let the relationship bend
within the observed score distribution instead of imposing one linear
employment response over the whole range. As in the paper, the fit uses all
eligible positive doses and displays the predictions only at observed scores
strictly between two and six.

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
    return {"curve": curve, "knots": knots}


short_fit = fit_window(short_sample)
long_fit = fit_window(long_sample)
print("Short-run knots:", short_fit["knots"].round(2))
print("Long-run knots:", long_fit["knots"].round(2))
```

The pointwise intervals for these dose curves come from the residual variation
in the pooled spline regression. They condition on the comparison counties'
estimated mean changes and do not add a separate uncertainty contribution for
estimating those means. Although this calculation reproduces the plotted
dose-response results, its intervals should not be read as the simultaneous
bands returned by `cont_did`.

In Figure 2, orange marks the short-run effects and blue the long-run effects.
The colors now distinguish time windows instead of low- and high-dose groups.

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
late employment gains that are not limited to the counties with the very
highest scores. The table below shows both curves at a score of 5.00.

```{code-cell} ipython3
# Read the short- and long-run curves at a score of 5.
pl.concat([
    short_fit["curve"].with_columns(window=pl.lit("short run")),
    long_fit["curve"].with_columns(window=pl.lit("long run")),
]).filter(pl.col("x") == 5.0)
```

The short-run estimate of 0.0125 log points has a pointwise interval from
−0.0007 to 0.0257. The long-run estimate of 0.0733 log points corresponds to
about 7.6 percent and has an interval from 0.0388 to 0.1079 log points. A
formal test of the difference would also need the covariance between the two
estimates.

:::{admonition} Score units differ across shale plays
:class: warning

Because the plays' scores sit on separate scales, the same increase in the
recorded index need not represent the same increase in physical resources
everywhere.
:::

## What the pooled figures leave out

The appendix averages over all positive doses for its overall event study,
Figure B.1, and over the adoption year through year four for its overall dose
curve, Figure B.3. Each gives a shorter account of the employment gains at the
cost of hiding differences across doses or exposure lengths. Since the
{ref}`continuous treatment example <example_cont_did>` draws the pooled event
study of Figure B.1 with `cont_did`, this page checks that figure only against
its published values below.

### Average the dose curve over the first five years

Appendix Figure B.3 uses the same adjusted-outcome construction as Figure 2
but averages event times zero through four. Its 307 treated counties and 60
coded-zero comparison counties match the long-run sample because both windows
require observation through year four.

```{code-cell} ipython3
# Average event times zero through four and read the curve at a score of 5.
pooled_sample = prepare_window(0, 4)
pooled_fit = fit_window(pooled_sample)
pooled_fit["curve"].filter(pl.col("x") == 5.0)
```

At a score of 5.00, the five-year average of 0.0393 log points corresponds to
about 4.0 percent and has a pointwise interval from 0.0177 to 0.0609 log points.
The green curve below gives an effect at each score averaged over those five
years.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {"image": {"alt": "County employment effects across prospectivity scores averaged over event times zero through four, with a green curve and shaded pointwise interval."}}

figure_b3 = comparison_plot(
    displayed_curve(pooled_fit, "Years 0–4"), {"Years 0–4": "#009E73"},
    "County prospectivity score",
)
figure_b3
```

Each point on the green curve combines two later annual effects with three
earlier effects for the same 307 treated counties. The curve therefore brings
the smaller early effects and larger later effects into one contrast, at the
cost of hiding when the employment changes emerged.

### Check the calculations against the published figures

The [reference values](../_static/fracking_paper_reference.csv) below come from
the vector paths in the
[authors' paper and appendix](https://psantanna.com/files/CGBS_AEAPP.pdf),
rather than from exact tabulated estimates. Appendix Figure B.4 repeats
Figure 2 at a larger scale and supplies its reference values. The comparison
checks every plotted point and both interval endpoints. The small dose-curve
differences reflect graphical rounding in the extracted paths. Recomputed
event-study intervals retain larger differences from the historical bootstrap
ribbons. The check also refits the pooled event study of Figure B.1 for its row.

```{code-cell} ipython3
:tags: [hide-input]

# Figure B.1 pools every positive dose into one event study.
all_group_time = did.att_gt(data, **event_spec)
all_event = paper_inference(did.aggte(all_group_time, **dynamic_spec), **bootstrap_spec)

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
rows = []
for figure, series, calculated in checks:
    published = reference.filter((pl.col("figure") == figure) & (pl.col("series") == series))
    nearest = np.abs(
        published["x"].to_numpy()[:, None] - calculated["x"].to_numpy()
    ).argmin(axis=1)
    actual = calculated[nearest]
    estimate_gap = (actual["estimate"] - published["estimate"]).abs().max()
    interval_gap = max(
        (actual["lower"] - published["lower"]).abs().max(),
        (actual["upper"] - published["upper"]).abs().max(),
    )
    assert estimate_gap < 1.5e-5
    if figure in {"Figure 2", "Figure B3"}:
        assert interval_gap < 1.5e-5
    rows.append((figure, series, published.height, estimate_gap, interval_gap))
pl.DataFrame(
    rows,
    schema=["figure", "series", "points", "max estimate gap", "max interval gap"],
    orient="row",
)
```

The checks recover the published point estimates and dose-curve intervals
within the figures' graphical precision. The larger event-study interval
differences persist despite the agreement in point estimates and the fixed
seed used to make this page reproducible.

Interpreting the larger employment gains in higher-prospectivity counties and
in later years as effects of fracking requires the assumptions about untreated
employment trends and the recorded adoption dates. The
{ref}`continuous-treatment background <background-didcont>` develops the
identification and aggregation results behind these choices.
