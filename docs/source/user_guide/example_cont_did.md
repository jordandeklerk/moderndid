---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_cont_did)=

# Continuous difference-in-differences

In this simulated panel, the treatment comes in doses between 0 and 1 rather
than as an on-off switch. Of its 2,000 units, 1,516 start treatment in period 2,
3, or 4, each at a dose of its own. We'll use the panel to see how closely
{func}`~moderndid.cont_did` recovers the dose-response the simulation planted.
That covers both the effect of each dose compared with no treatment and how
much more a slightly larger dose would do.

Applied work typically regresses the outcome on the dose along with unit and
period fixed effects. Its one coefficient blends the marginal effects of the
dose with selection bias and puts most of its weight on doses near the average,
as the {ref}`background page <background-didcont>` explains.

Our route is to estimate each cohort's dose-response in each period against
units that haven't started treatment yet and to average those curves into one.
You'll see the averaged curve and its slope inside bands that hold at every
dose together, two overall effects that sum them up, and event studies of both.
Each of them is then held up against the truth before a final round of checks
tries other comparison units and other splines.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## A simulated dose-response

To build a panel for this question, {func}`~moderndid.gen_cont_did_data` places
every unit in a cohort that starts treatment in period 2, 3, or 4 or never
starts it at all. Each treated unit gets a dose drawn uniformly between 0 and 1.
Once treatment starts, a unit at dose $d$ gains $0.5d + 0.3d^2$ in every
remaining period. Each unit also carries a fixed effect that grows with its
cohort. Later cohorts therefore sit at higher outcome levels even before anyone
is treated.

```{code-cell} ipython3
import moderndid as did
import numpy as np
import polars as pl

# Simulate 2,000 units over four periods, where a unit at dose d gains 0.5d + 0.3d^2 once treated.
data = did.gen_cont_did_data(
    n=2000,
    num_time_periods=4,
    dose_linear_effect=0.5,
    dose_quadratic_effect=0.3,
    seed=1234,
)
data.head()
```

Each row holds one unit in one period, along with its outcome `Y` and its dose
`D`. The `G` column gives the period in which the unit's treatment starts and
reads 0 for units that are never treated. A treated unit carries the same dose
in every period, even before its treatment begins.

:::{admonition} Give each unit one dose
:class: important

{func}`~moderndid.cont_did` raises an error unless `D` holds the same dose in
every period or a 0 until treatment starts. It sets the dose of never-treated
units to 0 on its own. A `G` of 0 or of a period after the last one marks a
unit that is never treated. Any other value of `G` must be an observed period.
:::

The cell below describes each cohort by its size, its average dose, and its
average outcome in period 1.

```{code-cell} ipython3
# Describe each cohort from the units' rows in the first period.
units = data.filter(pl.col("time_period") == 1)
units.group_by("G").agg(
    units=pl.len(),
    mean_dose=pl.col("D").mean().round(1),
    mean_outcome_period_1=pl.col("Y").mean().round(1),
).sort("G")
```

The 1,516 treated units split into cohorts of 490, 534, and 492 that start in
periods 2, 3, and 4, next to 484 units that are never treated. Every treated
cohort's doses average 0.5, as uniform doses between 0 and 1 should. In period
1, before anyone is treated, the never-treated units average an outcome of 1.0
against 3.0, 4.0, and 5.0 for cohorts 2, 3, and 4. A comparison of outcome
levels would mistake those gaps for treatment effects. Since the simulation
holds those gaps fixed over time, a comparison of changes in the outcome takes
them out.

## Level effects and causal responses

With a continuous treatment there are two effects to estimate at every dose.
The level effect compares the outcome a unit at dose $d$ has once treatment
starts with the outcome it would have had without treatment. Averaged over the
units that actually received dose $d$, it becomes the average treatment effect
on the treated at that dose,

$$
ATT(d \mid d) = \mathbb{E}\big[Y_t(d) - Y_t(0) \mid D = d\big].
$$

The average causal response on the treated, $ACRT(d \mid d)$, measures how much
more the outcome of the units at dose $d$ would rise if they got a slightly
larger dose. {func}`~moderndid.cont_did` estimates it from the slope of the
level curve. Unless the last assumption below holds, that slope also picks up
selection bias. Averaging each curve over the doses of the treated units gives
the two summaries that the estimator reports, $ATT^o$ and $ACRT^o$. Because
treatment starts at different times, each cohort gets its own pair of curves in
each period after it starts.

:::{admonition} How the cohorts are weighted
:class: note

The dose aggregation gives each cohort its share of the treated units and
spreads that share evenly over the cohort's treated periods. Cohort 2, treated
in periods 2 to 4, therefore enters through three periods at a third of its
weight each.
:::

Since you never see $Y_t(0)$ for a treated unit after its treatment starts, the
comparison units have to stand in for it under four assumptions.

- Each cohort's treatment starts after period 1 and stays on at a fixed dose.
- Outcomes don't react to the treatment in the periods before it starts.
- Without treatment, units at every dose would have followed the same average
  path as the comparison units. This parallel trends assumption identifies
  $ATT(d \mid d)$ and $ATT^o$.
- Had every unit received a given dose, the average path of all units would
  match the one that units at that dose actually followed. This strong parallel
  trends assumption rules out selection on gains and is what lets you read the
  slope, or a gap between two doses, as a causal response.

{ref}`DiD with continuous treatments <background-didcont>` writes out each
assumption in full and shows why parallel trends alone leaves the slope mixed
up with selection bias. Both the estimator and these assumptions come from
[Callaway, Goodman-Bacon, and Sant'Anna (2024)](https://arxiv.org/abs/2107.02637v4).

:::{admonition} Random doses satisfy both trend assumptions
:class: note

Since the simulation draws each treated unit's dose at random, independently of
its cohort and its unit effect, treated units at every dose share the same
potential outcomes on average. Every unit's untreated outcome also follows the
same trend over time.
:::

## Choices behind the estimates

The estimate rests on the answers to four questions, about which units stand in
for untreated outcomes, where each change in the outcome starts, how freely the
curve may bend, and whether its band should cover every dose at once. After the
column names, the dictionary below answers each question in a block of arguments
with its own comment. The checks in
[Back to the comparison units and the spline](#back-to-the-comparison-units-and-the-spline)
later test how much those two choices matter.

```{code-cell} ipython3
# The full specification. Each check below starts from a copy of it.
spec = dict(
    # The outcome, period, unit, starting period, and dose columns.
    yname="Y",
    tname="time_period",
    idname="id",
    gname="G",
    dname="D",
    # Compare each cohort with units that haven't started treatment yet.
    control_group="notyettreated",
    # Measure every period from the last one before the cohort starts and allow no anticipation.
    base_period="universal",
    anticipation=0,
    # Fit a cubic in the dose and average the cohorts' curves by dose.
    aggregation="dose",
    dose_est_method="parametric",
    degree=3,
    num_knots=0,
    # Bands over all doses from 10,000 bootstrap draws, seeded so the numbers reproduce.
    cband=True,
    biters=10000,
    random_state=7,
)
```

### Comparison units that haven't started treatment

Untreated outcomes for a cohort have to come from units that aren't treated
yet. Under `control_group="notyettreated"`, those are the never-treated units
and every cohort whose treatment starts after both periods of a comparison.
Borrowing the later cohorts adds comparisons in the early periods, before
cohorts 3 and 4 start. It also asks those cohorts, until they start, to have
trended the way the treated cohorts would have without treatment.
[Never-treated units alone](#never-treated-units-alone) drops them and keeps
only the 484 never-treated units.

:::{admonition} Comparison units and the slope
:class: note

Since the slope of each cohort's curve comes from its treated units alone, the
comparison units shift the level curve and the overall ATT but leave the slope
untouched.
:::

### Measuring from the last untreated period

Every comparison is a change in the outcome and needs a period to start from.
With `base_period="universal"`, every period gets measured from $g - 1$, the
last one before the cohort starts. Placebo estimates and effects in the event
studies then share one scale on which the period just before treatment reads
zero by construction. Under the default `"varying"` base, each placebo estimate
would cover a single period instead. Because periods after treatment starts use
$g - 1$ under both settings, the dose curves come out the same either way, as
the [base period check](example_staggered_did.md#the-base-period) of the
staggered example shows for a binary treatment.

Measuring from $g - 1$ also assumes outcomes didn't move ahead of treatment.
Setting `anticipation=0` records that the simulation plants no such early
reaction.

:::{admonition} Allow anticipation when outcomes move early
:class: tip

If outcomes in your data may react before treatment starts, `anticipation=1`
moves each cohort's base back one more observed period, to $g - 2$ on this
panel. What that does to the estimates appears in the
[anticipation check](example_staggered_did.md#anticipation) of the staggered
example.
:::

### A cubic in the dose

How freely the curve can bend across the doses depends on the spline behind it.
With `dose_est_method="parametric"`, each cohort's curve in each period comes
from a B-spline regression in the dose. With the defaults we keep, `degree=3`
and `num_knots=0`, that spline is a single cubic across the range of treated
doses. Setting `aggregation="dose"` then averages the cohorts' level and slope
curves and reports $ATT^o$ and $ACRT^o$ alongside them.
[Effects by periods since treatment began](#effects-by-periods-since-treatment-began)
later swaps in `aggregation="eventstudy"` to line the effects up by event time.
Both curves get evaluated at 50 evenly spaced doses between the smallest and
largest treated dose unless `dvals` names others.

A cubic can bend either way and includes straight lines and parabolas as
special cases. Each interior knot, placed at a quantile of the treated doses,
lets the curve change shape there at the cost of a noisier fit. A lower degree
buys precision by imposing a shape on the curve.

:::{admonition} A straight line hides a bending slope
:class: warning

With `degree=1` and no knots, the slope comes out the same at every dose
whatever the data say. Any bend in the true dose-response then disappears into
a single average slope.
:::

Later on, [A quadratic or an extra knot](#a-quadratic-or-an-extra-knot) tries a
simpler spline and a more flexible one.
[Letting the data pick the spline](#letting-the-data-pick-the-spline) hands the
choice over to a data-driven estimator.

### Bands that cover every dose

To judge the shape of a curve rather than its value at one dose, you need a band
that holds at every dose together. With `cband=True`, each band covers its whole
curve with 95 percent probability. Its width comes from the 95th percentile of
each bootstrap draw's largest standardized deviation across the 50 doses. We
raise `biters` to 10,000, ten times the default, to make that percentile settle
down. The default `cband=False` would instead give each dose a narrower
pointwise interval. That interval suits only a single dose chosen in advance, as
the note on simultaneous bands in the
[staggered example](example_staggered_did.md#group-time-effects) explains.

:::{admonition} Seed every call
:class: tip

{func}`~moderndid.cont_did` draws bootstrap samples on every call, whatever
the `boot` argument says. Its standard errors and bands therefore shift a little
from one run to the next unless `random_state` fixes the draws.
:::

One call to {func}`~moderndid.cont_did` with `**spec` estimates every cohort's
curve in every period and averages the curves by dose.

```{code-cell} ipython3
# Estimate each cohort's curve in each period and average the curves by dose.
result = did.cont_did(data, **spec)
print(result)
```

## Two numbers that sum up the curve

The report opens with the two summaries of the curve, the overall ATT and the
overall ACRT. At 0.3824, the overall ATT averages the level effect over the
doses of all 1,516 treated units. Its 95 percent interval, from 0.2853 to
0.4794, sits well above zero.

The overall ACRT of 0.7172 averages the slope of the curve over the same doses.
Since a slope is harder to pin down than a level, its standard error of 0.1979
is 28 percent of the estimate, against 13 percent for the overall ATT's 0.0495.
Even so, its interval from 0.3293 to 1.1051 stays clear of zero.

:::{admonition} The overall ATT ignores the spline
:class: note

Averaged over every treated dose, the level effect reduces to the gap between
the average outcome change of treated units and that of comparison units. The
overall ATT is therefore a binary comparison that no choice of spline can move.
:::

## The curve and its slope

The two summaries hide how the effect changes from one dose to the next.
{func}`~moderndid.plots.plot_dose_response` draws the level curve along with its
band over all 50 doses. Keep an eye on the width of the band as much as on the
curve itself.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Estimated ATT(d) curve rising with the dose inside a band that widens at both ends
---
# The level curve with its band over all 50 doses.
did.plot_dose_response(result) + did.theme_moderndid()
```

Passing `effect_type="acrt"` to the same function draws the slope of the curve
instead.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Estimated ACRT(d) curve with a band that balloons at both ends of the dose range
---
# The slope of the curve with its band.
did.plot_dose_response(result, effect_type="acrt") + did.theme_moderndid()
```

To put numbers on what the two plots show, we read both curves and their bands
at the smallest dose, one in the middle, and the largest.

```{code-cell} ipython3
# Each curve and its band at the smallest, a middle, and the largest dose on the grid.
middle = len(result.dose) // 2
for name, effect, se, crit in [
    ("ATT(d)", result.att_d, result.att_d_se, result.att_d_crit_val),
    ("ACRT(d)", result.acrt_d, result.acrt_d_se, result.acrt_d_crit_val),
]:
    for i in [0, middle, -1]:
        low = effect[i] - crit * se[i]
        high = effect[i] + crit * se[i]
        print(
            f"{name:<8} dose {result.dose[i]:.2f}  estimate {effect[i]:7.4f}  "
            f"band [{low:7.4f}, {high:7.4f}]  width {high - low:.3f}"
        )
```

The level curve climbs from 0.1223 at the smallest dose to 0.8107 at the
largest. Its band is 0.336 wide at a dose of 0.51 but more than twice that at
either end, 0.727 at the bottom and 0.707 at the top. Since a polynomial fit
has data on one side only at the edges of the dose range, it's least precise
there. With doses spread evenly between 0 and 1, a shortage of units near
either end can't explain the widening.

The slope comes out far noisier than the level at all three doses. At a dose
of 0.51 it reaches 0.8901 inside a band from 0.1357 to 1.6446. At the two ends
of the range the band grows to 5.879 and 5.851 across, enough room for almost
any shape. The band leaves the shape of the slope readable only over the middle
of the dose range, where it is far narrower.

## Effects by periods since treatment began

Since the dose curves average over time, they can't show how the effect evolves
once treatment starts. Setting `aggregation="eventstudy"` reorganizes the
estimates by event time instead, the number of periods since a cohort's
treatment started. Because the event study of level effects treats every unit in
a cohort as treated whatever its dose, no spline enters it.

```{code-cell} ipython3
# The same specification with the effects lined up by periods since treatment began.
event_study = did.cont_did(data, **(spec | {"aggregation": "eventstudy"}))
print(event_study)
```

Event time −1 reads 0.0000 and NA because every cohort is measured from the
period just before its treatment. The two placebo rows above it, −0.0433 and
−0.0014, test whether treated and comparison units trended alike before
treatment. Each of their bands covers zero, as it should when the trends match.
Once treatment begins, the effects of 0.3776, 0.3800, and 0.4211 barely change
from one period to the next, as you'd expect from an effect the simulation holds
constant over time.

The summary above the table, 0.3929, gives the three event times equal
weight. Because event time 2 rests on cohort 2 alone and carries the largest
estimate, it pulls this average above the overall ATT of 0.3824 from the dose
report, where cohorts count by their size.

In the plot, the dashed line marks the base period at −1. The navy placebo
estimates sit near zero and the red effects sit well above it.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Event study with placebo estimates near zero at -3 and -2 and effects near 0.4 at 0 to 2
---
# Placebo estimates before treatment in navy and effects after it in red.
did.plot_event_study(event_study) + did.theme_moderndid()
```

### The slope over time

Lining up the slope by event time shows whether a larger dose does more in some
periods than in others. With `target_parameter="slope"`, the event study
averages the slope of each cohort's dose curve instead of its level. Before
treatment starts, those placebo slopes should hover near zero as well.

```{code-cell} ipython3
# The average slope of the curve by periods since treatment began.
slope_study = did.cont_did(
    data, **(spec | {"aggregation": "eventstudy", "target_parameter": "slope"})
)
print(slope_study)
```

Both placebo slopes, 0.2447 and −0.2865, have bands that cover zero. After
treatment, the slopes of 0.7720, 0.2518, and 0.7017 swing far more than the
levels did. Only the band at event time 0, from 0.2115 to 1.3326, excludes
zero. The event-study average of 0.5752 sits below the dose report's 0.7172
because the two weight the cohorts' periods differently. In the dose report,
cohort 4's lone treated period carries the whole of that cohort's share. The
event study instead folds that period into event time 0 alongside the other two
cohorts' first treated periods.

:::{admonition} Placebo slopes can't test strong parallel trends
:class: warning

Since every outcome observed before treatment is an untreated one, parallel
trends and strong parallel trends predict the same zero slope there. Flat
placebo slopes therefore speak to parallel trends and say nothing about
selection on gains.
:::

## How close the estimates come to the truth

Because the simulation planted the effects, every estimate so far can be
checked against the truth. Averaging the planted curve $0.5d + 0.3d^2$ and its
slope $0.5 + 0.6d$ over doses spread evenly between 0 and 1 gives
$ATT^o = 0.35$ and $ACRT^o = 0.80$. Both 95 percent intervals from the
[two summaries](#two-numbers-that-sum-up-the-curve) cover these values. The
overall ATT sits 0.0324 above 0.35 and the overall ACRT 0.0828 below 0.80. The
lines below count the doses where each band from
[the curve and its slope](#the-curve-and-its-slope) covers the true curve. A
figure then draws the planted level curve in red over the estimated one.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: Estimated ATT(d) curve and its band with the planted curve drawn in red inside the band
---
from plotnine import aes, geom_line

# The planted curve and its slope at each dose on the grid.
doses = result.dose
true_att = 0.5 * doses + 0.3 * doses**2
true_acrt = 0.5 + 0.6 * doses

# How many doses each band covers, and where each curve strays furthest from the truth.
for name, estimate, se, crit, truth in [
    ("ATT(d)", result.att_d, result.att_d_se, result.att_d_crit_val, true_att),
    ("ACRT(d)", result.acrt_d, result.acrt_d_se, result.acrt_d_crit_val, true_acrt),
]:
    covered = np.sum(np.abs(estimate - truth) <= crit * se)
    gap = np.abs(estimate - truth)
    print(
        f"{name:<8} band covers the true curve at {covered} of {len(doses)} doses; "
        f"largest gap {gap.max():.4f} at dose {doses[np.argmax(gap)]:.2f}"
    )

# The level curve and its band with the planted curve in red.
truth_curve = pl.DataFrame({"dose": doses, "truth": true_att})
(
    did.plot_dose_response(result, title="Estimated and planted ATT(d)")
    + geom_line(aes(x="dose", y="truth"), data=truth_curve, color="#c0392b", size=1)
    + did.theme_moderndid()
)
```

Each band covers the true curve at all 50 doses of the grid. In the figure, the
red planted curve parts most from the estimate at the smallest doses, where the
estimate starts 0.1221 above it. The
[event studies](#effects-by-periods-since-treatment-began) pass the same test,
since every band after treatment covers the planted 0.35 for the level and 0.80
for the slope and every placebo band covers the zero effect planted before
treatment.

## Back to the comparison units and the spline

To find out which choices move each of the two summaries, the checks below
change the comparison units and then the shape of the curve, one argument of
`spec` at a time.

### Never-treated units alone

Dropping the later cohorts from the comparison leaves the 484 never-treated
units as the only controls.

```{code-cell} ipython3
# The same specification with only never-treated units as controls.
never = did.cont_did(data, **(spec | {"control_group": "nevertreated"}))
print(
    f"ATT^o {never.overall_att:.4f} ({never.overall_att_se:.4f}), "
    f"ACRT^o {never.overall_acrt:.4f} ({never.overall_acrt_se:.4f})"
)
```

Without the later cohorts, the overall ATT moves from 0.3824 to 0.3871. The
overall ACRT and its standard error don't move at all, as the note on
comparison units predicted.

### A quadratic or an extra knot

The next check turns from the comparison units to the shape of the curve. The
cell below tries a quadratic, the shape the simulation planted. It also keeps
the cubic but adds a knot at the median treated dose.

```{code-cell} ipython3
# Refit with a quadratic and, separately, with a knot at the median treated dose.
changes = {
    "quadratic": {"degree": 2},
    "one knot": {"num_knots": 1},
}
splines = {"cubic": result}
for name, change in changes.items():
    splines[name] = did.cont_did(data, **(spec | change))

# Both summaries and the average width of the slope's band under each spline.
for name, fit in splines.items():
    width = np.mean(2 * fit.acrt_d_crit_val * fit.acrt_d_se)
    print(
        f"{name:>9}  ATT^o {fit.overall_att:.4f}  ACRT^o {fit.overall_acrt:.4f} "
        f"({fit.overall_acrt_se:.4f})  band width {width:.3f}"
    )
```

The overall ATT stays at 0.3824 under all three splines, since it reduces to a
binary comparison. The quadratic cuts the overall ACRT's standard error from
0.1979 to 0.1181 and the slope's band from 2.335 wide on average to 1.253.
Adding the knot pushes that band the other way, to 3.581 wide on average, and
moves the overall ACRT to 0.7039.

:::{admonition} Don't choose the spline from its results
:class: danger

The quadratic does best here only because the simulation planted one. Picking
whichever spline gives the tightest band leaves the reported band too narrow,
since it ignores the search. Fix the spline before looking or let the data
choose it as in the next check.
:::

### Letting the data pick the spline

The last check leaves the choice of spline to the data. Setting
`dose_est_method="cck"` switches to the data-driven sieve estimator of
[Chen, Christensen, and Kankanala (2024)](https://doi.org/10.1093/restud/rdae025).
It picks the number of spline segments from the data and widens its band to
allow for that choice. The cell below builds a two-period panel from cohort 2
and the never-treated units. Each unit's outcome after treatment becomes its
average over periods 2 to 4, as Callaway, Goodman-Bacon, and Sant'Anna (2024) do
in their application.

:::{admonition} Give CCK two periods and one cohort
:class: important

With more periods or cohorts, {func}`~moderndid.cont_did` raises an error
under `dose_est_method="cck"`. Event studies also stay with the parametric
spline, since CCK supports only `aggregation="dose"`.
:::

```{code-cell} ipython3
# Keep cohort 2 and the never-treated units and average each unit's outcome over periods 2 to 4.
two_periods = (
    data.filter(pl.col("G").is_in([0, 2]))
    .with_columns(pl.when(pl.col("time_period") == 1).then(1).otherwise(2).alias("time_period"))
    .group_by("id", "G", "D", "time_period")
    .agg(pl.col("Y").mean())
    .sort("id", "time_period")
)
two_periods.head(4)
```

On this panel of 974 units, the CCK call changes a single entry of `spec`.

```{code-cell} ipython3
# Let the data choose the spline on the two-period panel.
cck = did.cont_did(two_periods, **(spec | {"dose_est_method": "cck"}))
print(cck)
```

The CCK fit's overall ATT of 0.3936 again comes from a binary comparison, now of
cohort 2 alone. With only 490 treated units instead of 1,516, the overall ACRT's
interval is wide enough to cover zero, from −0.0763 to 1.2183.

To see what the data-driven choice costs, the cell below fits our cubic to the
same panel and compares the two bands.

```{code-cell} ipython3
# Our cubic on the same two-period panel.
cubic = did.cont_did(two_periods, **spec)

# The critical value and the average width of each band.
for name, fit in [("cubic", cubic), ("CCK", cck)]:
    width = np.mean(2 * fit.att_d_crit_val * fit.att_d_se)
    print(f"{name:>5}  critical value {fit.att_d_crit_val:.3f}  average band width {width:.3f}")

# How far apart the two curves ever get.
gap = np.max(np.abs(cck.att_d - cubic.att_d))
print(f"largest gap between the two curves {gap:.4f}")
```

Since the largest gap between the two curves rounds to 0.0000, the data-driven
estimator evidently settled on a plain cubic too. Its band still comes out
wider, 1.046 on average compared with the cubic's 0.631. The extra width comes
from a critical value of 4.445 in place of 2.664. The larger value allows for
several candidate numbers of segments and adds a term that guards against the
bias of the one the data chose. That is the price of not fixing the spline in
advance.

### Every check against the planted effects

With the planted values in its first row, the table below shows how close each
check's two summaries come to the truth and whether their 95 percent intervals
cover it.

```{code-cell} ipython3
:tags: [hide-input]

from scipy.stats import norm

# Each check's two summaries and 95 percent intervals, under the values the simulation planted.
checks = {
    "our specification": result,
    "never-treated controls": never,
    "quadratic": splines["quadratic"],
    "one knot": splines["one knot"],
    "cubic, two periods": cubic,
    "CCK, two periods": cck,
}
z = norm.ppf(0.975)

print(f"{'check':<24}{'ATT^o':>8}   {'[95% Conf. Int.]':<18}  {'ACRT^o':>8}   [95% Conf. Int.]")
print(f"{'planted effects':<24}{0.35:>8.4f}{'':23}{0.80:>8.4f}")
for name, fit in checks.items():
    att_low = fit.overall_att - z * fit.overall_att_se
    att_high = fit.overall_att + z * fit.overall_att_se
    acrt_low = fit.overall_acrt - z * fit.overall_acrt_se
    acrt_high = fit.overall_acrt + z * fit.overall_acrt_se
    print(
        f"{name:<24}{fit.overall_att:>8.4f}   [{att_low:7.4f}, {att_high:7.4f}]"
        f"  {fit.overall_acrt:>8.4f}   [{acrt_low:7.4f}, {acrt_high:7.4f}]"
    )
```

Every interval in the table, the two-period ones included, covers its planted
value. Changing the comparison group moves only the overall ATT, by 0.0047.
Across the three spline shapes, only the overall ACRT moves, from 0.7039 to
0.7724. Resting on cohort 2's 490 treated units alone, the two rows from the
two-period panel have ACRT intervals that reach below zero.

In real data, where units seldom receive their doses at random, the overall ATT
and the level curve still rest only on parallel trends. Reading the slope as a
causal response takes an argument for strong parallel trends that no placebo
estimate can supply. {ref}`Nonparametric instrumental variables <example_npiv>`
covers the sieve estimator behind the CCK option in more depth. The
{ref}`staggered example <example_staggered_did>` works through the binary case,
where every treated unit gets the same dose.
