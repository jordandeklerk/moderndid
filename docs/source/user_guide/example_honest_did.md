---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_honest_did)=

# Sensitivity analysis for parallel trends

Between 2014 and 2019, 30 of the 46 states in this example's data expanded
Medicaid to adults with low incomes under the Affordable Care Act. An event
study of those expansions shows the share of low-income adults without children
who had health insurance jumping in the year a state expanded. That jump
measures what expansion did only if coverage in the expansion states would
otherwise have moved in parallel with coverage in the states they're compared
with. Our question is how far apart those two paths could have drifted before
the evidence that expanding Medicaid raised coverage falls away.

The usual reassurance is a pre-trend check that reads estimates near zero
before expansion as a sign that parallel trends held afterward. A check like
that can pass because it lacks the power to detect a violation large enough to
matter. Since parallel trends matters only in the years after expansion, even a
check with plenty of power can't confirm it. Rather than settle for a verdict of
pass or fail, we'll follow
[Rambachan and Roth (2023)](https://doi.org/10.1093/restud/rdad018) and use
{func}`~moderndid.honest_did` to compute confidence intervals that stay valid
when the trends depart from parallel by up to an amount you choose.

Once we're done, you'll know how sharply the trend could bend before the
interval for the effect in the year of expansion reaches zero. You'll also know
how far its yearly swings could outgrow the largest one before expansion without
that interval reaching zero. Repeating both analyses for the effect three years
after expansion shows that it tolerates far smaller violations. Checks on the
event window and the restriction then show how much each answer depends on
those choices.

```{code-cell} ipython3
:tags: [remove-cell]

import polars as pl
from plotnine import options
from prerun import stored

options.figure_size = (12, 5)
options.dpi = 100
pl.Config.set_float_precision(4)
```

## Medicaid expansion across the states

After a 2012 Supreme Court ruling left each state free to decide whether to
expand Medicaid under the Affordable Care Act, the expansions reached different
states in different years. The analysis follows low-income adults without
children because, before the expansion, most states' Medicaid programs didn't
cover them at any income. Because states chose for themselves whether and when
to expand, coverage in the expansion states need not have been on the same path
as coverage elsewhere.

The coverage shares in {func}`~moderndid.load_ehec` come from the American
Community Survey and fill one row for each of the 46 states in each year from
2008 to 2019.

```{code-cell} ipython3
import moderndid as did
import polars as pl

# States that hadn't expanded by 2019 get a 0, the value att_gt reads as never treated.
data = did.load_ehec().with_columns(expansion_year=pl.col("yexp2").fill_null(0).cast(pl.Int64))
data.head()
```

The outcome, `dins`, is the share of low-income adults without children who had
health insurance. The column `yexp2` records the year a state expanded and is
missing for states that hadn't expanded by 2019. The `W` column holds a
population weight for each state that this example doesn't use.

:::{admonition} Missing expansion years drop states
:class: warning

Because {func}`~moderndid.att_gt` drops every row with a missing value in the
columns it uses, it would throw out all 192 rows of the 16 states that hadn't
expanded by 2019 if you passed `yexp2` as `gname`.
:::

The expansion years tell us how much information supports each part of the
event study. The 2014 cohort holds 22 states, against just 3, 2, 1, and 2 in
the 2015, 2016, 2017, and 2019 cohorts. Because the data starts in 2008, the
2014 cohort is observed for six years before it expanded. Any estimate seven or
more years before expansion therefore rests only on states that expanded in 2015
or later, eight at most.

## The event study to stress-test

Since {func}`~moderndid.honest_did` works from an event study rather than
estimating effects itself, the first step is the event study you would report
anyway, built with {func}`~moderndid.att_gt` and {func}`~moderndid.aggte`.

Every estimate has to be measured from the same year for the restrictions to
make sense. With `base_period="universal"`, each year is compared with the year
before a state expanded, event time −1.

:::{admonition} Give the event study a universal base
:class: important

Under the default `base_period="varying"`, each estimate before expansion is a
change over a single year. {func}`~moderndid.honest_did` raises an error for an
event study built that way.
:::

To widen the comparison, `control_group="notyettreated"` compares each cohort
with the 16 states that hadn't expanded by 2019 and with each later cohort in the
years before that cohort expands. The price is that parallel trends has to hold
for those later cohorts as well. The staggered example's
[check of the comparison group](example_staggered_did.md#other-control-groups-and-estimators)
shows how this choice can play out.

Since each sensitivity analysis concerns a single event time,
{func}`~moderndid.honest_did` reports that estimate's pointwise interval as the
original one and never reads the event study's bands. Setting `cband=False` makes
the event study print those same pointwise intervals in place of simultaneous
bands.

Last, we keep the five years on either side of expansion with `min_e=-5` and
`max_e=5`, the furthest the data reaches after expansion. The matching years
before expansion stay clear of the earliest event times, where only the late
expanders contribute. [A longer event window](#a-longer-event-window) brings
those years back to show what they change.

```{code-cell} ipython3
# Compare every cohort with states that haven't expanded yet and measure each year from
# the year before expansion.
result = did.att_gt(
    data,
    yname="dins",
    tname="year",
    idname="stfips",
    gname="expansion_year",
    control_group="notyettreated",
    base_period="universal",
    cband=False,
)

# Line the effects up by years since expansion and keep five years on either side.
event_study = did.aggte(result, type="dynamic", min_e=-5, max_e=5)
print(event_study)
```

The four rows above the base year at event time −1 are placebo estimates, from
−0.0125 five years before expansion to −0.0049 two years before. Every one of
their intervals covers zero, although the interval at −5 only just does with an
upper end of 0.0003.

The estimate of 0.0453 in the year of expansion corresponds to a 4.53 percentage
point rise in coverage for low-income adults without children relative to the
comparison states. Since the fit leaves population weights unused, each of the
30 expansion states gets equal weight in that average. Its 95 percent interval
runs from 3.35 to 5.71 percentage points.

The event-study plot below shows the navy placebo estimates climbing toward zero
on their way to the dashed line at the base year. That climb hints that the
expansion states may already have been gaining coverage on their comparison
states. If so, part of the jump in the red effects after expansion could be that
same gain carrying on. The sensitivity analysis asks how large such a drift
would have to be before it could account for the jump.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Event study with placebo estimates at event times −5 to −2 and effects at 0 to 5
---
# The dashed line marks the base year, one year before expansion.
did.plot_event_study(event_study) + did.theme_moderndid()
```

## What the sensitivity analysis bounds

Each estimate in the event study targets a coefficient that mixes the effect of
expansion with a difference in trends,

$$
\beta_e = \tau_e + \delta_e .
$$

The first term, $\tau_e$, is the effect of expansion $e$ years after it. The
other term, $\delta_e$, is the gap between the coverage trends of the expansion
states and their comparison states that would have opened up without any
expansion. Since $\tau_e$ is zero before expansion, the placebo estimates
measure $\delta_e$ directly. Parallel trends is the assumption that $\delta_e$
stays at zero in every year after expansion.

{func}`~moderndid.honest_did` replaces that assumption with a limit on how far
$\delta_e$ after expansion can stray from what the placebo estimates show.
Under such a limit, the effect in a given year is only known to lie in an
interval called the identified set. The confidence interval that
{func}`~moderndid.honest_did` reports covers each value in that set with at
least 95 percent probability. That guarantee rests on three assumptions about
the expansions, the trends, and the estimates.

- Coverage doesn't respond to an expansion before the expansion takes effect.
- After expansion, $\delta_e$ obeys the restriction you choose, such as a cap
  on how fast its slope can change or on how large its yearly changes can grow
  relative to those before expansion.
- The event-study estimates have a nearly normal sampling distribution whose
  covariance moderndid estimates well from their influence functions.

:::{admonition} The reference year under anticipation
:class: note

If you estimate the event study with `anticipation=1`, its base year moves back
to event time −2. {func}`~moderndid.honest_did` then uses −2 as the reference
year and counts the estimate at −1 among the effects.
:::

{ref}`Honest DiD sensitivity analysis <background-didhonest>` gives each
restriction its formal definition and derives the identified sets and the
inference methods behind these intervals.

## The estimate, the restriction, and the interval

```{code-cell} ipython3
# One dictionary holds every decision, so later sections show only the entries they change.
spec = dict(
    # The effect in the year of expansion, from the event study above.
    event_study=event_study,
    event_time=0,
    # A difference in trends whose slope changes by at most M a year, for M from 0 to 0.03.
    sensitivity_type="smoothness",
    m_vec=[0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    # Fixed-length confidence intervals, the method that suits smoothness bounds.
    method="FLCI",
)
```

### The effect in the year of expansion

For the estimate to stress-test, `event_time=0` picks the effect in the year each
state expanded. No other effect after expansion draws on all 30 expansion
states. It's also where a departure from parallel trends has had the least time
to build up. [A longer event window](#a-longer-event-window) swaps in an event
study that keeps every event time.
[Three years after expansion](#three-years-after-expansion) reruns both
restrictions for the effect at `event_time=3`.

### A slope that changes by at most M a year

We start with smoothness, since it fits a worry about slow-moving differences
between the expansion states and the rest, the kind of gradual drift the placebo
estimates show. With `sensitivity_type="smoothness"`, the slope of $\delta_e$
may change by at most $M$ between consecutive years. Since $M$ is measured in
the outcome's units, $M = 0.01$ lets the slope change by one percentage point of
coverage each year.

:::{admonition} The linear trend at M = 0
:class: note

At $M = 0$ the gap can only continue along a straight line fitted to the
placebo estimates. That is the same assumption as the linear trend that applied
work often adds to a regression.
:::

The grid in `m_vec` runs from 0 to 0.03 in steps of 0.005, far enough for the
interval to reach zero. For a sense of scale, the sharpest bend among the
placebo estimates comes from −0.0038 at event time −3 through −0.0049 at −2 to 0
at the base year, a change in slope of about 0.006. Since those estimates are
noisy, treat that number as a rough guide rather than a bound. If you expect the
violation to have a known sign, [A sign on the bias](#a-sign-on-the-bias) adds
one to the restriction.
[Violations as large as before expansion](#violations-as-large-as-before-expansion)
replaces smoothness with a different restriction altogether.

:::{admonition} Report where the interval reaches zero
:class: tip

The smallest bound at which the interval includes zero is called the breakdown
value. Rambachan and Roth (2023) suggest reporting it next to the intervals to
let readers judge whether a violation that large is plausible.
:::

### Fixed-length confidence intervals

The last decision is how to turn a restriction into an interval. For
smoothness, Rambachan and Roth (2023) recommend fixed-length confidence
intervals, `method="FLCI"`, whose length is close to the shortest possible
under this restriction. The alternatives, the conditional test and its hybrids
`"C-F"` and `"C-LF"`, invert a test over a grid of candidate effects. They can
use sign and shape restrictions that a fixed-length interval ignores.
[A sign on the bias](#a-sign-on-the-bias) turns to the conditional FLCI hybrid
for that reason. Relative magnitudes rules out fixed-length intervals
altogether, since any interval of that kind would have to be infinitely wide.
{func}`~moderndid.honest_did` uses the conditional least favorable hybrid for
that restriction instead.

With all three decisions in `spec`, one call to {func}`~moderndid.honest_did`
computes an interval for each value of $M$.

```{code-cell} ipython3
# Compute a robust interval for each bound on the bend.
smoothness = did.honest_did(**spec)
smoothness.robust_ci
```

## How far the trend could bend

Each row holds the interval for one value of $M$, listed in the `m` column. At
$M = 0$ the interval runs from 0.0295 to 0.0546, below the original interval of
0.0335 to 0.0571. Continuing the straight line fitted to the placebo estimates
past the base year absorbs part of the jump, since that line keeps climbing.

On this grid, zero first enters at $M = 0.03$, where the lower bound reaches
−0.0027. The interval for the effect in the year of expansion therefore
excludes zero for every bend up to 0.025 a year, more than four times the
sharpest bend among the placebo estimates. At the smoothness bound of 0.025,
the interval still permits coverage gains as small as 0.23 percentage points.

:::{admonition} Breakdown values on a grid
:class: note

Each breakdown value lies between the largest bound whose interval excludes
zero and the next bound on the grid. The grid therefore brackets the threshold
rather than locating it exactly.
:::

The plot draws the gold original interval at the left and a navy fixed-length
interval for each value of $M$ to its right.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Original confidence interval and fixed-length intervals for M from 0 to 0.03
---
# The leftmost interval is the original one, valid only under exact parallel trends.
did.plot_sensitivity(smoothness) + did.theme_moderndid()
```

## Violations as large as before expansion

The relative magnitudes restriction bounds the same gap by tying what can happen
after expansion to the swings before it. It suits a worry about shocks that hit
the expansion states and their comparison states differently, as long as the
shocks after expansion stay within some multiple of the ones before it. With
`sensitivity_type="relative_magnitude"`, each yearly change in $\delta_e$ from
the step into the year of expansion onward can be at most $\bar{M}$ times the
largest yearly change up to the base year. In this event study that largest
change is 0.0058, from −0.0125 at event time −5 to −0.0067 at −4.

Rambachan and Roth (2023) suggest $\bar{M} = 1$ as a benchmark when the windows
before and after treatment are similar in length. With four placebo estimates
against six effects, this event study comes reasonably close. The grid in
`m_bar_vec` runs from 0.5 to 3, past the point where the interval reaches zero.

:::{admonition} A finer grid for relative magnitudes
:class: note

For relative magnitudes, {func}`~moderndid.honest_did` tests a grid of
candidate effects and reports the smallest and largest it can't reject. Since
its default of 100 candidates is too coarse to place some of these bounds on
either side of zero, the specification below passes `grid_points=1000` at the
cost of a slower run.
:::

```{code-cell} ipython3
# Cap each year's change after expansion at multiples of the largest change before it.
# Every relative magnitudes call below starts from this dictionary.
relative_spec = dict(
    event_study=event_study,
    event_time=0,
    sensitivity_type="relative_magnitude",
    m_bar_vec=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    grid_points=1000,
)
```

```{code-cell} ipython3
:tags: [skip-execution]

# The relative magnitudes intervals for the effect in the year of expansion.
relative = did.honest_did(**relative_spec)
```

```{code-cell} ipython3
:tags: [remove-cell]

relative = stored("honest_did_relative_magnitudes", lambda: did.honest_did(**relative_spec))
```

```{code-cell} ipython3
relative.robust_ci
```

With C-LF in the `method` column, the interval at the benchmark $\bar{M} = 1$
runs from 0.0234 to 0.0638, wider than the original 0.0335 to 0.0571 but well
clear of zero. It stays clear through $\bar{M} = 2$, comes within 0.0003 of zero
at $\bar{M} = 2.5$, and contains it at $\bar{M} = 3$. Under this restriction the
effect in the year of expansion holds up unless the yearly changes after
expansion could grow to about two and a half times the largest change before it.

In the plot, the red C-LF intervals widen to the right of the gold original
interval as $\bar{M}$ grows.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Original confidence interval and relative magnitudes intervals for Mbar from 0.5 to 3
---
# The gold interval at Mbar = 0 is the original one.
did.plot_sensitivity(relative) + did.theme_moderndid()
```

## Three years after expansion

Only the 27 states that expanded in 2014, 2015, or 2016 are observed three years
after expansion. Any departure from parallel trends has also had longer to build
by then.

```{code-cell} ipython3
# The same restriction applied to the effect three years after expansion.
later = did.honest_did(**(spec | {"event_time": 3}))
later.robust_ci
```

Where the effect in the year of expansion kept zero out for every bend up to
0.025, the later effect lets it in at the very next value on the grid. At
$M = 0.005$ the interval three years after expansion already stretches from
−0.0205 to 0.1387.

:::{admonition} Small bends add up over time
:class: warning

Under smoothness, a change in slope of $M$ a year keeps adding to the gap with
every year since expansion. A value of $M$ too small to matter in the year of
expansion can leave the effect three years later indistinguishable from zero.
:::

```{code-cell} ipython3
:tags: [skip-execution]

# The relative magnitudes analysis for the effect three years after expansion.
later_relative = did.honest_did(**(relative_spec | {"event_time": 3}))
```

```{code-cell} ipython3
:tags: [remove-cell]

later_relative = stored(
    "honest_did_relative_magnitudes_three_years",
    lambda: did.honest_did(**(relative_spec | {"event_time": 3})),
)
```

```{code-cell} ipython3
later_relative.robust_ci
```

Relative magnitudes also gives the later effect far less room than the effect
in the year of expansion. Its interval runs from 0.0331 to 0.1071 at
$\bar{M} = 0.5$ and keeps only 0.0013 above zero at $\bar{M} = 1$. From
$\bar{M} = 1.5$ on, where the effect in the year of expansion still had a lower
end of 0.0159, the later interval contains zero.

## Other windows and restrictions

### A longer event window

To bring the early years back, {func}`~moderndid.aggte` runs again without
`min_e` and `max_e` and keeps every event time the data reaches, back to eleven
years before expansion.

```{code-cell} ipython3
# The same event study with every event time the data reaches.
full_window = did.aggte(result, type="dynamic")

# Inspect the additional placebo estimates and their pointwise intervals.
did.to_df(full_window).filter(pl.col("event_time") < -5)
```

Three of the six new placebo estimates have intervals that exclude zero, among
them 0.0398 ten years before expansion and −0.0357 eight years before. None of
the three estimates draws on more than five of the expansion states.

```{code-cell} ipython3
# The smoothness analysis on the full event study.
full_smoothness = did.honest_did(**(spec | {"event_study": full_window}))
full_smoothness.robust_ci
```

From $M = 0.01$ on, these intervals match the ones from the five-year window to
the fourth decimal. At $M = 0$ the high early estimates tilt the straight line
fitted to all ten placebo estimates downward. Continuing that line adds to the
effect rather than subtracting from it and moves the interval up to 0.0474 to
0.0668.

```{code-cell} ipython3
:tags: [skip-execution]

# The relative magnitudes analysis on the full event study.
full_relative = did.honest_did(**(relative_spec | {"event_study": full_window}))
```

```{code-cell} ipython3
:tags: [remove-cell]

full_relative = stored(
    "honest_did_relative_magnitudes_full_window",
    lambda: did.honest_did(**(relative_spec | {"event_study": full_window})),
)
```

```{code-cell} ipython3
full_relative.robust_ci
```

Relative magnitudes reacts far more to the early years than smoothness does. The
largest change before expansion is now 0.0462, from 0.0398 at event time −10 to
−0.0064 at −9, nearly eight times the 0.0058 that set the scale on the five-year
window. Even at $\bar{M} = 0.5$ the interval stretches from 0.0015 to 0.0897,
its lower end just above zero. It contains zero from $\bar{M} = 1$ on, where the
five-year window still gave 0.0234 to 0.0638.

:::{admonition} One noisy swing sets the scale
:class: warning

Relative magnitudes takes its scale from the single largest change before
expansion. A swing in two or three states far from expansion can therefore
widen the interval for every event time.
:::

### A sign on the bias

The placebo estimates also hint at a direction for the bias. If their climb
toward the base year had continued, it would have pushed the estimates after
expansion upward. Rambachan and Roth (2023) formalize a trend that carries on as
a gap that keeps increasing. Setting `bias_direction="positive"` instead of
`monotonicity_direction="increasing"` keeps only what that implies after
expansion, $\delta_e \ge 0$. Because a positive bias means the estimates can
only overstate the effect, the sign can only cap the effect from above. Since a
fixed-length interval ignores sign restrictions, the cell below runs the
conditional FLCI hybrid twice and adds the sign to only one of the runs.

```{code-cell} ipython3
:tags: [skip-execution]

# The conditional FLCI hybrid with and without a positive bias.
hybrid = did.honest_did(**(spec | {"method": "C-F"}))
positive = did.honest_did(**(spec | {"method": "C-F", "bias_direction": "positive"}))
```

```{code-cell} ipython3
:tags: [remove-cell]

hybrid = stored("honest_did_hybrid", lambda: did.honest_did(**(spec | {"method": "C-F"})))
positive = stored(
    "honest_did_hybrid_positive_bias",
    lambda: did.honest_did(**(spec | {"method": "C-F", "bias_direction": "positive"})),
)
```

```{code-cell} ipython3
# The last two columns hold the hybrid with a positive bias.
hybrid.robust_ci.select("m", "lb", "ub").join(
    positive.robust_ci.select("m", "lb", "ub"), on="m", suffix="_positive"
)
```

Without the sign, the hybrid's intervals come within 0.0005 of the
fixed-length ones from $M = 0.01$ up. Adding the sign leaves every one of the
lower bounds exactly where it was. Instead the sign pulls down the upper ends,
from 0.0833 to 0.0554 at $M = 0.03$. At $M = 0$ the restricted interval even
reaches higher, 0.0599 against 0.0558. Intervals built by inverting two
different tests need not nest inside each other.

:::{admonition} Ground a sign in the setting
:class: tip

Rambachan and Roth (2023) motivate a sign with knowledge of the setting, such
as another policy known to push the outcome one way. A climb across four noisy
placebo estimates is weaker ground for one.
:::

Under smoothness, the longer window, the hybrid, and the hybrid with a sign all
keep zero out of the interval for the effect in the year of expansion up to
$M = 0.025$, as the main specification does. Under relative magnitudes the answer
hinges on the event window instead, since the interval excludes zero up to
$\bar{M} = 2.5$ on the five-year window but only to $\bar{M} = 0.5$ with every
event time.

(example_honest_did_external)=

## Event studies from outside moderndid

If your event study comes from a regression or another estimator rather than
from {func}`~moderndid.aggte`, the functions behind
{func}`~moderndid.honest_did` take the event-study coefficients and their
covariance matrix directly. To try them, we'll estimate a simpler design with a
single expansion year in an event-study regression with state and year fixed
effects. It compares the 22 states that expanded in 2014 with the 21 that hadn't
expanded by 2015.

```{code-cell} ipython3
import pyfixest as pf

# Keep the years through 2015 and drop the three states that expanded in 2015.
kept = (pl.col("year") <= 2015) & (pl.col("expansion_year") != 2015)
two_groups = data.filter(kept).with_columns(
    expanded_2014=(pl.col("expansion_year") == 2014).cast(pl.Int64)
)

# An event-study regression with state and year fixed effects, 2013 as the reference year,
# and standard errors clustered by state.
regression = pf.feols(
    "dins ~ i(year, expanded_2014, ref=2013) | stfips + year",
    data=two_groups.to_pandas(),
    vcov={"CRV1": "stfips"},
)

# The seven coefficients in time order without 2013, and their covariance matrix.
# The fitted regression keeps that matrix only in its private _vcov attribute.
betahat = regression.coef().to_numpy()
sigma = regression._vcov
regression.coef().round(4)
```

The first five coefficients are placebo estimates for 2008 through 2012, all
within 0.0113 of zero. The last two are the effects in 2014 and 2015, 0.0464
and 0.0692.

:::{admonition} Leave the reference year out of betahat
:class: important

The functions read the first `num_pre_periods` entries of `betahat` as placebo
estimates and the rest as effects. A coefficient for the reference year, or
coefficients out of time order, would put estimates in the wrong roles.
:::

From the same coefficients, {func}`~moderndid.construct_original_cs` builds the
interval that assumes exact parallel trends. The smoothness intervals for the
same grid of $M$ as before come from
{func}`~moderndid.create_sensitivity_results_sm`.

```{code-cell} ipython3
# The original interval and the smoothness intervals for the effect in 2014.
original = did.construct_original_cs(betahat, sigma, num_pre_periods=5, num_post_periods=2)
print(f"original interval [{original.lb:.4f}, {original.ub:.4f}]")

regression_smoothness = did.create_sensitivity_results_sm(
    betahat,
    sigma,
    num_pre_periods=5,
    num_post_periods=2,
    m_vec=[0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
)
regression_smoothness
```

The regression's original interval for the effect in 2014 runs from 0.0285 to
0.0644. Its smoothness intervals keep zero out up to $M = 0.02$, where the lower
bound is 0.0028, and let it in from $M = 0.025$ on. All three functions target
the first effect after expansion unless you pass weights in `l_vec`, such as
`np.array([0.5, 0.5])` for the average of the 2014 and 2015 effects.

For relative magnitudes, {func}`~moderndid.create_sensitivity_results_rm` takes
the same arguments with `m_bar_vec` in place of `m_vec` and tests 1000 candidate
effects by default.

```{code-cell} ipython3
:tags: [skip-execution]

# The relative magnitudes intervals for the regression's effect in 2014.
regression_relative = did.create_sensitivity_results_rm(
    betahat,
    sigma,
    num_pre_periods=5,
    num_post_periods=2,
    m_bar_vec=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
)
```

```{code-cell} ipython3
:tags: [remove-cell]

regression_relative = stored(
    "honest_did_regression_relative_magnitudes",
    lambda: did.create_sensitivity_results_rm(
        betahat,
        sigma,
        num_pre_periods=5,
        num_post_periods=2,
        m_bar_vec=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ),
)
```

```{code-cell} ipython3
regression_relative
```

Under relative magnitudes the regression's interval stays above zero through
$\bar{M} = 1.5$, where it runs from 0.0086 to 0.0796, before zero enters at
$\bar{M} = 2$. The event study's effect in the year of expansion held up longer
under both restrictions, through $M = 0.025$ and $\bar{M} = 2.5$.
{ref}`Honest DiD sensitivity analysis <background-didhonest>` defines both
restrictions formally and derives the identified sets behind every interval on
this page.
