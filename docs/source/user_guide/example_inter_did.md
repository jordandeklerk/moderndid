---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_inter_did)=

# Intertemporal treatment effects

The treatment in this example can change more than once and by different
amounts, because US states deregulated banking at their own pace. After a 1994
federal law allowed banks to operate across state lines, each state still kept
four restrictions on banks from other states and chose when to lift them.
Between 1995 and 2001, 42 states lifted at least one of them at different dates
and in steps of one to four restrictions. Eight states never lifted any of the
four in the years the data cover. We want to know whether lifting those
restrictions sped up mortgage lending by banks in the counties of the states
that acted and whether the boost lasted or wore off within a few years.

[Favara and Imbs (2015)](https://doi.org/10.1257/aer.20121416) studied the same
counties with local projections, regressions of loan growth in each of the
following years on the number of restrictions lifted so far. Because their
coefficients shrank after a few years, they concluded that the effect on credit
supply was short-lived. When a treatment rises in steps at different dates,
[de Chaisemartin and D'Haultfœuille (2024)](https://doi.org/10.1162/rest_a_01414)
show that those coefficients can shrink even if the effect holds steady.

{func}`~moderndid.did_multiplegt` gets around that problem by comparing each
county whose state lifted a restriction only with counties whose states hadn't
lifted any yet. You'll follow those comparisons from three placebo years before
deregulation through its first eight years, read them per restriction lifted,
and set them beside the local projections on the same counties. At the end,
separate checks vary the controls, the weights, and the standard errors to see
which of them the answer depends on.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options
from prerun import stored

options.figure_size = (12, 5)
options.dpi = 100
```

## The branching deregulations

The Interstate Banking and Branching Efficiency Act let each state keep those
four restrictions. A state could require its agreement before banks from other
states opened new branches, set a minimum age for a bank they bought in a
merger, forbid buying single branches without the whole bank, and cap the share
of the state's deposits that one bank controlled.

Every state starts this panel at zero, since no state had lifted any of the
restrictions as of 1994. With steps that differed across states in timing and
in size, the panel works as a natural experiment. Since the restrictions applied
to banks, the outcome is the growth of mortgage lending by banks. Because it is
already a growth rate, an effect that persists means the volume of loans keeps
pulling away from where it would have been.

{func}`~moderndid.load_favara_imbs` loads the county panel that Favara and Imbs
(2015) assembled. Each of its rows holds one county in one year between 1994 and
2005.

```{code-cell} ipython3
import moderndid as did
import polars as pl

# Load the panel and report how many counties and states it covers and over which years.
data = did.load_favara_imbs()
years = data["year"]
print(
    f"{data['county'].n_unique()} counties in {data['state_n'].n_unique()} states, "
    f"observed from {years.min()} to {years.max()}"
)
data.head()
```

The outcome, `Dl_vloans_b`, is the change in the log volume of mortgage loans
that banks originated in the county that year. The treatment, `inter_bra`,
counts how many of the four restrictions the county's state had lifted by then,
from 0 to 4. Each county's state appears as a numeric code in `state_n`, the
column the standard errors cluster on. For one of the checks, `w1` weights each
county by roughly the inverse of the number of counties in its state, the weight
Favara and Imbs use for house prices. Last, `Dl_hpi` holds the growth of house
prices, the second outcome of de Chaisemartin and D'Haultfœuille (2024).

:::{admonition} Pass the number lifted as dname
:class: important

{func}`~moderndid.did_multiplegt` reads each county's treatment in each year
from `dname`, finds the year of its first change, and compares only counties
whose treatment matched in their first year. Since an adoption year or a flag
for having ever deregulated never changes within a county, either one stops the
call with an error that no unit changes treatment. A yearly 0/1 flag would keep
the same effects but hide the size of each step and every later change from the
per-restriction effects and the average total effect.
:::

Since the estimator dates every county by its state's first change, the first
table shows when those first changes came and how many restrictions each one
lifted.

```{code-cell} ipython3
# A state sets the restrictions for all of its counties, so one row per state and year is enough.
state_years = (
    data.group_by("state_n", "year")
    .agg(pl.col("inter_bra").first())
    .sort("state_n", "year")
    .with_columns(step=pl.col("inter_bra").diff().over("state_n"))
)

# Note each state's first change, the size of that first step, how often the state changed, and
# where it ended up in 2005.
changed = pl.col("step") != 0
states = state_years.group_by("state_n").agg(
    first_change=pl.col("year").filter(changed).first(),
    first_step=pl.col("step").filter(changed).first(),
    changes=changed.sum(),
    falls=(pl.col("step") < 0).any(),
    final=pl.col("inter_bra").last(),
)
counties = data.group_by("state_n").agg(counties=pl.col("county").n_unique())
states = states.join(counties, on="state_n")

# Group the states by the year of their first change.
states.group_by("first_change").agg(
    states=pl.len(),
    counties=pl.col("counties").sum(),
    smallest_step=pl.col("first_step").min(),
    largest_step=pl.col("first_step").max(),
).sort("first_change", nulls_last=True)
```

Of the 42 states that lifted any restriction, 38 made their first change between
1995 and 1998 and the last four in 2000 or 2001. A first change could lift a
single restriction or all four at once. The row with a null first change holds
the eight never-deregulating states and their 130 counties.

Because every later change becomes part of the effects estimated below, the
second table lists the states that kept changing their count after the first
change.

```{code-cell} ipython3
# The states whose count changed more than once.
states.filter(pl.col("changes") > 1).sort("state_n")
```

Of these nine states, eight changed their count twice and one changed it three
times. Every one of those later changes lifted more restrictions, except in
Indiana (state code 18). Indiana lifted all four restrictions in 1998 and ended
the panel at three after one of them came back. Even so, no county's count ever
falls below its starting value.

## Switchers against the status quo

Call a county a switcher once its state lifts a restriction. Write $F_g$ for the
year in which county $g$'s state first lifts one and count that year as the first
horizon. At horizon $\ell$, the quantity to estimate averages the gap between
loan growth in year $F_g - 1 + \ell$ and the loan growth the county would have
had if its state had lifted nothing,

$$
\delta_\ell = \frac{1}{N_\ell} \sum_{g:\, F_g - 1 + \ell \le T_g}
\mathbb{E}\big[Y_{g,F_g-1+\ell} - Y_{g,F_g-1+\ell}(0, \ldots, 0) \mid \boldsymbol{D}\big].
$$

Here $N_\ell$ counts the switchers observed at horizon $\ell$. $T_g$ is the last
year in which county $g$ can still be compared with a county at zero. The
expectation holds fixed $\boldsymbol{D}$, the treatment of every county in every
year. The status quo $(0, \ldots, 0)$ in the display means zero restrictions
lifted in every year.

:::{admonition} Effects of whole deregulation paths
:class: note

Because states lifted different numbers of restrictions and some lifted more
later, $\delta_\ell$ is the average effect of $\ell$ years of deregulation,
whatever steps each state took along the way.
[Effects per restriction lifted](#effects-per-restriction-lifted) later puts
these effects on a per-restriction scale.
:::

To estimate $\delta_\ell$, {func}`~moderndid.did_multiplegt` takes each
switcher's change in loan growth from year $F_g - 1$ to year $F_g - 1 + \ell$.
From that it subtracts the average change over the same years among the counties
still at zero in year $F_g - 1 + \ell$. Averaging those differences over the
switchers gives an estimate $\text{DID}_\ell$ that recovers $\delta_\ell$ when
the four conditions below hold.

- Loan growth doesn't respond to restrictions a state will only lift in later
  years.
- Had nothing been lifted, loan growth in the switchers would have moved in
  parallel with loan growth in the counties still at zero.
- Counties that start at zero don't all see their first restriction lifted in
  the same year. The eight never-deregulating states keep their 130 counties at
  zero through 2005.
- No county's count is ever both above and below its starting value.

The {ref}`background page <background-didinter>` states these four conditions
formally and defines the estimator, the per-restriction effects, and the average
total effect used below. It also explains why local projections mix effects of
different lengths of exposure.

## Horizons, controls, and uncertainty

Before running the estimation, we settle how far to follow the switchers, how to
choose their controls, and how to measure the uncertainty.

```{code-cell} ipython3
# Each check in the last section starts from this dictionary and changes one entry.
spec = dict(
    # The outcome, year, county, and treatment columns.
    yname="Dl_vloans_b",
    tname="year",
    idname="county",
    dname="inter_bra",
    # Five years of effects and three placebo years before each first change.
    effects=5,
    placebo=3,
    # Controls from the pool of counties still at zero, each counted once, without covariates.
    only_never_switchers=False,
    weightsname=None,
    xformla="~1",
    # Analytic standard errors clustered by state, where deregulation was decided.
    cluster="state_n",
    boot=False,
)
```

### Five years of effects and three placebo years

To keep the same switchers behind every effect, `effects=5` stops at the fifth
year of deregulation. Since no state made its first change after 2001, four years
before the data end, every switcher reaches that fifth year. A change from one
horizon to the next therefore can't come from switchers dropping out. Going
further loses switchers, as the eight horizons in
[Whether the boost wore off](#whether-the-boost-wore-off) show.

To test the comparison before anything was lifted, `placebo=3` runs it over
three years before each first change.

:::{admonition} Placebo years run out early
:class: note

Because the data start in 1994, a county whose state made its first change in
1996 has only one placebo year to offer. Only the 132 counties of the four states
whose first change came in 2000 or 2001 could supply a fourth placebo year.
:::

### The pool of counties still at zero

To give each switcher as many controls as possible, `only_never_switchers=False`
takes every county still at zero in the year being compared. Besides the
counties of the eight never-deregulating states, that pool holds the counties
whose states changed later. They add many comparisons in the years right after
a first change. In return, their loan growth before their own first change has
to track what the switchers' growth would have been. If you'd rather compare the
switchers with the never-deregulating states alone, `only_never_switchers=True`
does that, as
[Never-deregulating states as the only controls](#never-deregulating-states-as-the-only-controls)
shows.

Since `weightsname=None` counts every county once, the effects describe the
average switcher rather than the average state. States with many counties
therefore weigh more in each average than states with few. The check in
[States weighted roughly equally](#states-weighted-roughly-equally) weights the
counties by `w1` instead. With `xformla="~1"`, no covariates enter, as in the
estimates of de Chaisemartin and D'Haultfœuille (2024).

:::{admonition} House prices are an outcome too
:class: warning

Since it comes with the data, `Dl_hpi` may look like a covariate to pass through
`xformla`. According to de Chaisemartin and D'Haultfœuille (2024), deregulation
raised house price growth as well. Adjusting for it would absorb some of the
effect on lending.
:::

### Analytic standard errors clustered by state

Each state deregulated all of its counties at once, in the same years and by the
same steps. A shock to a state's economy also reaches every one of its counties.
Setting `cluster="state_n"` lets the standard errors allow for the counties of
one state moving together.
[Standard errors that ignore states](#standard-errors-that-ignore-states) shows
how much narrower the intervals get without it.

As in de Chaisemartin and D'Haultfœuille (2024), `boot=False` takes the standard
errors from the estimator's analytic variance formula. That formula holds every
county's deregulation history fixed, as the display above does with
$\boldsymbol{D}$. The alternative, `boot=True`, redraws whole states and reruns
the estimation on each draw, as [A bootstrap over states](#a-bootstrap-over-states)
shows.

One call with `spec` estimates the five effects and the three placebos together.

```{code-cell} ipython3
# Estimate the five yearly effects and the three placebos.
result = did.did_multiplegt(data, **spec)
print(result)
```

## Loan growth in the first five years

Start with the placebo table at the bottom of the report, since it tests the
comparison before you read anything into the effects. Placebo $-\ell$ runs the
comparison over the $\ell$ years that end in the year before the first change,
when nothing had been lifted yet. All three estimates, 0.0548, −0.0718, and
−0.1095, have intervals that cover zero. The joint test that all three are zero
has a p-value of 0.4068. The placebos are far from precise, though, since the
third rests on 489 of the 905 switchers and its interval runs from −0.3126 to
0.0937.

:::{admonition} Three placebo years can't vouch for five
:class: warning

For $\text{DID}_5$ to be unbiased, parallel trends has to hold over five years.
The placebos can test it over spans of no more than three years. A pre-trend too
small to show up in them could still move the later effects.
:::

The effects table in the middle answers the first half of the question. In the
year of each state's first change, loan growth in its counties ran 0.0435 above
the status quo, about 4 percentage points. That first-year interval runs from
−0.0261 to 0.1131 and covers zero. The effect then climbs unevenly over the next
four years to 0.1476 in the fifth. That is the only horizon whose interval, from
0.0182 to 0.2770, excludes zero. Five years in, loan growth in the switchers ran
about 15 percentage points above where it would have been.

Down the Switchers column of the effects table, you'll see 905 at every horizon
rather than the 916 switchers that Data Info counts, since 11 of them lack loan
growth for the year before their first change. The N column counts the
county-year cells behind each estimate, from the switchers and their controls
together. With the switchers held at 905, the fall of N from 3,810 to 1,800 comes
from the controls, since fewer counties remain at zero in later years.

:::{admonition} Three counties the estimator leaves out
:class: note

Data Info counts 1,045 of the 1,048 counties because two have no loan data and
one enters the panel in 2000 with a restriction already lifted. Since no other
county shares that starting value, that county can't enter any comparison.
:::

The table at the top condenses all five horizons into the average total effect.
Call one restriction lifted for one year in one county a restriction-year. The
average total effect sums the effects over the switchers' first five years and
divides that sum by the restriction-years the switchers accumulated over the
same years. Its estimate of 0.0346 is the total rise in loan growth that one
restriction-year produced over that year and the later years through the fifth
horizon. The interval around that estimate runs from −0.0077 to 0.0769 and
covers zero.

In the plot below, the navy placebo points left of the dashed vertical line at
horizon 0 scatter around zero with wide bars. To the right of that line, you can
see the red effects drift upward until the fifth bar clears zero.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Placebo estimates at horizons -3 to -1 and effects at horizons 1 to 5 with confidence intervals
---
# Placebos sit left of zero and effects to its right.
did.plot_multiplegt(result) + did.theme_moderndid()
```

## Whether the boost wore off

If the boost had worn off, the estimates above would shrink toward zero as the
horizon grows instead of rising through the fifth year. To see past the fifth
year, we follow the switchers for eight years, as de Chaisemartin and
D'Haultfœuille (2024) do.

```{code-cell} ipython3
# The same specification followed for eight years.
eight_years = did.did_multiplegt(data, **(spec | {"effects": 8}))
print(eight_years)
```

The first five rows of the effects table match the five-year run, because the
estimate at one horizon doesn't depend on how many others are requested. Effects
six through eight come out at 0.1631, 0.1565, and 0.1858, above every one of the
first five. None of the three intervals at six to eight years covers zero. That
answers the second half of the question for this specification. In the eighth
year of deregulation, loan growth in the switchers still ran 0.1858 above the
status quo. Nothing in these estimates suggests that the boost to lending wore
off.

:::{admonition} Fewer switchers after five years
:class: note

Effects six to eight rest on 850 and then 773 of the 905 switchers, since the
data run out first for the three states whose first change came in 2001 and a
year later for the state whose first change came in 2000. Part of any rise after
the fifth year can come from that change in the counties behind the average.
:::

### Effects per restriction lifted

The effects so far mix counties whose states lifted one restriction with
counties whose states lifted four. With `normalized=True`, each effect is
divided by the average number of restriction-years the switchers had
accumulated by that horizon. The result is a weighted average of the effects of
this year's count and of the counts in earlier years, per restriction. If
restrictions lifted years ago mattered as much as recent ones, the normalized
effects would stay flat. Setting `effects_equal=True` adds a chi-squared test of
whether the five are equal.

```{code-cell} ipython3
# Divide each effect by the restriction-years so far and test whether the five are equal.
per_restriction = did.did_multiplegt(data, **spec, normalized=True, effects_equal=True)
print(per_restriction)
```

In the first year the divisor is the size of the first step, about 2.2
restrictions on average, as the ratio of 0.0435 to 0.0197 shows. From the second
year on, the normalized effects stay between 0.0080 and 0.0134. The first year's
0.0197 is about twice the later ones but rests on a wide interval from −0.0118
to 0.0513 that covers all of them. With a p-value of 0.2012, the test of equal
effects doesn't reject that all five are the same. Leaving that imprecise first
year aside, the data are consistent with each year of a lifted restriction
adding about the same amount to loan growth. In that case the effect builds with
exposure instead of fading. The average total effect at the top doesn't change,
because it already measures effects per restriction.

### What the local projections show

Each local projection regresses loan growth $\ell - 1$ years ahead on this
year's number of restrictions lifted and on fixed effects for counties and
years. Favara and Imbs (2015) used last year's count and a set of covariates.
For each horizon from 1 to 9, the cell below runs the simpler version without
covariates that de Chaisemartin and D'Haultfœuille (2024) analyze.

```{code-cell} ipython3
import pyfixest as pf

# Regress loan growth ell - 1 years ahead on this year's restrictions lifted, with county and year
# fixed effects and standard errors clustered by state.
panel = data.sort("county", "year")
projections = {}
for ell in range(1, 10):
    shifted = pl.col("Dl_vloans_b").shift(1 - ell).over("county")
    ahead = panel.with_columns(growth_ahead=shifted).to_pandas()
    projections[ell] = pf.feols(
        "growth_ahead ~ inter_bra | county + year", data=ahead, vcov={"CRV1": "state_n"}
    )
```

Each coefficient and its 95 percent interval appear below next to the effect at
the same horizon from the eight-year run, first in a table and then in a figure.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: Local-projection coefficients that fall below zero at horizons 7 and 8 beside did_multiplegt effects that rise through horizon 8
---
from plotnine import (
    aes,
    facet_wrap,
    geom_errorbar,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
)

# Line up each local projection with the eight-year effect at the same horizon.
panels = [
    "Local projections, per restriction lifted as of this year",
    "did_multiplegt, whole deregulation path",
]
effects = eight_years.effects
rows = []
print(f"{'horizon':>7}{'local projection':>18}   [95% Conf. Interval]{'did_multiplegt':>16}")
for ell, fit in projections.items():
    coefficient = fit.tidy().loc["inter_bra"]
    estimate, low, high = coefficient["Estimate"], coefficient["2.5%"], coefficient["97.5%"]
    rows.append((panels[0], ell, estimate, low, high))
    effect = f"{effects.estimates[ell - 1]:16.4f}" if ell <= len(effects.horizons) else ""
    print(f"{ell:>7}{estimate:>18.4f}   [{low:8.4f}, {high:8.4f}]{effect}")
for horizon, (effect, low, high) in enumerate(
    zip(effects.estimates, effects.ci_lower, effects.ci_upper), start=1
):
    rows.append((panels[1], horizon, effect, low, high))

estimates = pl.DataFrame(rows, schema=["panel", "horizon", "estimate", "low", "high"], orient="row")
estimates = estimates.with_columns(pl.col("panel").cast(pl.Enum(panels)))
(
    ggplot(estimates.to_pandas(), aes("horizon", "estimate"))
    + geom_hline(yintercept=0, color="#7f8c8d")
    + geom_errorbar(aes(ymin="low", ymax="high"), width=0.2, color="#1a3a5c")
    + geom_point(size=3, color="#1a3a5c")
    + facet_wrap("panel", scales="free_y")
    + scale_x_continuous(breaks=list(range(1, 10)))
    + labs(x="Horizon", y="Estimate")
    + did.theme_moderndid()
)
```

On their own, the local projections look like a short-lived effect. Their
coefficients exclude zero from the second horizon through the fifth and shrink
from 0.0382 at the second to 0.0104 at the sixth. The seventh and eighth
coefficients fall below zero, to −0.0099 and −0.0139. At the eighth horizon,
even the upper end of the interval stays below zero. The ninth coefficient
climbs back to 0.0047 with an interval from −0.0216 to 0.0309. Over the same
horizons, the effects from {func}`~moderndid.did_multiplegt` trend upward
through the eighth.

:::{admonition} Compare shapes, not levels
:class: tip

A local-projection coefficient measures the change in loan growth per
restriction lifted as of this year. Each effect in the right panel is the full
effect of a county's deregulation path up to that horizon. The two panels answer
the question of fading through their shapes, since their levels sit on different
scales.
:::

Decomposing these coefficients, de Chaisemartin and D'Haultfœuille (2024)
suggest that the regression itself may be why they shrink. When states make
their first change at different dates, each coefficient averages the effects of
several lengths of exposure. A county whose state deregulates between this year
and the year whose loan growth is regressed still counts as a control, because
its count this year is zero. Since the weights in that average add up to well
under one from the second horizon and to less than zero from the fourth, the
coefficients can shrink or flip sign even if the effect holds steady.

## Whether the answer survives other settings

Each of the four checks below reruns the estimation with one entry of `spec`
changed. The first two change who the controls are and how much each county
weighs. The last two keep every estimate and change only how its standard error
is computed.

### Never-deregulating states as the only controls

Dropping the counties whose states changed later from the controls shows how
much they shape the answer. With `only_never_switchers=True`, every comparison
rests on the 129 counties with loan data in the eight never-deregulating states,
for the effects and for their per-restriction version alike.

```{code-cell} ipython3
# The same specification with the never-deregulating states as the only controls.
never = did.did_multiplegt(data, **(spec | {"only_never_switchers": True}))
never_per_restriction = did.did_multiplegt(
    data, **(spec | {"only_never_switchers": True}), normalized=True, effects_equal=True
)

# The effects, the same effects per restriction lifted, and their test of equal effects.
p_value = never_per_restriction.effects_equal_test["p_value"]
print("effects:", ", ".join(f"{x:.4f}" for x in never.effects.estimates))
print("per restriction:", ", ".join(f"{x:.4f}" for x in never_per_restriction.effects.estimates))
print(f"test of equal effects: p-value = {p_value:.4f}")
```

Against the never-deregulating states alone, the effect is larger from the
start, 0.1087 in the first year against 0.0435. It then stays between 0.1087 and
0.1618 through the fifth year. Per restriction lifted, the effect now falls from
0.0494 in the first year to 0.0147 in the fifth. The test of equal effects has a
p-value of 0.0416 and rejects at the 5 percent level. Under these controls the
boost arrives at once and holds roughly steady instead of building with
exposure.

The choice of controls matters most in the first years of deregulation, when the
main specification leans on counties still at zero whose states changed later.
Whether the boost lasts through the fifth year doesn't depend on that choice.

### States weighted roughly equally

To see whether a few states with many counties drive the answer, this check
weights each county by `w1` instead of counting every county once. Under those
weights, a state's total weight no longer grows with its number of counties.

```{code-cell} ipython3
# The same specification weighted by w1, so that states with many counties weigh less per county.
state_weighted = did.did_multiplegt(data, **(spec | {"weightsname": "w1"}))
print("effects:", ", ".join(f"{x:.4f}" for x in state_weighted.effects.estimates))

# The joint placebo test under the same weights.
p_value = state_weighted.placebo_joint_test["p_value"]
print(f"joint test (placebos = 0): p-value = {p_value:.4f}")
```

Weighting the states roughly equally leaves the first and fifth years close to
where they were, at 0.0451 and 0.1608. The years in between rise, to 0.1166 and
0.1146 at three and four years against 0.0813 and 0.0706 under county weights.
The joint placebo test is even further from rejecting, at a p-value of 0.8039.

### Standard errors that ignore states

The standard errors below drop the state clusters and treat every county as an
independent draw.

```{code-cell} ipython3
# The same specification with standard errors that treat counties as independent.
unclustered = did.did_multiplegt(data, **(spec | {"cluster": None}))

# Set the two sets of standard errors side by side.
print(f"{'horizon':>7}{'by state':>10}{'by county':>11}")
for horizon in range(5):
    clustered_se = result.effects.std_errors[horizon]
    county_se = unclustered.effects.std_errors[horizon]
    print(f"{horizon + 1:>7}{clustered_se:>10.4f}{county_se:>11.4f}")

ate = unclustered.ate
print(f"\naverage total effect {ate.estimate:.4f} [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
```

Dropping the clusters shrinks the standard errors at every horizon, by 30
percent in the first year and by 40 percent in the fifth. The interval of the
average total effect then runs from 0.0104 to 0.0588 and excludes zero.

:::{admonition} County-level intervals overstate precision
:class: warning

Treating the counties of one state as independent counts that state's experience
many times over. The narrower intervals here come from that overcounting and not
from better evidence.
:::

### A bootstrap over states

The last check redraws the 50 states with replacement and reruns the whole
estimation on each of 1,000 draws.

```{code-cell} ipython3
:tags: [skip-execution]

# Bootstrap standard errors from 1,000 draws of whole states, seeded so they reproduce.
bootstrapped = did.did_multiplegt(
    data, **(spec | {"boot": True, "biters": 1000, "random_state": 7})
)
```

```{code-cell} ipython3
:tags: [remove-cell]

bootstrapped = stored(
    "inter_did_bootstrap",
    lambda: did.did_multiplegt(
        data, **(spec | {"boot": True, "biters": 1000, "random_state": 7})
    ),
)
```

```{code-cell} ipython3
# Set the bootstrap standard errors next to the analytic ones.
print(f"{'horizon':>7}{'analytic':>10}{'bootstrap':>11}")
for horizon in range(5):
    analytic_se = result.effects.std_errors[horizon]
    bootstrap_se = bootstrapped.effects.std_errors[horizon]
    print(f"{horizon + 1:>7}{analytic_se:>10.4f}{bootstrap_se:>11.4f}")

# The fifth year's interval and the average total effect under the bootstrap.
low, high = bootstrapped.effects.ci_lower[4], bootstrapped.effects.ci_upper[4]
ate = bootstrapped.ate
print(f"\nfifth year {bootstrapped.effects.estimates[4]:.4f} [{low:.4f}, {high:.4f}]")
print(f"average total effect {ate.estimate:.4f} [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
```

The bootstrap standard errors run 14 to 28 percent above the analytic ones. At
five years, the interval of −0.0072 to 0.3024 now covers zero. Under the
bootstrap, the average total effect's interval reaches from −0.0167 to 0.0858.

:::{admonition} Only the standard errors are bootstrapped
:class: note

With `boot=True`, the effects, the placebos, and the average total effect get
bootstrap standard errors. The joint placebo test and the test of equal effects
keep the analytic variance. Their p-values therefore stay the same as in the
analytic report above.
:::

### The five-year effect under each check

To show which settings move the answer, the last table sets every check's
one-year and five-year effects beside its 95 percent interval at five years.

```{code-cell} ipython3
:tags: [hide-input]

# The one-year and five-year effects under each check, with the 95 percent interval at five years.
checks = {
    "our specification": result,
    "never-deregulating controls": never,
    "states weighted roughly equally": state_weighted,
    "county-level standard errors": unclustered,
    "bootstrap over states": bootstrapped,
}
print(f"{'check':<33}{'one year':>10}{'five years':>12}   [95% Conf. Interval]")
for name, check in checks.items():
    effects = check.effects
    print(
        f"{name:<33}{effects.estimates[0]:>10.4f}{effects.estimates[4]:>12.4f}   "
        f"[{effects.ci_lower[4]:8.4f}, {effects.ci_upper[4]:8.4f}]"
    )
```

Since the five-year effect stays between 0.1476 and 0.1618 across the checks and
remains the largest of the five, none of these choices makes the boost to
mortgage lending wear off. The first year moves the most, from 0.0435 to 0.1087
when the controls come only from the never-deregulating states. Whether the
five-year effect can be told apart from zero depends on the standard errors.
Every interval built from the analytic standard errors excludes zero at five
years. Only the bootstrap's interval reaches down to −0.0072 and covers it.

All five rows of the table still assume parallel trends between the switchers
and the counties still at zero.
{ref}`Dynamic covariate balancing <example_dyn_balancing>` replaces that
assumption with one about how treatment is assigned, that each year's treatment
depends only on past treatments, outcomes, and covariates. That example applies
it to democracy, a treatment that countries move into and out of.
