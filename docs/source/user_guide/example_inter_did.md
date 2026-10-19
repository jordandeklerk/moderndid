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

{func}`~moderndid.did_multiplegt` avoids that problem by comparing each county
whose state lifted a restriction only with counties whose states hadn't lifted
any yet. You'll follow those comparisons from three placebo years before
deregulation through its first eight years, read them per restriction lifted,
and set them beside the local projections on the same counties. At the end,
separate checks vary the controls, the weights, and the standard errors to see
which of them the answer depends on.

```{code-cell} ipython3
:tags: [remove-cell]

import polars as pl
from plotnine import options
from prerun import stored

options.figure_size = (12, 5)
options.dpi = 100
pl.Config.set_float_precision(4)
```

## The branching deregulations

Under the Interstate Banking and Branching Efficiency Act, a state could require
its agreement before banks from other states opened new branches, set a minimum
age for a bank they bought in a merger, forbid buying single branches without the
whole bank, and cap the share of the state's deposits that one bank controlled.

Every state starts this panel at zero, since no state had lifted any of the
restrictions as of 1994. With steps that differed across states in timing and
in size, the panel works as a natural experiment. Since the restrictions applied
to banks, the outcome is the growth of mortgage lending by banks. Because it is
already a growth rate, an effect that persists means the volume of loans keeps
pulling away from where it would have been.

{func}`~moderndid.load_favara_imbs` loads the panel of 1,048 counties in 50 states
that Favara and Imbs (2015) assembled. Each of its rows holds one county in one
year between 1994 and 2005.

```{code-cell} ipython3
import moderndid as did
import polars as pl

data = did.load_favara_imbs()
data.head()
```

The outcome, `Dl_vloans_b`, is the change in the log volume of mortgage loans
that banks originated in the county that year. The treatment, `inter_bra`,
counts how many of the four restrictions the county's state had lifted by then,
from 0 to 4. Each county's state appears as a numeric code in `state_n`, the
column the standard errors cluster on. For one of the checks, `w1` weights each
county by roughly the inverse of the number of counties in its state. Last,
`Dl_hpi` holds the growth of house prices, the second outcome of de Chaisemartin
and D'Haultfœuille (2024).

:::{admonition} Give dname each year's treatment level
:class: important

{func}`~moderndid.did_multiplegt` reads each county's treatment in each year
from `dname` and finds the year of its first change. Since an adoption year or a
flag for having ever deregulated never changes within a county, either one stops
the call with an error that no unit changes treatment. A yearly 0/1 flag would
keep the same effects but hide the size of each step and every later change from
the per-restriction effects and the average total effect.
:::

Of the 42 states that lifted any restriction, 38 made their first change between
1995 and 1998 and the last four in 2000 or 2001. The eight never-deregulating
states supply 130 counties that remain at zero throughout the panel.

Because nine states changed their count more than once, their effects include
later policy changes as well as the first step. Every one of those later changes
lifted more restrictions, except in Indiana (state code 18). Indiana lifted all
four restrictions in 1998 and ended the panel at three after one of them came
back. Even so, no county's count ever falls below its starting value.

:::{admonition} Gaps in the panel
:class: note

Five counties miss some of the years between 1994 and 2005, 38 county-years in
all. {func}`~moderndid.did_multiplegt` uses the panel with those gaps and leaves
a county out of any comparison that needs one of its missing years. The guide to
{ref}`preparing panel data <panel-utilities-gaps>` lists these same gaps and
weighs filling them against dropping the five counties.
:::

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
year.

:::{admonition} Effects of whole deregulation paths
:class: note

Because states lifted different numbers of restrictions and some lifted more
later, $\delta_\ell$ is the average effect of $\ell$ years of deregulation,
whatever steps each state took along the way.
[Effects per restriction lifted](#effects-per-restriction-lifted) later puts
these effects on a per-restriction scale.
:::

{func}`~moderndid.did_multiplegt` compares each switcher's change in loan growth
from year $F_g - 1$ to year $F_g - 1 + \ell$ with the average change among the
counties still at zero over the same years. Averaged over the switchers, those
differences give an estimate $\text{DID}_\ell$ that recovers $\delta_\ell$ when
the four conditions below hold.

- Loan growth doesn't respond to restrictions a state will only lift in later
  years.
- Had nothing been lifted, loan growth in the switchers would have moved in
  parallel with loan growth in the counties still at zero.
- Counties that start at zero don't all see their first restriction lifted in
  the same year.
- No county's count is ever both above and below its starting value.

The {ref}`background page <background-didinter>` states these four conditions
formally and defines the estimator, the per-restriction effects, and the average
total effect used below.

## Horizons, controls, and uncertainty

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

To test the comparison before anything was lifted, `placebo=3` also runs it over
the three years before each first change.

:::{admonition} How far back the placebos reach
:class: note

Because the data start in 1994, a county whose state made its first change in
1996 has only one placebo year to offer. Only 128 counties, all in the four
states whose first change came in 2000 or 2001, could supply a fourth placebo
year.
:::

### The pool of counties still at zero

To give each switcher as many controls as possible, `only_never_switchers=False`
takes every county still at zero in the year being compared. Besides the
counties of the eight never-deregulating states, that pool holds the counties
whose states changed later. They add many comparisons in the years right after
a first change. In return, their loan growth before their own first change has
to track what the switchers' growth would have been.
[Never-deregulating states as the only controls](#never-deregulating-states-as-the-only-controls)
compares the switchers with the never-deregulating states alone.

Since `weightsname=None` counts every county once, the effects describe the
average switcher rather than the average state. The check in
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

Start with the placebo table at the bottom of the report. Placebo $-\ell$ runs
the comparison over the $\ell$ years that end in the year before the first
change, when nothing had been lifted yet. All three estimates, 0.0548, −0.0718,
and −0.1095, have intervals that cover zero. The joint test that all three are
zero has a p-value of 0.4068. The placebos are far from precise, though, since
the third rests on 489 of the 905 switchers and its interval runs from −0.3126
to 0.0937.

:::{admonition} Placebos cover fewer years than the effects
:class: warning

Since the placebos reach back three years at most, they can't test parallel
trends over the five years behind $\text{DID}_5$. Even placebos that pass don't
confirm it, as the {ref}`sensitivity analysis example <example_honest_did>`
explains.
:::

The effects table in the middle answers the first half of the question. In the
year of each state's first change, loan growth in its counties ran 0.0435 above
the status quo, about 4 percentage points. That first-year interval runs from
−0.0261 to 0.1131 and covers zero. The effect then climbs unevenly over the next
four years to 0.1476 in the fifth. That is the only horizon whose interval, from
0.0182 to 0.2770, excludes zero. Five years in, loan growth in the switchers ran
about 15 percentage points above where it would have been.

Data Info counts 1,045 of the 1,048 counties and 129 of the 130 never-switchers
because one never-switcher and one other county have no loan data and a third
county enters the panel in 2000 with a restriction already lifted. Down the
Switchers column, the same 905 of the 916 switchers that Data Info counts stand
behind all five effects, since the other 11 lack loan growth for the year before
their first change. The {ref}`results guide <working-with-results>` explains why
the set of units behind each estimate matters when you compare horizons.

At the top, the average total effect condenses all five horizons. Counting one
restriction lifted for one year in one county as a restriction-year, its
estimate of 0.0346 is the total rise in loan growth that one restriction-year
produced over that year and the later years through the fifth horizon. The
interval around that estimate runs from −0.0077 to 0.0769 and covers zero.

In the plot below, the navy placebo points left of the dashed line at horizon 0
scatter around zero with wide bars. To the right of that line, the red effects
drift upward until the fifth bar clears zero.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Placebo estimates at horizons −3 to −1 and effects at horizons 1 to 5 with confidence intervals
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
answers the second half of the question for this specification. Nothing in these
estimates suggests that the boost to lending wore off.

:::{admonition} Fewer switchers after five years
:class: note

Effects six to eight rest on 850 and then 773 of the 905 switchers, as the data
run out for the four states whose first change came in 2000 or 2001. Part of any
rise after the fifth year can come from that change in the counties behind the
average.
:::

### Effects per restriction lifted

With `normalized=True`, each effect is divided by the average number of
restriction-years the switchers had accumulated by that horizon. If restrictions
lifted years ago mattered as much as recent ones, the normalized effects would
stay flat. Setting `effects_equal=True` adds a chi-squared test of whether the
five are equal.

```{code-cell} ipython3
# Divide each effect by the restriction-years so far and test whether the five are equal.
per_restriction = did.did_multiplegt(data, **spec, normalized=True, effects_equal=True)
print(per_restriction)
```

From the second year on, the normalized effects stay between 0.0080 and 0.0134.
The first year's 0.0197 is about twice the later ones but rests on a wide
interval from −0.0118 to 0.0513 that covers all of them. With a p-value of
0.2012, the test of equal effects doesn't reject that all five are the same.
Leaving that imprecise first year aside, the data are consistent with each year
of a lifted restriction adding about the same amount to loan growth. In that
case the effect builds with exposure instead of fading. The average total effect
at the top doesn't change, because it already measures effects per restriction.

### What the local projections show

Each local projection regresses loan growth $\ell - 1$ years ahead on this
year's number of restrictions lifted and on fixed effects for counties and
years. Favara and Imbs (2015) used last year's count and a set of covariates.
For each horizon from 1 to 9, the cell below runs the simpler version without
covariates that de Chaisemartin and D'Haultfœuille (2024) analyze.

```{code-cell} ipython3
import pyfixest as pf

# Regress loan growth ell - 1 years ahead on this year's restrictions lifted and on county
# and year fixed effects. Each regression clusters its standard errors by state.
panel = data.sort("county", "year")
projections = {}
for ell in range(1, 10):
    shifted = pl.col("Dl_vloans_b").shift(1 - ell).over("county")
    ahead = panel.with_columns(growth_ahead=shifted).to_pandas()
    projections[ell] = pf.feols(
        "growth_ahead ~ inter_bra | county + year", data=ahead, vcov={"CRV1": "state_n"}
    )
```

The figure below places each coefficient and its 95 percent interval beside the
effect at the same horizon from the eight-year run.

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

# The coefficient and 95 percent interval of each local projection by horizon.
rows = []
for ell, fit in projections.items():
    coefficient = fit.tidy().loc["inter_bra"]
    rows.append((ell, coefficient["Estimate"], coefficient["2.5%"], coefficient["97.5%"]))
local = pl.DataFrame(rows, schema=["horizon", "att", "ci_lower", "ci_upper"], orient="row")

# Stack the eight-year effects under the local projections and label each panel.
effects = did.to_df(eight_years).filter(pl.col("horizon") > 0).select(local.columns)
estimates = pl.concat(
    [
        local.with_columns(
            panel=pl.lit("Local projections, per restriction lifted as of this year")
        ),
        effects.with_columns(panel=pl.lit("did_multiplegt, whole deregulation path")),
    ]
)
(
    ggplot(estimates, aes("horizon", "att"))
    + geom_hline(yintercept=0, color="#7f8c8d")
    + geom_errorbar(aes(ymin="ci_lower", ymax="ci_upper"), width=0.2, color="#1a3a5c")
    + geom_point(size=3, color="#1a3a5c")
    + facet_wrap("panel", scales="free_y")
    + scale_x_continuous(breaks=list(range(1, 10)))
    + labs(x="Horizon", y="Estimate")
    + did.theme_moderndid()
)
```

On their own, the local projections look like a short-lived effect. Their
coefficients exclude zero from the second horizon through the fifth and shrink
from 0.0382 at the second to 0.0104 at the sixth. By the seventh and eighth
horizons, they have fallen below zero. Over the same horizons, the effects from
{func}`~moderndid.did_multiplegt` trend upward through the eighth.

:::{admonition} Compare shapes, not levels
:class: tip

A local-projection coefficient measures the change in loan growth per
restriction lifted as of this year. Each effect in the right panel is the full
effect of a county's deregulation path up to that horizon. The two panels answer
the question of fading through their shapes, since their levels sit on different
scales.
:::

Because each coefficient mixes the effects of several lengths of exposure with
weights that can add up to less than one, the regression itself can make the
coefficients shrink, as the {ref}`background page <background-didinter>`
explains.

## Whether the answer survives other settings

### Never-deregulating states as the only controls

Dropping the counties whose states changed later from the controls shows how
much they shape the answer. With `only_never_switchers=True`, every comparison
takes its controls from the eight never-deregulating states.

```{code-cell} ipython3
# The same specification with the never-deregulating states as the only controls.
never = did.did_multiplegt(data, **(spec | {"only_never_switchers": True}))
did.to_df(never)
```

Against the never-deregulating states alone, the effect is larger from the
start, 0.1087 in the first year against 0.0435. It then stays between 0.1087 and
0.1618 through the fifth year.

```{code-cell} ipython3
# The per-restriction effects and the test of equal effects under the same controls.
never_per_restriction = did.did_multiplegt(
    data, **(spec | {"only_never_switchers": True}), normalized=True, effects_equal=True
)
ate = never_per_restriction.ate
print(f"average total effect {ate.estimate:.4f} [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
print(f"test of equal effects p-value {never_per_restriction.effects_equal_test['p_value']:.4f}")
did.to_df(never_per_restriction)
```

Per restriction lifted, the effect now falls from 0.0494 in the first year to
0.0147 in the fifth. The test of equal effects has a p-value of 0.0416 and
rejects at the 5 percent level. Under these controls the boost arrives at once
and holds roughly steady instead of building with exposure. The average total
effect rises to 0.0592 with an interval from 0.0007 to 0.1176 that excludes
zero.

The choice of controls matters most in the first years of deregulation, when the
main specification leans on counties still at zero whose states changed later.

### States weighted roughly equally

To see whether a few states with many counties drive the answer, this check
weights each county by `w1` instead of counting every county once.

```{code-cell} ipython3
# The same specification weighted by w1, so that states with many counties weigh less per county.
state_weighted = did.did_multiplegt(data, **(spec | {"weightsname": "w1"}))
print(f"joint placebo test p-value {state_weighted.placebo_joint_test['p_value']:.4f}")
did.to_df(state_weighted)
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
ate = unclustered.ate
print(f"average total effect {ate.estimate:.4f} [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
did.to_df(unclustered)
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
estimation on each of 1,000 draws. You can pass `n_jobs` to spread those draws
over several threads and get the same standard errors sooner, as the
{ref}`scaling guide <scaling>` explains.

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
ate = bootstrapped.ate
print(f"average total effect {ate.estimate:.4f} [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}]")
did.to_df(bootstrapped)
```

For the five effects, the bootstrap standard errors run 14 to 28 percent above
the analytic ones. At five years, the interval of −0.0072 to 0.3024 now covers
zero. Under the bootstrap, the average total effect's interval reaches from
−0.0167 to 0.0858.

:::{admonition} What the bootstrap leaves analytic
:class: note

With `boot=True`, the effects, the placebos, and the average total effect get
bootstrap standard errors. The joint placebo test and the test of equal effects
keep the analytic variance. Their p-values therefore stay the same as in the
analytic report above.
:::

Since the five-year effect stays between 0.1476 and 0.1618 under every check,
none of these choices makes the boost to mortgage lending wear off. Only the
standard errors decide whether that effect can be told apart from zero.
{ref}`Dynamic covariate balancing <example_dyn_balancing>` drops the assumption
of parallel trends between the switchers and the counties still at zero in favor
of one about how treatment is assigned.
