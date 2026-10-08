---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_staggered_did)=

# Staggered difference-in-differences

In the county data we'll use in this example, 13 of the 29 states raised their
minimum wage in 2004, 2006, or 2007. The central question we'll ask is whether
those increases cost teenagers jobs and whether the effect grew over time. With
the staggered timing of this policy change, the
{ref}`two-way fixed effects <background-did-twfe>` regression you'd usually run
measures later increases partly against counties still adjusting to earlier
ones. If the effect grows, those comparisons can hide it or even flip its sign.

We're going to avoid this common pitfall by using {func}`~moderndid.att_gt` to
measure each cohort's effect in each year only against counties whose state
never raised its minimum wage. What you'll see at the end of this is that
averaging those effects with {func}`~moderndid.aggte` gives an event study and
an overall effect that we can examine closer. Last, we'll change one choice at a
time to show you which parts of the answer to this natural experiment hold up.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## The data

The minimum wage changes in this data make for a good natural experiment. From
1997 until mid-2007 the federal minimum wage stayed at \$5.15 an hour. Every
increase in this data therefore came from a state choosing to raise its own
minimum above the federal level. Teenagers are a natural group to study, since
many of them work at or near the minimum wage.

{func}`~moderndid.load_mpdta` loads the county data, a panel of 500 US counties
observed every year from 2003 to 2007. Since these 500 counties are only a
subset of the 2,284 that [Callaway and Sant'Anna
(2021)](https://doi.org/10.1016/j.jeconom.2020.12.001) study, the numbers in
this example won't match the ones in their paper. Because states set the minimum
wage, it also helps to know which state each county belongs to. Dropping the
last three digits of a county's FIPS code gives its state code.

```{code-cell} ipython3
import moderndid as did
import polars as pl
from plotnine import theme

# The leading digits of a county FIPS code are its state's code.
data = did.load_mpdta().with_columns((pl.col("countyreal") // 1000).alias("state"))
data.head()
```

The outcome of interest is `lemp`, the log of teen employment in each county.
You'll also see `lpop`, the log of each county's population in thousands as
measured in 2000. The column that drives the whole analysis is `first.treat`,
the year a county's state first raised its minimum wage. Counties in states that
never raised it during these years have a 0 there instead.

:::{admonition} first.treat sets the timing
:class: important

{func}`~moderndid.att_gt` reads treatment timing from the column you pass as
`gname` and treats a 0 there as never treated. You might be tempted to use the
`treat` column for this instead. Since `treat` marks every county whose state ever acted
in all five years, it can't separate treated years from untreated ones and
moderndid never reads it.
:::

The 2004, 2006, and 2007 cohorts hold 20, 40, and 131 counties. The remaining
309 counties sit in the 16 states that never raised their minimum wage during
these years. One thing to keep in mind for later is that Illinois (state code
17) makes up the entire 2004 cohort. Since it's the only cohort observed two and
three years after adoption, anything this example says about the longer run is
really about Illinois.

On average the treated counties are also bigger than the never-treated ones.
Their mean `lpop` runs from 3.46 to 3.75, compared with 3.18 for the
never-treated counties. That size gap is the reason each cohort gets compared
only with never-treated counties of a similar size.

## What we're estimating

The quantity this example targets is the group-time average treatment effect.
For the cohort whose state raised its minimum wage in year $g$, it compares the
employment those counties had in year $t$ with the employment they would have
had without the increase,

$$
ATT(g, t) = \mathbb{E}\big[Y_t(g) - Y_t(0) \mid G = g\big].
$$

The catch is that you never observe $Y_t(0)$ for a treated county once its state
has raised the minimum wage. Never-treated counties of a similar size stand in
for it instead. That borrowing is only valid when the four assumptions below
hold together.

- No county is treated in 2003 and treatment never switches off once it starts.
- Employment doesn't react before a higher minimum wage takes effect.
- Among counties of similar size, treated and never-treated employment would
  have moved in parallel without the increases.
- Every treated county has never-treated counties of a similar size to compare
  with.

If you want the formal versions,
{ref}`DiD with multiple time periods <background-did-assumptions>` states each
one precisely. To build each comparison, moderndid uses the doubly robust
estimator of
[Sant'Anna and Zhao (2020)](https://doi.org/10.1016/j.jeconom.2020.06.003).
[Doubly robust DiD](../background/drdid) explains why that estimate stays
consistent when either its outcome regression or its propensity score is right.

## Choosing a specification

With the data and the target in place, there are a handful of choices to make
before estimating anything. Each one encodes an assumption about how minimum
wages and teen employment behave. In
[Pushing on the answer](#pushing-on-the-answer), we'll change them one at a time
to see whether the answer moves.

```{code-cell} ipython3
# Keep every choice in one dictionary so each check later on can change a single argument.
spec = dict(
    # The columns that hold the outcome, the year, the county, and its adoption year.
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    # Compare each cohort with never-treated counties of similar size.
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    # Measure every year against the year before adoption.
    base_period="universal",
    anticipation=0,
    # Bootstrap standard errors and simultaneous bands, seeded so the numbers reproduce.
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)
```

### Never-treated counties of a similar size

The first choice is which counties each cohort gets compared with. With
`control_group="nevertreated"`, the comparison counties are the 309
never-treated counties and nobody else. The alternative,
[`"notyettreated"`](#other-control-groups-and-estimators), would also borrow the
later cohorts in the years before they adopt. That buys more comparisons at the
price of a stronger assumption, since parallel trends would then have to hold
for those later adopters too.

Next comes the question of what makes two counties comparable in the first
place. Since big and small counties can follow different employment trends,
`xformla="~lpop"` conditions on size.
[Without the covariate](#without-the-covariate) later tests how much that
adjustment matters. Each comparison is then built with `est_method="dr"`,
the doubly robust estimator from the previous section. The alternatives
[`"ipw"` and `"reg"`](#other-control-groups-and-estimators) each lean on just one
of its two models.

:::{admonition} Covariates the policy moves bias the effect
:class: warning

A covariate that the policy itself can move would absorb part of the effect
you're trying to measure. Population is safe here because it was measured in
2000, before any of the increases took effect.
:::

### The year before adoption as the base

Each comparison runs from a base year to the year it estimates. With
`base_period="universal"`, every year gets measured against the year just before
adoption, $g-1$. That's what you want when you plot an event study, because it
puts the placebo estimates before adoption on the same scale as the effects
after it. The default, `"varying"`, instead compares each year before adoption
with the year right before it.

:::{admonition} What the base period changes
:class: note

The base period decides how the estimates before adoption are measured. The
effects after adoption and their standard errors come out the same under either
setting, as [The base period](#the-base-period) confirms near the end of the
guide.
:::

The year before adoption only works as a base if employment didn't react before
the increases took effect. Setting `anticipation=0` writes that assumption into
the specification for every cohort. Since it's also the easiest one to doubt, it
gets [its own test](#anticipation) near the end of the guide.

### Bootstrap standard errors and simultaneous bands

Last comes the question of how to measure the uncertainty in the estimates.
Setting `boot=True` and `cband=True` gives bootstrap standard errors along with
simultaneous bands that cover every estimate in a table at once. Raising the
number of draws from the default 1,000 to 10,000 makes the bands depend less on
the random seed. If shocks that hit a whole state worry you,
[Clustering by state](#clustering-by-state) reruns the bootstrap with one random
weight per state rather than per county.

:::{admonition} Seed the bootstrap
:class: tip

Without `random_state`, every run of the code draws new bootstrap weights and
prints new standard errors and bands. Even under the default `boot=False`,
{func}`~moderndid.aggte` draws a bootstrap for its simultaneous bands whenever
`cband=True`.
:::

With every choice in `spec`, the estimation itself takes a single call to
{func}`~moderndid.att_gt`.

```{code-cell} ipython3
# Estimate each cohort's effect in each year.
result = did.att_gt(data, **spec)
print(result)
```

## Group-time effects

Each row of this table is one cohort in one year, labeled by its Group and Time
columns. The rows that show 0.0000 and NA are each cohort's base year, zero by
construction. Start with the rows before the base years, since they're placebo
tests. If parallel trends held before adoption, those estimates should sit near
zero. All five placebo bands cover zero and the joint pre-test has a p-value of
0.2327. Neither check gives any sign of trouble in the years before adoption.

After adoption, Illinois's 2004 cohort is the only one that moves far from zero.
Its effect goes from −0.0145 in its adoption year to −0.1404 two years later.
Its two starred bands, two and three years after adoption, are the only ones in
the table that exclude zero. The 2006 and 2007 cohorts, by contrast, stay
between 0.0010 and −0.0413.

:::{admonition} Simultaneous bands are wider on purpose
:class: note

All 12 bands cover their effects together with 95 percent probability. That lets
you scan the table for stars without worrying that one in 20 excludes zero by
chance. For a single effect you chose in advance, `cband=False` gives the
narrower pointwise interval.
:::

It's easier to compare the cohorts year by year in a plot.
{func}`~moderndid.plots.plot_gt` draws one panel per cohort. As you look down
the panels, watch how far each cohort's effects drop below zero after adoption.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Group-time effects for the 2004, 2006, and 2007 cohorts in three stacked panels
---
# Three stacked panels need a taller canvas, and theme_moderndid drops the gray grid.
did.plot_gt(result) + did.theme_moderndid() + theme(figure_size=(12, 8))
```

## How the effect builds over time

The group-time table answers the question one cohort at a time. To see whether
the effect grows with time since adoption, you can line the cohorts up by event
time, the number of years since their state raised its minimum wage. Passing
`type="dynamic"` to {func}`~moderndid.aggte` averages the effects at each event
time and weights each cohort by its number of counties.

```{code-cell} ipython3
# Average the effects by years since adoption, over the cohorts observed that long.
event_study = did.aggte(result, type="dynamic")
print(event_study)
```

Before adoption, the estimates of 0.0063, 0.0269, and 0.0232 all sit near zero
with bands that cover it. After adoption, the effect grows from −0.0211 in the
adoption year to −0.0530, −0.1404, and −0.1069 over the next three years. At
first glance, that looks like an effect that builds over time.

:::{admonition} Event times 2 and 3 are one state
:class: warning

Event time 0 averages all three cohorts while event time 1 loses the 2007
cohort. By event times 2 and 3, only the Illinois counties are left in the
average. Because the growth after event time 1 mixes a changing effect with a
changing set of counties, read those points as Illinois's path.
:::

The same issue affects the overall number at the top of the report. Since the
event study weights every event time equally, more than half of its overall
estimate of −0.0804 comes from Illinois. That's why it overstates the average
effect across treated counties, a point that comes back when it's time to
measure how much employment fell.

In the plot of the event study, the estimates before adoption hover around zero
while those after adoption fall away from it.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Event study with placebo estimates at event times -4 to -2 and effects at 0 to 3
---
# The default reference line marks the base year at event time -1.
did.plot_event_study(event_study) + did.theme_moderndid()
```

### Holding the counties fixed

To separate real growth from a changing set of counties, you can hold the
counties fixed. With `balance_e=1`, {func}`~moderndid.aggte` keeps only the
cohorts observed at least one year after adoption. Event times 0 and 1 then rest
on exactly the same counties.

```{code-cell} ipython3
# Keep the cohorts observed at least one year after adoption.
balanced = did.aggte(result, type="dynamic", balance_e=1)
print(balanced)
```

On those 60 counties in four states, the effect goes from −0.0042 in the
adoption year to −0.0530 a year later. So even with the counties held fixed, the
effect still grows by 0.0488 in the year after adoption. If you want to see why
balancing matters in general,
{ref}`DiD with multiple time periods <background-did-balanced>` works through
the composition effects it removes.

## How much employment fell

Now we can answer the first part of the central question, whether the increases
cost teenagers jobs and by how much. With `type="group"`,
{func}`~moderndid.aggte` averages each cohort's effects after adoption and then
weights the cohorts by their number of counties. That way every treated county
counts the same in the overall effect.

```{code-cell} ipython3
# Average each cohort's effects after adoption and weight the cohorts by their counties.
by_cohort = did.aggte(result, type="group")
print(by_cohort)
```

The overall effect of −0.0328 means that teen employment in the treated counties
sat about 3.2 percent below where it would have been without the increases.
With a 95 percent interval from −0.0567 to −0.0089, this is our answer to how
much employment fell.

### Why the other averages come out larger

It's worth seeing why none of the other overall numbers that
{func}`~moderndid.aggte` can produce is the right answer here. The simple
aggregation is the most common alternative to the group aggregation.

```{code-cell} ipython3
# Weight every effect after adoption by its cohort's counties and average them.
simple = did.aggte(result, type="simple")
print(simple)
```

Both the simple aggregation at −0.0418 and the event study at −0.0804 come out
larger than −0.0328. The difference comes down to how much weight each one gives
Illinois, the cohort with the longest exposure and the largest effects.

Because the group aggregation counts each cohort by its counties, Illinois's 20
of the 191 treated counties earn it 10.5 percent of the weight. The simple
aggregation also counts every year a cohort spends after adoption. Since
Illinois has four of those years against two for the 2006 cohort and one for the
2007 cohort, its share rises to 27.5 percent. The event study gives each event
time from 0 to 3 a quarter of the weight. With Illinois the only cohort left at
event times 2 and 3, its share climbs to 61.0 percent. That's why we report
−0.0328 as the average effect across treated counties.

## Pushing on the answer

So far every number on this page has come from the single specification chosen
earlier. This section changes one choice at a time and keeps everything else the
same to show whether the answer moves. The checks follow the same order as the
choices in [Choosing a specification](#choosing-a-specification).

### Other control groups and estimators

The first check goes back to the comparison group and the estimator. The cell
below swaps in not-yet-treated controls and then each single-model estimator,
one at a time.

```{code-cell} ipython3
# The overall effect when only the control group or only the estimator changes.
changes = {
    "notyettreated": {"control_group": "notyettreated"},
    "ipw": {"est_method": "ipw"},
    "reg": {"est_method": "reg"},
}
variants = {}
for name, change in changes.items():
    variant = did.att_gt(data, **(spec | change))
    variants[name] = did.aggte(variant, type="group")
    print(f"{name:>14}  {variants[name].overall_att:.4f}")
```

Across all three, the overall effect stays between −0.0323 and −0.0329. Neither
the comparison group nor the estimator changes the answer here.

### Without the covariate

The next check asks how much the adjustment for county population really
matters. Without `lpop`, parallel trends would have to hold across all counties,
big and small.

```{code-cell} ipython3
# The same specification without lpop.
unadjusted = did.att_gt(data, **(spec | {"xformla": None}))
unadjusted_by_cohort = did.aggte(unadjusted, type="group")
print(
    f"pre-test p-value {unadjusted.wald_pvalue:.4f}, "
    f"overall effect {unadjusted_by_cohort.overall_att:.4f}"
)
```

Without the adjustment, the pre-test p-value falls to 0.1681 and the overall
effect moves only from −0.0328 to −0.0310. Adjusting for population makes the
placebo estimates a little more consistent with zero without driving the
answer.

### The base period

We chose a universal base period mostly for how the event study reads. To make
sure it doesn't change the answer, the cell below reruns the specification under
the default `base_period="varying"`. Each year before adoption is then compared
with the year just before it.

```{code-cell} ipython3
# The same specification under the default varying base.
varying = did.att_gt(data, **(spec | {"base_period": "varying"}))
varying_by_cohort = did.aggte(varying, type="group")
print(varying)
```

The estimates after adoption, their standard errors, and the pre-test p-value
of 0.2327 don't change at all. The rows before adoption now show one-year changes
such as the 2007 cohort's −0.0284 from 2005 to 2006. Because the simultaneous
bands cover those rows too, the bands after adoption widen slightly.

### Anticipation

The next assumption to push on is that employment didn't react before the
increases took effect. If you look back at the event study, the placebo
estimates drop from 0.0232 two years before adoption to zero in the base year.
Most of that drop is the 2007 cohort's −0.0284 between 2005 and 2006. If
employers cut teen hiring ahead of the increase, that drop belongs to the effect
rather than to a pre-trend. Setting `anticipation=1` moves each cohort's base
year back to $g-2$ so the drop counts toward the effect.

```{code-cell} ipython3
# The same specification with one year of anticipation, so each cohort's base year is g - 2.
anticipating = did.att_gt(data, **(spec | {"anticipation": 1}))
anticipating_by_cohort = did.aggte(anticipating, type="group")
print(anticipating_by_cohort)
```

The 2004 cohort drops out here because it would need 2002 as its base year. One
year of anticipation roughly doubles the 2007 cohort's effect from −0.0288 to
−0.0572 and moves the overall effect from −0.0328 to −0.0500.

:::{admonition} Anticipation looks like a pre-trend
:class: warning

The data can't tell anticipation apart from a downward trend that was already
under way. Which reading you accept decides whether the overall effect is
−0.0328 or −0.0500. This part of the answer rests on what you're willing to
assume about the year before adoption.
:::

### Clustering by state

The last check is about how much to trust the estimate rather than the estimate
itself. Since counties in the same state share the policy and whatever else hit
that state's economy, treating them as independent can make the standard errors
too small. Setting `clustervars=["state"]` makes the bootstrap draw one random
weight per state instead of one per county.

```{code-cell} ipython3
# The same specification, except that the bootstrap weights whole states.
clustered = did.att_gt(data, **spec, clustervars=["state"])
clustered_by_cohort = did.aggte(clustered, type="group")
print(clustered_by_cohort)
```

With state clusters, the standard error of the overall effect rises from 0.0122
to 0.0176. Its interval now runs from −0.0674 to 0.0018 and just barely covers
zero.

:::{admonition} Don't trust Illinois's clustered bands
:class: danger

Illinois's standard error falls from 0.0265 to 0.0190 under state clusters
because all its counties share one cluster. With a single treated state the
bootstrap can't see state-level shocks. The 2004 cohort's clustered bands and
those at event times 2 and 3 are too narrow as a result.
:::

To see what clustering does to each event time, the cell below reruns the event
study on the clustered estimates.

```{code-cell} ipython3
# The event study from the state-clustered estimates.
clustered_event_study = did.aggte(clustered, type="dynamic")
print(clustered_event_study)
```

Every placebo band still covers zero once the bootstrap clusters by state. The
band at event time 1 widens to run from −0.1185 to 0.0125 and now covers zero as
well. Clustering also drops the Wald pre-test and leaves the placebo bands as
the only check.

:::{admonition} Report the state-clustered overall effect
:class: tip

Since the overall effect pools all 13 treated states, its state-clustered
interval is the safer one to report. For single event times, read the
county-level and state-clustered bands side by side. With so few treated states,
neither one settles them on its own.
:::

### Putting the checks together

The checks help distinguish choices that move the employment estimate from
choices that change how precisely we can measure it. Changing the comparison
group, the estimator, the covariate, or the base period leaves the overall effect
between −0.0310 and −0.0329.

The two choices that do matter end up working in quite different ways. Allowing
one year of anticipation moves the estimate itself to −0.0500 and keeps its
interval below zero, from −0.0839 to −0.0161. Clustering by state instead leaves
the estimate at −0.0328 but widens its interval until it reaches 0.0018. How
large the effect is depends on what you're willing to assume about the year
before adoption. How confident you can be in the estimate depends on how you
cluster.

Every number on this page still leans on parallel trends with the never-treated
counties. If you want to know how far the event study holds up when that
assumption is only approximately true,
{ref}`Sensitivity analysis <example_honest_did>` shows how to bound the effects.
{ref}`Extended TWFE <example_etwfe>` takes the same data in a different
direction and estimates it with a regression that gives each cohort and year its
own effect.
