---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_etwfe)=

# Extended two-way fixed effects

The county data of the {ref}`staggered example <example_staggered_did>` covers
13 states that raised their minimum wage above the federal level in 2004, 2006,
or 2007 and 16 that left theirs alone. This time a single regression answers
whether those increases lowered teen employment and by how much. The textbook
version of that regression puts one treatment indicator next to county and year
fixed effects. Its lone coefficient has to stand for every cohort in every year
after adoption. When states raise their minimum wages in different years, that
coefficient also treats counties whose minimum wage has already gone up as
controls for the later increases.
[Wooldridge (2025)](https://doi.org/10.1007/s00181-025-02807-z) argues that the
fault lies with that restrictive model and not with two-way fixed effects
themselves.

{func}`~moderndid.etwfe` keeps the regression but gives each treated cohort in
each year its own coefficient. You'll see {func}`~moderndid.emfx` average the
seven cohort-year effects, first by years since adoption and then into one
overall effect. We set that effect beside the single-coefficient regression
before the checks at the end revisit our choices one by one. The choice that
matters most is which county-years serve as controls, since the 2007 cohort's
employment had already slipped relative to the never-treated counties in the
year before its increase.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## The county data

With the federal minimum wage held at \$5.15 an hour from 1997 until mid-2007,
each increase in this panel was a state's own decision. Because many of the jobs
that pay close to the minimum wage go to teenagers, the outcome here is teen
employment. {func}`~moderndid.load_mpdta` loads the panel, five years of data
from 2003 through 2007 on each of 500 US counties. We also add each county's
state, the level at which minimum wages are set.

```{code-cell} ipython3
import moderndid as did
import polars as pl

# A county's state code is its FIPS code without the last three digits.
data = did.load_mpdta().with_columns(state=pl.col("countyreal") // 1000)
data.head()
```

`lemp` holds teen employment in logs for each county and year. Next to it,
`lpop` is the log of the county's population in 2000. The timing comes from
`first.treat`, the year the county's state lifted its minimum wage, or 0 if that
never happened in these years.

:::{admonition} etwfe reads the timing from gname
:class: important

{func}`~moderndid.etwfe` takes each county's adoption year from whichever column
`gname` names and reads a 0, infinity, or a year after the panel ends as never
treated. Rows with a missing adoption year leave the sample with a warning. So
do counties already treated in the first year, since no untreated year is left
to identify their effects.
:::

Of the 191 treated counties, 131 adopted in 2007, 40 in 2006, and 20 in 2004.
The 16 states without an increase supply the other 309 counties as never-treated
controls. Because all 20 counties of the 2004 cohort lie in Illinois (state code
17), every estimate two or three years after adoption describes that one state.
Average `lpop` is 3.18 among the never-treated counties and between 3.46 and
3.75 in the treated cohorts. The specification below therefore lets employment
trends differ with county population.

## The target of each coefficient

Each treatment coefficient in the regression stands for one cohort in one year.
For the counties whose minimum wage first rose in year $g$, the target in any
year $t$ from $g$ on is the gap between their employment and what it would have
been had their minimum wage never gone up,

$$
\tau_{g,t} = \mathbb{E}\big[y_t(g) - y_t(\infty) \mid d_g = 1\big], \qquad t \ge g,
$$

where $d_g$ flags those counties and $y_t(\infty)$ is employment in year $t$
without any increase.

After an increase, $y_t(\infty)$ is never observed for the counties it reached.
The regression predicts it from the county-years that are still untreated. Those
are all five years of the never-treated counties plus every cohort's years
before its increase. In practice, {func}`~moderndid.etwfe` gives each of the
seven treated cohort-years, or cells, its own indicator in a regression with
county and year fixed effects and year-specific slopes on population. Wooldridge
(2025) proves that the coefficients on those indicators equal an imputation
estimate. That estimate fits a regression to the untreated county-years,
predicts untreated employment for the treated ones, and averages the gaps within
each cohort and year.

:::{admonition} County or cohort fixed effects
:class: note

In a balanced panel like this one, Wooldridge (2025) shows that cohort dummies
give the same coefficients as county fixed effects. Without `idname`,
{func}`~moderndid.etwfe` uses cohort dummies and heteroskedasticity-robust
standard errors by default. Since those errors ignore how a county's shocks
persist over time, pass `vcov={"CRV1": "countyreal"}` in that case.
:::

Four assumptions have to hold for that prediction to recover $y_t(\infty)$.

- Every county starts untreated in 2003 and stays treated once its minimum wage
  rises.
- Teen employment does not respond in any year before the increase takes effect.
- Without the increases, counties of equal population would have followed one
  employment trend in every year, whatever their cohort.
- The employment trend without the increases changes linearly with county
  population.

{ref}`Extended TWFE <background-etwfe>` in the background section gives the
formal assumptions in the paper's notation and lays out the imputation steps
that the regression reproduces.

## Specifying the regression

Three choices remain before the regression can run, two about how it predicts
employment without the increases and one about its standard errors. We gather
all three and the columns the data uses in the dictionary below.

```{code-cell} ipython3
# Three design choices follow the column names, one for each section below.
spec = dict(
    # Columns for teen employment, the year, the adoption year, and the county.
    yname="lemp",
    tname="year",
    gname="first.treat",
    idname="countyreal",
    # Use every untreated county-year as a control, even a cohort's own years before adoption.
    cgroup="notyet",
    # Let employment trends differ with county population.
    xformla="~lpop",
    # Cluster the standard errors by county.
    vcov={"CRV1": "countyreal"},
)
```

### Every untreated county-year as a control

With `cgroup="notyet"`, each cohort's own years before adoption join the
never-treated counties in predicting employment without the increase. Each
cohort's baseline then rests on all of its years before the increase rather than
on the last one alone. The price is a version of parallel trends that has to
hold through every one of those years.

The alternative, `cgroup="never"`, compares each cohort only with the
never-treated counties and measures every year from the one just before
adoption. It also estimates a placebo effect for each earlier year, as
[Never-treated controls and placebo estimates](#never-treated-controls-and-placebo-estimates)
shows.

:::{admonition} No placebo estimates under the default
:class: note

With every year before adoption on the control side, the not-yet-treated
regression has no cells left to test parallel trends in those years.
{func}`~moderndid.emfx` returns no event times below zero for it, even with
`post_only=False`.
:::

### Trends that differ with population

Because employment can trend differently in large and small counties and the
treated counties are larger on average, `xformla="~lpop"` lets each year carry
its own slope on population. Parallel trends then only has to hold among
counties of similar size. Inside the treated cells, population enters centered
at its cohort mean. That centering makes each cell's coefficient the average
effect over the cohort's counties rather than the effect at a log population of
zero. The check in
[One trend for counties of every size](#one-trend-for-counties-of-every-size)
shows how much the adjustment matters here.

:::{admonition} A covariate can carry the effect away
:class: warning

If a minimum wage increase could change a covariate, an adjustment for that
covariate would strip part of the increase's effect out of the estimate.
Population measured in 2000, four years before the first increase, can't have
responded to any increase.
:::

### Standard errors clustered by county

Treating a county's five years as independent draws would ignore that shocks to
its employment persist from year to year. Clustering by county allows for that
persistence, as Wooldridge (2025) recommends for this regression. Because
`idname` names the counties, {func}`~moderndid.etwfe` clusters by county by
default. The specification spells out `vcov` anyway to keep the choice in view.
When {func}`~moderndid.emfx` averages the cells later, it carries this variance
matrix into the averages with the delta method and treats the cohort sizes and
the cohort means of `lpop` as known. The check in
[Standard errors clustered by state](#standard-errors-clustered-by-state) uses
clusters of whole states instead, since each state sets its own minimum wage.

With `spec` complete, {func}`~moderndid.etwfe` fits the regression and returns
one estimate for each of the seven treated cells.

```{code-cell} ipython3
# Fit one regression with a separate effect for every treated cohort-year.
result = did.etwfe(data, **spec)
print(result)
```

## One effect per cohort and year

The Group and Time columns name the cohort and year behind each of the seven
coefficients. The 2004 cohort's effect deepens from −0.0212 in its first year to
−0.0818 and −0.1379 over the next two years before easing to −0.1095 in 2007.
The 2006 cohort moves from 0.0025 in its first year to −0.0451 in its second. In
2007, its only treated year, the 2007 cohort comes in at −0.0460. Of the seven
bands, only the two for the first treated year of the 2004 and 2006 cohorts
cover zero.

:::{admonition} Analytic, pointwise bands
:class: note

Since the standard errors come straight from the regression's clustered variance
matrix, no bootstrap is drawn and no seed is needed. Each band holds its 95
percent coverage for its own effect only, not for all seven at once.
:::

## The effect over years of exposure

Event time counts how long a cohort has lived with its higher minimum wage.
Grouping the cells by it shows whether the effect changes with exposure. With
`type="event"`, {func}`~moderndid.emfx` averages the cohorts at each event time
in proportion to their counties.

```{code-cell} ipython3
# Group the cells by event time and average the cohorts at each one.
event_study = did.emfx(result, type="event")
print(event_study)
```

The effect starts at −0.0332 in the year of the increase and deepens to −0.0573
a year later. Event times 2 and 3 reproduce the 2004 cohort's cells for 2006 and
2007, −0.1379 and −0.1095, because Illinois is the only cohort observed that
long after adoption. The summary of −0.0845 at the top gives the four event
times equal weight. Half of that weight therefore falls on event times that
describe Illinois alone. The staggered example shows how to
[hold the counties fixed](example_staggered_did.md#holding-the-counties-fixed)
and separate a changing mix of counties from an effect that grows.

## The overall effect on teen employment

For one number that answers the question, {func}`~moderndid.emfx` with
`type="group"` first reduces each cohort to the mean of its treated cells. It
then weights the three cohorts by how many counties each holds. No treated
county ends up weighing more than any other in that average.

```{code-cell} ipython3
# One effect per cohort, then one overall effect with cohorts weighted by size.
by_cohort = did.emfx(result, type="group")
print(by_cohort)
```

The overall effect of −0.0452 has a 95 percent interval from −0.0723 to
−0.0180. Since `lemp` is in logs, the effect converts to $e^{-0.0452} - 1$, or
about 4.4 percent fewer teenagers employed in the treated counties than without
the increases. At −0.0876, the 2004 counties fell furthest of the three cohorts.
The 2006 cohort's effect of −0.0213 has a band that covers zero. For the 2007
cohort, the effect equals its one treated cell at −0.0460. With 131 of the 191
treated counties, the 2007 cohort carries most of the weight in the overall
effect.

### Counting treated years instead of counties

The default of {func}`~moderndid.emfx`, `type="simple"`, weights the cells
differently and gives a larger number.

```{code-cell} ipython3
# The default simple type averages over every treated county-year instead.
simple = did.emfx(result, type="simple")
print(f"simple average {simple.overall_att:.4f} ({simple.overall_se:.4f})")
```

The simple average of −0.0506 counts every treated county-year once rather than
every treated county. Illinois's counties each add four treated years to that
average, twice as many as the 2006 counties and four times as many as the 2007
counties. Their share of the weight thus grows from 20 of the 191 counties to 80
of the 291 treated county-years. We keep the group summary of −0.0452 as our
answer, because it gives Illinois the weight of its 20 counties and no more.

### The one-coefficient regression

The regression from the introduction has a single indicator that switches on in
each county's first treated year and stays on. To compare like with like, the
regression below gets the same county and year fixed effects, the same
year-by-population slopes, and the same county clusters.

```{code-cell} ipython3
import pyfixest as pf

# Mark every county-year from its state's increase on with a single indicator.
adopted = (pl.col("first.treat") > 0) & (pl.col("year") >= pl.col("first.treat"))
twfe_data = data.with_columns(treated=adopted.cast(pl.Int64)).to_pandas()

# One treatment coefficient with county and year fixed effects, the same
# population trends, and errors clustered by county.
conventional = pf.feols(
    "lemp ~ treated + i(year, lpop, ref=2003) | countyreal + year",
    data=twfe_data,
    vcov={"CRV1": "countyreal"},
)
print(f"one coefficient {conventional.coef()['treated']:.4f} ({conventional.se()['treated']:.4f})")
```

The lone coefficient of −0.0387 is smaller in magnitude than both the
county-weighted −0.0452 and the county-year-weighted −0.0506. It leans partly on
comparisons that use counties already treated as controls for later adopters.
When effects grow over time, as Illinois's do,
{ref}`such comparisons <background-did-twfe>` push the estimate toward zero.
Giving each cell its own coefficient removes those comparisons and measures
every cell only against county-years that are still untreated.

## Testing the three choices

Each check below refits the regression with one argument of `spec` changed. The
controls come up first, the population trends second, and the clusters last,
just as [Specifying the regression](#specifying-the-regression) set them out.

### Never-treated controls and placebo estimates

The first check limits the controls to the never-treated counties with
`cgroup="never"` and prints every cell, the placebo rows included.

```{code-cell} ipython3
# The same specification with only the never-treated counties as controls.
never = did.etwfe(data, **(spec | {"cgroup": "never"}))
print(never)
```

Each of the five rows dated before its cohort's increase, the 2006 cohort's two
and the 2007 cohort's three, is a placebo estimate whose band covers zero.
Relative to the never-treated counties and measured against 2006, the 2007
cohort's employment ran 0.0333 higher in 2004 and 0.0285 higher in 2005. Its
relative employment therefore slipped in the year before its increase.

That slip is why the 2007 cohort's effect shrinks from −0.0460 under the default
to −0.0288 here. The default builds that cohort's baseline from all four of its
years before adoption. Averaging its placebo estimates of 0.0069, 0.0333, and
0.0285 with the zero that its base year carries by construction gives 0.0172.
Subtracting that 0.0172 from −0.0288 gives exactly the default's estimate of
−0.0460.

```{code-cell} ipython3
# The overall effect and an event study that keeps the placebo estimates before adoption.
never_by_cohort = did.emfx(never, type="group")
never_event = did.emfx(never, type="event", post_only=False)
print(f"overall effect {never_by_cohort.overall_att:.4f} ({never_by_cohort.overall_se:.4f})")
```

Across all cohorts, the overall effect moves from −0.0452 to −0.0329. The 2007
cohort accounts for most of that change, since its one cell moved the furthest
and it carries most of the weight. Wooldridge (2025) shows that this design
gives the same estimates as the outcome regression version of the
[Callaway and Sant'Anna (2021)](https://doi.org/10.1016/j.jeconom.2020.12.001)
estimator with never-treated controls and a universal base period.

In the figure below, the blue triangles show the default's event study and the
gray circles show the never-treated design's. The gray circles to the left of
the dotted line at event time −1 are its placebo estimates.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: Event studies from the default and never-treated designs, with placebo estimates at event times -4 to -2 and effects at 0 to 3
---
from plotnine import (
    aes,
    geom_errorbar,
    geom_hline,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_shape_manual,
    scale_x_continuous,
)

# Stack both event studies and label each by the controls its design uses.
default = "every untreated county-year (default)"
colors = {default: "#315bc4", "never-treated counties only": "#7f8c8d"}
shapes = {default: "^", "never-treated counties only": "o"}
studies = {default: event_study, "never-treated counties only": never_event}
comparison = pl.concat(
    [did.to_df(study).with_columns(design=pl.lit(name)) for name, study in studies.items()]
)

dodge = position_dodge(width=0.3)
(
    ggplot(comparison, aes("event_time", "att", color="design", shape="design"))
    + geom_hline(yintercept=0, color="#7f8c8d")
    + geom_vline(xintercept=-1, linetype="dotted", color="#7f8c8d")
    + geom_errorbar(aes(ymin="ci_lower", ymax="ci_upper"), width=0.2, position=dodge)
    + geom_point(size=3, position=dodge)
    + scale_color_manual(values=colors)
    + scale_shape_manual(values=shapes)
    + scale_x_continuous(breaks=list(range(-4, 4)))
    + labs(
        x="Years since the minimum wage increase",
        y="Effect on log teen employment",
        color="",
        shape="",
    )
    + did.theme_moderndid()
)
```

All three gray placebo estimates sit above zero, though each band still reaches
it. After adoption the two designs nearly coincide, except in the year of the
increase. Since the 2007 cohort makes up most of the counties at event time 0,
the gray circle sits closer to zero there than the blue triangle at −0.0332.

### One trend for counties of every size

Dropping the covariate asks parallel trends to hold for large and small counties
alike.

```{code-cell} ipython3
# The same specification without the population covariate.
unadjusted = did.etwfe(data, **(spec | {"xformla": None}))
unadjusted_by_cohort = did.emfx(unadjusted, type="group")
print(
    f"overall effect {unadjusted_by_cohort.overall_att:.4f} ({unadjusted_by_cohort.overall_se:.4f})"
)
```

Without `lpop`, the overall effect comes in at −0.0423 rather than −0.0452. The
year-by-population slopes therefore make little difference here, even though the
treated counties are larger.

### Standard errors clustered by state

Setting `vcov={"CRV1": "state"}` clusters by the 29 states rather than the 500
counties and leaves every coefficient as it was.

```{code-cell} ipython3
# The same specification with whole states as the clusters.
clustered = did.etwfe(data, **(spec | {"vcov": {"CRV1": "state"}}))
clustered_by_cohort = did.emfx(clustered, type="group")
print(clustered_by_cohort)
```

State clusters leave the overall effect at −0.0452 and raise its standard error
from 0.0139 to 0.0226. The 95 percent interval of −0.0895 to −0.0008 now
excludes zero by less than a thousandth.

:::{admonition} Don't rely on a one-state cluster
:class: danger

With all 20 of its counties in Illinois, the 2004 cohort gets a state-clustered
standard error of 0.0130, well below the 0.0230 from county clusters. That band
is too narrow to trust, because a single cluster gives the variance estimate no
way to see shocks to Illinois as a whole.
:::

### Four estimates of the overall effect

Across these four specifications the estimate ranges from −0.0329 to −0.0452
with every interval below zero. Only the switch to never-treated controls moves the
estimate much, by 0.0123 against 0.0029 for dropping the covariate. The size of
the drop in teen employment therefore turns on which years count as controls,
about 3.2 percent with never-treated counties alone against 4.4 percent under
the default.

:::{admonition} Pair the default with placebo estimates
:class: tip

When you report the default design, fit the never-treated design next to it. Its
placebo estimates show whether the years the default treats as controls moved in
step with the never-treated counties.
:::

All of these answers assume that teen employment in counties of similar size
would have trended alike without the increases. The data alone can't settle
whether the 2007 cohort's slip before its increase was employers holding back on
teen jobs ahead of time. The staggered example's
[anticipation check](example_staggered_did.md#anticipation) shows how far an
answer on this data moves if it was. For counts or binary outcomes, `family`
fits a Poisson, logit, or probit version of the same regression. The
{ref}`background page <background-etwfe>` explains which parallel trends
assumption each of those models imposes.
