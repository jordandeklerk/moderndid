---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_etwfe)=

# Extended two-way fixed effects

The county data from the {ref}`staggered example <example_staggered_did>` cover
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

{func}`~moderndid.load_mpdta` loads the county panel that the staggered
example's [data section](example_staggered_did.md#the-data) describes, including
the Illinois counties that make up its whole 2004 cohort. We also add each
county's state, the level at which minimum wages are set.

```{code-cell} ipython3
import moderndid as did
import polars as pl

# A county's state code is its FIPS code without the last three digits.
data = did.load_mpdta().with_columns(state=pl.col("countyreal") // 1000)
data.head()
```

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
estimate.

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
counties of similar size. Since population was measured in 2000, before any
increase, it avoids the problem in the staggered example's
[covariate warning](example_staggered_did.md#never-treated-counties-of-a-similar-size).
Inside the treated cells, population enters centered at its cohort mean. That
centering makes each cell's coefficient the average effect over the cohort's
counties rather than the effect at a log population of zero. The check in
[One trend for counties of every size](#one-trend-for-counties-of-every-size)
shows how much the adjustment matters here.

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
−0.1379 in its third. Of the seven bands, only the two for the first treated
year of the 2004 and 2006 cohorts cover zero.

:::{admonition} Pointwise bands from the regression
:class: note

Since the standard errors come straight from the regression's clustered variance
matrix, each band is pointwise and needs no seed, unlike the simultaneous bands
that the {ref}`bootstrap background <background-did-bootstrap>` explains.
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

To estimate an average across treated counties, {func}`~moderndid.emfx` with
`type="group"` first averages each cohort's treated cells. It then weights
the three cohorts by their numbers of counties. Each treated county therefore
contributes equally to the overall effect.

```{code-cell} ipython3
# One effect per cohort, then one overall effect with cohorts weighted by size.
by_cohort = did.emfx(result, type="group")
print(by_cohort)
```

The overall effect of −0.0452 has a 95 percent interval from −0.0723 to
−0.0180. Since `lemp` is in logs, the effect converts to $e^{-0.0452} - 1$, or
about 4.4 percent fewer teenagers employed in the treated counties than without
the increases. With 131 of the 191 treated counties, the 2007 cohort and its
effect of −0.0460 carry most of the weight in the overall effect.

### Counting treated years instead of counties

The default of {func}`~moderndid.emfx`, `type="simple"`, weights the cells
differently and finds a larger drop in teen employment.

```{code-cell} ipython3
# The default simple type averages over every treated county-year instead.
simple = did.emfx(result, type="simple")
print(f"simple average {simple.overall_att:.4f} ({simple.overall_se:.4f})")
```

Because the simple average of −0.0506 counts every treated county-year once,
Illinois's counties make up 80 of its 291 treated county-years instead of 20 of
the 191 counties. We keep the group summary of −0.0452 as our answer, because it
gives Illinois the weight of its 20 counties and no more. The
{ref}`results guide <working-with-results>` compares the aggregation types by
their weights and the question each one answers.

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
    alt: Event studies from the default and never-treated designs, with placebo estimates at event times −4 to −2 and effects at 0 to 3
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
comparison = pl.concat(
    [
        did.to_df(event_study).with_columns(design=pl.lit(default)),
        did.to_df(never_event).with_columns(design=pl.lit("never-treated counties only")),
    ]
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

:::{admonition} Pair the default with placebo estimates
:class: tip

When you report the default design, fit the never-treated design next to it. Its
placebo estimates show whether the years the default treats as controls moved in
step with the never-treated counties. To show both designs in one table, give
each regression its own column as in the
{ref}`publication tables guide <publication_tables>`.
:::

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

:::{admonition} One treated state cannot reveal state-level variation
:class: warning

The 2004 cohort's standard error falls from 0.0230 under county clusters to
0.0130 under state clusters. The staggered example's
[clustering check](example_staggered_did.md#clustering-by-state) explains why a
single treated state makes that band too narrow to trust.
:::

Only the choice of controls moves the overall effect much, from −0.0452 under
the default to −0.0329 with never-treated counties alone. The data alone can't
tell whether the 2007 cohort's slip before its increase was a pre-trend or
employers holding back on teen jobs ahead of time. The staggered example's
[anticipation check](example_staggered_did.md#anticipation) shows how far an
answer on this data moves if employers did hold back.
