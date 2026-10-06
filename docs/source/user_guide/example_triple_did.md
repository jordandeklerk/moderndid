---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_triple_did)=

# Triple differences

The tobacco farmers we'll follow here could lose a whole year's crop to bad
weather in a single season. Until 2003, the farmers of Jiangxi province in China
carried that risk on their own. That year the People's Insurance Company of
China began selling weather insurance to the tobacco farmers of one county
there. This example asks whether the insurance changed how those households
saved, measured by how much of their new savings they kept in checking accounts
they could draw on at any time.

Two simple comparisons get this question wrong in opposite ways. Setting the insured
farmers against tobacco farmers in other counties would credit the insurance
with anything else that hit their county. Setting them against their own
neighbors who grew other crops would credit it with anything that hit tobacco
farmers everywhere.

{func}`~moderndid.ddd` sidesteps both by measuring how the
gap between tobacco farmers and their neighbors changed in the insured county
and subtracting how the same gap changed in counties without insurance. By the
end you'll have a year-by-year event study from {func}`~moderndid.agg_ddd` and a
single overall effect that answers the question. After that, we'll vary the
specification one choice at a time and hold the answer up against the three-way
fixed effects regression of the original study.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## The insurance rollout

The way the insurance reached farmers makes this data a natural experiment. In
2003 the company launched its first weather-indexed crop
insurance for tobacco farmers in selected counties of Jiangxi province. Every
tobacco grower in those counties had to take a contract. Households growing
other crops were left out even in the same counties. So was every household in
the counties that never joined the program.

{func}`~moderndid.load_cai2016` returns yearly records for 3,659 households
between 2000 and 2008. These are the households
[Cai (2016)](https://doi.org/10.1257/pol.20130371) studied and
[Ortiz-Villavicencio and Sant'Anna (2025)](https://arxiv.org/abs/2505.09942)
later revisited with the estimator used here.

```{code-cell} ipython3
import moderndid as did
import polars as pl

data = did.load_cai2016()
data.head()
```

Each row of the data is one household observed in one year. The
`checksaving_ratio` column records
the share of that year's net savings the household put into checking accounts,
the money it could withdraw on demand. Ortiz-Villavicencio and Sant'Anna (2025)
treat a rise in this share as a sign that insured households moved their savings
toward more flexible accounts. Whether a household could get the insurance
depends on two columns, `treatment` for the county that offered it and `sector`
for tobacco growers. The covariates `hhsize` and `age` hold the size of the
household and the age of its head.

:::{admonition} ddd needs the adoption year, not a flag
:class: important

The `gname` column tells {func}`~moderndid.ddd` when a household's county
started offering insurance. A 0 in that column means the county never did. A
flag like `treatment` won't do, because it reads 1 for the insured county even
in 2000, 2001, and 2002.
:::

{func}`~moderndid.core.panel.get_group` turns the flag into that year. With
`treat_period=2003`, households in the insured county get 2003 and everyone else
gets 0. Because no household ever switches county or crop, a single row per
household is enough to size up the four groups the design compares.

```{code-cell} ipython3
# Give households in the insured county the year insurance arrived and every other household a 0.
data = did.get_group(data, idname="hhno", tname="year", treatname="treatment", treat_period=2003)
data = data.rename({"G": "group"})

# Count the households and counties in each of the four groups, and how many appear every year.
households = data.group_by("hhno").agg(pl.col("group", "sector", "county").first(), years=pl.len())
print(f"{(households['years'] == 9).sum()} of {households.height} households appear in all nine years")
households.group_by("group", "sector").agg(
    households=pl.len(),
    counties=pl.col("county").n_unique(),
).sort("group", "sector")
```

Inside the insured county sit 837 tobacco farmers and 161 households growing
something else. The 11 counties without insurance contribute another 1,271
tobacco farmers. Ten of those counties also hold the 1,390 other households. The
smallest group, the 161 non-tobacco households in the insured county, enters
every comparison below and caps how precise the answer can get. The panel is
also unbalanced, since 361 of the 3,659 households are missing from at least one
year.

## The effect we're after

For each year $t$, the target is the average effect of the insurance on the
insured tobacco farmers. It's the gap between the share of savings they kept in
checking accounts and the share they would have kept without coverage,

$$
ATT(2003, t) = \mathbb{E}\big[Y_t(2003) - Y_t(\infty) \mid S = 2003, Q = 1\big].
$$

Here $S$ is the year a household's county began offering insurance and $Q = 1$
flags tobacco farmers. With a single adoption year, every year after 2003 doubles
as an event time that counts the years since the insurance arrived.

Once 2003 has passed, nobody observes $Y_t(\infty)$, the share an insured farmer
would have kept without insurance. The triple difference fills it in from the
other three groups. That step only works if four assumptions hold at once.

- Coverage begins in 2003 for every insured household and never lapses.
- Households don't change how they save in anticipation of the insurance.
- Without the insurance, the gap between tobacco farmers and other households
  of similar size and head's age would have moved the same way in the insured
  county as elsewhere.
- All four groups contain households of comparable size and age.

The background page on {ref}`triple differences <background-tripledid>` writes
these out formally. The estimator is the doubly robust one from
[Ortiz-Villavicencio and Sant'Anna (2025)](https://arxiv.org/abs/2505.09942). It
builds every comparison from an outcome model and a propensity model and stays
consistent if either one is right.

:::{admonition} Less to assume than in a plain DiD
:class: note

Neither tobacco farmers nor the insured county has to trend like anyone else.
All the triple difference needs is for the gap between the two kinds of
households to move alike across counties.
:::

## Setting up the estimation

Apart from the first four, which only name columns in the data, every argument
below settles a question about this design. We write them all out first so that
you can see every answer in one place.

```{code-cell} ipython3
# The whole specification lives in one dictionary.
# Each later check swaps out a single argument.
spec = dict(
    # The outcome, year, household, and adoption-year columns.
    yname="checksaving_ratio",
    tname="year",
    idname="hhno",
    gname="group",
    # Mark tobacco farmers and compare with counties without insurance.
    pname="sector",
    control_group="nevertreated",
    # Adjust for household size and age with the doubly robust estimator.
    xformla="~ hhsize + age",
    est_method="dr",
    # Anchor every year to 2002 and allow the unbalanced panel.
    base_period="universal",
    allow_unbalanced_panel=True,
    # Bootstrap the standard errors with 999 draws and a fixed seed.
    boot=True,
    biters=999,
    random_state=7,
)
```

### Tobacco farmers and the counties without insurance

Since the insurance was meant for tobacco farmers alone, `pname="sector"` splits
every county's households into tobacco farmers and everyone else.
[Two differences instead of three](#two-differences-instead-of-three) tests what
that split adds over a plain DiD.

For the comparison counties, `control_group="nevertreated"` takes the 11 that
never offered insurance. The other setting, `"notyettreated"`, would also let in
counties that adopt insurance later. Since none do in this data, the two
settings give identical results.

### Adjusting for household size and age

Beyond their crop and county, households differ in ways that could shape how
they save. Following Ortiz-Villavicencio and Sant'Anna (2025), we adjust for
household size and the age of the household head with `xformla="~ hhsize + age"`.
Under `est_method="dr"`, each comparison combines an outcome model with a
propensity model. Switching to `"ipw"` or `"reg"` would keep only one of the
two. [Other estimators and no covariates](#other-estimators-and-no-covariates)
tries both and then drops the covariates.

:::{admonition} Choose covariates the insurance can't change
:class: tip

Conditioning on something the insurance itself affects would soak up part of the
very effect being measured. Household size and the head's age are safe on that
count. Household income, which the insurance protects, would not be safe to
adjust for.
:::

### One base year and an unbalanced panel

Each yearly estimate is a change measured from a base year. Because the default
`base_period="universal"` measures every year from 2002, the last year before
coverage, the placebo years and the treated years share one reference point in
the event study. If you
[set `base_period="varying"`](#the-trend-before-2003) instead, each year before
2003 takes the year that precedes it as its base.

For the 361 households missing from at least one year, some comparisons have no
change to measure. With `allow_unbalanced_panel=True`, each comparison pools
everyone observed in either of its two years and treats those years as separate
samples. The default `False` would keep only the households seen in both.
Because moderndid adds up each household's contributions before computing
standard errors, the household stays the unit of inference throughout.
[Households seen every year](#households-seen-every-year) drops those 361
households from the data altogether.

### Standard errors from 999 bootstrap draws

To repeat the 999 draws that Ortiz-Villavicencio and Sant'Anna (2025) used, the
specification sets `boot=True` and `biters=999`. Fixing `random_state` keeps the
printed numbers from changing between runs. The simultaneous bands come in
later, once {func}`~moderndid.agg_ddd` builds the event study.

:::{admonition} Every interval here may be too narrow
:class: warning

Every interval on this page treats each household as an independent draw. A
shock that hit the insured county's tobacco farmers differently from its other
households would make them too narrow. With a single insured county, clustering
by county couldn't account for it anyway.
:::

Passing `spec` to {func}`~moderndid.ddd` runs the estimation under every choice
above.

```{code-cell} ipython3
# Estimate the effect of the insurance in each year.
result = did.ddd(data, **spec)
print(result)
```

## Year by year

There's one row per year, all for the same insured tobacco farmers. The 2002
row reads 0.0000 with no standard error because 2002 is the anchor year. The two
rows above it serve as placebo tests for the years before coverage. Had the gap between tobacco farmers and other
households moved alike in every county before 2003, both would hover near zero.

Instead, the 2000 estimate of −0.0599 has an interval from −0.1089 to −0.0108
that excludes zero. The 2001 estimate of −0.0264 sits between it and the base
year, as if the gap were already widening on its way to 2002. We'll come back to
that pattern near the end, since it shapes how to read everything after 2003.
After 2003 the estimates rise from 0.0118 in the first year to 0.1553 by 2008.
The one break in that climb comes in 2006, when the estimate slips from 0.0525
to 0.0494. Only the intervals for 2007 and 2008, the last two years, exclude
zero.

To read these effects by time since the insurance arrived and to get one overall
number, `type="eventstudy"` in {func}`~moderndid.agg_ddd` lines the years up by
event time. It also replaces the intervals with simultaneous bands that cover
every event time at once.

```{code-cell} ipython3
# Line the years up by time since the insurance arrived and average them into an overall effect.
event_study = did.agg_ddd(result, type="eventstudy", biters=999, random_state=7)
print(event_study)
```

The estimates are the same as in the table above, since event time $e$ is just
the year $2003 + e$. The bands, however, now cover all eight estimates together
with 95 percent probability and come out wider as a result. Under these bands,
only the effects at event times 4 and 5 exclude zero. The placebo band at event
time −3 now runs from −0.1268 to 0.0071 and just covers zero.

:::{admonition} One adoption year makes every aggregation agree
:class: note

Since every insured household got coverage in 2003, the `"simple"` and `"group"`
aggregations of {func}`~moderndid.agg_ddd` give the same overall effect as the
event study. With several adoption years they would weight the cohorts
differently, as the {ref}`staggered example <example_staggered_did>` shows.
:::

Plotted, the event study shows the placebo estimates rising into the base year
and the effects carrying on upward after it.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Event study with placebo estimates at event times -3 and -2 and effects at 0 to 5
---
# The dashed line at event time -1 marks 2002, the base year.
did.plot_event_study(event_study) + did.theme_moderndid()
```

## How much saving shifted

The overall effect at the top of the event study answers the central question.
It averages the effects of the six years after the insurance arrived. At 0.0652,
it says insured tobacco farmers kept about 6.5 percentage points more of their
net savings in checking accounts than they would have without the insurance. Its
95 percent interval runs from 0.0261 to 0.1044, entirely above zero. Like
everything here, it holds only if the four assumptions above do.

### Against the original regression

[Cai (2016)](https://doi.org/10.1257/pol.20130371) estimated this event study
with a three-way fixed effects regression. Besides the same two covariates, its
fixed effects absorb each household, each county in each year, and each crop in
each year. {ref}`Triple differences <background-tripledid>` explains how a
regression like this can go wrong once covariates enter.

```{code-cell} ipython3
import pyfixest as pf

# Count the years since 2003 for insured tobacco farmers, with -99 for every other household.
regression_data = data.with_columns(
    rel_time=pl.when((pl.col("treatment") == 1) & (pl.col("sector") == 1))
    .then(pl.col("year") - 2003)
    .otherwise(-99)
).to_pandas()

# Household, county-by-year, and crop-by-year fixed effects with errors clustered by household.
regression = pf.feols(
    "checksaving_ratio ~ i(rel_time, ref=-1) + hhsize + age | hhno + county^year + sector^year",
    data=regression_data,
    vcov={"CRV1": "hhno"},
)
```

The table and figure below line up both sets of estimates with 95 percent
pointwise intervals, as Figure 5 of Ortiz-Villavicencio and Sant'Anna (2025)
does. The last column divides the width of the regression's interval by the
width of the triple difference's.

```{code-cell} ipython3
:tags: [hide-input]

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
from scipy.stats import norm

z = norm.ppf(0.975)

# The regression's coefficients by event time, without the reference year and the -99 placeholder.
coefficients = regression.tidy()
coefficients = coefficients[coefficients.index.str.contains(r"\[T\.-?[0-5]\]")]
event_times = coefficients.index.str.extract(r"\[T\.(-?\d+)\]")[0].astype(int).to_list()
regression_rows = {
    e: (row["Estimate"], row["2.5%"], row["97.5%"])
    for e, row in zip(event_times, coefficients.to_dict("records"))
}

rows = []
print(f"{'event time':>10}{'triple diff':>14}{'regression':>13}{'width ratio':>14}")
for e, att, se in zip(event_study.egt, event_study.att_egt, event_study.se_egt):
    if e == -1:
        continue
    estimate, low, high = regression_rows[int(e)]
    rows.append(("triple difference", int(e), att, att - z * se, att + z * se))
    rows.append(("three-way fixed effects", int(e), estimate, low, high))
    print(f"{int(e):>10}{att:>14.4f}{estimate:>13.4f}{(high - low) / (2 * z * se):>14.2f}")

comparison = pl.DataFrame(rows, schema=["estimator", "event_time", "att", "low", "high"], orient="row")
dodge = position_dodge(width=0.3)
(
    ggplot(comparison, aes("event_time", "att", color="estimator", shape="estimator"))
    + geom_hline(yintercept=0, color="#7f8c8d")
    + geom_vline(xintercept=-1, linetype="dotted", color="#7f8c8d")
    + geom_errorbar(aes(ymin="low", ymax="high"), width=0.2, position=dodge)
    + geom_point(size=3, position=dodge)
    + scale_color_manual(values={"triple difference": "#315bc4", "three-way fixed effects": "#7f8c8d"})
    + scale_shape_manual(values={"triple difference": "^", "three-way fixed effects": "o"})
    + scale_x_continuous(breaks=list(range(-3, 6)))
    + labs(x="Years since the insurance arrived", y="Effect on the checking-account share", color="", shape="")
    + did.theme_moderndid()
)
```

Although only the triple difference dips in 2006, its blue triangles and the
gray circles of the regression rise together in the figure after 2003. They part
most at the
ends, where the regression's 2000 placebo is −0.0175 against −0.0599 and its
2008 effect is 0.1047 against 0.1553. On this data the regression's intervals
also come out narrower, between 0.74 and 0.95 times as wide as the triple
difference's.

:::{admonition} Narrower isn't the same as better
:class: note

The regression gets its narrower intervals by leaning on one linear adjustment
for the covariates. The doubly robust triple difference stays consistent when
either of its two models is right. On this data, that protection shows up as
somewhat wider intervals.
:::

## How sturdy the answer is

Every estimate so far rests on the one specification set up earlier. Each check
below changes a
single piece of it and reports what happens to the overall effect. The checks
run in the order that [Setting up the estimation](#setting-up-the-estimation)
introduced the choices.

### Two differences instead of three

The first check asks what the third difference buys over a plain DiD. The cell
below runs {func}`~moderndid.att_gt` with the same choices on two narrower
comparisons. The first compares tobacco farmers across counties and the second
compares both kinds of households inside the insured county.

```{code-cell} ipython3
# The same choices without the eligibility column, on two comparisons that each skip one difference.
did_spec = {name: value for name, value in spec.items() if name != "pname"}
inside_county = data.filter(pl.col("treatment") == 1).with_columns(
    group=pl.when(pl.col("sector") == 1).then(2003).otherwise(0)
)
comparisons = {
    "tobacco farmers across counties": data.filter(pl.col("sector") == 1),
    "both crops inside the insured county": inside_county,
}
two_differences = {}
for name, frame in comparisons.items():
    estimates = did.att_gt(frame, **did_spec)
    two_differences[name] = did.aggte(estimates, type="dynamic", random_state=7)
    print(f"{name:<38}{two_differences[name].overall_att:.4f}")
```

Comparing tobacco farmers across counties gives 0.1111, about 1.7 times the
triple difference's 0.0652. That comparison credits the insurance with whatever
lifted the share for every household in the insured county. Comparing the two
kinds of households inside that county gives 0.0679, close to the triple
difference. That closeness says the gap between tobacco farmers and other
households barely moved in the counties without insurance.

### Other estimators and no covariates

The second check revisits how the specification makes two households comparable
to each other. The cell below tries each single-model estimator in place of the
doubly robust one and then removes the covariates.

```{code-cell} ipython3
# Swap the estimator or drop the covariates, one change at a time.
changes = {
    "ipw": {"est_method": "ipw"},
    "reg": {"est_method": "reg"},
    "without covariates": {"xformla": None},
}
variants = {}
for name, change in changes.items():
    variant = did.ddd(data, **(spec | change))
    variants[name] = did.agg_ddd(variant, type="eventstudy", biters=999, random_state=7)
    print(f"{name:>18}  {variants[name].overall_att:.4f}  ({variants[name].overall_se:.4f})")
```

Each of the three swaps lowers the overall effect from 0.0652 to somewhere
between 0.0565 and 0.0608. The estimator and the adjustment for size and age
matter little here.
Outcome regression does give the least precise estimate of the three. Its
standard error of
0.0220 compares with 0.0200 for the doubly robust one.

### The trend before 2003

A universal base suits the event study but blurs how the gap moved from one year
to the next before 2003. Switching to `base_period="varying"` turns each
placebo into a one-year change and leaves every estimate from 2003 on untouched.

```{code-cell} ipython3
# The same specification under a varying base, so each placebo is a one-year change.
varying = did.ddd(data, **(spec | {"base_period": "varying"}))
varying_event_study = did.agg_ddd(varying, type="eventstudy", biters=999, random_state=7)
print(varying)
```

Relative to the same gap elsewhere, the gap between tobacco farmers and other
households in the insured county grew by 0.0304 from 2000 to 2001 and by 0.0264
from 2001 to 2002. The estimates from 2003 on match the ones under the
universal base. The overall effect therefore stays at 0.0652 under either
setting.

:::{admonition} A widening gap can pass for an effect
:class: warning

From −0.0599 in 2000 to zero in 2002, the gap widened by about 0.03 a year. At
that pace it would have reached 0.1797 by 2008 without any insurance, above the
estimate in every year after 2003. The answer on this page rests on that
widening having stopped when the insurance arrived.
:::

### Households seen every year

The last check drops the 361 households that miss a year and keeps the 3,298
seen in all nine. Each comparison then follows the same households in both of
its years.

```{code-cell} ipython3
# The same specification on the households observed in every year.
complete_households = households.filter(pl.col("years") == 9).get_column("hhno").to_list()
complete = data.filter(pl.col("hhno").is_in(complete_households))
balanced = did.ddd(complete, **(spec | {"allow_unbalanced_panel": False}))
balanced_event_study = did.agg_ddd(balanced, type="eventstudy", biters=999, random_state=7)
print(
    f"placebo in 2000 {balanced.att[0]:.4f}, "
    f"overall effect {balanced_event_study.overall_att:.4f} ({balanced_event_study.overall_se:.4f})"
)
```

On those households the overall effect falls from 0.0652 to 0.0429. Their 2000
placebo of −0.0564 shows the same widening before 2003. Dropping the households
that miss a year doesn't remove the trend.

### The checks side by side

The table below starts with our own specification and then collects the overall
effect and its 95 percent interval from every check.

```{code-cell} ipython3
:tags: [hide-input]

# Every check's overall effect and 95 percent interval, starting with our specification.
checks = {
    "our specification": event_study,
    "tobacco farmers across counties": two_differences["tobacco farmers across counties"],
    "both crops inside the insured county": two_differences["both crops inside the insured county"],
    "inverse probability weighting": variants["ipw"],
    "outcome regression": variants["reg"],
    "without covariates": variants["without covariates"],
    "varying base period": varying_event_study,
    "households seen every year": balanced_event_study,
}

print(f"{'check':<38}{'overall effect':>15}   [95% Conf. Interval]")
for name, check in checks.items():
    low = check.overall_att - z * check.overall_se
    high = check.overall_att + z * check.overall_se
    print(f"{name:<38}{check.overall_att:>15.4f}   [{low:8.4f}, {high:8.4f}]")
```

Read together, the checks separate the choices that matter from the ones that
don't. The estimator, the covariates, and the base period keep the overall effect
between 0.0565 and 0.0652. Keeping only the households seen every year lowers it
to 0.0429 without pushing its interval down to zero. Dropping the third
difference moves the answer the most, since comparing tobacco farmers across
counties alone gives 0.1111.

What none of these checks can settle is the widening gap before 2003. Every
number on this page leans on that trend having stopped when the insurance
arrived. Bounding how much a trend like that could change an event study is the
job of {ref}`sensitivity analysis <example_honest_did>`. The
{ref}`staggered example <example_staggered_did>` shows what changes when groups
adopt in different years. With only two periods, {func}`~moderndid.ddd` reports a
single effect instead of one per year. For repeated cross sections you pass
`panel=False`, as the {ref}`API reference <api-didtriple>` describes.
