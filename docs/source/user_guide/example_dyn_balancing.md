---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_dyn_balancing)=

# Dynamic covariate balancing

Democracy is not a treatment that a country adopts once and keeps for good. Over
six consecutive years, 12 of the 141 countries in the data move into or out of
democracy at least once. A country's status in a given year may also respond to
how its economy fared in the years before. The question here is whether a
stretch of democracy leaves a country with higher GDP per capita than the same
stretch under autocracy would.

Switching and feedback both break the difference-in-differences designs you'd
usually reach for. A staggered design assumes that each country adopts democracy
once and stays a democracy. Its parallel trends assumption also fails when
countries choose democracy in response to their past growth. A plain comparison
of democracies with autocracies fails for a different reason, because the
democracies in this data were far richer before the years being compared.
Instead, we'll use {func}`~moderndid.diddynamic.dyn_balancing`, the dynamic
covariate balancing estimator of [Viviano and Bradic
(2026)](https://doi.org/10.1093/biomet/asag016). It weights countries one year
at a time so that democracies and autocracies look alike in past GDP per capita
and, where the data allows, in region. It then compares their GDP per capita in
the final year.

The answer you'll reach is the effect of two years of democracy, against two
years of autocracy, on GDP per capita in the last year of the panel. On the way,
you'll see how closely the weights match the two groups of countries and how the
effect changes as the stretch grows from one year to five. Each closing check
reruns the estimate with one choice changed to find out whether that choice can
move the answer.

```{code-cell} ipython3
:tags: [remove-cell]

import polars as pl
from plotnine import options
from prerun import stored

options.figure_size = (12, 5)
options.dpi = 100
pl.Config.set_tbl_rows(20)
```

## Democracy and growth across 141 countries

The data comes from the panel that
[Acemoglu, Naidu, Restrepo, and Robinson (2019)](https://doi.org/10.1086/700936)
assembled to study whether democracy causes growth. Its outcome, log GDP per
capita, turns a gap between two groups of countries into a rough percent
difference in income. What suits the data to this method is how that study
treats democracy. It takes a country's democracy status in a given year to be as
good as random once its past GDP per capita and democracy are accounted for.
That assumption, known as sequential ignorability, lets democracy respond to how
the economy did in the past while ruling out hidden factors that drive both.

{func}`~moderndid.load_acemoglu` loads the data with one row for each country in
each year. The first country's six years show how the columns fit together.

```{code-cell} ipython3
import moderndid as did
import polars as pl

data = did.load_acemoglu()
data.select("Unit", "Time", "region", "D", "Y", "lag1.Value1", "lag2.Value1").head(6)
```

In each row, `Y` holds the country's log GDP per capita and `D` holds a 1 if the
country was a democracy that year. The row also records the year from 0 to 5 in
`Time` and one of seven world regions in `region`. The four columns from
`lag1.Value1` to `lag4.Value1` hold log GDP per capita one to four years
earlier. Because `lag1.Value1` in each of the rows above repeats `Y` from the
year before, you can also tell that the six years are consecutive.

Since only a few countries ever change status, we start by counting the
treatment histories that the last two years contain.

```{code-cell} ipython3
# Each country's democracy status in the last two years, how often it changed status,
# and its log GDP per capita in the year before the last two.
countries = data.group_by("Unit").agg(
    last_two_years=pl.col("D").sort_by("Time").tail(2).cast(pl.String).str.join(""),
    changes=pl.col("D").sort_by("Time").diff().abs().sum(),
    gdp_year_before=pl.col("Y").filter(pl.col("Time") == 3).first(),
    final_gdp_missing=pl.col("Y").filter(pl.col("Time") == 5).first().is_null(),
    region=pl.col("region").first(),
)
changed = (countries["changes"] > 0).sum()
print(f"{changed} of {countries.height} countries change status at least once")

countries.group_by("last_two_years").agg(
    countries=pl.len(),
    mean_gdp_year_before=pl.col("gdp_year_before").mean().round(1),
    final_gdp_missing=pl.col("final_gdp_missing").sum(),
).sort("last_two_years")
```

In the last two years, 91 countries were democracies throughout and 46 were
autocracies throughout, against three that became democracies and one that
stopped being one. Before those two years began, the democracies' mean log GDP
per capita stood at 8.3 against 7.1 for the autocracies. That head start of 1.2
log points is what a plain comparison would mistake for an effect of democracy.

Four countries, two on each of those main histories, have no GDP per capita
figure for the last year. Since the estimator drops them with a warning, the
main estimates rest on 137 countries. Splitting the same histories by region
shows where the autocracies are.

```{code-cell} ipython3
# The number of countries in each region that followed each history over the last two years.
countries.pivot(
    on="last_two_years",
    index="region",
    values="Unit",
    aggregate_function="len",
    sort_columns=True,
).fill_null(0).sort("region")
```

Every industrialized country (INL) and every South Asian one (SAS) was a
democracy in both years. The Middle East and North Africa (MNA) is the mirror
image, since 10 of its 11 countries were autocracies in both. Those gaps come
back later, when the weights try to match the autocracies to the full set of
countries.

## The effect of a stretch of democracy

With so few switches, the comparison worth making is between whole stretches of
status at the end of the panel. For the last year of the panel, $T$, the target
is the gap between two average potential outcomes,

$$
\text{ATE} = \mathbb{E}\big[Y_T(D_{1:T-2},\, 1,\, 1)\big]
- \mathbb{E}\big[Y_T(D_{1:T-2},\, 0,\, 0)\big].
$$

In that display, $Y_T(D_{1:T-2}, d_{T-1}, d_T)$ is the log GDP per capita a
country would have in the last year with status $d_{T-1}$ and $d_T$ in the last
two years. Because every country keeps its own democracy history from before
that two-year window, the comparison averages over those earlier years.

Since each country followed only one history, its outcome under the other is
never observed. Filling that outcome in from countries on the other history
rests on four assumptions.

- A country's GDP per capita doesn't respond to its democracy status before that
  status takes effect.
- In each of the last two years, democracy is as good as random among countries
  that share a region, their GDP per capita in the four previous years, and any
  earlier status in the window.
- In each year of the window, a country's expected GDP per capita in the last
  year under either history is linear in its covariates and earlier status.
- Every country had some chance of being a democracy and some chance of being an
  autocracy in each year, given its covariates and earlier status.

Each assumption gets its formal statement in
{ref}`Dynamic covariate balancing DiD <background-diddynamic>`. That page also
shows how the estimator recovers both potential outcomes under them.

:::{admonition} A different assumption from parallel trends
:class: note

Nothing in these assumptions asks democracies and autocracies to follow parallel
paths in GDP per capita. The comparison relies instead on region and past GDP per
capita accounting for whatever ties a country's democracy to its growth.
:::

## Pinning down the comparison

Each of the few decisions that turn that target into an estimate rests on an
assumption of its own. We keep all of them in the dictionary below and group its
arguments by the question each one settles.
[Which choices move the answer](#which-choices-move-the-answer) comes back to
each decision in the order set here.

```{code-cell} ipython3
# Every choice sits in this dictionary so that each check can swap out one of them.
spec = dict(
    # The outcome, year, country, and democracy columns.
    yname="Y",
    tname="Time",
    idname="Unit",
    treatment_name="D",
    # Two years of democracy against two years of autocracy at the end of the panel.
    ds1=[1, 1],
    ds2=[0, 0],
    # Hold four years of past GDP per capita and the region fixed.
    xformla="~ lag1.Value1 + lag2.Value1 + lag3.Value1 + lag4.Value1",
    fixed_effects=["region"],
    # A lasso projection for GDP per capita with adaptive balance.
    method="lasso_plain",
    regularization=True,
    balancing="dcb",
    adaptive_balancing=True,
    # Analytic standard errors with Gaussian intervals.
    robust_quantile=False,
)
```

### Two years at the end of the panel

The comparison starts from the two treatment histories it sets side by side.
With `ds1=[1, 1]` and `ds2=[0, 0]`, those are two years of democracy and two
years of autocracy. Two years is long enough for the dynamic part of the method
to matter. In the second year, the weights must match the weighted averages from
the first year rather than the plain averages of the whole sample. The window is
also short enough that 91 and 46 countries followed the two histories. Histories
that switch status inside the window, such as autocracy followed by democracy,
hold too few countries to compare.

:::{admonition} Count histories back from the final period
:class: important

{func}`~moderndid.diddynamic.dyn_balancing` lines up the last entry of `ds1` and
`ds2` with `final_period`, the last year by default, and counts back one period
per entry. That's why the time column must hold consecutive integers, as `Time`
does here.
:::

[Longer stretches of democracy](#longer-stretches-of-democracy) repeats the
comparison for stretches of one to five years.
[Stacking the earlier windows](#stacking-the-earlier-windows) instead adds every
earlier two-year window to the data.

### Four years of past GDP per capita and the region

Since democracy may respond to how the economy did, `xformla` holds fixed each
country's log GDP per capita in the four previous years. Four years follows the
preferred specification of the original study, as Viviano and Bradic (2026) do.
`fixed_effects=["region"]` adds a dummy for each of the seven regions on top of
them. Holding these fixed assumes that nothing else drives both democracy and
growth. [Leaving out past GDP per capita](#leaving-out-past-gdp-per-capita)
shows what happens when only the region stays.

:::{admonition} Write dotted names as they are
:class: tip

`xformla` reads dotted names such as `lag1.Value1` as plain column names joined
by `+`. Since a transformation such as `log(x)` raises an error, add the
transformed values to the data as a column of their own.
:::

Unlike Viviano and Bradic (2026), this example doesn't hold democracy in the
four previous years fixed. Nearly every country had the same status in those
years as in the window's first year. Holding it fixed would therefore push the
autocracy weights onto the handful of autocracies with a democratic past.
Leaving it out relies instead on past GDP per capita capturing whatever earlier
democracy did to the economy.

:::{admonition} A covariate that democracy may have moved
:class: note

In the second year, `lag1.Value1` holds a GDP per capita that democracy in the
first year may already have moved. The second year's weights only compare
countries that shared the first year's status. Balancing them on that outcome
therefore adjusts for which countries kept their first-year status without
removing the first year's effect.
:::

### A lasso projection with adaptive balance

The first of the estimator's two stages fits an outcome model for each year.
Working backward from the last year, it projects GDP per capita onto everything
known about a country by then. With `method="lasso_plain"`, each projection fits
one linear model to all countries at once. The democracy indicators enter it as
regressors that the lasso leaves unpenalized. With `regularization=True`,
cross-validation picks a single penalty for the four lags and the region dummies
alike.

That shared model assumes that the effects of democracy add up the same way in
every country. The fully interacted model of `method="lasso_subsample"` drops
that assumption by fitting each year only on the countries still on the history.
Its fits rest on fewer countries with every year the history adds. Viviano and
Bradic (2026) use the linear model in their own analysis of the democracy panel.

In the second stage, `balancing="dcb"` makes each year's weights as even as
possible. They sum to one, stay between zero and a cap, give nothing to
countries off the history, and keep each covariate's weighted mean close to a
target. For the first year, that target is the mean over all countries. For the
second, it's the mean under the first year's weights.

:::{admonition} Why inverse probability weights fail here
:class: note

If you switch to `balancing="ipw"` or `"aipw"`, the call stops with an error on
this data because the logistic model for the propensity score has no estimate.
Since no industrialized or South Asian country was an autocracy in either year,
those two region dummies predict democracy perfectly in the first year. Without
region, past GDP per capita still singles out the one country that left
democracy in the second year.
:::

With `adaptive_balancing=True`, the tolerance on each covariate's distance from
its target stays tight for covariates the lasso kept and widens for the rest
until weights can be found.
[Swapping the outcome model or the tolerance](#swapping-the-outcome-model-or-the-tolerance)
tries the fully interacted model, a nearly unpenalized ridge fit in place of the
lasso, and one tolerance for every covariate.

### Analytic standard errors with Gaussian intervals

The standard errors come from the analytic variance of the estimator. Because
that variance treats each country's first-year covariates as fixed, it measures
uncertainty about the effect for these particular countries. With
`robust_quantile=False`, the 95 percent interval uses the Gaussian critical
value that the inference theorem of Viviano and Bradic (2026) supports.
[Two other measures of uncertainty](#two-other-measures-of-uncertainty) tries
the larger chi-squared critical value and standard errors clustered by region.

{func}`~moderndid.diddynamic.dyn_balancing` then runs both stages for each
history under `spec` and reports the gap between the two estimates.

```{code-cell} ipython3
# Estimate mean GDP per capita in the last year under each history and the gap between them.
result = did.dyn_balancing(data, **spec)
print(result)
```

## Two years of democracy against two of autocracy

According to the table that opens the report, two years of democracy change log
GDP per capita in the last year by −0.0244 compared with two years of autocracy,
about 2.4 percent less. Since the 95 percent interval from −0.0639 to 0.0151
covers zero and the p-value is 0.2259, the data gives no reason to reject an
effect of zero.

Below the table, `mu(ds1)` estimates the mean log GDP per capita of all 137
countries in the last year at 7.8179 had every one of them been a democracy in
both years. Under two years of autocracy, `mu(ds2)` puts the same mean at
7.8423. The effect of −0.0244 is the gap between those two means.

### How closely the weights match the two groups

Each set of weights tries to make the countries on its history look like all 137
countries in every covariate. The `imbalances` field of the result measures how
far each one falls short, as the gap between a covariate's weighted mean and its
target in standard deviations of the covariate. Below, you'll see the largest
imbalance the democracy weights leave, followed by the first-year imbalances of
the autocracy weights next to whether the lasso kept each covariate.

```{code-cell} ipython3
# The largest imbalance the democracy weights leave on any covariate in either year.
democracy_imbalance = result.imbalances["ds1"]["imbalance"].abs().max()
print(f"largest imbalance under the democracy weights: {democracy_imbalance:.4f}\n")

# The autocracy weights' first-year imbalances next to the covariates the lasso kept.
# The first year's lasso fit sits at index 0 and starts with its intercept.
first_year = result.imbalances["ds2"].filter(pl.col("period") == 1)
kept = result.coefficients["ds2"][0][1:] != 0
print(f"{'covariate':<14}{'imbalance':>10}   kept by the lasso")
rows = zip(first_year["covariate"], first_year["imbalance"], kept)
for covariate, imbalance, in_model in rows:
    print(f"{covariate:<14}{imbalance:>10.4f}   {'yes' if in_model else 'no'}")
```

The democracy weights match every covariate to within 0.0013 standard deviations
in both years. The autocracy weights come close on past GDP per capita too,
where the largest gap is −0.0150 for `lag4.Value1`. The region dummies are where
the autocracy weights fall short of their targets. INL sits at −0.4707 and SAS
at −0.1939 because neither region had an autocracy in either year for the
weights to draw on. MNA at 0.6870 and EAP, East Asia and the Pacific, at 0.5105
end up overrepresented instead.

Those gaps are allowed to stand because the lasso kept only `lag1.Value1`.
Adaptive balancing held that one covariate to the tight tolerance and let the
tolerance on every other covariate, the three older lags included, widen until
weights could be found. The older lags still come out well balanced because
they move almost in step with `lag1.Value1`.

:::{admonition} Overlap fails for INL and SAS countries
:class: warning

The autocracy weights give INL and SAS countries no weight at all. Their GDP per
capita under autocracy comes from the outcome model alone. That estimate holds
only if region really adds nothing once the previous year's GDP per capita is
known, as the lasso found.
:::

## Longer stretches of democracy

If democracy raises GDP per capita only slowly, two years may be too short to
show it. Passing `histories_length` makes
{func}`~moderndid.diddynamic.dyn_balancing` repeat the comparison for each
length in the list. A run of length $h$ keeps only the last $h$ entries of `ds1`
and `ds2` and estimates the target above with the last $h$ years in place of
the last two.

```{code-cell} ipython3
# One to five years of democracy against as many of autocracy, all ending in the last year.
history = did.dyn_balancing(
    data, **(spec | {"ds1": [1] * 5, "ds2": [0] * 5, "histories_length": [1, 2, 3, 4, 5]})
)
print(history)
```

One year of democracy changes log GDP per capita by −0.0021. Longer stretches
pull the estimate down to −0.0312 at five years. The standard error doubles from
0.0149 at one year to 0.0297 at five. Each extra year adds another round of
weights and another term to the variance.

The figure that {func}`~moderndid.plots.plot_dyn_balancing_history` draws from
these estimates lets you watch the bars widen with each added year while the
points stay close to zero.

```{code-cell} ipython3
---
mystnb:
  image:
    alt: Effects of one to five years of democracy on log GDP per capita with 95 percent intervals
---
# Bars mark each length's 95 percent interval and the dashed line an effect of zero.
did.plot_dyn_balancing_history(history) + did.theme_moderndid()
```

Since every bar in the figure crosses zero, none of the five intervals excludes
it.

:::{admonition} Each length is a separate comparison
:class: note

Read the points as five separate answers rather than one effect unfolding over
time. Since each length draws on the countries that kept one status for that
long, the five estimates rest on slightly different groups of countries.
:::

## Which choices move the answer

An effect this close to zero could still hide one that another reasonable
specification would find. The checks below alter one choice in `spec` apiece
and keep the others.

### Stacking the earlier windows

Nothing forces the comparison to use only the last two years of the panel. With
`pooled=True`, every earlier two-year window joins the comparison as a separate
country history, from the window ending in the second year of the panel to the
one ending in the last. Because the windows end in different years, the check
also adds `Time` to `fixed_effects` to give each year its own dummy, as the
pooled regression of Viviano and Bradic (2026) does. A shock that hits every
country in the same year can then be absorbed by that year's dummy.

```{code-cell} ipython3
:tags: [skip-execution]

# Stack every earlier two-year window as its own country history and give each year a dummy.
pooled = did.dyn_balancing(data, **(spec | {"pooled": True, "fixed_effects": ["region", "Time"]}))
```

```{code-cell} ipython3
:tags: [remove-cell]

pooled = stored(
    "dyn_balancing_pooled",
    lambda: did.dyn_balancing(
        data, **(spec | {"pooled": True, "fixed_effects": ["region", "Time"]})
    ),
)
```

```{code-cell} ipython3
print(pooled)
```

The report now counts 141 countries and 701 stacked histories, since the four
countries missing the last year's figure still contribute their earlier windows.
The effect moves to −0.0158 and its interval narrows to run from −0.0404 to
0.0089. Because the same country appears in several windows, the standard errors
now cluster by country. Pooling buys that precision by assuming that the
two-year effect and the outcome model are the same in every window apart from
each year's own intercept.

### Leaving out past GDP per capita

To see how much of the answer past GDP per capita carries, this check drops the
four lags from `xformla` and keeps only the region dummies. Without past GDP per
capita, the weights can match democracies and autocracies on region alone.

```{code-cell} ipython3
# The same specification without past GDP per capita, so only the region is balanced.
without_past_gdp = did.dyn_balancing(data, **(spec | {"xformla": None}))
print(f"effect {without_past_gdp.att:.4f}, standard error {without_past_gdp.se:.4f}")
```

The effect jumps from −0.0244 to 0.3067, or about 36 percent more GDP per capita
for the democracy history. Its standard error rises from 0.0201 to 0.2153
because the outcome model no longer knows any past GDP per capita.

:::{admonition} Balancing only region keeps the head start
:class: warning

The democracies were already 1.2 log points richer before the last two years
began. A comparison that balances only region carries the part of that head
start that region doesn't explain into the estimate of 0.3067.
:::

### Swapping the outcome model or the tolerance

This check turns to the settings that shape each of the estimator's two stages.
Below our specification, the fully interacted model, a nearly unpenalized ridge
fit, and a single tolerance for every covariate each take a row of their own.

```{code-cell} ipython3
# The two-year effect when only the outcome model or the balance tolerance changes.
changes = {
    "our specification": {},
    "lasso_subsample": {"method": "lasso_subsample"},
    "ridge": {"regularization": False},
    "one tolerance": {"adaptive_balancing": False},
}
variants = {}
for name, change in changes.items():
    variant = did.dyn_balancing(data, **(spec | change))
    variants[name] = variant
    # The worst match on past GDP per capita that the autocracy weights leave in either year.
    past_gdp = variant.imbalances["ds2"].filter(pl.col("covariate").str.starts_with("lag"))
    print(
        f"{name:>17}  effect {variant.att:.4f}  standard error {variant.se:.4f}"
        f"  past GDP imbalance {past_gdp['imbalance'].abs().max():.4f}"
    )
```

Under these swaps the effect runs from −0.0241 to −0.0121, a spread smaller than
the standard error of 0.0201 in our specification. The fully interacted model
and ridge report smaller standard errors of 0.0125 and 0.0105. Because both fit
GDP per capita more closely than a lasso that keeps only `lag1.Value1`, their
residuals in the last year and the year-to-year changes in their projections
both come out smaller.

In the last column you can see what adaptive balancing protects. When one
tolerance applies to every covariate, the INL gap forces it wide enough to leave
past GDP per capita off by as much as 0.1220 standard deviations, against 0.0150
in our specification. Ridge leaves the same gap, since none of its coefficients
is zero and adaptive balancing therefore has no covariate to loosen.

### Two other measures of uncertainty

Only the measure of uncertainty differs under the two settings below. With
`robust_quantile=True`, the interval uses the square root of a chi-squared
critical value with twice as many degrees of freedom as there are years. With
`clustervars=["region"]`, the standard error lets countries in the same region
share shocks.

```{code-cell} ipython3
# The same estimate under a chi-squared critical value and under region clusters.
changes = {
    "chi-squared": {"robust_quantile": True},
    "region clusters": {"clustervars": ["region"]},
}
uncertainty = {}
for name, change in changes.items():
    estimate = did.dyn_balancing(data, **(spec | change))
    uncertainty[name] = estimate
    # robust_quantile holds the critical value behind each printed interval.
    low = estimate.att - estimate.robust_quantile * estimate.se
    high = estimate.att + estimate.robust_quantile * estimate.se
    print(
        f"{name:>16}  standard error {estimate.se:.4f}"
        f"  critical value {estimate.robust_quantile:.4f}  interval [{low:.4f}, {high:.4f}]"
    )
```

The chi-squared critical value of 3.0802 widens the interval to run from −0.0864
to 0.0377. Clustering by region raises the standard error from 0.0201 to 0.0358
and stretches the interval to span −0.0945 to 0.0457.

:::{admonition} Seven clusters make a fragile standard error
:class: warning

With seven regions, the clustered variance rests on seven cluster sums and has
no small-sample correction. Treat the larger standard error as a hint that
regional shocks could matter rather than as a measure of how much.
:::

### The spread across the checks

For our specification and every check, the table below gives the two-year
effect, its standard error, and its 95 percent interval.

```{code-cell} ipython3
:tags: [hide-input]

# The two-year effect, its standard error, and its 95 percent interval under each check.
checks = {
    "our specification": result,
    "stacked earlier windows": pooled,
    "without past GDP per capita": without_past_gdp,
    "fully interacted outcome model": variants["lasso_subsample"],
    "ridge outcome model": variants["ridge"],
    "one balance tolerance": variants["one tolerance"],
    "chi-squared critical value": uncertainty["chi-squared"],
    "clustered by region": uncertainty["region clusters"],
}

print(f"{'check':<32}{'effect':>9}{'std. error':>12}   [95% Conf. Interval]")
for name, check in checks.items():
    low = check.att - check.robust_quantile * check.se
    high = check.att + check.robust_quantile * check.se
    print(f"{name:<32}{check.att:>9.4f}{check.se:>12.4f}   [{low:8.4f}, {high:8.4f}]")
```

Whenever past GDP per capita stays among the covariates, the two-year effect
runs from −0.0244 to −0.0121 and every interval covers zero. The fully
interacted model comes closest to excluding zero, since its interval tops out at
0.0004. Only leaving out past GDP per capita moves the estimate much, to 0.3067
with an interval from −0.1153 to 0.7286.

Sequential ignorability given region and past GDP per capita still carries every
estimate on this page. A hidden factor that moved democracy and growth together
would bias all of them. If you'd rather assume parallel trends among countries
that start from the same status,
{ref}`intertemporal treatment effects <example_inter_did>` estimates the effects
of a treatment that switches on and off under that assumption.
