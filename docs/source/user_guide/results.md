---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(working-with-results)=

# Working with estimates

An estimator's report can contain many effects even when your research
question calls for a single answer. In the minimum wage analysis, you might
want the effect in an adoption cohort's first year, the average across treated
county-years, or the employment response as time passes after adoption.
Choosing among those summaries changes both the population and the periods
your estimate describes.

We'll follow a minimum wage fit from its printed report into the result
object, the averages you can construct from it, and the tables you export.
This gives you a way to work with estimates before moving into a full
application. Other estimators return different fields, but you still need to
identify the effect being reported and understand how its uncertainty was
calculated.

## Fit the comparison you intend to report

Before averaging the effects, we need a fit that records the comparison and
inference choices we intend to use. This specification compares counties
whose states raised the minimum wage with counties whose states never raised
it during the observed years. It adjusts for pre-policy population and
clusters the bootstrap by state because counties in the same state share a
policy and may share employment shocks.

```{code-cell} ipython3
import moderndid as did
import numpy as np
import polars as pl

data = did.load_mpdta().with_columns(
    (pl.col("countyreal") // 1000).alias("state")
)

spec = dict(
    # Identify the outcome, period, county, and adoption year.
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    # Adjust for population using the never-treated comparison group.
    xformla="~lpop",
    control_group="nevertreated",
    est_method="dr",
    # Use the year before adoption as the untreated reference.
    base_period="universal",
    anticipation=0,
    # Account for state dependence in the bootstrap and bands.
    boot=True,
    biters=999,
    cband=True,
    clustervars=["state"],
    random_state=42,
)

result = did.att_gt(data=data, **spec)
print(result)
```

The doubly robust method combines outcome regression and propensity score
weighting to adjust for population. Its consistency still depends on
conditional parallel trends and no anticipation, alongside a correctly
specified model for at least one of those adjustments. The universal base
period expresses each cohort's comparisons relative to the year before
adoption. The {ref}`staggered adoption example <example_staggered_did>`
explains these choices and checks how the results change under other
specifications.

## Read the report and its result object

The report pairs each cohort's adoption year with an observation year.
For a post-treatment row, the estimate is the cohort's log employment effect
in that year relative to its estimated untreated path. A pre-treatment row
compares untreated outcome changes under the same base-period convention.
The zero rows immediately before adoption are imposed reference values, so
their standard errors appear as `NA`.

The confidence limits describe uncertainty around the estimates under our
state-clustered bootstrap. Here they form a 95 percent simultaneous band
over the group-time effects, rather than a separate pointwise interval for
each row. A star marks a band that excludes zero under those inference
choices. If a band's endpoints straddle zero, the estimate remains
compatible with effects of either sign at the chosen confidence level.

The returned {class}`~moderndid.did.container.MPResult` keeps the same rows
in its `groups`, `times`, `att_gt`, and `se_gt` arrays. To inspect those
rows as a table, you can convert the result to a Polars DataFrame.

```{code-cell} ipython3
group_time = did.to_df(result)
print(group_time.head())
```

The `group` and `time` columns identify the comparison alongside its
estimate in `att` and standard error in `se`. Because `ci_lower` and
`ci_upper` use the critical value stored in the result, the conversion
retains the fit's band convention.

Alongside the estimation settings, the result object keeps influence
functions that summarize how each unit contributes to sampling
variation in the estimates. Aggregation uses this information to account for
dependence between effects estimated from the same data. If you plan to
aggregate the effects or conduct sensitivity analysis later, keep the object
because a table of estimates leaves this information out.

## Choose the average for your question

Once you have read the cohort-year effects, {func}`~moderndid.aggte` can
summarize them without refitting the comparison. We choose its `type`
argument according to the question the average should answer.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``type``
     - Question
     - How the effects are averaged
   * - ``"dynamic"``
     - How does the effect vary with time since adoption?
     - Cohort-size weights among cohorts observed at each event time.
   * - ``"group"``
     - What is the average effect for each adoption cohort?
     - An average over its post-treatment periods, followed by cohort-size weights for the overall effect.
   * - ``"calendar"``
     - What is the average effect among treated units in a given year?
     - Cohort-size weights among cohorts already treated in that year.
   * - ``"simple"``
     - What is the average across all post-treatment cohort-period cells?
     - A weight proportional to cohort size for each included cell.
```

### Follow the response after adoption

For the employment question, an event study shows how the effect varies
after a state's minimum wage increase. Event time zero is the adoption
year, followed by positive values that count the years since adoption.

```{code-cell} ipython3
event_study = did.aggte(result, type="dynamic", random_state=42)
print(event_study)
```

The {class}`~moderndid.did.container.AGGTEResult` stores this curve in
`event_times`, `att_by_event`, and `se_by_event`. Its `overall_att`
averages the included post-treatment event-time estimates, giving each
exposure length equal weight. That summary can differ from an average
across treated counties because each horizon may include different cohorts.

Here the overall event-study average is -0.0804 log points. The estimates
at event times 2 and 3 are more negative than those in the first two
treated years, so they pull that average downward. Before reading this
as evidence of a growing employment response, we need to check which
cohorts remain in those later averages.

### Give each cohort an average over its treated years

To compare adoption cohorts, we first average each cohort's observed
post-treatment effects. The group aggregation then weights those cohort
averages by cohort size for its overall effect.

```{code-cell} ipython3
cohort_effects = did.aggte(result, type="group", random_state=42)
print(cohort_effects)
```

The cohort rows let you read each average before interpreting the overall
estimate of -0.0328 log points. The 2004 cohort's average of -0.0846 log
points covers four treated years, whereas the 2007 cohort's -0.0288 log
points covers only its adoption year. Since the panel ends in the same year
for every county, differences between these cohort averages can reflect
exposure length as well as differences between the counties in each cohort.

### Average across county-years or within calendar years

If your question concerns the effects across all observed treated
county-years, the simple aggregation gives each included cohort-year effect
a weight proportional to cohort size. Earlier adopters receive more total
weight because their counties contribute more treated years.

```{code-cell} ipython3
county_year_average = did.aggte(result, type="simple", random_state=42)
print(county_year_average)
```

The estimate of -0.0418 log points describes the average across treated
county-years. Its 95 percent interval of -0.0770 to -0.0065 log points
excludes zero under the state-clustered bootstrap, although the group
average's interval includes zero. Those reports answer different questions
because their weights assign different importance to earlier adopters.

For a question about the treated counties in a particular year, the calendar
aggregation instead averages across the cohorts already treated in that
year. Its overall effect then averages the included calendar-year effects.

```{code-cell} ipython3
calendar_effects = did.aggte(result, type="calendar", random_state=42)
print(calendar_effects)
```

The calendar report's -0.0442 log-point overall effect averages the effects
for 2004 through 2007. These reports summarize the same underlying
estimates using different weights, so choose the target before comparing
their numbers. A change in the population or exposure lengths you average
does not represent a change in the fitted comparison.

For other estimators, use the aggregation function or option documented for
that result. Triple differences uses {func}`~moderndid.agg_ddd`, continuous
DiD chooses an aggregation within {func}`~moderndid.cont_did`, and extended
TWFE uses {func}`~moderndid.emfx` to summarize its fitted regression. Those
result objects need their own aggregation methods rather than `aggte`.

## Track which cohorts contribute to an event study

The event-study averages need particular care because longer horizons are
often observed for fewer cohorts. In this panel, the 2007 cohort contributes
in its adoption year but has no observation one year afterward. Only the
2004 cohort is observed two or three years after adoption, so those parts
of the curve concern its counties. Because this cohort comes from Illinois,
only one treated state contributes to these later horizons. Their narrow
bands deserve caution even though they exclude zero in the report.

To compare the first two post-treatment years using the same cohorts, we
set `balance_e=1` to keep cohorts observed at event times zero and one.
The `min_e` and `max_e` arguments also limit the displayed estimates to
the reference year and those two treated years.

```{code-cell} ipython3
balanced = did.aggte(
    result,
    type="dynamic",
    balance_e=1,
    min_e=-1,
    max_e=1,
    random_state=42,
)
print(balanced)
```

This drops the 2007 cohort and changes the population described by the
curve. The adoption-year effect is now -0.0042 log points, compared with
-0.0211 log points when the 2007 cohort was included. Setting only `min_e`
and `max_e` would trim the event times without holding cohort composition
fixed. That distinction matters when you want to interpret movement along
the curve as a change with exposure.

This report labels its bands as pointwise even though the original fit
requests simultaneous bands. For this window and seed, the bootstrap
produces a simultaneous critical value below the usual pointwise critical
value, so the package falls back to pointwise inference. The displayed
label records that fallback; read the report's band type before describing
the uncertainty in your analysis.

## Interpret the units of the outcome

The outcome in this fit is log employment, so a negative post-treatment
estimate corresponds to lower teen employment relative to the estimated
untreated path. The estimate describes that comparison rather than the raw
change in a county's employment between two years.

For a log effect {math}`a`, the corresponding percentage change is
{math}`100(\exp(a)-1)`. Multiplying the log effect by 100 gives a close
approximation when its magnitude is small. We can apply the exact
transformation to the overall cohort average from the group report.

```{code-cell} ipython3
percent_effect = 100 * np.expm1(cohort_effects.overall_att)
print(f"Percentage change corresponding to the average log effect: {percent_effect:.2f}")
```

The group average corresponds to a 3.23 percent decline in employment
relative to the estimated untreated path. Averaging log effects and
averaging county percentage changes produce different summaries, so
describe this transformation as the percentage change corresponding to
the average log effect. For outcomes recorded as proportions, a change of
0.01 instead corresponds to one percentage point; a count outcome retains
the units of the count.

## Match the uncertainty to the comparison

The reports above carry the state-clustered bootstrap and inference
settings from the original fit. For `att_gt`, `boot=True` uses
multiplier bootstrap standard errors, while `cband=True` adds a simultaneous
critical value for the group-time effects. Simultaneous bands account for
examining a collection of effects, whereas pointwise intervals describe
each effect separately.

If you use the default `boot=False`, `att_gt` calculates analytical standard
errors. Although `cband=True` is also a default, its group-time bands use a
simultaneous critical value only when the bootstrap is enabled. Specify
both settings when the distinction matters to your analysis.

`aggte` inherits those inference settings unless you override them.
It can also draw a bootstrap for simultaneous bands when its standard
errors remain analytical, so give the aggregation a `random_state` as well
as the original fit to reproduce those draws. The overall scalar summary
has its own pointwise uncertainty; the curve's simultaneous critical value
applies to the collection of event-time estimates.

:::{admonition} Cluster at the level of dependence
:class: important

Specifying `clustervars` in `att_gt` does not account for an additional
clustering variable unless `boot=True`. With state clustering, the
analytical Wald pre-test is omitted because its covariance does not account
for dependence between counties in the same state.
:::

Pre-treatment bands can help assess the comparison without proving
post-treatment parallel trends. If you want to examine bounded departures
from that assumption, the {ref}`sensitivity analysis example
<example_honest_did>` explains the inputs that {func}`~moderndid.honest_did`
needs, including a suitable base period and consecutive event times.

To carry our state-clustered uncertainty into that analysis, use the
{ref}`external-estimate route <example_honest_did_external>` and supply a
covariance matrix that accounts for state clustering. The automatic
`honest_did` wrapper forms its covariance from unit influence functions and
does not incorporate the fit's additional state clustering.

## Export estimates and keep a reproducible specification

After choosing the summary you want to report, use {func}`~moderndid.to_df`
to export a supported result's estimate rows. We inspect the event-study
table before saving it because the available columns depend on the result
type.

```{code-cell} ipython3
estimates = did.to_df(event_study)
print(estimates)
```

The normalized reference at event time -1 has an undefined standard error,
so it is omitted from this table. A `type="simple"` aggregation contains
only a scalar effect and cannot be converted with `to_df`; read
`overall_att` and `overall_se` from its result instead.

Once you have checked the rows and their units, the DataFrame can be saved
to a CSV for use in another part of your analysis.

```python
estimates.write_csv("employment-event-study.csv")
```

Save the input data version, full specification, package version, and seeds
alongside the exported estimates. An integer seed reproduces a bootstrap
call under the same settings; changing the seed can change its standard
errors and bands without changing the underlying point estimates. A CSV
records the estimates but does not retain the influence functions needed
for later aggregation or sensitivity analysis.

The {doc}`plotting` guide draws these results and explains how to save a
figure. For publication tables that retain the estimator's labels and
uncertainty, continue with {doc}`publication_tables`.
