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

We'll use a minimum wage fit to compare those summaries and follow the
estimates from the printed report into the result object and exported tables.
Although other estimators return different fields, interpreting their reports
still means identifying the effect being reported and understanding how its
uncertainty was calculated.

## Start from the first analysis's fit

We start from a fit that keeps every setting of the
{ref}`first analysis <quickstart>`. It compares counties whose states raised the
minimum wage with counties whose states never raised it during the observed
years and adjusts for pre-policy population. The
{ref}`staggered DiD example <example_staggered_did>` explains these choices and
checks how the results change under other specifications.

```{code-cell} ipython3
import moderndid as did

# Load the county panel.
data = did.load_mpdta()

spec = dict(
    # Identify the outcome, period, county, and adoption year.
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    # Adjust for population using the never-treated comparison group.
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    # Use the year before adoption as the untreated reference.
    base_period="universal",
    anticipation=0,
    # Bootstrap 10,000 times with one weight per county and the first analysis's seed.
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)

# Estimate the group-time effects that every summary below starts from.
result = did.att_gt(data, **spec)
print(result)
```

## Read the report and its result object

The report pairs each cohort's adoption year with an observation year.
For a post-treatment row, the estimate is the cohort's log employment effect
in that year relative to its estimated untreated path. A pre-treatment row
compares untreated outcome changes under the same base-period convention.
Because the zero rows immediately before adoption are imposed reference
values, their standard errors appear as `NA`.

Because the fit uses the bootstrap with `cband=True`, the confidence limits
form a 95 percent simultaneous band that covers all the group-time effects at
once rather than row by row. Each report's column header names the kind of
band it shows. A star marks a band that excludes zero under
those inference choices. When a band's endpoints straddle zero, the estimate
remains compatible with effects of either sign at the chosen confidence level.
The p-value of 0.2327 printed below the table comes from a joint Wald test
that the five placebo estimates are all zero.

The returned {class}`~moderndid.did.container.MPResult` keeps the same rows
in its `groups`, `times`, `att_gt`, and `se_gt` arrays. To inspect those
rows as a table, you can convert the result to a Polars DataFrame.

```{code-cell} ipython3
# One row per cohort and year, as a Polars DataFrame.
group_time = did.to_df(result)
group_time.head()
```

Because `ci_lower` and `ci_upper` use the critical value stored in the result,
the conversion retains the fit's band convention.

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

Other estimators summarize their results with their own functions rather than
`aggte`. Triple differences uses {func}`~moderndid.agg_ddd`, continuous DiD
chooses an aggregation within {func}`~moderndid.cont_did`, and extended TWFE
uses {func}`~moderndid.emfx` on its fitted regression.

### Follow the response after adoption

For the employment question, an event study shows how the effect varies
after a state's minimum wage increase. Event time counts the years since a
state's increase and is zero in the adoption year itself.

```{code-cell} ipython3
# Line the cohorts up by years since adoption.
event_study = did.aggte(result, type="dynamic")
print(event_study)
```

The {class}`~moderndid.did.container.AGGTEResult` stores this curve in
`event_times`, `att_by_event`, and `se_by_event`. Its `overall_att` gives each
exposure length equal weight in an average of the included post-treatment
event-time estimates. That summary can differ from an average across treated
counties because each horizon may include different cohorts.

The estimates for event times 2 and 3 run more negative than those in the
first two treated years and pull the overall event-study average down to
−0.0804 log points. Before reading this as evidence of a growing employment
response, we need to check which cohorts remain in those later averages.

Since the panel ends in 2007, only the 2004 cohort is observed two and three
years after adoption. Because `balance_e=1` keeps only the cohorts observed at
event times zero and one, the estimates at those two event times then rest on
the same counties. The `min_e` and `max_e` arguments only trim the event window
and leave its cohort mix unchanged.
[Holding the counties fixed](example_staggered_did.md#holding-the-counties-fixed)
in the staggered DiD example shows how much the effect still grows once the
counties stay the same.

```{code-cell} ipython3
# Hold the cohorts fixed over event times 0 and 1 and show only that window.
balanced = did.aggte(result, type="dynamic", balance_e=1, min_e=-1, max_e=1)
print(balanced)
```

Once `balance_e=1` drops the 2007 cohort, the adoption-year estimate moves from
−0.0211 to −0.0042 log points.

### Give each cohort an average over its treated years

To compare adoption cohorts, we first average each cohort's observed
post-treatment effects. The group aggregation then weights the cohort
averages by cohort size to form its overall effect.

```{code-cell} ipython3
# Average each cohort over its treated years and weight the cohorts by size.
cohort_effects = did.aggte(result, type="group")
print(cohort_effects)
```

The cohort rows let you read each average before interpreting the overall
estimate of −0.0328 log points. The 2004 cohort's average of −0.0846 log
points covers four treated years. The 2007 cohort's average of −0.0288 log
points covers only its adoption year. Since the panel ends in the same year
for every county, differences between these cohort averages can reflect
exposure length as well as differences between the counties in each cohort.

### Average across county-years or within calendar years

If your question concerns the effects across all observed treated
county-years, the simple aggregation gives each included cohort-year effect
a weight proportional to cohort size. Earlier adopters receive more total
weight because their counties contribute more treated years.

```{code-cell} ipython3
# Weight every treated cohort-year effect by the size of its cohort.
county_year_average = did.aggte(result, type="simple")
print(county_year_average)
```

The simple estimate of −0.0418 log points sits further from zero than the
group average of −0.0328. The extra weight on earlier adopters goes mostly to
the 2004 cohort and its larger effects. The staggered DiD example works out
[why the other averages come out larger](example_staggered_did.md#why-the-other-averages-come-out-larger).

For a question about the treated counties in a particular year, the calendar
aggregation instead averages across the cohorts already treated in that
year. Its overall effect is the average of those calendar-year effects.

```{code-cell} ipython3
# Average the cohorts already treated in each calendar year.
calendar_effects = did.aggte(result, type="calendar")
print(calendar_effects)
```

The calendar report's −0.0442 log-point overall effect averages the effects
for 2004 through 2007.

### When some cells fail to estimate

A cohort-year comparison can fail when too few comparison counties remain or
when a cohort's covariates leave no comparable never-treated counties. For each
failed cell, {func}`~moderndid.att_gt` issues a warning that names its cohort
and year, as in "Overlap condition violated for 2004 in time period 2005".
With the universal base used here, the failed cells stay in the result as
`NaN` and print as NA in the report. Every `aggte` type then raises the error
"Missing values at att_gt found" rather than averaging around the gap.

Passing `na_rm=True` to `aggte` leaves the failed cells out and averages the
rest. If every 2004 cell had failed in this data, the group average would give
−0.0268 instead of −0.0328 because it would weight only the 2006 cohort's 40
counties and the 2007 cohort's 131. Since `aggte` drops any cohort, event time,
or year that the failures leave without an estimate, the event study would stop
at event time 1. For the same reason, the calendar report would start in 2006
rather than 2004.

:::{admonition} Dropping failed cells changes the average
:class: warning

The average that `na_rm=True` returns covers fewer cohorts or years than the
one you set out to report. Recovering a failed comparison takes a change to the
fit itself, such as fewer covariates or `control_group="notyettreated"`.
:::

Under the default varying base, `aggte` runs without `na_rm=True` because
`att_gt` leaves the failed cells out of the result. The warnings from `att_gt`
and the missing rows are then the only signs that an average covers less than
you intended. If the failures leave no effect after adoption to average,
`aggte` says so in a warning or an error rather than returning an estimate.

## Interpret the units of the outcome

Since the outcome in this fit is log employment, a negative post-treatment
estimate corresponds to lower teen employment relative to the estimated
untreated path. The estimate describes that comparison rather than the raw
change in a county's employment between two years.

For a log effect {math}`a`, the corresponding percentage change is
{math}`100(\exp(a)-1)`. Multiplying the log effect by 100 gives a close
approximation when its magnitude is small. Applying the exact transformation
to the overall cohort average gives a 3.23 percent decline in employment
relative to the estimated untreated path.

:::{admonition} Name the average you transformed
:class: tip

Averaging log effects and averaging county percentage changes produce
different summaries. Describe the 3.23 percent figure as the percentage change
that corresponds to the average log effect.
:::

For outcomes recorded as proportions, a change of 0.01 instead corresponds to
one percentage point. An effect on a count outcome stays in the units of the
count.

## Match the uncertainty to the comparison

If you use the default `boot=False`, `att_gt` calculates analytical standard
errors. Although `cband=True` is also a default, its group-time bands use a
simultaneous critical value only when the bootstrap is enabled. Specify
both settings when the distinction matters to your analysis.

`aggte` inherits those inference settings and the fit's `random_state` unless
you override them. That seed matters even under analytical standard errors,
since `aggte` can still draw a bootstrap for simultaneous bands. While the
overall scalar summary has its own pointwise uncertainty, the curve's
simultaneous critical value applies only to the collection of event-time
estimates.

Because the minimum wage is set state by state, treatment arrives for whole
states rather than for single counties. Clustering the bootstrap by state
matches the uncertainty to that assignment. When you pass
`clustervars=["state"]` to `att_gt`, every county in a state gets the same
bootstrap weight in the fit and in each summary that `aggte` computes from it.
[Clustering by state](example_staggered_did.md#clustering-by-state) in the
staggered DiD example runs that fit and shows why a cohort from a single
treated state still gets too narrow a band.

:::{admonition} Check clustered pre-trends with the bands
:class: important

Instead of a Wald pre-test, the report of a fit clustered by state points you to
the bootstrap's uniform bands over the pre-treatment cells.
:::

If you want to examine bounded departures from parallel trends, the
{ref}`sensitivity analysis example <example_honest_did>` explains the inputs
that {func}`~moderndid.honest_did` needs, such as a universal base period and
an event window. Since `honest_did` sums the influence function within the
fit's clusters before it forms the covariance, an event study from a
state-clustered fit keeps that clustering in the sensitivity analysis.

## Export estimates and keep a reproducible specification

After choosing the summary you want to report, use {func}`~moderndid.to_df`
to export a supported result's estimate rows. We inspect the event-study
table before saving it because the available columns depend on the result
type.

```{code-cell} ipython3
# The event study's estimated rows, ready to save.
estimates = did.to_df(event_study)
estimates
```

Because the normalized reference at event time −1 has an undefined standard
error, the table omits it. `to_df` cannot convert a `type="simple"`
aggregation, since that result holds only a scalar effect in `overall_att`
and its standard error in `overall_se`.

Once you have checked the rows and their units, the DataFrame can be saved
to a CSV for use in another part of your analysis.

```python
estimates.write_csv("employment-event-study.csv")
```

Save the input data version, full specification, package version, and seeds
alongside the exported estimates. The {ref}`plotting guide <plotting>` draws
these results and explains how to save a figure. For publication tables that
retain the estimator's labels and uncertainty, continue with the
{ref}`publication tables guide <publication_tables>`.
