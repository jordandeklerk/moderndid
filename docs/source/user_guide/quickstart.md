---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(quickstart)=

# Your first analysis

We'll start with the county employment data included in ModernDiD and carry
one fit through the tools you'll use in later analyses. The question is how
teen employment changed after state minimum wage increases. As you follow the
code, you'll see what {func}`~moderndid.att_gt` returns, how
{func}`~moderndid.aggte` summarizes those estimates, and how to display them
in an event study.

Before running the code, use {doc}`installation` to install the package and
the plotting dependency. If the logic of the comparison is new to you,
{doc}`causal_inference` explains how untreated outcomes help identify a
treatment effect.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Load the county data

{func}`~moderndid.load_mpdta` returns a Polars DataFrame containing 500 counties
observed from 2003 through 2007. Each observation occupies a single row for
its county and year.
Because the policy is set by states, we also derive a state identifier from
the county's FIPS code for the standard errors.

```{code-cell} ipython3
import moderndid as did
import polars as pl

data = did.load_mpdta().with_columns(
    (pl.col("countyreal") // 1000).alias("state")
)
print(data.head())
```

You can follow a county through the data using `countyreal` to identify it
and `year` to locate each observation in time. To study its employment
response, we'll use `lemp`, the log of teen employment, as the outcome.
The population measure in `lpop` records log county population in 2000,
before any of the increases in this sample, so it can enter the adjustment
without depending on a response to the policy.

To tell when a county's state raised its minimum wage, the estimator reads
`first.treat`. That adoption year appears on every row for the county,
including the years before the increase; a value of 0 identifies counties
whose states never increased their minimum wage during these years.
Although the separate `treat` column marks counties that ever received
treatment, it does not supply the timing we need for this comparison.

(quickstart-dataframes)=

### Use the data format you already have

Your own data can come from Polars, pandas, or an Arrow table that exposes the
Arrow PyCapsule interface. You pass the data directly to `att_gt` and name
the columns it should use. The {doc}`data` guide shows supported input formats
and explains how to encode treatment timing, covariates, and sampling weights.

## Estimate the cohort effects

Each cohort contains the counties whose state first increased its minimum wage
in the same year. The estimator reports a separate average treatment effect
for each cohort in each observed year so you can keep differences across
cohorts and exposure lengths visible.

```{code-cell} ipython3
spec = dict(
    # Identify the outcome, period, county, and adoption year.
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    # Compare with never-treated counties after adjusting for population.
    xformla="~lpop",
    control_group="nevertreated",
    est_method="dr",
    # Measure every year relative to the year before adoption.
    base_period="universal",
    anticipation=0,
    # Keep counties in the same state together in the bootstrap.
    boot=True,
    biters=999,
    clustervars=["state"],
    cband=True,
    random_state=42,
)

result = did.att_gt(data=data, **spec)
print(result)
```

The comparison assumes that counties of similar population would have followed
parallel employment trends without the minimum wage increases and did not
react to the policy before adoption. The doubly robust method combines
an outcome regression with propensity score weighting. Under these identifying
assumptions, its estimate is consistent if either of those two models is
correctly specified.

For each cohort, the universal base period sets the year before its increase
as the reference. Although `idname="countyreal"` identifies counties as the
panel units, the multiplier bootstrap keeps counties in the same state
together because they share the policy and may share employment shocks.
With `cband=True` and `alp=0.05`, the default significance level, the report
provides a 95 percent simultaneous band over its effects. Keeping the seed
fixed makes the bootstrap calculation reproducible when you rerun the same
analysis. Since the analytical covariance does not account for dependence
between counties in the same state, the report omits the Wald pre-test.

In the printed report, each row pairs an adoption year with an observation
year. Rows in and after the adoption year estimate effects on treated counties;
rows before adoption estimate placebo contrasts that help you assess the comparison.
The zero rows just before each cohort's adoption are the reference periods
we chose. Their standard errors display `NA` because those zeroes are imposed
by the normalization. Since `lemp` is a log outcome, the effects are in log
points. The
{doc}`results` guide explains the report, the stored estimates, and the
difference between pointwise intervals and simultaneous bands.

(quickstart-panel-data)=

### Panels and repeated cross-sections

Because this dataset observes every county in every year, it is balanced.
With your own panel,
`att_gt` drops units missing from any period unless you set
`allow_unbalanced_panel=True`. Repeated cross-sections instead use
`panel=False` because their rows don't follow the same units over time.
Before choosing a setting, use {doc}`data` to understand how these samples
differ and {doc}`panel_utilities` to inspect an incomplete panel.

## Average the effects for your question

The cohort effects let you ask several different questions about the same
policy. We'll first average them by time relative to adoption to see how
the estimated employment effects differ across exposure lengths.

```{code-cell} ipython3
event_study = did.aggte(result, type="dynamic", random_state=42)
print(event_study)
```

Event time puts each cohort on a common scale according to the years since
its state's first minimum wage increase. The adoption year is event time 0
and the following year is event time 1.
The negative event times show the placebo contrasts before adoption, except
for event time -1, the normalized zero reference with no estimated standard
error. Because each event time includes only the cohorts observed at that
horizon, the counties behind the average can change as exposure length
increases. The overall estimate at the top of this report averages the
post-treatment event-time estimates; it answers a different question from the
cohort-year average below.

To summarize the effects across the observed treated years, we'll also
calculate the `"simple"` aggregation. Each cohort-year effect receives a
weight proportional to its cohort's size. Earlier adopters therefore
contribute more to the overall average because they appear in more treated
years.

```{code-cell} ipython3
overall = did.aggte(result, type="simple", random_state=42)
print(overall)
```

The overall estimate of -0.0418 log points indicates lower teen employment in
the treated counties relative to their estimated employment without the
minimum wage increases. Its 95 percent interval runs from -0.0795 to -0.0040
log points under the state-clustered bootstrap we chose. This uncertainty
concerns the average across treated county-years rather than an effect shared
by every cohort. The
{doc}`results` guide also explains the `"group"` and `"calendar"`
aggregations so you can pick the average that answers your own question.

## Plot the event study

The event study plot puts the estimates and their bands on the same axis.
The plot function returns a plotnine object that you can display in a notebook
or save to a file.

```{code-cell} ipython3
plot = did.plot_event_study(
    event_study,
    ref_period=-1,
    title="Minimum wage increases and teen employment",
    xlab="Years relative to the first increase",
    ylab="Effect on log teen employment",
) + did.theme_moderndid()
plot
```

In this plot, a band that covers zero does not rule out a zero effect at that
horizon. The placebo points before adoption give you a check on the comparison,
although their proximity to zero cannot establish parallel trends after
adoption.

:::{admonition} Read the late horizons cautiously
:class: warning

Event times 2 and 3 come only from the 2004 Illinois cohort. Although their
bands exclude zero, inference at these horizons needs particular care because
only one treated state contributes to those estimates.
:::

With a first fit in hand, the {doc}`fundamentals` pages help you adapt the
analysis to your own study, beginning with {doc}`estimator_overview` and
the data your chosen method needs. Once those choices are familiar, the
{ref}`staggered DiD example <example_staggered_did>` examines the employment
results more closely and changes the specification to see how the conclusions
move.
