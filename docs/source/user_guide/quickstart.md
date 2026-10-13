---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(quickstart)=

# Your first analysis

This page carries one question from the county employment data that comes with
ModernDiD to a finished plot. The question is whether the states that raised
their minimum wage lost teen employment as a result. Since the states raised it
in different years, a single regression over the whole panel would compare some
newly treated counties with counties already treated. To keep those counties
out of the comparison, {func}`~moderndid.att_gt` compares each group of
counties, year by year, only with counties in states that had no increase.
You'll then average the resulting effects into one number with
{func}`~moderndid.aggte` and plot how the effect develops after the increases.

Before you start, follow the {ref}`installation guide <installation>` to install
the package and its plotting dependency. If the comparison behind
difference-in-differences is new to you, the
{ref}`DiD introduction <causal_inference>` explains how untreated counties stand
in for what the treated counties would have done without the policy.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Load the county data

{func}`~moderndid.load_mpdta` returns the panel as a Polars DataFrame in which
each row records one of 500 counties in one year from 2003 to 2007.

```{code-cell} ipython3
import moderndid as did

# Load the county panel and show its first rows.
data = did.load_mpdta()
data.head()
```

We'll use the log of each county's teen employment, `lemp`, as the outcome. The
covariate `lpop` holds each county's log population from 2000, before the first
increase. The column `first.treat` holds the first year in which a county's
state raised its minimum wage. A 0 there marks counties whose states didn't
raise it at any point in the panel. The
{ref}`data guide <data-for-estimation>` covers coding these columns for your own
study and passing a pandas DataFrame or an Arrow table instead.

## Estimate the cohort effects

A cohort groups the counties that saw their state's minimum wage rise in the
same year. {func}`~moderndid.att_gt` estimates a population-adjusted effect for
each cohort in each year of the panel. Since the
{ref}`staggered DiD example <example_staggered_did>` explains each choice in its
specification and tests the alternatives, we fit that same specification here.

```{code-cell} ipython3
spec = dict(
    # Name the columns for the outcome, year, county, and adoption year.
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    # Compare with never-treated counties after adjusting for population.
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    # Use the year before each cohort's adoption as its base.
    base_period="universal",
    anticipation=0,
    # Draw 10,000 seeded bootstrap samples for the standard errors and simultaneous bands.
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)

# Fit one comparison for every cohort and year.
result = did.att_gt(data, **spec)
print(result)
```

Each row of the report pairs a cohort in the Group column with a year in the
Time column. Rows before a cohort's adoption hold placebo estimates that should
sit near zero if the comparison counties trend like the treated ones. The zero
rows with NA standard errors are the base years that every other row is
measured against. Since every placebo band covers zero, the report shows no
evidence that trends differed before adoption. The pre-test of parallel trends
printed below the table gives a p-value of 0.2327. The
{ref}`results guide <working-with-results>` explains the rest of the report and
the result object behind it.

## Average the effects into one number

Most questions about the policy call for fewer numbers than this report holds.
Passing `type="group"` to {func}`~moderndid.aggte` first averages each cohort's
effects over its years after adoption. The overall effect then combines those
cohort averages in proportion to cohort size. We use this summary because every
treated county carries an equal weight in it.

```{code-cell} ipython3
# One average per cohort and a county-weighted overall effect.
overall = did.aggte(result, type="group")
print(overall)
```

The overall effect of −0.0328 log points puts the treated counties' teen
employment about 3.2 percent below the level that the comparison counties
imply for them. The 95 percent interval for the overall effect runs from −0.0567
to −0.0089.
[Match the uncertainty to the comparison](results.md#match-the-uncertainty-to-the-comparison)
in the results guide shows how that interval changes once the bootstrap stops
treating counties as independent and clusters them by state.

## Plot the event study

With `type="dynamic"`, `aggte` instead averages the effects by the number of
years since each state's increase. The plot of that event study puts the placebo
estimates and the effects after adoption on one axis.

```{code-cell} ipython3
# Average the effects at each number of years since adoption.
event_study = did.aggte(result, type="dynamic")

# Draw the event study on white panels without grid lines.
did.plot_event_study(event_study) + did.theme_moderndid()
```

Before adoption, the navy placebo estimates stay close to the zero line. After
adoption the red estimates fall below zero and reach their lowest values in the
second and third years after the increase.

Only the 2004 cohort adopted early enough for the panel to reach those years.
Since its 20 counties all lie in Illinois, those two points trace one state's
path rather than an average over the cohorts. The staggered DiD example [holds
the counties fixed](example_staggered_did.md#holding-the-counties-fixed) to
separate growth in the effect from that change in composition. When you're
ready to fit your own data, the {ref}`Fundamentals <fundamentals>` guides start
with {ref}`choosing an estimator <estimator-overview>` and the columns your
method needs.
