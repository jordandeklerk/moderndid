---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(data-for-estimation)=

# Data for estimation

Before fitting a difference-in-differences model, you need to tell the
estimator which outcome to study, which units to follow, and when treatment
begins. Those choices also determine which observations can enter the
comparison. Duplicate rows, inconsistent adoption dates, and missing outcomes
can change that comparison even when the data appears to be a complete panel.

We'll use the bundled minimum wage data to connect each data argument to the
observations it describes. The checks below cover the table's layout,
treatment timing, covariates, and sampling weights before we add a clustering
variable for inference. If your treatment varies in dose or applies only to
an eligible subgroup, the corresponding {doc}`example <../examples/index>`
shows the additional columns that design needs.

## Describe one observation per row

A county observed in five years contributes five observations to a panel
that follows the same units through time. ModernDiD's panel estimators take
data in long format, where each row records one unit in one period rather
than placing each year's outcome in a separate column. In the minimum wage
data, each row therefore describes one county in one year.

```{code-cell} ipython3
import moderndid as did
import polars as pl

data = did.load_mpdta()
print(data.head())
print(
    f"{data['countyreal'].n_unique()} counties observed "
    f"from {data['year'].min()} through {data['year'].max()}"
)
```

You can see that `countyreal` and `year` identify the county and year on
each row of this dataset. It follows 500 counties from 2003 through 2007 and
records log teen employment in `lemp`. Because that outcome is a logarithm,
an estimated effect measures log points rather than a number of jobs in the
county.

For a staggered adoption fit, the names in the call have these roles.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Argument
     - County data column
     - What the column records
   * - ``yname``
     - ``lemp``
     - The outcome whose change you want to explain.
   * - ``tname``
     - ``year``
     - The period of the observation.
   * - ``idname``
     - ``countyreal``
     - The unit followed across periods.
   * - ``gname``
     - ``first.treat``
     - The first treatment period for that unit, or zero if it is never treated.
```

Although those roles recur in several estimators, their treatment conventions
depend on the design. For example, {func}`~moderndid.did_multiplegt` uses
`dname` for a treatment value that can change over time. For
{func}`~moderndid.ddd`, you also need `pname` to record eligibility. Choose the design
with {doc}`estimator_overview` before adapting the column names.

(data-formats)=

## Use the DataFrame format you work with

Although the dataset loaders return polars DataFrames, you can use the
DataFrame format you already work with. The DataFrame-based estimators also
accept an eager pandas DataFrame or a pyarrow Table through the Arrow
PyCapsule interface, `__arrow_c_stream__`. We can convert the county data
to either format without changing its observations or their meaning.

```{code-cell} ipython3
pandas_data = data.to_pandas()
arrow_data = data.to_arrow()

print(pandas_data.shape)
print(arrow_data.shape)
```

Both representations retain the same rows and columns as the original data.
This applies to functions that take a `data` argument; lower-level
estimation functions that take arrays instead describe their inputs in the
API reference.

If a table is lazy or distributed, materialize it before passing it to an
estimator. Accepting an Arrow table does not mean estimation runs inside the
database or a distributed scheduler. The {doc}`computation guide <scaling>`
explains the execution options supported by the current package.

The outcome, time, cohort, and panel identifier columns should be numeric.
If your identifiers are strings, map them to numeric codes once across the
whole dataset and keep that mapping for later merges. Assigning codes
separately within each year can make different units share an identifier.

## Record treatment timing consistently

For {func}`~moderndid.att_gt`, `gname` identifies a unit's adoption cohort
by holding its first treatment date on every row, including its untreated
rows. The minimum wage data's `first.treat` therefore stays constant within
each county. A zero-one indicator that switches on during treated years
cannot provide the cohort information this argument needs.

You can inspect those dates using one row per county rather than counting
the same county once for every year.

```{code-cell} ipython3
counties = data.select("countyreal", "first.treat").unique()
print(counties.group_by("first.treat").len().sort("first.treat"))
```

The zero cohort contains the counties whose states never raised their
minimum wage during the observed years. Because each county belongs to one
adoption cohort, the other rows count counties by the year their state's
increase began.

With `anticipation=0`, `att_gt` drops units treated in or before the first
observed period because they have no observed untreated baseline. If outcomes
may respond before formal adoption, `anticipation` specifies how many earlier
periods can already be affected, and the untreated baseline must precede that
window. Shortening your panel or allowing a longer anticipation window can
therefore change which cohorts remain usable.

Time and adoption dates must use the same numeric scale. If you replace
calendar years by positive period indices, apply the same mapping to positive
adoption dates and keep zero for never-treated units. With irregularly spaced dates,
check how the estimator defines event time and anticipation; `att_gt` uses
differences in the numeric labels, whereas `cont_did` recodes observed
periods to consecutive positions.

## Check what will remain in the sample

An estimator can only use observations with the columns its comparison
requires. Before fitting, inspect duplicate unit-period pairs, missing
values, and the number of periods observed per unit. This small check counts
the duplicate pairs and reports null values in the county columns we need.

```{code-cell} ipython3
repeated_pairs = (
    data.group_by("countyreal", "year")
    .len()
    .filter(pl.col("len") > 1)
)
print(repeated_pairs)
print(data.select("countyreal", "year", "first.treat", "lemp", "lpop").null_count())

period_counts = data.group_by("countyreal").agg(periods=pl.col("year").n_unique())
print(period_counts.group_by("periods").len().sort("periods"))
```

The empty duplicate table and zero null counts show that these county
columns pass the checks. The final table also confirms that all counties
appear in each of the five years, so balancing the panel will not remove
any of them. For your own data, the {doc}`panel utilities <panel_utilities>`
provide fuller diagnostics and tools for inspecting gaps.
If duplicates appear, resolve them according to what each row means in your
study rather than keeping one arbitrarily. Null counts alone miss
floating-point infinities and NaNs. If your outcome construction can produce
them, check for those values too because preprocessing treats them as missing.

For `att_gt`, the default `allow_unbalanced_panel=False` drops units
missing any observed period after the required data have been cleaned.
Setting `allow_unbalanced_panel=True` keeps partially observed units and
uses repeated-cross-section estimation for an unbalanced sample with
unit-level contributions retained for inference. That choice requires the
repeated-cross-section conditions in the {doc}`two-period background <../background/drdid>`; it does not recover missing outcomes.

:::{admonition} Missing rows are not zero outcomes
:class: warning

Filling a panel gap creates a row without an observed outcome. Treating
that outcome as zero changes the data and can change the effect you
estimate. Decide how missing observations arose before using the panel
utilities to fill or remove gaps.
:::

If your study samples different units in each period, it is a repeated
cross-section and needs `panel=False`. Each row then represents a separate
observation in `att_gt` even if you also supply an identifier. Choose this
setting based on how your sample was collected rather than using it to
bypass a panel validation error.

## Choose covariates measured before treatment

Once treatment timing and the sample are settled, we need to decide which
characteristics make the treated and comparison counties comparable. In the
county data, `lpop` records log county population in 2000, before the policy
changes we study. Using that fixed measure adjusts for differences in county
size rather than population changes that could themselves respond to
treatment.

You pass those adjustment variables through `xformla`, as in
`xformla="~ lpop"`. For `att_gt`, the formula lists existing covariate
columns joined by `+` and includes an intercept; the outcome goes in
`yname` instead of on the formula's left-hand side. Leaving `xformla`
unset, or using `"~1"`, fits without covariates. If you need a transformation
or an interaction, create it as a column before naming it in the formula.

Covariate adjustment asks for parallel trends conditional on those
characteristics and enough overlap between treated and comparison units.
Adding more variables does not establish either condition or solve a lack of
comparable observations. In particular, a variable affected by treatment can
remove part of the effect you intended to estimate or make the comparison
depend on a treatment response.

On a balanced panel, each `att_gt` two-period comparison uses covariates
from the earlier of its two periods. A time-varying column therefore needs
careful interpretation, especially for pre-treatment comparisons whose
earlier period changes. The {ref}`staggered adoption example <example_staggered_did>` shows the population adjustment and then checks an
unadjusted comparison.

## Keep sampling weights and inference groups distinct

Sampling weights describe how each observation contributes to the population
you want your analysis to represent. In
estimators that accept `weightsname`, pass the name of the column containing
those weights. For `att_gt`, they must be finite and nonnegative with a
positive mean; leaving the argument unset gives observations equal weight.

If weights change within a balanced panel unit, `att_gt` uses the earlier
period's weights for each two-period comparison and issues a warning.
For a balanced panel, `aggte` uses each unit's first-period weight when
computing cohort shares. For an unbalanced panel, those shares instead use
each unit's mean weight across its retained rows.
Record that rule when a time-varying survey weight changes the population
represented by your analysis.

Clustering concerns dependence between observations rather than population
representation. Since states set the minimum wage in this application,
counties in the same state may share shocks. We derive the state identifier
from the county code before using it as the clustering variable.

```{code-cell} ipython3
data = data.with_columns((pl.col("countyreal") // 1000).alias("state"))
print(data.select("countyreal", "state").unique().sort("countyreal").head())
```

For `att_gt`, `clustervars=["state"]` needs `boot=True` to account for
that dependence. Other estimators expose different clustering arguments or
limitations that you should check in their API entry before transferring the setting.
The {doc}`results guide <results>` uses this prepared data to explain
aggregation and confidence bands in the context of the employment question.
