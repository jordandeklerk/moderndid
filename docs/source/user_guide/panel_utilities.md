---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(panel-utilities)=

# Preparing panel data

When a panel has missing years or duplicate records, the comparisons an
estimator can make may differ from those you intended. The panel utilities
help you check what each observation represents, whether the same units
appear throughout the study, and how treatment changes over time before
deciding how to handle the data.

We'll work with the county panel from Favara and Imbs (2015), where states
lifted restrictions on interstate bank branching between 1994 and 2005.
The data record the number of restrictions lifted and the growth of mortgage
lending in each county. Because some counties have missing years, we can use
this panel to compare filling gaps with dropping incomplete units.

Since functions that return data preserve your input's DataFrame format, you
can pass pandas, Polars, or another supported Arrow-compatible object without
converting it first, as described in {ref}`Data formats <data-formats>`.

(panel-utilities-diagnosing)=

## Reading the panel diagnostics

The first check is whether the county and year columns identify each row
uniquely and cover the years you expect. We use
{func}`~moderndid.core.panel.diagnose_panel` to collect those checks in a
single report that also shows whether the treatment column changes within
counties.

```{code-cell} ipython3
import moderndid as did

# Diagnose the county-year structure and whether inter_bra changes within counties.
data = did.load_favara_imbs()
diagnostics = did.diagnose_panel(
    data,
    idname="county",
    tname="year",
    treatname="inter_bra",
)
print(diagnostics)
```

A complete panel of 1,048 counties over 12 years would contain 12,576 rows.
Of the five counties behind the 38 missing county-year pairs, one is observed
only once. The report finds no duplicate rows to resolve before working with
the panel.

Missing years and missing values affect the sample in different ways. Of the
524 rows the report flags, 200 lack the lending outcome and the other 324 lack
only the house price change in `Dl_hpi`. A specification that leaves out
`Dl_hpi` is unaffected by those 324 missing values.

The time-varying treatment flag is expected for `inter_bra`, since a state
can lift further restrictions after its first deregulation. That matters for
{ref}`choosing an estimator <estimator-overview>` as well as for cleaning the
data. A treatment history that rises from zero to one and then to two contains
more information than a single adoption date can represent.

(panel-utilities-gaps)=

## Deciding how to handle missing years

You can inspect the missing pairs with {func}`~moderndid.core.panel.scan_gaps` before
choosing whether incomplete counties should stay in the analysis. Filling
those gaps keeps every county and inserts a row for each missing year.

```{code-cell} ipython3
# List the missing county-year pairs and insert a row for each of them.
gaps = did.scan_gaps(data, idname="county", tname="year")
filled = did.fill_panel_gaps(data, idname="county", tname="year")
gaps.sort("county", "year")
```

Although filling the 38 pairs that the scan returns gives the expected 12,576
county-year rows, the added rows contain nulls outside the county and year
columns. This representation can help you inspect missingness or prepare a
separate imputation procedure.

:::{admonition} Filling gaps does not impute outcomes
:class: warning

{func}`~moderndid.core.panel.fill_panel_gaps` inserts the missing rows without
estimating the values they should hold. Treating those missing outcomes as zero
would change the effect you estimate. Even an estimator that allows an
unbalanced panel still needs observed outcomes for the comparisons it makes.
:::

If your analysis requires the same counties in every year,
{func}`~moderndid.core.panel.make_balanced_panel` instead keeps only the counties
with exactly one row per year across all 12 years. We apply it to the original
data so inserted rows cannot be mistaken for observed years.

```{code-cell} ipython3
# Keep only the counties observed in every one of the 12 years.
balanced = did.make_balanced_panel(data, idname="county", tname="year")
```

The remaining 1,043 counties contribute 12,516 rows, or 12 years each.
This changes the population in your analysis by excluding all five incomplete
counties. Before estimation, check the outcome and covariates your
specification uses for missing values, since balancing the county-year
structure leaves those values in place.

(panel-utilities-group-timing)=

## Recording the first treatment period

For an absorbing binary treatment, {func}`~moderndid.att_gt` needs to know
when each unit first became treated so it can form the cohort comparisons.
If your data record only a period-specific treatment indicator,
{func}`~moderndid.core.panel.get_group` builds that timing column by finding
the first period when the indicator is positive. As in `att_gt`, the new `G`
column uses zero for units whose indicator is never positive.

The same operation on `inter_bra` records when a county's state first lifted
any restriction. You can inspect those dates without changing the original
treatment history.

```{code-cell} ipython3
# Record the first year each county's state lifted any restriction.
groups = did.get_group(
    data,
    idname="county",
    tname="year",
    treatname="inter_bra",
)
print(groups["G"].unique().sort().to_list())
```

The six adoption dates and the untreated group record only when each county's
state first lifted a restriction.

:::{admonition} Supply adoption dates for an ever-treated indicator
:class: important

If your indicator labels the units that ever receive treatment and is
constant over time, its first positive row cannot reveal when treatment
began. For a common adoption date you already know, pass it as
`treat_period` to {func}`~moderndid.core.panel.get_group`. When adoption dates
differ across units, supply those dates from the policy records themselves.
:::

(panel-utilities-inspection)=

## Checking specific features of the data

You may only need to check one aspect of a panel rather than print the full
report. The functions below check whether every county has every year and
whether a column changes within counties.

```{code-cell} ipython3
# Balance, gaps, and within-county variation, each as a single answer.
print(did.is_balanced_panel(data, idname="county", tname="year"))
print(did.has_gaps(data, idname="county", tname="year"))
print(did.are_varying(data, idname="county", cols=["inter_bra", "state_n"]))
```

While the treatment varies within counties, the state identifier stays fixed
as you would expect. If retaining counties with a few missing years is
appropriate for your design, {func}`~moderndid.core.panel.complete_data` lets you set
how many distinct years a county must have rather than require all 12.

```{code-cell} ipython3
# Keep the counties observed in at least ten of the 12 years.
trimmed = did.complete_data(data, idname="county", tname="year", min_periods=10)
```

The ten-year threshold keeps 12,527 rows, 11 more than the balanced panel,
since the county observed in 11 of the 12 years now stays.

Duplicate records require a decision about what the repeated rows mean.
{func}`~moderndid.core.panel.deduplicate_panel` can keep the last occurrence of each
county-year pair or average its numeric columns with `strategy="mean"`.

:::{admonition} Averaging duplicates also averages treatment
:class: warning

Choose `strategy="mean"` only when averaging makes sense for the records you
have, because it also averages treatment values and numeric identifiers.
:::

```{code-cell} ipython3
# Keep the last row of any repeated county-year pair.
deduplicated = did.deduplicate_panel(
    data,
    idname="county",
    tname="year",
    strategy="last",
)
```

With no duplicate county-year pairs in this panel, all 12,538 rows remain.

(panel-utilities-reshaping)=

## Reshaping data and inspecting changes

The estimators take long data, where each county-year pair has its own row.
If you need wide data for another part of your analysis,
{func}`~moderndid.core.panel.panel_to_wide` creates one row per county and one column
per year for every variable that changes over time. We keep just the lending
outcome and state identifier here so the return to long format is explicit.

```{code-cell} ipython3
# Spread the lending outcome across years and gather it back into one column.
lending = balanced.select("county", "year", "Dl_vloans_b", "state_n")
wide = did.panel_to_wide(lending, idname="county", tname="year")
long = did.wide_to_panel(
    wide,
    idname="county",
    stub_names=["Dl_vloans_b"],
    tname="year",
)
display(wide.head(3))
long.head(3)
```

Since `state_n` never changes within a county, it stays as a single column
in wide data and is repeated across that county's years in long data.

You can also inspect differences between consecutive recorded outcomes with
{func}`~moderndid.core.panel.get_first_difference`. Because the outcome in this
data is already a change in log lending, its `dy` column measures the change
in lending growth rather than the change in lending levels.

```{code-cell} ipython3
# Difference each county's lending growth from one recorded year to the next.
differenced = did.get_first_difference(
    balanced,
    idname="county",
    yname="Dl_vloans_b",
    tname="year",
)
differenced.select("county", "year", "Dl_vloans_b", "dy").head(3)
```

The first recorded year for each county has no preceding outcome and receives
a null difference. Every later difference spans one year, since the balanced
data keep the recorded years adjacent.

:::{admonition} Differences across a gap
:class: note

On a panel with gaps, `get_first_difference` subtracts the previous observed
outcome even if more than one year has elapsed.
:::

For repeated cross-sections, {func}`~moderndid.core.panel.assign_rc_ids` provides a
unique `rowid` for every observation when you do not track the same units
over time. {func}`~moderndid.att_gt` still needs `panel=False` for that
design, because those identifiers label observations without creating a panel.

The {ref}`intertemporal treatment example <example_inter_did>` uses
{func}`~moderndid.did_multiplegt` to take this county panel through an analysis
of the deregulations' effects on lending.
