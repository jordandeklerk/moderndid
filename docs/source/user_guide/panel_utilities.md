---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(panel-utilities)=

# Preparing panel data

Before estimating a treatment effect, you need to know what an observation
represents and whether the same units appear throughout the study. Missing
years, duplicate records, and changes in treatment can each alter the
comparisons an estimator can make. The panel utilities help you inspect those
features before deciding how to handle them.

We will work with the county panel from Favara and Imbs (2015), where states
lifted restrictions on interstate bank branching between 1994 and 2005.
The data record the number of restrictions lifted and the growth of mortgage
lending in each county. A few counties have missing years, so this panel gives
you a chance to see how filling gaps differs from dropping incomplete units.
The {ref}`intertemporal treatment example <example_inter_did>` takes the same
data through an analysis of the deregulations' effects on lending.

Since functions that return data preserve your input's DataFrame format, you
can pass pandas, Polars, or another supported Arrow-compatible object without
converting it first, as described in {ref}`Data formats <data-formats>`.

(panel-utilities-diagnosing)=

## Reading the panel diagnostics

The first check is whether the county and year columns identify each row
uniquely and cover the years you expect. We use
{func}`~moderndid.core.panel.diagnose_panel` to collect those checks in a single report,
including whether the treatment column changes within counties.

```{code-cell} ipython3
import moderndid as did

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
The 38 missing county-year pairs belong to five counties, including one county
observed only once. There are no duplicate county-year pairs to resolve before
working with the panel.

Missing years and missing values affect the sample in different ways. The 524 rows
flagged here contain a null in at least one column of the supplied data.
An estimator's preprocessing checks the columns its specification uses, so
this broad diagnostic count need not equal the number of rows it removes.
In particular, balancing the county-year structure does not repair missing
values in the lending outcome.

The time-varying treatment flag is expected for `inter_bra`, since a state
can lift further restrictions after its first deregulation. That matters for
choosing an estimator as well as for cleaning the data. A treatment history
that rises from zero to one and then to two contains more information than a
single adoption date can represent.

(panel-utilities-gaps)=

## Deciding how to handle missing years

You can inspect the missing pairs with {func}`~moderndid.core.panel.scan_gaps` before
choosing whether incomplete counties should stay in the analysis. Filling
those gaps keeps every county and inserts a row for each missing year.

```{code-cell} ipython3
gaps = did.scan_gaps(data, idname="county", tname="year")
filled = did.fill_panel_gaps(data, idname="county", tname="year")
print(filled.shape)
```

The filled data have the expected 12,576 county-year rows, but the added rows
contain nulls outside the county and year columns. This representation can
help you inspect missingness or prepare a separate imputation procedure.

:::{admonition} Filling gaps does not impute outcomes
:class: warning

{func}`~moderndid.core.panel.fill_panel_gaps` inserts missing rows without estimating
their values. Even an estimator that allows an unbalanced panel still needs
observed outcomes for the comparisons it makes.
:::

If your analysis requires the same counties in every year,
{func}`~moderndid.core.panel.make_balanced_panel` instead keeps only counties observed
once in each of the 12 years. We apply it to the original data so inserted
rows cannot be mistaken for observed years.

```{code-cell} ipython3
balanced = did.make_balanced_panel(data, idname="county", tname="year")
print(balanced.shape)
```

The remaining 1,043 counties contribute 12,516 rows, or 12 years each.
This changes the population in your analysis by excluding all five incomplete
counties. Any nulls in their recorded outcomes or covariates remain a separate
issue, so check the variables your specification needs before estimation.

(panel-utilities-group-timing)=

## Recording the first treatment period

For an absorbing binary treatment, {func}`~moderndid.att_gt` needs to know
when each unit first became treated so it can form the cohort comparisons.
If your data record only a period-specific treatment indicator,
{func}`~moderndid.core.panel.get_group` builds that timing column by finding
the first period when the indicator is positive. The new `G` column uses
zero for units whose indicator is never positive, following the convention
in `att_gt`.

The same operation on `inter_bra` records when a county's state first lifted
any restriction. You can inspect those dates without changing the original
treatment history.

```{code-cell} ipython3
groups = did.get_group(
    data,
    idname="county",
    tname="year",
    treatname="inter_bra",
)
print(groups["G"].unique().sort().to_list())
```

These six adoption dates and the untreated group describe the timing of the
first deregulation. Those dates do not capture later increases in the number
of restrictions a state has lifted.
For the lending analysis, {func}`~moderndid.did_multiplegt` uses the full
period-specific dose rather than treating every positive dose as the same
absorbing policy.

If your indicator only labels the units that ever receive treatment and is
constant over time, the first positive row cannot reveal when treatment
began. For a common adoption date you already know, pass it as
`treat_period` to {func}`~moderndid.core.panel.get_group`. When adoption dates differ
across units, supply those dates from the policy records themselves.

(panel-utilities-inspection)=

## Checking specific features of the data

You may only need to check one aspect of a panel rather than print the full
report. The functions below check whether every county has every year and
whether a column changes within counties.

```{code-cell} ipython3
print(did.is_balanced_panel(data, idname="county", tname="year"))
print(did.has_gaps(data, idname="county", tname="year"))
print(did.are_varying(data, idname="county", cols=["inter_bra", "state_n"]))
```

The treatment varies within counties, while the state identifier stays fixed
as you would expect. If retaining counties with a few missing years is
appropriate for your design, {func}`~moderndid.core.panel.complete_data` lets you set
how many distinct years a county must have rather than require all 12.

```{code-cell} ipython3
trimmed = did.complete_data(data, idname="county", tname="year", min_periods=10)
```

Duplicate records require a decision about what the repeated rows mean.
{func}`~moderndid.core.panel.deduplicate_panel` can keep the last occurrence of each
county-year pair or average its numeric columns with `strategy="mean"`.
Averaging would also affect treatment values and numeric identifiers, so
choose it only when that operation makes sense for the records you have.

```{code-cell} ipython3
deduplicated = did.deduplicate_panel(
    data,
    idname="county",
    tname="year",
    strategy="last",
)
```

(panel-utilities-reshaping)=

## Reshaping data and inspecting changes

The estimators take long data, where each county-year pair has its own row.
If you need wide data for another part of your analysis,
{func}`~moderndid.core.panel.panel_to_wide` creates one row per county and one column
per year for every variable that changes over time. We keep just the lending
outcome and state identifier here so the return to long format is explicit.

```{code-cell} ipython3
lending = balanced.select("county", "year", "Dl_vloans_b", "state_n")
wide = did.panel_to_wide(lending, idname="county", tname="year")
long = did.wide_to_panel(
    wide,
    idname="county",
    stub_names=["Dl_vloans_b"],
    tname="year",
)
print(long.shape)
```

The year-specific lending columns are gathered back into the single
`Dl_vloans_b` outcome column.
Since `state_n` never changes within a county, it stays as a single column
in wide data and is repeated across that county's years in long data.

You can also inspect differences between consecutive recorded outcomes with
{func}`~moderndid.core.panel.get_first_difference`. In this data the outcome is already
a change in log lending, so its `dy` column measures the change in lending
growth rather than the change in lending levels.

```{code-cell} ipython3
differenced = did.get_first_difference(
    balanced,
    idname="county",
    yname="Dl_vloans_b",
    tname="year",
)
```

The first recorded year for each county has no preceding outcome and receives
a null difference. Using the balanced data keeps the recorded years adjacent;
on a panel with gaps, this function subtracts the previous observed outcome
even if more than one year has elapsed.

For repeated cross sections, {func}`~moderndid.core.panel.assign_rc_ids` provides a
unique `rowid` for every observation when you do not track the same units
over time. Those identifiers label observations without creating a panel;
{func}`~moderndid.att_gt` still needs `panel=False` for that design.
After these checks, the {ref}`estimator guide <estimator-overview>` helps you
choose a method that fits the treatment history in your data. The
{ref}`quickstart <quickstart>` shows how to pass the prepared columns to a
fit. Once you've estimated those effects, the {doc}`results guide <results>`
explains how to read the report and choose an aggregation.
