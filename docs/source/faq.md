(faq)=

# FAQ

## Preparing your data

### How should I code units that are never treated?

`gname` holds the period in which each unit is first treated, on the same scale
as `tname`. Give never-treated units a 0 there, as `first.treat` does in
[the staggered example](user_guide/example_staggered_did.md#the-data).
{func}`~moderndid.att_gt` also reads `np.inf` or any period after the last one
in your data as never treated. Since units treated in or before the first period
have no untreated year to compare against, `att_gt` drops them with a warning.

:::{admonition} A missing code removes the never-treated units
:class: warning

If never-treated units carry a null or `NaN` in `gname`, `att_gt` drops their
rows as missing data before it reads any codes. On mpdta the sample then shrinks
from 500 counties to 191 as the last cohort stands in for the comparison group.
Fill those values with 0 before you estimate, as in `pl.col("g").fill_null(0)`.
:::

### How do I turn a 0/1 treatment dummy into treatment timing?

{func}`~moderndid.core.panel.get_group` reads a 0/1 treatment indicator and adds
a `G` column holding the period each unit is first treated, or 0 if it never is.

```python
data = did.get_group(data, idname="unit_id", tname="year", treatname="treated")
result = did.att_gt(data, yname="outcome", tname="year", idname="unit_id", gname="G")
```

If your dummy marks treated units in every period, pass the known start date as
`treat_period` so that `get_group` doesn't read the first period as the start.
{func}`~moderndid.did_multiplegt` handles a treatment that can switch off again
and takes the dummy itself as `dname`.

### Which DataFrames and column types does moderndid accept?

moderndid accepts any DataFrame from a library that implements the Arrow PyCapsule
interface.
{ref}`The quickstart <quickstart-dataframes>` shows polars, pandas, and pyarrow
inputs side by side. A polars LazyFrame works
too once you call `.collect()` on it. If a pandas DataFrame keeps the unit or
period in its index, call `reset_index()` first, since only columns survive the
conversion. The id, time, outcome, and timing columns must all be numeric. Map
text ids to integers with `pl.col("id").rank("dense")` and turn dates into
integer periods such as `pl.col("date").dt.year()`.

### What happens to an unbalanced panel or rows with missing values?

By default {func}`~moderndid.att_gt` keeps only the units observed in every
period. Since it first drops every row with a missing value in a column the
model uses, one missing outcome removes a whole unit. Pass
`allow_unbalanced_panel=True` to keep the incomplete units, as in
{ref}`the quickstart <quickstart-panel-data>`. The estimates then come from the
repeated cross-section form of the estimator and lose some precision. Before you
estimate, {func}`~moderndid.core.panel.diagnose_panel` counts the gaps,
duplicates, and missing values that would cost you units.
{ref}`Panel data utilities <panel-utilities>` shows how to fill, drop, or
deduplicate those rows yourself.

### How do I estimate with repeated cross sections instead of a panel?

For repeated cross sections, set `panel=False` and leave `idname` out of the
call. The estimator then treats each row as its own observation and compares
cohort means across periods instead of changes within units. Don't switch to
`panel=False` just to get past an error on panel data. On mpdta that throws away
the pairing within counties and makes the standard error of the overall effect
13 times larger.

## Choosing an estimator and its settings

### Which estimator fits my design?

Pick the function by how your treatment behaves over time and across units.

- {func}`~moderndid.att_gt` fits a binary treatment that switches on for good at
  staggered times.
- {func}`~moderndid.etwfe` fits the same design as a regression and also handles
  binary and count outcomes.
- {func}`~moderndid.cont_did` fits a continuous dose.
- {func}`~moderndid.ddd` fits a policy that reaches only an eligible subgroup.
- {func}`~moderndid.did_multiplegt` fits a treatment that switches on and off or
  changes intensity.

{ref}`The estimator overview <estimator-overview>` compares them and adds
two-period designs, dynamic covariate balancing, and sensitivity analysis.
Neither the estimator of Sun and Abraham (2021) nor the imputation estimator of
Borusyak, Jaravel, and Spiess (2024) has its own function, since
{func}`~moderndid.etwfe` reproduces both. With `cgroup="never"`, its event study
matches the Sun and Abraham estimator. Its default cohort-time effects match
those of the imputation estimator, as {ref}`the ETWFE background
<background-etwfe>` shows.

### Should I compare with never-treated or not-yet-treated units?

By default {func}`~moderndid.att_gt` compares each cohort with the
never-treated units in your data. With `control_group="notyettreated"` it also uses cohorts that adopt
later, in the years before they do. Those extra comparisons can add precision
but extend parallel trends to the later adopters.
{ref}`The background <background-did-comparison-groups>` explains why
never-treated units are the safer choice when you have enough of them and they
resemble the treated ones. With no never-treated units at all, `att_gt` drops
every period from the last cohort's adoption on, since nothing is left to
compare against. Under the default it also makes that cohort the comparison
group and warns you.

### Why do `att_gt` and `etwfe` give different answers on the same data?

Most of the gap comes from their default comparison groups rather than from the
methods. With its default not-yet-treated comparison, {func}`~moderndid.etwfe`
followed by `emfx(type="simple")` gives −0.0477 on mpdta. Under its default
never-treated comparison, {func}`~moderndid.att_gt` with
`aggte(type="simple")` gives −0.0400, the same number `etwfe` gives with
`cgroup="never"`. {ref}`Extended TWFE <example_etwfe>` walks through the
comparison on the same county data.

### Should I use a varying or universal base period?

`base_period` changes only the estimates before adoption, since every effect
after adoption uses $g-1$ as its base under either setting. The default
`"varying"` makes each estimate before adoption a change over one period. Under
`"universal"` every period is measured against $g-1$ and the placebo estimates
sit on the same scale as the effects. Choose `"universal"` for an event study
you plot or set beside a regression event study. Always choose it before
{func}`~moderndid.honest_did`, since `honest_did` treats event time −1 as the
reference and drops whatever estimate sits there.
[The base period](user_guide/example_staggered_did.md#the-base-period) puts the
two settings side by side on the county data.

### What can I put in `xformla`?

{func}`~moderndid.att_gt`, {func}`~moderndid.ddd`, {func}`~moderndid.etwfe`, and
{func}`~moderndid.did_multiplegt` read `xformla` as plain column names joined by
`+`. Build any transformation, dummy, or interaction as its own column first,
such as `pl.col("pop").log().alias("lpop")`.

:::{admonition} Formula terms are dropped without a warning
:class: warning

The parser keeps only the column names inside `log()`, `C()`, `I()`, powers, and
interactions. On mpdta, for example, `"~ lpop + I(lpop**2)"` gives exactly the
same estimates as `"~ lpop"`. In the same way `"~ C(region)"` enters a numeric `region` code as one
covariate rather than a set of dummies. The two-period {func}`~moderndid.drdid`,
{func}`~moderndid.ipwdid`, and {func}`~moderndid.ordid` are the exception and
evaluate the full formula.
:::

### Can my covariates change over time?

In {func}`~moderndid.att_gt` each comparison reads covariates from the earlier
of its two periods, the base period for every effect after adoption. A covariate
that changes over time therefore enters at its value before treatment, even if
treatment changes it later. {ref}`Conditional parallel trends
<conditional-parallel-trends>` explains why the covariates should be fixed
before treatment in the first place.

### Does `est_method` matter if I have no covariates?

Without covariates, `"dr"`, `"ipw"`, and `"reg"` reduce to the same comparison
of mean changes and give identical estimates. Once you add covariates, `"dr"`
stays consistent if either the outcome model or the propensity score model is
right. That double protection is why `"dr"` is the default, as
{ref}`Doubly robust DiD <background-drdid>` explains in detail.

### What if the policy was announced before it took effect?

Set `anticipation` to the number of periods before adoption in which units might
respond. Each cohort's base period then moves back from $g-1$ to $g-1-k$ for
`anticipation=k`. Any response in those periods counts toward the treatment
effect instead of looking like a pre-trend. A cohort without an earlier base
period drops out, as Illinois does in
[Anticipation](user_guide/example_staggered_did.md#anticipation).

## Reading the results

### Why do my standard errors change every time I rerun the code?

While point estimates stay fixed, anything drawn from the multiplier bootstrap
moves between runs unless you seed it. {func}`~moderndid.aggte` draws a
bootstrap for its simultaneous bands even when `boot=False`. If you pass
`random_state` to {func}`~moderndid.att_gt`, every `aggte` call on the result
reuses it, as
[the staggered example](user_guide/example_staggered_did.md#choosing-a-specification)
does. Since {func}`~moderndid.agg_ddd` doesn't reuse the seed you gave
{func}`~moderndid.ddd`, give it its own `random_state`.
{func}`~moderndid.cont_did` bootstraps its dose-response standard errors even
with `boot=False` and needs a seed of its own too.

### How do I cluster standard errors by state?

Since {func}`~moderndid.att_gt` clusters inside the multiplier bootstrap,
`clustervars=["state"]` takes effect only with `boot=True`. Without the
bootstrap you get a warning and standard errors that ignore the clusters.
Besides `idname` you can name one more variable that stays constant within each
unit. You set the clusters once, since every {func}`~moderndid.aggte` call on
that result reuses them. The other estimators that cluster spell the option in their own way.

- {func}`~moderndid.etwfe` takes `vcov={"CRV1": "state"}`.
- {func}`~moderndid.did_multiplegt` takes `cluster="state"`.
- {func}`~moderndid.ddd` takes `cluster="state"` together with `boot=True`.

For now, {func}`~moderndid.agg_ddd` and {func}`~moderndid.cont_did` can't
cluster standard errors above the level of the unit.
[Clustering by state](user_guide/example_staggered_did.md#clustering-by-state)
clusters the county data by state and shows why the bands for a lone treated
state come out too narrow.

### Why is an estimate significant in the `att_gt` table but not in the event study?

Under the defaults, {func}`~moderndid.att_gt` prints pointwise intervals and
keeps simultaneous bands for `boot=True`. {func}`~moderndid.aggte` prints
simultaneous bands built on a bootstrap critical value above 1.96. On mpdta the
2007 cohort's 2004 estimate of 0.0305 carries a star in the `att_gt` table. At
event time −3 the same estimate gets a simultaneous band about a third wider and
loses its star. Since a simultaneous band covers every estimate in a table at
once with 95 percent probability, it suits scanning a table or plot for effects.
A pointwise interval suits a single estimate that you picked out before looking.
To make the two tables agree, pass `boot=True` to `att_gt` or `cband=False` to
`aggte`.

### Why do the simple, group, and dynamic aggregations give different overall effects?

Each `type` in {func}`~moderndid.aggte` weights the same group-time effects in
a different way. On mpdta the default {func}`~moderndid.att_gt` fit gives −0.0310
for `"group"`, −0.0400 for `"simple"`, and −0.0772 for `"dynamic"`. `"group"`
is the default and the usual answer to how large the effect was. It averages
each cohort's effects after adoption and weights the cohorts by size so that
every treated unit counts once. Although `"dynamic"` gives the event study, its
overall number weights event times equally and leans on the cohorts observed
longest.
[How much employment fell](user_guide/example_staggered_did.md#how-much-employment-fell)
works out each type's weights on the county data.

### Why are some group-time estimates missing?

When a cell's comparison can't be estimated, {func}`~moderndid.att_gt` names the
cell in a warning such as "Overlap condition violated for 2004.0 in time period
2005". Under the default varying base the failed cells drop out of the result.
Under `base_period="universal"` they stay as `NaN` and {func}`~moderndid.aggte`
stops with "Missing values at att_gt found". Pass `na_rm=True` to `aggte` to
aggregate the cells that did estimate. To recover the failing cells instead, try
fewer covariates for small cohorts or `control_group="notyettreated"` for more
comparison units.

### How do I get results into a DataFrame or a paper table?

Since every result is a `NamedTuple`, you can read fields such as `overall_att`
and `att_by_event` as attributes. {func}`~moderndid.to_df` turns a result into a
polars DataFrame with one row per estimate and columns for the estimate, its
standard error, and its band. For a `type="simple"` aggregation `to_df` raises
an error instead, since the single number already sits in `overall_att`.
{ref}`Publication tables <publication_tables>` shows how to combine several
results into one table for a paper.

## Installing and scaling

### Which extra do I need for a function?

The base install runs the core estimators, such as {func}`~moderndid.att_gt`,
{func}`~moderndid.ddd`, and {func}`~moderndid.did_multiplegt`, along with
{func}`~moderndid.npiv` and the panel utilities. Because the `gpu` extra needs
CUDA, `"moderndid[all]"` leaves it out and adds every other extra. A function
whose extra is missing raises an `ImportError` that names the install command,
such as `uv add 'moderndid[didcont]'` for {func}`~moderndid.cont_did`. If
{func}`~moderndid.etwfe` or {func}`~moderndid.diddynamic.dyn_balancing` raises a
plain `ModuleNotFoundError` instead, install the `etwfe` or `diddynamic` extra.
{doc}`Installation <getting_started/installation>` lists every extra along with
fixes for common install failures.

### How do I speed up estimation on a large panel?

On one machine, `n_jobs=-1` lets {func}`~moderndid.att_gt` and
{func}`~moderndid.ddd` estimate the group-time cells in parallel without
changing the results. Since the multiplier bootstrap is often the slowest step,
the `numba` extra and a smaller `biters` both help. With an NVIDIA GPU and the
`gpu` extra, `backend="cupy"` runs `att_gt`, `ddd`, and
{func}`~moderndid.cont_did` on the GPU. {ref}`GPU acceleration <gpu>` explains
why that pays off only once cells hold thousands of units.
