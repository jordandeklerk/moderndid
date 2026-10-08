---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(faq)=

# FAQ

(faq-example-setup)=

The Python examples use one shared setup with the bundled panel of 500
counties observed from 2003 through 2007.
After running this cell, you can copy any recipe without running the
intervening answers.
Unless an answer changes a setting, the estimates use no covariates and
analytical standard errors. Each bootstrap example also sets its inference
options and uses an integer seed to reproduce its uncertainty.

```{code-cell} ipython3
import moderndid as did
import numpy as np
import polars as pl

data = did.load_mpdta()
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla=None,
    control_group="nevertreated",
    base_period="varying",
    boot=False,
    cband=False,
    random_state=42,
)
```

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
Only replace missing codes when you know those units are never treated.
:::

For a floating-point timing column, `pl.col("g").fill_null(0)` leaves `NaN`
values unchanged. After the {ref}`shared setup <faq-example-setup>`, you can
run this example to replace both kinds of missing code when they mean never
treated.

```{code-cell} ipython3
timing = pl.DataFrame({"g": [None, float("nan"), 2004.0, 2007.0]})
timing = timing.with_columns(pl.col("g").fill_nan(0).fill_null(0))
print(timing)
```

### How do I turn a 0/1 treatment dummy into treatment timing?

{func}`~moderndid.core.panel.get_group` reads a 0/1 treatment indicator and adds
a `G` column holding the period each unit is first treated, or 0 if it never is.

In the county data, `treat` labels counties that ever receive treatment rather
than the years when treatment is active. We construct a period-specific dummy
from the recorded adoption dates to demonstrate the conversion.

```{code-cell} ipython3
treated = (pl.col("first.treat") > 0) & (pl.col("year") >= pl.col("first.treat"))
with_dummy = data.with_columns(treated.cast(pl.Int64).alias("treated_now"))
with_timing = did.get_group(
    with_dummy, idname="countyreal", tname="year", treatname="treated_now"
)
county = with_timing.filter(pl.col("first.treat") == 2004)["countyreal"][0]
print(
    with_timing.filter(pl.col("countyreal") == county)
    .select("year", "treated_now", "G")
    .sort("year")
)
```

Although the dummy switches on in 2004, `G` records that adoption year on every
row for the county. You would then pass `gname="G"` to `att_gt` rather than
using `treated_now` as the cohort column.

If your dummy marks treated units in every period, pass the known start date as
`treat_period` so that `get_group` doesn't read the first period as the start.
{func}`~moderndid.did_multiplegt` handles a treatment that can switch off again
and takes the dummy itself as `dname`.

### Which DataFrames and column types does moderndid accept?

moderndid accepts any DataFrame from a library that implements the Arrow PyCapsule
interface.
{ref}`The data guide <data-formats>` shows how to prepare polars, pandas, and
pyarrow inputs. A polars LazyFrame works
too once you call `.collect()` on it. If a pandas DataFrame keeps the unit or
period in its index, call `reset_index()` first, since only columns survive the
conversion. The id, time, outcome, and timing columns must all be numeric. Map
text ids to integers with `pl.col("id").rank("dense")` and turn dates into
integer periods such as `pl.col("date").dt.year()`.

You can convert the bundled data to either of these supported formats without
changing its rows or columns.

```{code-cell} ipython3
pandas_data = data.to_pandas().reset_index(drop=True)
arrow_data = data.to_arrow()
```

Both representations keep the 2,500 county-year rows and six columns of the
original data.

### What happens to an unbalanced panel or rows with missing values?

By default {func}`~moderndid.att_gt` keeps only the units observed in every
period. Since it first drops every row with a missing value in a column the
model uses, one missing outcome removes a whole unit. Pass
`allow_unbalanced_panel=True` to keep the incomplete units, as in
{ref}`the quickstart <quickstart-panel-data>`. The estimates then come from the
repeated cross-section form of the estimator rather than paired outcome changes.
Before you
estimate, {func}`~moderndid.core.panel.diagnose_panel` counts the gaps,
duplicate unit-year pairs, and rows containing nulls. Its missing-value count
does not include floating-point `NaN` values.
{ref}`Panel data utilities <panel-utilities>` shows how to fill, drop, or
deduplicate those rows yourself.

To see how a gap affects the sample, we remove one county's 2005 observation
from a copy of {ref}`the setup data <faq-example-setup>` and inspect the
diagnostic counts.

```{code-cell} ipython3
county = data["countyreal"][0]
incomplete = data.filter(
    ~((pl.col("countyreal") == county) & (pl.col("year") == 2005))
)
diagnostics = did.diagnose_panel(incomplete, idname="countyreal", tname="year")
print(f"Missing county-year pairs: {diagnostics.n_gaps}")
print(f"Incomplete counties:      {diagnostics.n_unbalanced_units}")

for allow in [False, True]:
    fit = did.att_gt(incomplete, **(spec | {"allow_unbalanced_panel": allow}))
    print(f"allow_unbalanced_panel={str(allow):5}: {fit.n_units} counties retained")
```

The default removes the county with the missing year, whereas allowing an
unbalanced panel retains it for comparisons with the observations it has.
Keeping a county does not supply an outcome for its missing year.

### How do I estimate with repeated cross sections instead of a panel?

For repeated cross sections, set `panel=False` and leave `idname` out of the
call. The estimator then treats each row as its own observation and compares
cohort means across periods instead of changes within units. Don't switch to
`panel=False` just to get past an error on panel data. On mpdta that throws away
the pairing within counties and makes the standard error of the overall effect
about 13.5 times as large under the specification used here.

Applying `panel=False` to the county panel lets you see how discarding the
pairing changes the reported standard error of the simple average.

```{code-cell} ipython3
rc_spec = {key: value for key, value in spec.items() if key != "idname"}
rc_fit = did.att_gt(data, panel=False, **rc_spec)
rc_overall = did.aggte(rc_fit, type="simple", cband=False)
print(rc_overall)

panel_fit = did.att_gt(data, **spec)
panel_overall = did.aggte(panel_fit, type="simple", cband=False)
print(f"Panel standard error:          {panel_overall.overall_se:.4f}")
print(f"Cross-section standard error:  {rc_overall.overall_se:.4f}")
print(f"Ratio: {rc_overall.overall_se / panel_overall.overall_se:.1f}")
```

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

Holding the other settings in the {ref}`shared specification
<faq-example-setup>` fixed makes the effect of this choice easier to see.
Both rows below report the simple average effect in log employment units.

```{code-cell} ipython3
for comparison in ["nevertreated", "notyettreated"]:
    fit = did.att_gt(data, **(spec | {"control_group": comparison}))
    overall = did.aggte(fit, type="simple", cband=False)
    print(f"{comparison:14} {overall.overall_att: .4f}")
```

Changing the eligible comparison units changes the estimated untreated path;
the choice still needs a parallel trends argument for your data.

### Why do `att_gt` and `etwfe` give different answers on the same data?

Most of the gap comes from their default comparison groups rather than from the
methods. With its default not-yet-treated comparison, {func}`~moderndid.etwfe`
followed by `emfx(type="simple")` gives −0.0477 on mpdta. Under its default
never-treated comparison, {func}`~moderndid.att_gt` with
`aggte(type="simple")` gives −0.0400, the same number `etwfe` gives with
`cgroup="never"`. {ref}`Extended TWFE <example_etwfe>` walks through the
comparison on the same county data.

We leave covariates out of both fits here so the comparison isolates their
default control conventions and the never-treated alternative. This recipe
also needs the `etwfe` extra, installed with `uv add 'moderndid[etwfe]'` or
`pip install 'moderndid[etwfe]'`.

```{code-cell} ipython3
fit = did.att_gt(data, **spec)
overall = did.aggte(fit, type="simple", cband=False)
print(f"att_gt, never-treated: {overall.overall_att: .4f}")
for comparison in ["notyet", "never"]:
    regression = did.etwfe(
        data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        cgroup=comparison,
    )
    average = did.emfx(regression, type="simple")
    print(f"etwfe, {comparison:6}:     {average.overall_att: .4f}")
```

The agreement under never-treated controls is specific to this specification;
adding covariates can make the estimators differ in how they adjust for them.

### Should I use a varying or universal base period?

With no anticipation, `base_period` changes only the estimates before adoption,
since every effect after adoption uses $g-1$ as its base under either setting. The default
`"varying"` makes each estimate before adoption a change over one period. Under
`"universal"` every period is measured against $g-1$ and the placebo estimates
sit on the same scale as the effects. Choose `"universal"` for an event study
you plot or set beside a regression event study. Always choose it before
{func}`~moderndid.honest_did`, since it removes the normalized reference period
at event time $-1-\text{anticipation}$. With a positive anticipation setting,
the universal base is $g-\text{anticipation}-1$ rather than $g-1$.
[The base period](user_guide/example_staggered_did.md#the-base-period) puts the
two settings side by side on the county data.

For the 2007 cohort in 2005, the varying base compares 2005 with 2004 while
the universal base compares 2005 with 2006.

```{code-cell} ipython3
for base in ["varying", "universal"]:
    fit = did.att_gt(data, **(spec | {"base_period": base}))
    row = (fit.groups == 2007) & (fit.times == 2005)
    print(f"{base:9} {fit.att_gt[row][0]: .4f}")
```

### What can I put in `xformla`?

{func}`~moderndid.att_gt`, {func}`~moderndid.ddd`, {func}`~moderndid.etwfe`, and
{func}`~moderndid.did_multiplegt` read `xformla` as plain column names joined by
`+`. Build any transformation, dummy, or interaction as its own column first,
such as `pl.col("pop").log().alias("lpop")`.

:::{admonition} Build transformed covariates before estimation
:class: warning

The named-column parser rejects transformations, categorical terms, powers,
and interactions rather than evaluating them. For an `att_gt` fit with a
squared population covariate, create that column first and name it in
`xformla`. The two-period {func}`~moderndid.drdid`, {func}`~moderndid.ipwdid`,
and {func}`~moderndid.ordid` can evaluate a full formula when the `formulaic`
dependency is installed.
:::

For example, you can add a squared term for the recorded log population and
name both columns in the specification.

```{code-cell} ipython3
with_square = data.with_columns((pl.col("lpop") ** 2).alias("lpop_sq"))
adjusted = did.att_gt(with_square, **(spec | {"xformla": "~lpop + lpop_sq"}))
adjusted_overall = did.aggte(adjusted, type="simple", cband=False)
print(f"Population-adjusted average: {adjusted_overall.overall_att:.4f} log points")
```

Choose the population adjustment according to your argument for conditional
parallel trends in the application you are studying.

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

```{code-cell} ipython3
for method in ["dr", "ipw", "reg"]:
    fit = did.att_gt(data, **(spec | {"est_method": method}))
    overall = did.aggte(fit, type="simple", cband=False)
    print(f"{method:3} {overall.overall_att: .4f}")
```

### What if the policy was announced before it took effect?

Set `anticipation` to the number of periods before adoption in which units might
respond. Each cohort's base period then moves back from $g-1$ to $g-1-k$ for
`anticipation=k`. Any response in those periods counts toward the treatment
effect instead of looking like a pre-trend. A cohort without an earlier base
period drops out, as Illinois does in
[Anticipation](user_guide/example_staggered_did.md#anticipation).

Because one year of anticipation requires a baseline before 2003 for the 2004
cohort, those counties leave this five-year panel's estimation sample.

```{code-cell} ipython3
for years in [0, 1]:
    fit = did.att_gt(data, **(spec | {"anticipation": years}))
    cohorts = [int(g) for g in np.unique(fit.groups)]
    print(f"anticipation={years}: {fit.n_units} counties; treated cohorts {cohorts}")
```

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

An integer seed allows repeated aggregations of the same fit to reproduce
their bootstrap uncertainty. We enable the bootstrap in the
{ref}`shared specification <faq-example-setup>` and compare the standard
errors and simultaneous critical values from two aggregation calls.

```{code-cell} ipython3
seeded = did.att_gt(data, **(spec | {"boot": True, "cband": True, "biters": 999}))
first = did.aggte(seeded, type="dynamic")
second = did.aggte(seeded, type="dynamic")
print(f"Same standard errors: {np.array_equal(first.se_by_event, second.se_by_event)}")
print(f"Same critical values: {np.array_equal(first.critical_values, second.critical_values)}")
```

Keep the specification and the seed together when saving or sharing an
analysis. Passing the same NumPy random generator object twice continues
its sequence of draws instead of restarting it.

### How do I cluster standard errors by state?

Since {func}`~moderndid.att_gt` clusters inside the multiplier bootstrap,
`clustervars=["state"]` takes effect only with `boot=True`. Without the
bootstrap you get a warning and standard errors that ignore the clusters.
Besides `idname` you can name one more variable that stays constant within each
unit. You set the clusters once, since every {func}`~moderndid.aggte` call on
that result reuses them. The other estimators that cluster spell the option in their own way.

- {func}`~moderndid.etwfe` takes `vcov={"CRV1": "state"}`.
- {func}`~moderndid.did_multiplegt` takes `cluster="state"`.
- {func}`~moderndid.ddd` takes `cluster="state"` for itself and for {func}`~moderndid.agg_ddd`.

For now, {func}`~moderndid.cont_did` can't cluster standard errors above the
level of the unit.
[Clustering by state](user_guide/example_staggered_did.md#clustering-by-state)
clusters the county data by state and shows why the bands for a lone treated
state come out too narrow.

Since the county identifiers are FIPS codes, their leading digits identify the
state. We use that code to cluster the bootstrap and then print the overall
average with its resulting uncertainty.

```{code-cell} ipython3
state_data = data.with_columns((pl.col("countyreal") // 1000).alias("state"))
clustered = did.att_gt(
    state_data,
    **(spec | {"boot": True, "cband": True, "clustervars": ["state"], "biters": 999}),
)
clustered_overall = did.aggte(clustered, type="simple")
print(clustered_overall)
```

The interval in this report concerns the overall average across treated
county-years. It does not establish reliable inference for the longer event
times that depend on Illinois alone.

### Why is an estimate significant in the `att_gt` table but not in the event study?

Under the defaults, {func}`~moderndid.att_gt` prints pointwise intervals and
keeps simultaneous bands for `boot=True`. {func}`~moderndid.aggte` prints
simultaneous bands built on a bootstrap critical value above 1.96. On mpdta the
2007 cohort's 2004 estimate of 0.0305 carries a star in the `att_gt` table. At
event time −3 the same estimate gets a wider simultaneous band and
loses its star. Since a simultaneous band covers every estimate in a table at
once at the nominal 95 percent confidence level, it suits scanning a table or plot for effects.
A pointwise interval suits a single estimate that you picked out before looking.
To use the same interval convention, pass `boot=True` to `att_gt` for
simultaneous bands or `cband=False` to `aggte` for pointwise intervals.
Simultaneous critical values can still differ because the group-time table
and the event study cover different collections of effects.

The output below follows the 2007 cohort's 2004 estimate into event time -3.
The point estimate stays the same while the confidence limits change.

```{code-cell} ipython3
fit = did.att_gt(data, **(spec | {"cband": True, "biters": 999}))
event_study = did.aggte(fit, type="dynamic")
cell = did.to_df(fit).filter((pl.col("group") == 2007) & (pl.col("time") == 2004))
event = did.to_df(event_study).filter(pl.col("event_time") == -3)
for label, row in [("Pointwise", cell), ("Simultaneous", event)]:
    estimate, lower, upper = row.select("att", "ci_lower", "ci_upper").row(0)
    print(f"{label:12} {estimate:.4f} [{lower:.4f}, {upper:.4f}]")
```

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

Since all three rows below use one fitted result, their different overall
effects come from aggregation rather than from refitting the cohort-year
comparisons.

```{code-cell} ipython3
fit = did.att_gt(data, **spec)
for kind in ["group", "simple", "dynamic"]:
    average = did.aggte(fit, type=kind, cband=False)
    print(f"{kind:7} {average.overall_att: .4f} log points")
```

### Why are some group-time estimates missing?

When a cell's comparison can't be estimated, {func}`~moderndid.att_gt` names the
cell in a warning such as "Overlap condition violated for 2004.0 in time period
2005". Under the default varying base the failed cells drop out of the result.
Under `base_period="universal"` they stay as `NaN` and {func}`~moderndid.aggte`
stops with "Missing values at att_gt found". Pass `na_rm=True` to `aggte` to
aggregate the cells that did estimate. To recover the failing cells instead, try
fewer covariates for small cohorts or `control_group="notyettreated"` for more
comparison units.

Once you have inspected the failed cells, the aggregation call can explicitly
exclude them from the average.

```python
available_effects = did.aggte(result, type="group", na_rm=True)
print(available_effects)
```

The `result` in this snippet refers to the object returned by your estimation
call. Although excluding cells can change the periods and cohorts represented
in the summary, `na_rm=True` does not repair the comparison that failed.

### How do I get results into a DataFrame or a paper table?

Results expose their estimates as attributes whose names depend on the
estimator. For an `aggte` result, those include `overall_att` and
`att_by_event`. {func}`~moderndid.to_df` converts supported result types to a
polars DataFrame containing estimates and uncertainty. For a `type="simple"`
aggregation, read `overall_att` and `overall_se` directly because that scalar
result cannot be converted with `to_df`. The
{doc}`results guide <user_guide/results>` explains the fields and aggregation choices.
{ref}`Publication tables <publication_tables>` shows how to combine several
results into one table for a paper.

For an event study, the converted data expose event time, the estimate, and
its confidence limits as named columns.

```{code-cell} ipython3
fit = did.att_gt(data, **(spec | {"base_period": "universal"}))
event_study = did.aggte(fit, type="dynamic", cband=False)
print(did.to_df(event_study).select("event_time", "att", "se", "ci_lower", "ci_upper"))
```

For a formatted table, install `maketables` and pass the result object directly
to its `ETable` constructor. The rendered table below retains the event study's
analytical standard errors and pointwise intervals.

```{code-cell} ipython3
import maketables as mt

table = mt.ETable(
    [event_study],
    keep=[r"^Event "],
    drop=[r"^Event -1$"],
    coef_fmt="b:.3f \\n [ci95l:.3f, ci95u:.3f]",
    model_stats=["n_units", "se_type"],
    model_stats_labels={"n_units": "Counties"},
    caption="Minimum wage effects on log teen employment",
)
table.make("gt")
```

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
{doc}`Installation <user_guide/installation>` lists every extra along with
fixes for common install failures.

For continuous treatment, use `uv add` in a uv project or `pip install` in your
current Python environment.

```bash
uv add 'moderndid[didcont]'
```

```bash
pip install 'moderndid[didcont]'
```

### How do I speed up estimation on a large panel?

On one machine, `n_jobs=-1` lets {func}`~moderndid.att_gt` and
{func}`~moderndid.ddd` estimate the group-time cells in parallel without
changing the results. Since the multiplier bootstrap is often the slowest step,
the `numba` extra and a smaller `biters` both help. With an NVIDIA GPU and the
`gpu` extra, `backend="cupy"` runs `att_gt`, `ddd`, and
{func}`~moderndid.cont_did` through supported GPU calculations.
{ref}`GPU acceleration <gpu>` explains how to benchmark the full call,
including preparation and transfers, to find out whether it helps your specification.

To use all available CPU cores for the group-time comparisons, change only
`n_jobs` in the shared specification. You can verify that parallel execution
preserves the effect estimates on the county panel.

```{code-cell} ipython3
serial = did.att_gt(data, **spec)
parallel = did.att_gt(data, **(spec | {"n_jobs": -1}))
print(f"Same estimates: {np.allclose(serial.att_gt, parallel.att_gt)}")
```

Reducing bootstrap iterations trades precision in the simulated uncertainty
for speed. Keep enough draws for the inference you plan to report, as
{ref}`Computation on larger panels <scaling>` discusses.
