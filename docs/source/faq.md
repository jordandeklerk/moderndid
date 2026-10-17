---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(faq)=

# FAQ

(faq-example-setup)=

The code in these answers runs on the county panel that
{func}`~moderndid.load_mpdta` loads and starts from the specification below.
Unless an answer changes a setting, the estimates use no covariates and
analytical standard errors.

```{code-cell} ipython3
import moderndid as did

# The county panel and the specification that the code in these answers starts from.
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
as `tname`. {func}`~moderndid.att_gt` reads a 0, `np.inf`, or any period after
the last one in your data as never treated. Since a null or `NaN` in `gname`
drops the unit's rows as missing data, never-treated units coded that way leave
the sample. When a missing code does mean never treated, replace both kinds with
`pl.col("g").fill_nan(0).fill_null(0)`. The data guide's
[Record treatment timing consistently](user_guide/data.md#record-treatment-timing-consistently)
covers the other requirements for `gname`.

### How do I turn a 0/1 treatment dummy into treatment timing?

{func}`~moderndid.core.panel.get_group` reads a 0/1 treatment indicator and adds
a `G` column holding the period each unit is first treated, or 0 if it never is.
You then pass `gname="G"` to `att_gt` in place of the dummy. If your dummy marks
treated units in every period, pass the known start date as `treat_period`. For
a treatment that can switch off again, {func}`~moderndid.did_multiplegt` takes
the dummy itself as `dname`.
{ref}`Recording the first treatment period <panel-utilities-group-timing>`
walks through the conversion on a county panel.

### Which DataFrames and column types does ModernDiD accept?

ModernDiD accepts DataFrames from libraries that implement the Arrow PyCapsule
interface. If a pandas DataFrame keeps the unit or period in its index, call
`reset_index()` first, since only columns survive the conversion. The unit
identifier, time, outcome, and timing columns must all be numeric. Map text
identifiers to integers with `pl.col("id").rank("dense")` across the whole
dataset and turn dates into integer periods such as `pl.col("date").dt.year()`.
The {ref}`data guide <data-formats>` shows how to prepare Polars, pandas, and
pyarrow inputs and what to do with a lazy table.

### What happens to an unbalanced panel or rows with missing values?

By default {func}`~moderndid.att_gt` keeps only the units observed in every
period. Since it first drops every row with a missing value in a column the
model uses, one missing outcome removes a whole unit. With
`allow_unbalanced_panel=True` it keeps the incomplete units and estimates from
the repeated cross-section form of the estimator, as
[the data guide](user_guide/data.md#check-what-will-remain-in-the-sample)
explains. {ref}`Preparing panel data <panel-utilities>` shows how to find those
rows with {func}`~moderndid.core.panel.diagnose_panel` and how to fill, drop,
or deduplicate them.

### How do I estimate with repeated cross sections instead of a panel?

For repeated cross sections, set `panel=False` and leave `idname` out of the
call. The estimator then treats each row as its own observation and compares
cohort means across periods instead of changes within units. Don't switch to
`panel=False` just to get past an error on panel data, since that throws away
the pairing within units. Under the {ref}`shared specification <faq-example-setup>`,
the switch makes the standard error of the simple average on the county data
about 13.5 times as large.
[The data guide](user_guide/data.md#check-what-will-remain-in-the-sample)
explains which sampling designs call for `panel=False`.

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

{ref}`The estimator guide <estimator-overview>` compares them and adds
two-period designs, dynamic covariate balancing, and sensitivity analysis.
Neither the estimator of Sun and Abraham (2021) nor the imputation estimator of
Borusyak, Jaravel, and Spiess (2024) has its own function, since
{func}`~moderndid.etwfe` reproduces both in the matched specifications that
{ref}`the ETWFE background <background-etwfe>` sets out. With `cgroup="never"`
and no covariates, its event study matches the Sun and Abraham estimator.

### Should I compare with never-treated or not-yet-treated units?

By default {func}`~moderndid.att_gt` compares each cohort with the
never-treated units in your data. With `control_group="notyettreated"` it also
uses cohorts that adopt later, in the years before they do. Those extra
comparisons can add precision but extend parallel trends to the later adopters.
{ref}`The staggered DiD background <background-did-comparison-groups>` explains
why never-treated units are the safer choice when you have enough of them and
they resemble the treated ones.

### Why do `att_gt` and `etwfe` give different answers on the same data?

Most of the gap comes from their default comparison groups rather than from the
methods. Where {func}`~moderndid.att_gt` compares each cohort with never-treated
units by default, {func}`~moderndid.etwfe` defaults to `cgroup="notyet"` and
also counts cohorts in their years before adoption as controls. On a balanced
panel without covariates, `etwfe` with `cgroup="never"` gives the same simple
average as `att_gt` under its defaults. Once you add covariates, the two
estimators adjust for them in different ways and may no longer agree. The
{ref}`extended TWFE example <example_etwfe>` compares the two comparison groups
on the county data.

### Should I use a varying or universal base period?

With no anticipation, `base_period` changes only the estimates before adoption,
since every effect after adoption uses $g-1$ as its base under either setting.
The default `"varying"` makes each estimate before adoption a change over one
period. Under `"universal"` every period is measured against $g-1$ and the
placebo estimates sit on the same scale as the effects. Choose `"universal"` for
an event study you plot, set beside a regression event study, or pass to
{func}`~moderndid.honest_did`, since `honest_did` raises an error for one
estimated with the varying base.
[The base period](user_guide/example_staggered_did.md#the-base-period) puts the
two settings side by side on the county data.

### What can I put in `xformla`?

{func}`~moderndid.att_gt`, {func}`~moderndid.ddd`, {func}`~moderndid.etwfe`, and
{func}`~moderndid.did_multiplegt` read `xformla` as plain column names joined by
`+`. Since that parser rejects transformations, categorical terms, powers, and
interactions, build each one as its own column first, such as
`pl.col("pop").log().alias("lpop")`. The two-period {func}`~moderndid.drdid`,
{func}`~moderndid.ipwdid`, and {func}`~moderndid.ordid` can evaluate a full
formula when the `formulaic` dependency is installed. The data guide's
[Choose covariates measured before treatment](user_guide/data.md#choose-covariates-measured-before-treatment)
shows how to pass covariates through `xformla` for a staggered design.

### Can my covariates change over time?

On a balanced panel, {func}`~moderndid.att_gt` reads each covariate in the
earlier of a comparison's two periods. Every effect after adoption therefore
adjusts for the covariate's value before treatment, even if treatment changes it
later.
[The data guide](user_guide/data.md#choose-covariates-measured-before-treatment)
explains why the covariates should be fixed before treatment in the first
place.

### Does `est_method` matter if I have no covariates?

Without covariates, `"dr"`, `"ipw"`, and `"reg"` reduce to the same comparison
of mean changes and give identical estimates. Once you add covariates, `"dr"`
stays consistent if either the outcome model or the propensity score model is
right. That double protection is why `"dr"` is the default, as
{ref}`Doubly robust DiD <background-drdid>` explains.

### What if the policy was announced before it took effect?

Set `anticipation` to the number of periods before adoption in which units might
respond. Each cohort's base period then moves back from $g-1$ to $g-1-k$ for
`anticipation=k`. Any response in those periods counts toward the treatment
effect instead of looking like a pre-trend. A cohort without an earlier base
period drops out, as Illinois does in
[Anticipation](user_guide/example_staggered_did.md#anticipation).

## Reading the results

### Why do my standard errors change every time I rerun the code?

While point estimates stay fixed, uncertainty calculated by the multiplier
bootstrap changes between runs unless you seed it. Because
{func}`~moderndid.aggte` draws a bootstrap for its simultaneous bands even when
`boot=False`, those bands change between runs too. If you pass `random_state`
to {func}`~moderndid.att_gt`, every `aggte` call on the result reuses it, as
[the staggered example](user_guide/example_staggered_did.md#choosing-a-specification)
does. Since {func}`~moderndid.agg_ddd` doesn't reuse the seed you gave
{func}`~moderndid.ddd`, give it its own `random_state`.
{func}`~moderndid.cont_did` always bootstraps its dose-response standard errors
and needs a seed of its own too.

An integer seed allows repeated aggregations of the same fit to reproduce
their bootstrap uncertainty. We enable the bootstrap in the
{ref}`shared specification <faq-example-setup>` and compare the standard
errors and simultaneous critical values from two aggregation calls.

```{code-cell} ipython3
# Fit once with the bootstrap and the shared seed and aggregate the result twice.
seeded = did.att_gt(data, **(spec | {"boot": True, "cband": True, "biters": 999}))
first = did.aggte(seeded, type="dynamic")
second = did.aggte(seeded, type="dynamic")
print(f"Same standard errors: {np.array_equal(first.se_by_event, second.se_by_event)}")
print(f"Same critical values: {np.array_equal(first.critical_values, second.critical_values)}")
```

Both checks return `True` because the integer seed restarts the draws for
each aggregation. Keep the specification and the seed together when saving
or sharing an analysis.

:::{admonition} Seeding with a NumPy generator
:class: note

Passing the same NumPy random generator object twice continues its sequence
of draws instead of restarting it as an integer seed does.
:::

### How do I cluster standard errors by state?

With `clustervars=["state"]`, both the analytic standard errors of
{func}`~moderndid.att_gt` and its multiplier bootstrap treat each state as one
cluster. Besides `idname` you can name one more variable that stays constant
within each unit. You set the clusters once, since every
{func}`~moderndid.aggte` call on that result reuses them. Because `aggte` has
no data of its own, it raises an error for a variable the fit didn't cluster
on. Since intervals and bands from few clusters can come out too narrow,
`att_gt` warns when a treated cohort's counties fall in fewer than 5 states or
all counties in fewer than 30. The state-clustered fit below meets both
conditions and gets the warning.
{func}`~moderndid.honest_did` carries the same clusters into the covariance it
builds for an event study. The other estimators that cluster spell the option
in their own way.

- {func}`~moderndid.etwfe` takes `vcov={"CRV1": "state"}`.
- {func}`~moderndid.did_multiplegt` takes `cluster="state"`.
- {func}`~moderndid.ddd` takes `cluster="state"` for itself and for {func}`~moderndid.agg_ddd`.

For now, {func}`~moderndid.cont_did` can't cluster standard errors above the
level of the unit.
In the results guide,
[Match the uncertainty to the comparison](user_guide/results.md#match-the-uncertainty-to-the-comparison)
explains why a policy set by states calls for state clusters and how they change
the standard errors. The staggered example's
[Clustering by state](user_guide/example_staggered_did.md#clustering-by-state)
shows why the bands for a lone treated state come out too narrow.

Since the county identifiers are FIPS codes, their leading digits identify the
state. We use that code to cluster the bootstrap and then print the overall
average with its resulting uncertainty.

```{code-cell} ipython3
# Integer division by 1000 keeps the state part of each county FIPS code.
state_data = data.with_columns((pl.col("countyreal") // 1000).alias("state"))

# Bootstrap with whole states as the clusters and average the effects overall.
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
once at the nominal 95 percent confidence level, it suits scanning a table or
plot for effects. A pointwise interval suits a single estimate that you picked
out before looking.
To use the same interval convention, pass `boot=True` to `att_gt` for
simultaneous bands or `cband=False` to `aggte` for pointwise intervals.
Simultaneous critical values can still differ because the group-time table
and the event study cover different collections of effects. The
{doc}`results guide <user_guide/results>` explains how to match the interval
to the comparison you report.

The output below follows the 2007 cohort's 2004 estimate into event time −3.
The point estimate stays the same while the confidence limits change.

```{code-cell} ipython3
# Pointwise intervals for the group-time fit and simultaneous bands for its event study.
fit = did.att_gt(data, **(spec | {"cband": True, "biters": 999}))
event_study = did.aggte(fit, type="dynamic")

# The 2007 cohort's 2004 estimate in each table.
cell = did.to_df(fit).filter((pl.col("group") == 2007) & (pl.col("time") == 2004))
event = did.to_df(event_study).filter(pl.col("event_time") == -3)
for label, row in [("Pointwise", cell), ("Simultaneous", event)]:
    estimate, lower, upper = row.select("att", "ci_lower", "ci_upper").row(0)
    print(f"{label}: {estimate:.4f} [{lower:.4f}, {upper:.4f}]")
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
# Three overall effects from one fitted result.
fit = did.att_gt(data, **spec)
for kind in ["group", "simple", "dynamic"]:
    average = did.aggte(fit, type=kind, cband=False)
    print(f"{kind}: {average.overall_att:.4f} log points")
```

### Why are some group-time estimates missing?

When a cell's comparison can't be estimated, {func}`~moderndid.att_gt` names the
cell in a warning such as "Overlap condition violated for 2004 in time period
2005". Under the default varying base the failed cells drop out of the result.
Under `base_period="universal"` they stay as `NaN` and make
{func}`~moderndid.aggte` stop with "Missing values at att_gt found" unless you
pass `na_rm=True`.

Either way, the averages then cover only the cells that did estimate. A cohort
or year whose effects after adoption all failed has no row in the group or
calendar summary. If only cells before adoption are left, the overall effect
comes back as `NaN` with a warning that names the cause. Although
{func}`~moderndid.ddd` leaves failed cells out under either base period,
{func}`~moderndid.agg_ddd` summarizes the rest of a triple-difference fit by
the same rules.
[When some cells fail to estimate](user_guide/results.md#when-some-cells-fail-to-estimate)
in the results guide shows the county data's averages once a whole cohort drops
out. To recover the failing cells instead, try fewer covariates for small
cohorts or `control_group="notyettreated"` for more comparison units.

### How do I get results into a DataFrame or a paper table?

Results expose their estimates as attributes whose names depend on the
estimator. For an `aggte` result, those include `overall_att` and
`att_by_event`. {func}`~moderndid.to_df` converts supported result types to a
Polars DataFrame containing estimates and uncertainty. For a `type="simple"`
aggregation, read `overall_att` and `overall_se` directly because that scalar
result cannot be converted with `to_df`. The
{doc}`results guide <user_guide/results>` explains those fields and how to
choose an aggregation.
The {ref}`publication tables guide <publication_tables>` shows how to combine
several results into one table for a paper.

For an event study, the converted data expose event time, the estimate, and
its confidence limits as named columns.

```{code-cell} ipython3
# An event study with pointwise intervals, converted to a Polars DataFrame.
fit = did.att_gt(data, **(spec | {"base_period": "universal"}))
event_study = did.aggte(fit, type="dynamic", cband=False)
did.to_df(event_study).select("event_time", "att", "se", "ci_lower", "ci_upper")
```

For a formatted table, install `maketables` and pass the result object directly
to its `ETable` constructor. The rendered table below retains the event study's
analytical standard errors and pointwise intervals.

```{code-cell} ipython3
import maketables as mt

# Each event time's estimate above its 95 percent interval, without the reference row.
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

### How do I know the estimates match the original implementations?

The {ref}`worked comparisons <r-comparisons>` run the staggered,
two-period, triple-difference, continuous-treatment, intertemporal,
sensitivity-analysis, extended TWFE, and nonparametric IV estimators on the
same application data in ModernDiD and in reference implementations of the
same methods. Each page shows both sets of results side by side and explains
where and why they differ. Because agreement in one specification doesn't carry
over to every option, compare the settings on the page for your function with
the ones you use. Dynamic covariate balancing is checked against its reference
implementation in the test suite instead. Machine learning DiD has no
comparison with a reference implementation yet.

## Installing and scaling

### Which extra do I need for a function?

The base install runs the core estimators, such as {func}`~moderndid.att_gt`,
{func}`~moderndid.ddd`, and {func}`~moderndid.did_multiplegt`, along with
{func}`~moderndid.npiv` and the panel utilities. Because the `gpu` extra needs
CUDA, `"moderndid[all]"` leaves it out and adds every other extra. A function
whose extra is missing raises an `ImportError` that names the extra and the
commands that install it, such as the `didcont` extra for
{func}`~moderndid.cont_did`. If {func}`~moderndid.etwfe` or
{func}`~moderndid.diddynamic.dyn_balancing` raises a plain `ModuleNotFoundError`
instead, install the `etwfe` or `diddynamic` extra.
{doc}`Installation <user_guide/installation>` lists every extra along with
fixes for common install failures.

For continuous treatment, use `uv add` in a uv project or
`python -m pip install` in your current Python environment.

```bash
uv add "moderndid[didcont]"
```

```bash
python -m pip install "moderndid[didcont]"
```

### How do I speed up estimation on a large panel?

On one machine, `n_jobs=-1` lets {func}`~moderndid.att_gt` and
{func}`~moderndid.ddd` estimate the group-time cells in parallel without
changing the results. Since the multiplier bootstrap can take a large share of
the time on a large panel, a smaller `biters` helps. With an NVIDIA GPU and the
`gpu` extra, `backend="cupy"` runs `att_gt`, `ddd`, and
{func}`~moderndid.cont_did` through supported GPU calculations.
{ref}`GPU acceleration <gpu>` explains how to benchmark the full call,
including preparation and transfers, to find out whether it helps your
specification.

To use all available CPU cores for the group-time comparisons, change only
`n_jobs` in the shared specification. You can verify that parallel execution
preserves the effect estimates on the county panel.

```{code-cell} ipython3
# The same specification fitted serially and on all CPU cores.
serial = did.att_gt(data, **spec)
parallel = did.att_gt(data, **(spec | {"n_jobs": -1}))
print(f"Same estimates: {np.allclose(serial.att_gt, parallel.att_gt)}")
```

Since reducing bootstrap iterations trades precision in the simulated
uncertainty for speed, keep enough draws for the inference you plan to report.
The {ref}`scaling guide <scaling>` covers parallel fits and the GPU backend.
