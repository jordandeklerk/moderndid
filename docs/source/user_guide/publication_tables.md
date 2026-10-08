---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(publication_tables)=

# Reporting results in publication tables

A results table needs to make clear which effect each row estimates and how
its uncertainty was calculated. Copying estimates by hand makes it easy to
lose that connection when you change a specification. ModernDiD result
objects can pass their estimates and inference information directly to
[maketables](https://py-econometrics.github.io/maketables/), so the table
can be rebuilt from the analysis that produced it.

We begin with the minimum wage county sample so readers know whose employment
the estimates describe. The event-study table then puts each estimate beside
its uncertainty before we compare specifications in separate columns. Each
table renders directly from the code so you can see how the rows, labels, and
uncertainty appear together.
`ETable` reads supported result objects, `DTable` describes the data, and
`MTable` gives you control over custom panels. All three can produce LaTeX,
HTML, Word, and Typst output for the document you're preparing.


## Installing the table package

To follow the table examples, add `maketables` to the environment where you
run your analysis. The ordinary ModernDiD installation does not include it,
so use the command that matches how you manage that environment.

```bash
uv add maketables
```

Or install the package into your current Python environment with pip.

```bash
pip install maketables
```

## Describing the sample before the estimates

Before presenting an effect, give your reader a sense of the counties in the
analysis. The {ref}`staggered DiD example <example_staggered_did>` uses log teen
employment as the outcome and adjusts for population measured before the
minimum wage increases. We summarize both variables in the first observed
year so each county appears once in the sample description.

```{code-cell} ipython3
import maketables as mt
import moderndid as did
import pandas as pd
import polars as pl

data = did.load_mpdta()
baseline = data.filter(pl.col("year") == 2003).to_pandas()
sample_table = mt.DTable(
    baseline,
    vars=["lemp", "lpop"],
    stats=["count", "mean", "std", "min", "max"],
    labels={
        "lemp": "Log teen employment in 2003",
        "lpop": "Log population in thousands in 2000",
    },
    stats_labels={"count": "Counties"},
    format_spec={"count": ",.0f", "mean": ".2f", "std": ".2f", "min": ".2f", "max": ".2f"},
    caption="County sample before the minimum wage increases",
    notes="One observation per county. The summaries describe log values, not levels.",
)
sample_table.make("gt")
```

Because each county appears once in this table, its count describes the
baseline sample rather than the county-year observations used for estimation.
The means and standard deviations also retain the variables' log units.
These summaries help readers understand the sample, although they do not
establish the parallel trends assumption needed to interpret an effect.

## Making a table from an event study

With the sample described, we can turn to whether state minimum wage increases
reduced teen employment. The fit uses the county panel and population
adjustment from the worked example. Analytical standard errors and pointwise
intervals keep this formatting exercise independent of bootstrap draws.
These county-level intervals differ from the state-clustered simultaneous
bands used in the worked analysis; choose inference for your study before
formatting its results, as the {doc}`results` guide explains.

```{code-cell} ipython3
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    xformla="~lpop",
    est_method="dr",
    control_group="nevertreated",
    base_period="universal",
    boot=False,
    cband=False,
)
result = did.att_gt(data, **spec)
event_study = did.aggte(result, type="dynamic", cband=False)
print(event_study)
```

The event times are years relative to a state's first increase. The outcome
is a log, so the estimates remain in log-employment units when you put them
in a table. A negative estimate represents lower employment relative to the
untreated counterfactual under the specification's parallel trends assumption.
We put the estimate above its confidence interval so readers can assess both
without looking between separate columns.

```{code-cell} ipython3
tab = mt.ETable(
    [event_study],
    coef_fmt="b:.3f \\n [ci95l:.3f, ci95u:.3f]",
    keep=[r"^Event "],
    drop=[r"^Event -1$"],
    labels={"lemp": "Log teen employment"},
    model_stats=["n_units", "se_type"],
    model_stats_labels={"n_units": "Counties"},
    caption="Minimum wage effects by event time",
    notes="Estimates in log-employment units. Pointwise 95 percent intervals in brackets.",
)
tab.make("gt")
```

The `coef_fmt` string puts the estimate on the first line and its interval
on the second. Its tokens name the values supplied by the result object;
`b` is the estimate, `se` is its standard error, and `ci95l` and
`ci95u` are the interval bounds. We use `.3f` to keep three decimal places
and `\\n` to place the interval on its own line within the cell.

For the first year after adoption, the table reports -0.053 log points and a
pointwise interval from -0.085 to -0.021 log points. The interval excludes
zero under the county-level analytical inference chosen here. These are the
same estimates shown in the report, rounded for the publication display.

Because `keep` selects rows by their names, `r"^Event "` leaves out the
separate overall ATT. We also omit event time minus one because that reference
year is normalized to zero rather than estimated as a separate effect.
For this panel result, `n_units` counts counties and `N` counts county-year
rows in the estimation sample. Labeling the requested unit count as Counties
keeps readers from confusing those two sample sizes.


## Comparing specifications in columns

Placing specifications side by side helps readers see how a particular choice
affects the estimates. We keep the data, population adjustment, and inference
settings fixed so the columns isolate a change in either the comparison group
or the estimation method. In the two doubly robust columns, that choice is
whether later adopters can serve as controls before their own minimum wage
increases.

```{code-cell} ipython3
changes = [
    {},
    {"control_group": "notyettreated"},
    {"est_method": "ipw"},
    {"est_method": "reg"},
]
models = [result] + [did.att_gt(data, **(spec | change)) for change in changes[1:]]
event_studies = [did.aggte(model, type="dynamic", cband=False) for model in models]
```

The inverse probability weighting and outcome regression columns each rely
on one of the nuisance models used by the doubly robust estimator.
The {ref}`estimator guide <estimator-overview>` explains those choices;
this table keeps their labels alongside the estimates so the specifications
remain identifiable after export.

```{code-cell} ipython3
comparison_table = mt.ETable(
    event_studies,
    coef_fmt="b:.3f \\n (se:.3f)",
    keep=[r"^Event "],
    drop=[r"^Event -1$"],
    labels={"lemp": "Log teen employment"},
    model_heads=[
        "Doubly robust",
        "Doubly robust",
        "Inverse probability\nweighting",
        "Outcome regression",
    ],
    head_order="dh",
    model_stats=["n_units", "control_group", "estimation_method", "se_type"],
    model_stats_labels={
        "n_units": "Counties",
        "control_group": "Comparison group",
        "estimation_method": "Estimation method",
    },
    caption="Minimum wage event studies across specifications",
    notes=(
        "All columns adjust for baseline log population. "
        "Analytical standard errors at the county level appear in parentheses."
    ),
)
comparison_table.make("gt")
```

The two doubly robust columns share a heading because adjacent identical
entries in `model_heads` merge into a column spanner. Their different
comparison groups remain visible in the footer, where you can also check
that the uncertainty method stays fixed across specifications.
`head_order="dh"` displays the outcome and model headings together; use
`"h"` or `"d"` when only one of those levels helps identify the columns.

At event time one, the doubly robust estimates are -0.053 log points using
never-treated counties and -0.055 using not-yet-treated counties. That
closeness is a feature of this sample and these specifications; a clear
comparison table lets you assess such changes without treating agreement
between estimators as evidence that parallel trends holds.


## Choosing rows and reporting uncertainty

Once the full table is available, you can choose a shorter set of rows for a
particular display. Use literal coefficient names with `exact_match=True`
and regex patterns when it is false. We retain one pre-adoption estimate and
the first two effects after adoption below, in their event-time order.

```{code-cell} ipython3
selected_table = mt.ETable(
    [event_study],
    coef_fmt="b:.3f \\n [ci95l:.3f, ci95u:.3f]",
    keep=["Event -2", "Event 0", "Event 1"],
    order=["Event -2", "Event 0", "Event 1"],
    exact_match=True,
    labels={
        "Event -2": "Two years before adoption",
        "Event 0": "Adoption year",
        "Event 1": "One year after adoption",
        "lemp": "Log teen employment",
    },
    model_stats=["n_units", "se_type"],
    model_stats_labels={"n_units": "Counties"},
    caption="Selected minimum wage effects",
    notes="Pointwise 95 percent intervals in brackets. Rows follow event-time order.",
)
selected_table.make("gt")
```

Selecting rows changes the display rather than the estimates or their
confidence bounds. If the result uses simultaneous bands, the exported
bounds retain the critical values from that result. A table containing fewer
rows does not automatically recompute a band for that smaller set.

:::{admonition} Stars use pointwise p-values
:class: important

The table interface computes p-values from the estimate and standard error
using a normal approximation. Stars can therefore disagree with a
simultaneous band that contains zero. Report the bands and omit stars when
you want the table's significance cues to follow simultaneous inference.

:::

For a table that intentionally reports pointwise tests, append `*` to the
estimate token and use `signif_code` to set the three p-value cutoffs.
Omitting `*` from the format suppresses stars, as in the tables above.

```{code-cell} ipython3
starred_table = mt.ETable(
    [event_study],
    coef_fmt="b:.3f* \\n (se:.3f)",
    signif_code=[0.01, 0.05, 0.10],
    keep=[r"^Event "],
    drop=[r"^Event -1$"],
    labels={"lemp": "Log teen employment"},
    model_stats=["n_units", "se_type"],
    model_stats_labels={"n_units": "Counties"},
    caption="Event-study estimates with pointwise significance stars",
    notes=(
        "Analytical county-level standard errors in parentheses. "
        "Pointwise significance levels: * p < 0.10, ** p < 0.05, *** p < 0.01."
    ),
)
starred_table.make("gt")
```

## Adding statistics and table metadata

An overall effect needs enough context for readers to identify the comparison
it summarizes. The table interface can report details available in each
result, such as the aggregation type, comparison group, or standard error
method. We use `custom_model_stats` to add joint pre-trend p-values directly
from the underlying group-time fits so the diagnostic updates alongside the
analysis.

```{code-cell} ipython3
overall = [did.aggte(model, type="group", cband=False) for model in models[:2]]
report = mt.ETable(
    overall,
    coef_fmt="b:.3f \\n (se:.3f)",
    keep=["Overall ATT"],
    exact_match=True,
    model_heads=["Never treated", "Not yet treated"],
    model_stats=["n_units", "control_group", "se_type"],
    model_stats_labels={"n_units": "Counties", "control_group": "Comparison group"},
    custom_model_stats={
        "Joint pre-trend p-value": [f"{model.wald_pvalue:.3f}" for model in models[:2]],
        "Population adjustment": ["Yes", "Yes"],
    },
    caption="Average minimum wage effects across treated counties",
    labels={"lemp": "Log teen employment"},
    tab_label="tab:minimum-wage",
    notes="Group aggregation. Analytical county-level standard errors in parentheses.",
)
report.make("gt")
```

Group aggregation gives each treated county the same weight in the overall
effect after averaging over its cohort's observed post-adoption years.
That differs from the overall event-study effect, which averages across
post-adoption event times. Naming the aggregation in the notes helps readers
understand why two overall numbers from the same analysis can differ.

The group averages here are -0.033 and -0.032 log points under the two
comparison groups. The joint pre-trend p-values come from the original
group-time fits and do not test these overall effects. Keeping that
distinction in the row label helps readers use the statistic for the
comparison it was designed to assess.

Use the `caption` to describe the question the table answers and `tab_label`
to identify it for cross-referencing or replacement in a document. The notes
are where you record the outcome's units, the inference method, and any
restrictions readers need to interpret the rows.


## Arranging custom panels with MTable

A table with several aggregation types needs more control over its rows than
one column per result can provide. `MTable` accepts a pandas DataFrame whose
cells you construct yourself. We use a row `MultiIndex` for the panel
headings and a column `MultiIndex` for spanners that group related columns.

The panels below compare the population-adjusted estimates with estimates
under unconditional parallel trends. Within each panel, simple, group,
dynamic, and calendar aggregations answer different averaging questions.
Keeping the aggregation in the row label tells readers which effect each
estimate describes even when several summaries appear in the same table.

```{code-cell} ipython3
unadjusted = did.att_gt(data, **(spec | {"xformla": None}))
specifications = {
    "Without population adjustment": unadjusted,
    "With population adjustment": result,
}
aggregations = {
    "simple": ("Simple average", None),
    "group": ("Cohort effects", "g"),
    "dynamic": ("Post-adoption event study", "e"),
    "calendar": ("Calendar-year effects", "t"),
}
results = {
    panel: {
        kind: did.aggte(
            model,
            type=kind,
            cband=False,
            **({"min_e": 0, "max_e": 3} if kind == "dynamic" else {}),
        )
        for kind in aggregations
    }
    for panel, model in specifications.items()
}
```

There are at most four component estimates in each of these rows. The
post-adoption event study is restricted to event times zero through three,
while cohort and calendar rows report their observed cohorts or years.
The simple aggregation has no separate component estimates to display.

```{code-cell} ipython3
:tags: [hide-input]

n_components = 4
row_index = []
table_rows = []
for panel, panel_results in results.items():
    for kind, (label, prefix) in aggregations.items():
        aggregated = panel_results[kind]
        components = []
        if aggregated.event_times is not None:
            components = [
                f"{prefix} = {int(value)}\n{estimate:.3f}\n({se:.3f})"
                for value, estimate, se in zip(
                    aggregated.event_times,
                    aggregated.att_by_event,
                    aggregated.se_by_event,
                )
            ]
        components += [""] * (n_components - len(components))
        row_index.append((panel, label))
        table_rows.append(
            components + [f"{aggregated.overall_att:.3f}\n({aggregated.overall_se:.3f})"]
        )

columns = pd.MultiIndex.from_tuples(
    [("Component effects", str(index + 1)) for index in range(n_components)]
    + [("Overall effect", "")]
)
table_data = pd.DataFrame(
    table_rows,
    index=pd.MultiIndex.from_tuples(row_index, names=["Specification", "Aggregation"]),
    columns=columns,
)
panel_table = mt.MTable(
    table_data,
    caption="Minimum wage estimates under two parallel trends specifications",
    notes="Effects in log-employment units. Analytical standard errors in parentheses.",
    rgroup_sep="tb",
    rgroup_display=True,
    tex_style={"group_header_format": r"\textbf{%s}"},
)
panel_table.make("gt")
```

The component cells show the cohort, event time, or calendar year above
each estimate and standard error. Keeping those labels in the cells avoids
aligning a cohort date with an event time as though they shared an axis.
The panel headings separate the two parallel trends specifications and the
overall column identifies each aggregation's average. The
[MTable documentation](https://py-econometrics.github.io/maketables/docs/MTable.html)
covers further row-group and column-spanner options.

In the population-adjusted panel, the group average of -0.033 log points and
the post-adoption event-study average of -0.080 log points summarize the same
fit with different weights. Their separation shows why each row needs an
aggregation label. The {doc}`results` guide explains which counties and
exposure lengths contribute to those averages.


## Exporting and updating a table

The previews above use `make("gt")` to render HTML through Great Tables.
You can render the same rows and notes in the format your document needs.
Specifying the format makes the call independent of whether you run it in a
notebook, script, or document renderer.

```python
tex = report.make("tex")
html = report.make("html")
word = report.make("docx")
typst = report.make("typst")
```

To adjust a renderer's layout, pass its style dictionary when you call
`make`. You can set the LaTeX table width or the HTML font size without
changing the stored estimates.

```python
tex = report.make("tex", tex_style={"tab_width": r"0.9\linewidth", "tabcolsep": "2pt"})
html = report.make("html", gt_style={"table_width": "100%", "table_font_size_all": "14px"})
word = report.make("docx", docx_style={"font_name": "Times New Roman", "font_size_pt": 11})
```

For these LaTeX tables, load `booktabs`, `threeparttable`, and `makecell` in
your document's preamble so the rules, notes, and stacked cells can render.
Add `tabularx` and `array` when you specify a table width, as in the styled
export above.

To write the rendered table to disk, pass a format and file name to `save`.
The `show=False` setting keeps the export from opening a viewer.

```python
report.save("tex", "./results.tex", show=False)
report.save("docx", "./results.docx", show=False)
```

You can rerun `save` to replace the exported table file after updating your
analysis. For a table embedded in an existing LaTeX document, `update_tex`
instead looks for its `tab_label` and replaces that table. If the label is
absent, it inserts the table, so keep labels stable as your analysis changes.
For Word, `update_docx` uses a one-based table position rather than a label.
The call below replaces the first table in the exported document; omitting
`tab_num` would append a table instead.

```python
report.update_tex("./paper.tex")
report.update_docx("./results.docx", tab_num=1)
```

The [maketables reference](https://py-econometrics.github.io/maketables/docs/ETable.html)
lists the full set of coefficient and metadata formatting options. For a figure
built from the same estimates,
{ref}`plotting treatment effects <plotting>` shows how to carry the result's
confidence bounds into a plot without copying them by hand.
