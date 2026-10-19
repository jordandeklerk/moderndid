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
lose that connection when you change a specification. Since ModernDiD result
objects pass their estimates and inference information directly to
[maketables](https://py-econometrics.github.io/maketables/), you can rebuild
a table from the analysis that produced it.

We begin with the minimum wage county sample so readers know whose employment
the estimates describe. The event-study table then puts each estimate beside
its uncertainty before we compare specifications in separate columns.
`ETable` reads supported result objects, `DTable` describes the data, and
`MTable` gives you control over custom panels. All three can produce LaTeX,
HTML, Word, and Typst output for the document you're preparing.

Since the ordinary ModernDiD installation leaves `maketables` out, add it to
the environment where you run your analysis before following the examples. Use
`uv add` in a uv project and pip in any other Python environment.

```bash
uv add maketables
```

```bash
python -m pip install maketables
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

# Keep one row per county from the first observed year, in the pandas format DTable reads.
data = did.load_mpdta()
baseline = data.filter(pl.col("year") == 2003).to_pandas()

# Summarize the outcome and the population covariate in their log units.
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

## Making a table from an event study

Since the fit below reuses the specification of the
{ref}`first analysis <quickstart>`, the tables report county-level bootstrap
standard errors and show simultaneous bands wherever they print intervals. For
a paper, format the fit whose inference you chose to report. The results
guide's section on
[matching the uncertainty to the comparison](results.md#match-the-uncertainty-to-the-comparison)
covers that choice.

```{code-cell} ipython3
# Every table below formats this fit or a variant with one setting changed.
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    base_period="universal",
    anticipation=0,
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)

# Fit the cohort effects and average them by years since adoption.
result = did.att_gt(data, **spec)
event_study = did.aggte(result, type="dynamic")
print(event_study)
```

We put the estimate above its simultaneous band so readers can assess both
without looking between separate columns.

```{code-cell} ipython3
# Stack each interval under its estimate and leave out the normalized base year.
tab = mt.ETable(
    [event_study],
    coef_fmt="b:.3f \\n [ci95l:.3f, ci95u:.3f]",
    keep=[r"^Event "],
    drop=[r"^Event -1$"],
    labels={"lemp": "Log teen employment"},
    model_stats=["n_units", "se_type"],
    model_stats_labels={"n_units": "Counties"},
    caption="Minimum wage effects by event time",
    notes="Estimates in log-employment units. Simultaneous 95 percent bands in brackets.",
)
tab.make("gt")
```

The `coef_fmt` string puts the estimate on the first line and its interval
on the second. Its tokens name values that the result object supplies, as in
`b` for the estimate, `se` for its standard error, and `ci95l` and `ci95u` for
the interval bounds. Because `keep` selects rows by their names, `r"^Event "`
leaves out the separate overall ATT.

For the first year after adoption, the table reports −0.053 log points with a
simultaneous band from −0.094 to −0.012 that excludes zero. These are the same
estimates shown in the report, rounded for the publication display.

:::{admonition} Say which sample size you report
:class: tip

For this panel result, `n_units` counts counties and `N` counts county-year
rows in the estimation sample. Labeling the requested unit count as Counties
keeps readers from confusing those two sample sizes.
:::


## Comparing specifications in columns

Placing specifications side by side helps readers see how a particular choice
affects the estimates. We keep the data, population adjustment, and inference
settings fixed so the columns isolate a change in either the comparison group
or the estimation method. In the two doubly robust columns, that choice is
whether later adopters can serve as controls before their own minimum wage
increases.

```{code-cell} ipython3
# Each variant changes one setting of the original fit.
changes = [
    {"control_group": "notyettreated"},
    {"est_method": "ipw"},
    {"est_method": "reg"},
]

# Keep the original fit as the first column and turn every fit into an event study.
models = [result] + [did.att_gt(data, **(spec | change)) for change in changes]
event_studies = [did.aggte(model, type="dynamic") for model in models]
```

```{code-cell} ipython3
# One column per specification and one shared heading over the two doubly robust fits.
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
        "Bootstrap standard errors at the county level appear in parentheses."
    ),
)
comparison_table.make("gt")
```

The two doubly robust columns share a heading because adjacent identical
entries in `model_heads` merge into a column spanner. Their different
comparison groups remain visible in the footer, where you can also check
that the uncertainty method stays fixed across specifications.
With `head_order="dh"`, the table displays the outcome and model headings
together. Use `"h"` or `"d"` when only one of those levels helps identify the
columns.

At event time one, the doubly robust estimates are −0.053 log points using
never-treated counties and −0.055 using not-yet-treated counties. That
closeness is a feature of this sample and these specifications.

:::{admonition} Agreement is not evidence of parallel trends
:class: warning

Every column relies on a parallel trends assumption for its comparison group.
Since columns can agree while sharing a violation of that assumption, their
agreement shows only that the specification choice matters little here.
:::


## Choosing rows and reporting uncertainty

Once the full table is available, you can choose a shorter set of rows for a
particular display. Use literal coefficient names with `exact_match=True`
and regex patterns when it is false.

```{code-cell} ipython3
# Keep three rows by exact name, in event-time order, and relabel them in words.
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
    notes="Simultaneous 95 percent bands in brackets. Rows follow event-time order.",
)
selected_table.make("gt")
```

Selecting rows changes the display rather than the estimates or their
confidence bounds. The three rows keep the critical value computed for all
seven event times. Their bands therefore match the first table rather than a
band recomputed for three effects.

:::{admonition} Stars use pointwise p-values
:class: warning

The table interface computes p-values for {func}`~moderndid.att_gt` and
{func}`~moderndid.aggte` results such as these from the estimate and standard
error using a normal approximation. Stars can therefore disagree with a
simultaneous band that contains zero. Report the bands and omit stars when
you want the table's significance cues to follow simultaneous inference.
:::

For a table that intentionally reports pointwise tests, append `*` to the
estimate token and use `signif_code` to set the three p-value cutoffs.
Omitting `*` from the format suppresses stars, as in the tables above.

```{code-cell} ipython3
# The star after the estimate token asks for pointwise significance marks.
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
        "Bootstrap standard errors at the county level in parentheses. "
        "Pointwise significance levels: * p < 0.10, ** p < 0.05, *** p < 0.01."
    ),
)
starred_table.make("gt")
```

Although its simultaneous band in the first table covers zero, the
adoption-year estimate earns a star at the 10 percent level here.

## Adding statistics and table metadata

An overall effect needs enough context for readers to identify the comparison
it summarizes. The table interface can report details available in each
result, such as the aggregation type, comparison group, or standard error
method. We use `custom_model_stats` to add joint pre-trend p-values directly
from the underlying group-time fits so the diagnostic updates alongside the
analysis.

```{code-cell} ipython3
# Group averages for both comparison groups and each fit's pre-trend p-value.
overall = [did.aggte(model, type="group") for model in models[:2]]
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
    notes="Group aggregation. Bootstrap standard errors at the county level in parentheses.",
)
report.make("gt")
```

The group averages here are −0.033 and −0.032 log points under the two
comparison groups. The joint pre-trend p-values come from the original
group-time fits and do not test these overall effects. Keeping that
distinction in the row label helps readers use the statistic for the
comparison it was designed to assess.

Use the `caption` to describe the question the table answers and `tab_label`
to identify it for cross-referencing or replacement in a document. The notes
are where you record the outcome's units, the aggregation, the inference
method, and any restrictions readers need to interpret the rows.

:::{admonition} Name the clustering level in the notes
:class: tip

The footer's S.E. type names the cluster variable of a clustered fit, as in
"Bootstrap clustered by state". Since a county-level fit's S.E. type reads only
Bootstrap or Analytical, give that level in the notes, as the notes above do.
:::


## Arranging custom panels with MTable

A table with several aggregation types needs more control over its rows than
one column per result can provide. `MTable` accepts a pandas DataFrame whose
cells you construct yourself. We use a row `MultiIndex` for the panel
headings and a column `MultiIndex` for spanners that group related columns.

The panels below compare the population-adjusted estimates with estimates
under unconditional parallel trends. Within each panel, simple, group,
dynamic, and calendar aggregations answer different averaging questions.

```{code-cell} ipython3
# Fit the same specification under unconditional parallel trends.
unadjusted = did.att_gt(data, **(spec | {"xformla": None}))

# Label the panels, the aggregations, and the prefix each component cell uses.
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

# Aggregate both fits four ways and keep event times 0 through 3 in the event study.
results = {}
for panel, model in specifications.items():
    results[panel] = {
        "simple": did.aggte(model, type="simple"),
        "group": did.aggte(model, type="group"),
        "dynamic": did.aggte(model, type="dynamic", min_e=0, max_e=3),
        "calendar": did.aggte(model, type="calendar"),
    }
```

For the cohort and calendar rows, `to_df` returns the cohorts and years in the
same `event_time` column that holds an event study's event times.

```{code-cell} ipython3
# Write each component's label, estimate, and standard error into one cell and
# pad the shorter rows to four components.
n_components = 4
row_index = []
table_rows = []
for panel, panel_results in results.items():
    for kind, (label, prefix) in aggregations.items():
        aggregated = panel_results[kind]
        components = []
        if kind != "simple":
            rows = did.to_df(aggregated).select("event_time", "att", "se").rows()
            components = [
                f"{prefix} = {int(value)}\n{estimate:.3f}\n({se:.3f})"
                for value, estimate, se in rows
            ]
        components += [""] * (n_components - len(components))
        row_index.append((panel, label))
        table_rows.append(
            components + [f"{aggregated.overall_att:.3f}\n({aggregated.overall_se:.3f})"]
        )

# Row groups for the two specifications and a spanner over the component columns.
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
    notes=(
        "Effects in log-employment units. "
        "Bootstrap standard errors at the county level in parentheses."
    ),
    rgroup_sep="tb",
    rgroup_display=True,
    tex_style={"group_header_format": r"\textbf{%s}"},
)
panel_table.make("gt")
```

The component cells show the cohort, event time, or calendar year above
each estimate and standard error. Keeping those labels in the cells avoids
aligning a cohort date with an event time as though they shared an axis. The
[MTable documentation](https://py-econometrics.github.io/maketables/docs/MTable.html)
covers further options for row groups and column spanners.

In the population-adjusted panel, the group average of −0.033 log points and
the post-adoption event-study average of −0.080 log points summarize the same
fit with different weights. Their separation shows why each row needs an
aggregation label. The {ref}`results guide <working-with-results>` explains
which counties and exposure lengths contribute to those averages.


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

For these LaTeX tables, load `booktabs`, `threeparttable`, `makecell`,
`tabularx`, and `array` in your document's preamble so the rules, notes,
stacked cells, and column widths can render. Even when you leave `tab_width`
unset, the default export spans the line width with `tabularx` and needs the
last two packages.

To write the rendered table to disk, pass a format and file name to `save`.
With `show=False`, `save` writes the file without also returning the table
for display.

```python
report.save("tex", "./results.tex", show=False)
report.save("docx", "./results.docx", show=False)
```

:::{admonition} Pass `replace=True` to overwrite a table
:class: important

To rerun `save` after updating your analysis, pass `replace=True`, because by
default it refuses to overwrite an existing file.
:::

For a table embedded in an existing LaTeX document, `update_tex` instead
looks for its `tab_label` and replaces that table. If the label is missing, it
inserts a new table before `\end{document}`. Keep labels stable as
your analysis changes so each update finds its table. For Word,
`update_docx` uses a one-based table position rather than a label. The call
below replaces the first table in the exported document. Without `tab_num`,
`update_docx` would append a new table to the document instead.

```python
report.update_tex("./paper.tex")
report.update_docx("./results.docx", tab_num=1)
```

The [maketables reference](https://py-econometrics.github.io/maketables/docs/ETable.html)
lists the full set of coefficient and metadata formatting options.
