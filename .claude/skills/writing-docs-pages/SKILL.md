---
name: writing-docs-pages
description: Adds, edits, and verifies pages in the moderndid Sphinx docs under docs/source, including the executed MyST-NB example pages, the numbers their prose quotes, and their figures. Use when the user asks to write, edit, restructure, rename, or check docs, a docs page, an Examples, User Guide, Getting Started, or Background page, an index table or toctree, or the docs build and its warnings, or when a change to an estimator, a dataset, or plotting code could leave a page's outputs, numbers, or plots stale.
---

# Writing docs pages

The Prose section of `.claude/CLAUDE.md` and the rules below govern every page, and the sections after them are the procedure. Run commands from the repository root.

## Rules

- Guide prose is a conversation between us and the reader. We say what we're doing and why in the first person plural, and we talk to the reader as "you" about what they see, what it means, and what they could do differently, the way you'd walk a colleague through an analysis. Tell the reader what to look for before an output and what it means after, tie each step back to the question the page is answering, and let that reasoning carry the transitions, so the page never reads as a list of findings. A clipped statement such as "The estimates are negative." becomes a sentence that explains, such as "You can see that every estimate after adoption sits below zero." No standalone sentence runs under about ten words, not even for emphasis, so fold a short one into its neighbor or give it its reason. A continuation after a display equation or a list item doesn't count. Vary the openers, and keep clear of chatty tics such as "Let's", "Notice that", "simply", and rhetorical questions. Installation pages and quickstarts stay short and direct, since they are instructions and code.
- The introduction of `user_guide/example_staggered_did.md` was written by the maintainer and sets the voice for every page. It names the question outright ("The central question we'll ask is whether..."), names the problem a method avoids ("We're going to avoid this common pitfall by..."), and tells the reader what they'll see before they see it ("What you'll see at the end of this is..."). It prefers the full phrase ("the staggered timing of this policy change", "minimum wage") to a clipped one. Each step opens with its purpose in plain words rather than with an argument in code. "We" marks a decision the guide makes or the path it takes, about once a section. Other sentences take the data, the estimator, the result, or "you" as their subject. Match that voice, never its wording. Each page finds its own phrasing for the intro, the section headings, and the transitions between choices and checks, since sentences carried over from another guide make the pages read like a template.
- Say each thing once and briefly, since readers shouldn't have to read for long to get the answer. Brevity comes from cutting repetition, preamble, and recaps, never from compressing sentences. A page skips preamble and recaps, quotes only the numbers the reader needs from an output, gives each design choice a sentence or two, and keeps each check to its call and a short reading of what changed.
- Guides teach the design choices they make. Where a page picks a comparison group, an estimation method, covariates, a base period, anticipation periods, clustering, a bootstrap or simultaneous bands, an aggregation, a balanced event window, or a tuning setting, it says why, what the choice assumes, and what it implies for the results. It also names the alternative a reader could pick and links to where the docs show the consequence. That reasoning replaces line-by-line restatement of math or code the reader can already see.
- Each example threads one question from start to end. On data a moderndid generator simulates, the question is how well the estimator recovers the effects planted in it, and the page checks that only after the reader has seen the estimates, linking back to where they were shown. On real data, such as `load_mpdta()`, it is the question the data was collected to answer.
- Example pages are MyST-NB pages executed during the build, so an example that stops working fails the build. Keep each page's cells quick, since Read the Docs stops a build at 15 minutes and a cold build already spends about six on the API pages. An estimation too slow for that budget loads a stored result, as Stored results describes.
- Verify every numerical claim against executed data or results and keep its units and precision consistent. Sample sizes, cohort composition, ranges, and similar data qualities belong in prose; their verification runs in scratch code rather than display-only notebook cells. Estimated effects and uncertainty should be visible in the actual result reports, plots, or short public converter calls where they help the reader. A claim about behavior such as how bands move across bootstrap seeds needs a check broad enough to carry it. Lead with runnable code, and use no Markdown tables except a reference list of names, an index page's table of pages, or a short table of parallel facts that prose would bury. Keep those last ones to a few across the docs, since a page full of tables and lists reads as sloppy. A table that would run past about eight rows splits by group into a `tab-set`, one table per tab.
- Print moderndid's results in full with `print(result)`, so readers learn to read the real report rather than a hand-picked subset of it. A variant cell that changes one argument may print only the numbers its comparison reads.
- An example writes out its full specification before the first estimation. Each variant then shows only the arguments that change and points back to it.
- Treat the old pages as raw material and pick the presentation that teaches best. A section of design choices opens with the full specification in one cell, a `spec` dictionary whose comments group its arguments by the question they answer. Two to four H3s follow in the same order, one per group. Each names its group's answer in words the comment echoes. The paragraph or two under each explains the choices rather than listing the arguments and links every alternative to the check that tries it. A setup heading never repeats a check's heading, since MyST would then move the check's anchor. A one-sentence lead-in and the call that passes `spec` close the section. Checks run in the same order as the choices they revisit. Explain the differences beside their results and add a compact comparison table only when it makes the comparison easier to read. A section that answers the page's question states the answer before any aside, and an aside such as why other aggregations differ gets its own subsection. A page on real data says early what makes the data a natural experiment and why its outcome is the one to study.
- Math stays light on example pages, since the Background pages hold the derivations. An example states its estimand in a display and its assumptions in words, links to `../background/<page>` for the identification and aggregation formulas, and explains each argument in prose where the page sets it. A page follows its analysis, from the data through the specification and the results to the checks, rather than a template borrowed from another library's docs.
- Styling in `custom.css` keeps one accent color, works in both color schemes, and gives every element a job, with motion that eases and turns off for readers who ask for reduced motion. Test interactive features over HTTP, as readers load the site, rather than from local files.
- The FAQ in `faq.md` follows marimo's: H2 topic sections, each question an H3 phrased the way a DiD practitioner would ask it and ending in a question mark, and short answers that link to the guide covering the topic. That's the one place a heading ends in "?".
- A page's H1 is a short sentence-case phrase with no colon. Section headings are phrases that may name a question without a "?", nothing is numbered, and headings stop at H3. No page has a Summary, Conclusion, Overview, or Next steps section, because the last paragraph reads the last result and links to the page that takes it further.
- `_ext/last_updated.py` puts the date of each page's last commit at its foot, so a new page shows no date until it's committed.

## Place the page

Example pages live in `docs/source/user_guide/` as `example_<name>.md`, and the READMEs under `moderndid/*/` link to their URLs, so a rewrite keeps each page's name. Each page keeps its `(example_<name>)=` label above the H1, since other pages link to it with `{ref}` and `:ref:`. Public functions' docstrings link to example pages and guide sections by label, as in `` :ref:`example_honest_did_external` ``, so a rewrite keeps every label that `grep -rn ":ref:" moderndid` finds, written `(label)=` in MyST. Converting a page from reST deletes its `.rst` in the same change, because two sources with one name fail the build. `examples/index.rst` lists every example in its hidden toctree and in its two-column `section-index-table` of a link and a one-line description, so a new example adds both, and a page in no toctree fails the build (`toc.not_included`). `_user_guide_sections` in `conf.py` leaves `user_guide/example_*` pages out of the User Guide's section headings, so they sit under Examples in the sidebar. Every index page keeps a hidden toctree, the two-column table, and the `mdid-footer-logo` paragraph that closes it. Pages about working on the project, such as setup, the Git workflow, testing, reviewing, and releasing, go in `contributing/`, and pages about how the library works inside go in `dev/`, so each section's index table and toctree gain the new page. A long page gets subheadings before it splits into new pages.

## Page skeleton

An example page opens like this, with the hidden setup cell after the intro and before the first `##`.

````markdown
---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_staggered_did)=

# Title in sentence case

An intro paragraph that names the design and the question the page answers.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```
````

Without the front matter the cells never run. A page has one H1 and skips no heading level (`myst.header`). The setup cell holds only what readers don't need to copy, and the page imports moderndid as `did` in its first visible cell. `user_guide/example_staggered_did.md` shows the whole pattern, from the data through the specification in one dictionary to checks that each change one argument.

## Cells

- End each cell in one expression or a `print`, rounded to what the prose quotes, as in `round(float(x), 3)`. The cell that defines `spec` is the one exception and ends in its assignment.
- Prefer the package's result converters or a table API when a numeric display helps. Avoid loops and custom formatting that only restate prose or earlier results. Round quoted estimates consistently with the displayed values and use a stated precision when a table needs fixed decimals.
- Every estimation that draws a bootstrap passes `random_state`, so the build reproduces the numbers the prose quotes. That includes `att_gt` under its default `cband=True`, because `aggte` then draws one for the simultaneous bands even when `boot=False`.
- moderndid's plot functions, such as `did.plot_gt` and `did.plot_event_study`, return plotnine plots, so a plot cell ends with the plot as its last expression and draws at the setup cell's size. Add `+ did.theme_moderndid()` to every plot for white panels without grid lines. Prefer moderndid's plot functions wherever one draws what the page needs. Every figure is 12 inches wide at 100 dpi so it fills the column, a single panel is 12 by 5, and a faceted plot takes a taller size that keeps its panels from squeezing. Pages name points and bands by color, so a palette change means rereading the prose.
- Example code reads in steps separated by blank lines, with a comment of one to three lines above each step that says what it computes and, where it matters, why. Name each value on its own line, so the code reads as a sequence of definitions.
- Code a reader may want but need not read goes in a cell tagged `hide-input`, which folds it behind a Show code line. That covers hand-made figures, tables built for display, and checks against a simulation's truth, while moderndid calls and short reads of their results stay visible. Split a cell when a moderndid call sits among display code, and never let a visible cell use a name defined only in a folded one. Prose before a folded cell describes the table or figure, not the cell. Don't tag cells `hide-output`, whose toggle would read the same.
- Keep display code to calculations the page teaches or output the reader needs. Sample descriptions and arithmetic a reader can follow from existing results belong in prose, even when their former code could be folded. Verification of those facts belongs in scratch scripts. Method-essential loops, custom plotting examples, and table-construction tutorials remain useful when that computation is the point of the page.
- A cell meant to fail needs `:tags: [raises-exception]` and a hidden `%xmode minimal` cell earlier on the page. Any other error stops the build. Stderr is dropped, so warnings and progress bars never show.
- Code cells are not linted. Write them in ruff style by hand, with two blank lines after a top-level `def`, and wrap lines at about 100 characters, well before the code box scrolls sideways.
- A new import needs its package in pixi's `[feature.docs]` dependencies and in the `doc` extra of `pyproject.toml`, which Read the Docs installs along with `all`.

Callout boxes are written `:::{admonition} Title` with the `:class:` that fits what the box does, so a page shows a mix of colors. `important` (purple, styled in `conf.py`) states a rule the code depends on, `warning` (orange) a mistake that gives wrong results, `tip` (teal) practical advice, `note` (blue) a side detail, and `danger` (red) only a "Don't" that would invalidate a result. Boxes break up dense stretches and pull out what a reader must not miss, so a long example page carries about one for every 300 words of prose and never two in a row. A stretch of prose that runs past about 250 words without a box, a code cell, or a figure needs breaking up. The box takes its sentence out of the paragraph rather than repeating it. A box title is two to seven words in sentence case, imperative for a rule or advice, a claim for a warning, and a topic for a note, and a danger title starts with "Don't". The body runs one to three sentences in the page's voice with no list, and the box follows the paragraph whose point it sharpens, never another box. Put estimator logic in `$$` displays rather than long inline math.

## Stored results

- An estimation too slow for the build, such as a long bootstrap or a dynamic balancing fit, runs once in a local build and saves its result in `docs/source/prerun`, which `conf.py` puts on the kernels' `PYTHONPATH`. The page shows the call in a cell tagged `skip-execution`, and the hidden cell after it, tagged `remove-cell`, runs `result = stored("name", lambda: <the visible call>)`. The hidden setup cell imports it with `from prerun import stored`, and the call keeps the visible cell's seed and settings.
- The `skip-execution` cell holds only the estimation. Setup the call needs runs in a cell before it, and the printout or plot of the result in a cell after the hidden one, so the hidden cell repeats nothing but the call. A fit of several statements goes in a `def fit_name()` inside the hidden cell.
- `stored` pickles the result object and loads an existing file without looking at the call. Delete a result's file after changing its call, the estimator, or the result's container, since an old file loads stale numbers or fails to unpickle. Find the stored results with `grep -rn "stored(" docs/source --include="*.md"`.
- A missing file is computed and saved by whichever local run reaches it first, the draft below or the build. On Read the Docs `stored` raises instead, so a file left out of a commit fails the build at once. Tell the user each new `.pkl` must be committed with its page.

## Draft numbers before building

Candidate numbers come from running the page's cells in the docs environment.

```bash
pixi run -e docs python - <<'EOF'
import moderndid as did
# the page's cells, printing what the prose will quote
EOF
```

## Build

Build into a scratch directory, never `docs/_build` or `docs/_doctree`, which hold the maintainer's own build. Run one build at a time, since two builds can corrupt a shared execution cache, and run it through pixi so the notebook kernel is the docs environment.

```bash
pixi run -e docs sphinx-build -b html --keep-going -d <scratch>/doctree docs/source <scratch>/html 2>&1 | grep -E "(WARNING|ERROR):|build (succeeded|finished)"
```

Reuse the same scratch folders from build to build, because a cold build spends about six minutes on the API pages and a later one rereads only the pages that changed. The execution cache in `<scratch>/.jupyter_cache` keys each page on its own cells, so delete it after changing library code a page calls. A failing cell's traceback is in the build log. Other pages already raise warnings, so a page passes when no warning or error names its file. Link another example page by its label, as in `` {ref}`Extended TWFE <example_etwfe>` ``, since a bare name also matches the label and MyST warns that the link is ambiguous. Link other sections by relative path, as in `../background/drdid`, a section of the same page by its anchor, as in `[The data](#the-data)`, and the API with roles such as `` {func}`~moderndid.att_gt` ``. Intersphinx covers Python, NumPy, pandas, polars, SciPy, matplotlib, and plotnine.

## Verify

Read the outputs in `<scratch>/html/user_guide/<page>.html`.

1. A role such as `{func}` that finds no target renders as plain code without a warning. A resolved one sits inside an `<a>` tag, so `grep -o '.\{2\}<code class="xref' <page>.html | grep -v '">'` prints only the unresolved ones.
2. Match every number in the prose to an output and recompute derived ones. After a page's numbers move, grep the other pages for them, since `estimator_overview.rst`, `plotting.rst`, and the example pages quote each other.
3. Read each figure the page's `<img>` tags point to in `<scratch>/html/_images`, and get its pixel size from `file`. plotnine stores each image at twice its display size, so a 12-inch figure measures about 2400 pixels wide, and a height near 1000 is one panel. Look for grid lines, overlapping labels or legends, and squeezed panels.
4. Screenshot the page into the scratch directory in light mode (`preferredColorScheme=1`) and dark mode (`0`), crop each tall image into pieces, and read every piece. The window must be as tall as the page, and the virtual time budget lets MathJax finish, since without it the math comes out blank. Tabs and folded cells need a click, so check those over HTTP by serving `<scratch>/html` with `python -m http.server`.

   ```bash
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=20000 --window-size=1400,22000 --blink-settings=preferredColorScheme=1 --screenshot=<scratch>/light.png "file://<scratch>/html/user_guide/<page>.html"
   ```

5. Copy the page's prose into a scratch file, leaving out code cells, math, front matter, tables, and link targets, and read it against the Prose rules in `.claude/CLAUDE.md` and the avoid-ai-writing skill. Run its detector on that file with `--context technical --source-mode rendered-markdown`, since on the raw page it scores code and math and counts each anchor link as a hashtag. A low vocabulary-diversity flag is a prompt to look rather than a verdict, since the ratio falls as any page gets longer. Clear pages measure about 0.5 to 0.6 in 250-word windows, and a three-word phrase repeated four or more times is worth a look.
6. Have a second reviewer check each explanation of a design choice against the code, the docstring, the paper, and the outputs, since a teaching pass tends to add reasons that sound right and aren't.
