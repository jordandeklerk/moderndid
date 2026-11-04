---
name: writing-moderndid-code
description: Use when writing, changing, fixing, refactoring, porting, or reviewing any Python code in moderndid/ or tests/, such as an estimator, function, argument, result container, printed report, to_df converter, plot, public export, docstring, or test, or when the user asks about moderndid's code conventions. Holds the project's standards for package layout, validation and error messages, warnings, results and reports, exports and API pages, numpydoc docstrings, and comments, plus the checklist for reviews. Tests and the test-first workflow belong to writing-moderndid-tests, docs pages to writing-docs-pages, and comparisons with R packages to comparing-with-r.
---

# Writing moderndid code

The Code section of `.claude/CLAUDE.md` sets the rules, and this skill adds the specifics behind them, taken from the code. Ruff and the pre-commit hooks already enforce formatting, import order, numpydoc layout, line length, and the `print` ban, so this skill leaves those out. `moderndid/did/` shows most of what follows in one family, and `docs/source/dev/architecture.rst` and `docs/source/dev/new_estimator.rst` explain the design behind it.

The Bad and Good pairs cover what reviews keep catching. Bad lines come from older modules and past reviews. A neighbor that reads like one is no precedent.

## Before handing over

1. Run `pixi run lint`. The hooks see only tracked files, so also run `pixi run -e check pre-commit run --files <path>` on each new file. The hooks rewrite what they touch, so while another session edits the tree, check with `pixi run -e check ruff check --no-fix <path>` and `pixi run -e check ruff format --check <path>` instead.
2. Run only the test files that cover the change, found by grepping `tests/` for the changed names, as in `pixi run -e dev pytest tests/did/test_aggte.py`. Never run the whole suite.
3. Go through the checklist under Reviewing a diff for your own changes, since no tool checks those items.
4. List every new public name, argument, and result field in your report, so the maintainer can confirm or rename it.

## Reviewing a diff

Reviews of this repository keep finding the same problems, so check every new or changed function and test against this list and say which items you checked. Older modules break many of these rules, which never excuses new code, so flag every item the new code misses. Read the current code before saying what an existing function, file, or test folder does.

- Signature: no type hints on a documented function, `alpha` and `biters` for new arguments, and the shared names listed under Signatures and arguments for shared ideas.
- Checks and errors: `ValueError` or `TypeError` rather than a bare `Exception`, a message that names the argument and its value and says what to pass, a warning in the same form with a `stacklevel`, no suppressed warning, and no silent NaN or empty result for an edge case.
- Placement: result types in `container.py` with Attributes and `#:` comments, a formatter registered with `attach_format`, a converter in `_DISPATCH`, and the export steps and API page for every new public name.
- Docstrings: a summary line with no comma, no clause tacked on after a comma, no Latin abbreviation, no Raises section, no pointer to its own sections such as "See Notes", Returns bullets as `- **name**: description`, long math in a `.. math::` display in Notes, authors only in an entry point's References, and no mention of R.
- Code: imports at the top, no module-level constant for a one-off number, and comments that say why in plain sentences without a template opener such as "Since X, Y".
- Tests: the checks under Reviewing a test in the writing-moderndid-tests skill, starting with the red run and the mutations each new test reports.

## Package layout

```text
moderndid/
├── did/                    # an estimator family
│   ├── att_gt.py           # public entry point
│   ├── compute_att_gt.py   # numerical work as plain functions of arrays
│   ├── container.py        # NamedTuple results and their factories
│   ├── format.py           # printed reports
│   └── README.md           # short overview that the docs link to
├── core/                   # code the families share
│   ├── preprocess/         # checks and prepares data (validators.py, config.py, builders.py, transformers.py)
│   ├── bootstrap.py        # bootstrap and variance helpers
│   ├── converters.py       # to_df converters
│   ├── data.py             # load_* datasets and gen_* simulators
│   ├── format.py           # report helpers
│   ├── maketables.py       # publication-table builders
│   └── parallel.py         # parallel_map for group-time cells and other independent tasks
├── plots/                  # plot functions
├── cupy/                   # GPU backend
├── spark/                  # Spark execution of att_gt
└── dev/                    # placeholders for planned families
```

In `did`, `att_gt.py` checks the arguments, builds the config, runs the preprocessing builder, and returns the result through the `mp()` factory. Larger families split the work into subfolders such as `didtriple/estimators/` and `didtriple/bootstrap/`.

Some families predate this layout. drdid defines a result beside each estimator in `drdid/estimators/`, didhonest keeps its results in the modules that compute them, and neither didhonest nor npiv has a `format.py`. Don't copy those layouts. Move an old type only in a change of its own, since a move changes import paths, `_lazy_imports`, and the API pages.

## Signatures and arguments

- A function whose docstring has Parameters and Returns sections takes no type hints, since the docstring owns the types. Hints stay on NamedTuple and dataclass fields, on `__maketables_*__` methods with one-line docstrings, and on module-internal helpers with one-line docstrings. The heavily hinted modules in `core/preprocess/` predate the rule, so don't copy them.
- Reuse the names the estimators share. `data`, `yname`, `tname`, `idname`, `gname`, `xformla`, `weightsname`, `control_group`, `anticipation`, `base_period`, `est_method`, `boot`, `cband`, `clustervars`, `random_state`, `n_jobs`, and `backend` each mean the same thing everywhere, and `docs/source/dev/architecture.rst` explains the treatment arguments that differ by design (`dname`, `pname`).
- Name the significance level `alpha` and the number of bootstrap draws `biters` in every function you add. Where an older function still spells them `alp`, `n_boot`, or `nboot`, rename it only in a change of its own, outright with no deprecated alias, and list the break in your report for the maintainer's release notes rather than editing the release notes.
- Randomness enters as `random_state` and reaches every draw through one `np.random.default_rng(random_state)`, never the global `np.random` functions or `RandomState`.

## Validation and errors

Check arguments at the top of a public function, before any computation. `check_columns(data, yname=yname, tname=tname, ...)` in `core/preprocess/validators.py` names every missing column and suggests the closest one, and `_option_errors` words every invalid option the same way. A family with a config dataclass in `core/preprocess/config.py` gets the remaining checks from the builder.

- Raise `ValueError` for a bad value, shape, or option and `TypeError` for a wrong type. A missing optional dependency raises `ImportError` naming its extra, and `RuntimeError` is for a failure after valid input. There are no custom exception classes.
- A message starts with the argument as the user wrote it, says what is wrong, says what to pass instead, and ends with a period.

  ```python
  # Bad
  raise ValueError("The dose aggregation doesn't use min_e.")
  raise ValueError("No treated units found. Cannot estimate treatment effect.")

  # Good
  raise ValueError(f"control_group={control_group!r} is not valid. Must be 'nevertreated' or 'notyettreated'.")
  raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")
  raise ValueError("dvals holds missing or infinite values. Pass finite doses at which to evaluate the curves.")
  ```

- A helper that checks arguments for several public functions takes the caller's argument names as parameters, so its message names only arguments the user passed. `ddd` passes its names to the shared helpers rather than rewriting their messages.
- Re-raise with `raise ... from err` so the cause survives.

## Warnings

- A public function warns with `warnings.warn(message, UserWarning, stacklevel=2)`. A helper adds one to `stacklevel` for each frame below the public function.
- A warning names what happened and says what the function did about it or what to pass instead.

  ```python
  # Bad
  warnings.warn(f"Propensity score estimation (IPT) failed in bootstrap {b}: {e}", UserWarning)
  warnings.warn("Bootstrap standard error is zero or NaN.", UserWarning)

  # Good
  warnings.warn(f"alpha={alpha} is above 0.10. Using alpha=0.05.", UserWarning, stacklevel=2)
  warnings.warn(
      f"Units in id_col='{id_col}' appear in every period. For panel data, use ddd_mp.", UserWarning, stacklevel=3
  )
  warnings.warn(
      f"The regression dropped the treatment cells {_format_cells(dropped)} as collinear with its other "
      "terms. Controls that span a cell or a cohort with no untreated period cause this. Their estimates "
      "are NaN.",
      UserWarning,
      stacklevel=2,
  )
  ```

- Never suppress a warning in library code, whether with `warnings.catch_warnings`, `simplefilter`, or `np.errstate`. Guard the computation so the warning can't arise, such as dividing only where the denominator is positive, or let the warning show. A guard keeps every number identical wherever the old expression was defined.

## Results, reports, and conversions

- A result is a NamedTuple in the family's `container.py`. Its class docstring lists the fields under Attributes, and a `#:` comment above each field gives the one-line description autodoc shows, as `AGGTEResult` in `did/container.py` does. A factory such as `mp()` assembles the result when an entry point would otherwise repeat that work.
- A new field on an existing result changes a public type, and the pickled results in `docs/source/prerun` stop loading when their container changes. Prefer a design that leaves the fields alone, such as an argument on the converter, and propose the field in your report.
- Leave what an existing report prints alone unless the task is that report, and never add an HTML repr.
- A printed report is `format_<name>_result(result)` in the family's `format.py`. A `to_df` converter is `<classname>_to_polars(result)` in `core/converters.py`. Each follows this shape.

  ```python
  # did/format.py
  def format_aggte_result(result):
      """Format an aggregated treatment effect result for display."""
      lines = []

      lines.extend(format_title("Aggregate Treatment Effects"))
      lines.extend(format_event_table(...))
      lines.extend(format_significance_note(band=True))

      lines.extend(format_section_header("Data Info"))
      lines.append(f" Control Group: {control_text}")

      lines.extend(format_section_header("Estimation Details"))
      lines.append(f" Estimation Method: {method_text}")

      lines.extend(format_section_header("Inference"))
      lines.append(f" Significance level: {alpha}")

      lines.extend(format_footer("Reference: Callaway and Sant'Anna (2021)"))

      return "\n".join(lines)


  attach_format(AGGTEResult, format_aggte_result)


  # core/converters.py
  def aggteresult_to_polars(result):
      data = {
          "event_time": event_times,
          "att": att,
          "se": se,
          "ci_lower": ci_lower,
          "ci_upper": ci_upper,
      }
      df = pl.DataFrame(data).filter(~pl.col("se").is_nan())
      return _whole_numbers_as_int(df, ["event_time"])


  _DISPATCH = {
      "MPResult": mpresult_to_polars,
      "AGGTEResult": aggteresult_to_polars,
  }
  ```

- A report ends with `format_footer("Reference: <Authors> (<Year>)")`. That printed line is the one place outside an entry point's References section where authors appear, so formatter docstrings still name no one.
- The family's `__init__.py` imports `format.py` with a relative import, as in `from .format import format_aggte_result`, or `attach_format` never runs. A dotted import such as `import moderndid.npiv.format as f` fails, since `moderndid.npiv`, like `drdid`, `didml`, and `etwfe`, names the function rather than its subpackage.
- `to_df` finds a converter by class name in `_DISPATCH`. A new result gets a `<classname>_to_polars` function there with columns named like the other converters' (`att`, `se`, `ci_lower`, `ci_upper` beside the cell's keys). The filter drops a reference row whose standard error is NaN. The docstring lists each column as a Returns bullet.
- A plot is a function in `moderndid/plots/plots.py` that returns a plotnine `ggplot`, exported through `moderndid/plots/__init__.py` and the root `_optional_imports` with the extra `"plots"`.

## Adding a public name

1. Define it in its family, then import it and add it to `__all__` in the family's `__init__.py`.
2. In `moderndid/__init__.py`, add it to `__all__` at its sorted position and map it in `_lazy_imports`, in `_optional_imports` with its extra, or in `_aliases` for a renamed export. A function named like a subpackage, such as `npiv`, is imported eagerly at the bottom of the file instead.
3. Add it to the autosummary list of `docs/source/api/<family>.rst` under `.. currentmodule:: moderndid`. Converters go in `docs/source/api/results.rst`.
4. A new dependency goes in the extras of `pyproject.toml` and the matching pixi feature, and the import error names the extra.

Most families leave their formatters and converters out of both `__all__` lists, since printing and `to_df` reach them.

## Adding an estimator family

`docs/source/dev/new_estimator.rst` develops an estimator from its target parameter to its result. In code, a family needs the folder layout above, an entry point whose docstring holds the paper's References, argument checks through `check_columns` and a config or a `preprocess_*` wrapper, a compute module, `container.py`, `format.py`, a converter, the export steps above, an API page added to the table and toctree of `docs/source/api/index.rst`, and a `README.md`. Tests come first under the writing-moderndid-tests skill, the comparison with the reference R package follows the comparing-with-r skill, and the example page follows writing-docs-pages. A port describes the method in its own terms and keeps every mention of the source R package out of the code.

## Numerics

- Group-time cells, the bootstrap draws of `did_multiplegt`, and the separate fits of `dyn_balancing` run through `parallel_map` in `core/parallel.py`, which takes `n_jobs`.
- Bootstrap helpers in `core/bootstrap.py` are plain NumPy functions that branch to a CuPy version, where one exists, when `get_backend()` isn't NumPy.
- A fit that can fail on a cell, such as a propensity score reaching one, names the cell in a warning and leaves the cell out or NaN, as `att_gt` does, rather than clipping the value silently.

## Comments

A comment explains why in one or two lines and states the reason as a plain fact, as in `# Rows that differ only in the sign of a zero share a key because adding 0.0 turns -0.0 into 0.0.` (`did/shards/layout.py`). Its opener follows from its subject. A run of comments that open with one template, such as "Since X, Y" or "As in X, Y", reads as machine-written. So does a comment that narrates what the next line does. There are no section dividers, no TODO markers, and no field comments except the `#:` lines in containers.

```python
# Bad
# Influence function of treated component
# Since each draw solves a small dual problem, the bootstrap runs on the CPU under every backend.
# Since statsmodels stops on an absolute change in the deviance, the screen rescales the weights too.
# Every route reports pointwise intervals, and agg_ddd builds the simultaneous bands.

# Good
# The bootstrap runs on the CPU because each draw solves a small dual problem.
# Settling the inference options before routing gives every data layout the same interval level.
# Every route reports pointwise intervals. agg_ddd builds the simultaneous bands.
```

## Docstrings

Every public docstring starts from a template in [docstrings.md](docstrings.md), which also holds the type spellings, default forms, math, and citation rules. Read it before writing or editing a docstring.

## Tests

Every test, and every change to behavior that a test should pin, follows the writing-moderndid-tests skill. Load it before writing, changing, or reviewing a test and before fixing a bug. R validation tests in `tests/validation/` follow the comparing-with-r skill.
