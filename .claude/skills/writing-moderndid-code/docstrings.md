# moderndid docstrings

## Contents

- Estimator entry point, the template for a public function that runs an estimator
- Computation or helper function, the template for a public function of arrays or results
- Result container, the template for a NamedTuple result
- One-line docstrings, for private helpers, formatters, fixtures, and modules
- Every docstring, the summary, body, type spellings, Returns bullets, math, and citation rules

Sphinx renders these docstrings through numpydoc, and signatures carry no type hints, so the docstring is the only place a reader sees types. Start from the template for the kind of object and replace each `<placeholder>`. Text outside the angle brackets, including section underlines, type spellings, and default forms, is the wording new code uses.

## Estimator entry point

A user-facing function such as `att_gt`, `ddd`, or `cont_did`.

```python
def <name>(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    alpha=0.05,
    boot=False,
    biters=1000,
    random_state=None,
):
    r"""<Estimate> <what> <for which design>.

    <What the function estimates and the problem the method solves, citing the
    paper as [1]_.>

    <What it returns and how it computes the uncertainty.>

    <The choices the user controls, and what each warning it raises means.>

    See the :ref:`<guide name> <example_<label>>` for <a full analysis of which data>.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome column.
    idname : str, optional
        The name of the unit identifier column. Required for panel data.
    alpha : float, default=0.05
        The significance level.
    biters : int, default=1000
        The number of bootstrap draws. Only applicable if ``boot=True``.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across calls.

    Returns
    -------
    <Name>Result
        Object containing <what the result holds>:

        - **<field>**: <Description with no final period>
        - **<field>**: <Description>

    See Also
    --------
    <next_function> : <What the reader does next with the result>.

    Notes
    -----
    Let :math:`<symbol>` denote <meaning>.

    <A sentence that leads into the display without a colon>

    .. math::

        <equation>.

    References
    ----------

    .. [1] <Author, A., & Author, B.> (<Year>). "<Title>." <Journal>, <vol>(<issue>),
           <pages>. https://doi.org/<doi>
    """
```

- The summary names what the function estimates. `att_gt` and `aggte` show the full pattern, and `docs/source/api/<family>.rst` renders this docstring as the function's page.
- A function that a guide walks through has no Examples section. Its main body ends with the `See the :ref:` sentence instead, since docstring code runs on every docs build and the guide carries the worked example. A function no guide covers may keep an Examples section in the `.. ipython::` directive, as the drdid wrappers do.
- The References section is the only place the paper appears in full, and the body cites it as `[1]_`. Add the blank line after the dashes as shown.

## Computation or helper function

A function of arrays or results, such as `compute_aggte` or `mboot`.

```python
def <name>(<array>, <setting>=<default>):
    """<Compute> <what> <for which input>.

    <One or two short paragraphs on what the function computes and which
    inputs it holds fixed. Link to the function that owns the math with
    :func:`~moderndid.<owner>` rather than restating it.>

    Parameters
    ----------
    <array> : ndarray
        <Meaning, shape, and requirements, as in "Influence function matrix of
        shape (n_units, n_cells).">
    <setting> : int, default=<default>
        <Meaning and allowed values.>

    Returns
    -------
    ndarray
        <What it holds and its shape.>
    """
```

- A helper names no authors and cites no paper. It describes the method in its own terms, since the entry point's References section is the one place the paper appears.
- A function that returns a dict documents each key with the same bullets as a result, after an intro line such as "Dictionary containing the cell estimates:".

## Result container

```python
class <Name>Result(NamedTuple):
    """Container for <what the result holds>.

    Attributes
    ----------
    <field> : <type>
        <Description.>
    <field> : ndarray, optional
        <Description.>
    """

    #: <One-line description.>
    <field>: <type>
    #: <One-line description.>
    <field>: np.ndarray | None = None
```

- The class lives in its family's `container.py`, with its factory function if it has one.
- A result that implements the publication-table interface says so in one sentence after the summary, as `AGGTEResult` does.

## One-line docstrings

- A private helper gets one line that states its intent or the constraint it keeps, as in `"""Return the count with the noun in the singular or the plural."""`.
- A formatter gets one line such as `"""Format an aggregated treatment effect result for display."""`.
- A conftest fixture gets one line such as `"""Generate DDD data with covariates."""`.
- A module docstring is one line, such as `"""Result containers for multi-period DiD estimators."""`.

## Every docstring

- The summary line has no comma, not even in a list, and reads as one clause in the imperative ("Compute", "Return", "Format") or, for a container, "Container for ...".
- The body runs in short paragraphs of one idea each, such as what the function computes, which inputs it holds fixed, how uncertainty carries through, or what a warning means. It uses no bulleted or numbered lists outside the Returns bullets and never points to its own sections, as in "See Notes".
- A Parameters entry says only what the argument is and what it requires, such as its meaning, units, shape, and allowed values. What the function as a whole does with it goes in the body.
- Type spellings follow the existing entries: `DataFrame`, `str`, `str, optional`, `float, default=0.05`, `bool, default=True`, `list[str], optional`, `{"dr", "ipw", "reg"}, default="dr"`, `int, Generator, optional`, `ndarray`. `optional` marks a `None` default, and every other default is written `default=<value>`.
- Returns opens with the type on its own line, then an intro sentence ending in a colon, then one bullet per field as `- **name**: Description` with no final period. Never join two fields in one bullet or quote a name as `'name'`. A plain array or scalar gets a one-line description and no bullets.
- The sections in use are Parameters, Returns, See Also, Notes, References, Attributes for containers, and Examples where no guide covers the function. No docstring has a Raises, Warns, or Warnings section. The body says in prose what the function raises or warns about.
- Heavy math goes in Notes as `.. math::` blocks, with a blank line after the directive, the equation indented, and the display ending in a period or comma. A sentence leads into each display without a colon, as in `aggte`. Inline `:math:` is for single symbols and short fragments such as :math:`0 < \zeta < 1`. Use an `r"""` docstring when it holds a backslash.
- Each piece of math lives in one docstring. Other functions name the quantity and link to the owner with `:func:`, since a copy goes stale when one of them changes.
- Never mention R, an R package, or another library's behavior. Describe what the code computes, as in "performs k-fold grid search over the penalty".
- The Prose rules of `.claude/CLAUDE.md` apply in full, so write "such as", "for example", or "that is" instead of Latin abbreviations, "percent" instead of "%", and no em dashes or tacked-on ", which" and ", <verb>ing" clauses.
