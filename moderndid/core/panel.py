"""Panel data utility functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from moderndid.core.dataframe import from_polars, to_polars
from moderndid.core.format import (
    _make_table,
    adjust_separators,
    format_footer,
    format_section_header,
    format_title,
)
from moderndid.core.preprocess.utils import (
    get_first_difference as _get_first_difference_impl,
)
from moderndid.core.preprocess.utils import (
    get_group as _get_group_impl,
)
from moderndid.core.preprocess.utils import (
    is_balanced_panel as _is_balanced_panel_impl,
)
from moderndid.core.preprocess.utils import (
    make_balanced_panel as _make_balanced_panel_impl,
)

__all__ = [
    "PanelDiagnostics",
    "are_varying",
    "assign_rc_ids",
    "complete_data",
    "deduplicate_panel",
    "diagnose_panel",
    "fill_panel_gaps",
    "get_first_difference",
    "get_group",
    "has_gaps",
    "is_balanced_panel",
    "make_balanced_panel",
    "panel_to_wide",
    "scan_gaps",
    "wide_to_panel",
]


@dataclass
class PanelDiagnostics:
    """Structured report returned by :func:`diagnose_panel`.

    Attributes
    ----------
    n_units : int
        Number of unique cross-sectional units.
    n_periods : int
        Number of unique time periods.
    n_observations : int
        Total row count.
    is_balanced : bool
        Whether every unit is observed in every period.
    n_duplicate_unit_time : int
        Number of rows that share a unit-time pair with another row.
    n_unbalanced_units : int
        Units observed in fewer than *n_periods* periods.
    n_gaps : int
        Missing unit-time combinations in the full cross-product.
    n_missing_rows : int
        Rows containing at least one null value.
    n_single_period_units : int
        Units observed in only one period.
    n_early_treated : int or None
        Units already treated in the first observed period.
        ``None`` when no treatment column is provided.
    treatment_time_varying : bool or None
        Whether the treatment indicator changes within units.
        ``None`` when no treatment column is provided.
    suggestions : list[str]
        Actionable remediation messages.
    """

    n_units: int
    n_periods: int
    n_observations: int
    is_balanced: bool
    n_duplicate_unit_time: int
    n_unbalanced_units: int
    n_gaps: int
    n_missing_rows: int
    n_single_period_units: int
    n_early_treated: int | None
    treatment_time_varying: bool | None
    suggestions: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a formatted string representation."""
        return _format_panel_diagnostics(self)

    def __str__(self) -> str:  # pragma: no cover
        """Return a human-readable summary."""
        return _format_panel_diagnostics(self)


def diagnose_panel(
    data: Any,
    idname: str,
    tname: str,
    treatname: str | None = None,
) -> PanelDiagnostics:
    """Run a diagnostic battery on panel data.

    Inspects the data for common issues that would cause estimation to fail
    or produce misleading results, including duplicate unit-time pairs,
    unbalanced units, gaps in the panel, missing values, single-period
    units, and early-treated units. When a treatment column is provided,
    the check also flags whether treatment varies within units over time
    (which usually indicates the data needs :func:`get_group` to derive the
    group-timing variable).

    The returned :class:`PanelDiagnostics` object includes a
    ``suggestions`` list that maps each detected problem to the
    appropriate remediation function (e.g., :func:`deduplicate_panel`,
    :func:`fill_panel_gaps`, :func:`make_balanced_panel`), making it a
    natural first step before calling any estimator.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    treatname : str or None
        Treatment indicator column.  If provided, checks whether treatment
        varies within units over time.

    Returns
    -------
    PanelDiagnostics
        Structured report with counts and actionable suggestions.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import diagnose_panel, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: diag = diagnose_panel(df, idname="county", tname="year", treatname="inter_bra")
           ...: diag

    See Also
    --------
    deduplicate_panel : Remove duplicate unit-time pairs.
    fill_panel_gaps : Insert null rows for missing pairs.
    make_balanced_panel : Drop units not observed in every period.
    get_group : Derive group-timing from a binary treatment indicator.
    """
    df = to_polars(data)

    n_units = df[idname].n_unique()
    n_periods = df[tname].n_unique()
    n_obs = len(df)

    balanced = bool(_is_balanced_panel_impl(df, tname, idname))

    deduped = df.unique(subset=[idname, tname])
    n_dups = n_obs - len(deduped)

    counts = df.group_by(idname).agg(pl.col(tname).n_unique().alias("_n"))
    n_unbalanced = int((counts["_n"] < n_periods).sum())

    expected_full = n_units * n_periods
    n_gaps = expected_full - len(deduped)

    n_missing_rows = len(df) - len(df.drop_nulls())

    n_single = int((counts["_n"] == 1).sum())

    n_early: int | None = None
    if treatname is not None and treatname in df.columns:
        first_period = df[tname].min()
        n_early = int(df.filter((pl.col(tname) == first_period) & (pl.col(treatname) > 0))[idname].n_unique())

    treat_varying: bool | None = None
    if treatname is not None and treatname in df.columns:
        nuniq = df.group_by(idname).agg(pl.col(treatname).n_unique().alias("_nu"))
        treat_varying = bool((nuniq["_nu"] > 1).any())

    suggestions: list[str] = []
    if n_dups > 0:
        suggestions.append(f"Call deduplicate_panel() to remove {n_dups} duplicate unit-time pairs")
    if n_gaps > 0:
        suggestions.append(f"Call fill_panel_gaps() to fill {n_gaps} missing unit-time pairs")
    if n_unbalanced > 0 and n_dups == 0:
        suggestions.append(f"Call make_balanced_panel() to drop {n_unbalanced} units not observed in all periods")
    if n_missing_rows > 0:
        suggestions.append(f"{n_missing_rows} rows contain missing values and will be dropped during preprocessing")
    if n_single > 0:
        suggestions.append(
            f"Call complete_data() or make_balanced_panel() to drop {n_single} units observed in only one period"
        )
    if n_early is not None and n_early > 0:
        suggestions.append(
            f"{n_early} units are already treated in the first period and will be dropped during preprocessing"
        )
    if treat_varying:
        suggestions.append("Treatment varies within units — verify this is expected or call get_group()")

    return PanelDiagnostics(
        n_units=n_units,
        n_periods=n_periods,
        n_observations=n_obs,
        is_balanced=balanced,
        n_duplicate_unit_time=n_dups,
        n_unbalanced_units=n_unbalanced,
        n_gaps=n_gaps,
        n_missing_rows=n_missing_rows,
        n_single_period_units=n_single,
        n_early_treated=n_early,
        treatment_time_varying=treat_varying,
        suggestions=suggestions,
    )


def get_group(data: Any, idname: str, tname: str, treatname: str, treat_period: int | None = None) -> Any:
    """Extract treatment-group timing into a ``"G"`` column.

    Staggered difference-in-differences estimators like :func:`att_gt`
    require a *group* variable (``gname``) that records the first period
    each unit receives treatment. Many real-world datasets instead contain
    a binary treatment indicator that switches from 0 to 1 when treatment
    begins. This function converts that indicator into the group-timing
    variable ``"G"`` expected by the estimator. For each treated unit,
    ``G`` equals the first period where the treatment indicator is
    positive. For never-treated units, ``G`` is 0.

    When the treatment indicator is *static* (e.g., a region dummy that
    equals 1 in every period for treated units), the first-switch logic
    would incorrectly assign ``G`` to the earliest observed period.  In
    this case, pass ``treat_period`` to directly specify the known
    treatment onset: any unit with a positive value of *treatname* in any
    period receives ``G = treat_period``, and all others receive ``G = 0``.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    treatname : str
        Binary treatment indicator column.
    treat_period : int or None
        Known treatment onset period.  When provided, units with any
        positive value of *treatname* are assigned ``G = treat_period``
        and all others receive ``G = 0``, bypassing the first-switch
        detection logic.  Useful for static treatment indicators that do
        not switch on at a specific time.

    Returns
    -------
    DataFrame
        Original columns plus ``"G"``, in the same format as *data*.

    Examples
    --------
    When the treatment indicator switches on at a specific period, the
    default behaviour detects the first switch automatically:

    .. ipython::

        In [1]: from moderndid import get_group, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df = get_group(df, idname="county", tname="year", treatname="inter_bra")
           ...: df.select("county", "year", "inter_bra", "G").head(10)

    When the treatment indicator is static (e.g., a region dummy), pass
    ``treat_period`` to specify the known onset:

    .. ipython::

        In [2]: from moderndid import get_group, load_cai2016
           ...:
           ...: df = load_cai2016()
           ...: df = get_group(df, idname="hhno", tname="year",
           ...:                treatname="treatment", treat_period=2003)
           ...: df.select("hhno", "year", "treatment", "G").head(10)

    See Also
    --------
    att_gt : Estimate group-time average treatment effects.
    diagnose_panel : Check whether treatment varies within units.
    """
    result = _get_group_impl(data, idname, tname, treatname, treat_period=treat_period)
    return from_polars(result, data)


def get_first_difference(data: Any, idname: str, yname: str, tname: str) -> Any:
    r"""Add a ``"dy"`` column containing first-differenced outcomes.

    First-differencing computes :math:`\Delta Y_{it} = Y_{it} - Y_{i,t-1}`
    for each unit, removing time-invariant unit fixed effects. The
    :func:`att_gt` estimator performs this step internally, but exposing it
    here allows inspection of the transformed data before estimation.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    yname : str
        Outcome column.
    tname : str
        Time period column.

    Returns
    -------
    DataFrame
        Original columns plus ``"dy"``, in the same format as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import get_first_difference, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df = get_first_difference(df, idname="county", yname="Dl_vloans_b", tname="year")
           ...: df.select("county", "year", "Dl_vloans_b", "dy").head(10)

    See Also
    --------
    att_gt : Estimate group-time average treatment effects.
    """
    result = _get_first_difference_impl(data, idname, yname, tname)
    return from_polars(result, data)


def make_balanced_panel(data: Any, idname: str, tname: str) -> Any:
    """Drop units not observed in every time period.

    Many difference-in-differences estimators require a strictly balanced
    panel where every unit appears in every time period. When
    ``allow_unbalanced_panel=False`` (the default in :func:`att_gt`), the
    preprocessing pipeline calls this function automatically. Calling it
    beforehand lets you inspect how many units will be dropped and decide
    whether balancing, gap-filling with :func:`fill_panel_gaps`, or a
    flexible threshold via :func:`complete_data` is more appropriate.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    DataFrame
        Balanced panel in the same format as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import make_balanced_panel, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: balanced = make_balanced_panel(df, idname="county", tname="year")
           ...: print(f"Before: {df.shape[0]} rows, After: {balanced.shape[0]} rows")

    See Also
    --------
    complete_data : Keep units observed in at least *min_periods* periods.
    fill_panel_gaps : Insert null rows instead of dropping units.
    is_balanced_panel : Check whether the panel is already balanced.
    """
    result = _make_balanced_panel_impl(data, idname, tname)
    return from_polars(result, data)


def is_balanced_panel(data: Any, idname: str, tname: str) -> bool:
    """Check whether the panel is balanced.

    A balanced panel has exactly one observation for every unit-period
    combination. This is a quick Boolean check you can run before passing
    data to an estimator. If the panel is unbalanced, use
    :func:`make_balanced_panel` to drop incomplete units or
    :func:`fill_panel_gaps` to insert null rows for the missing pairs.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    bool
        ``True`` if every unit is observed in every period.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import is_balanced_panel, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: is_balanced_panel(df, idname="county", tname="year")

    See Also
    --------
    make_balanced_panel : Drop units not observed in every period.
    diagnose_panel : Full diagnostic battery including balance checks.
    """
    return _is_balanced_panel_impl(data, tname, idname)


def deduplicate_panel(data: Any, idname: str, tname: str, strategy: str = "last") -> Any:
    """Remove duplicate unit-time pairs.

    Duplicate unit-time rows cause hard errors during the preprocessing
    pipeline because the data cannot be unambiguously reshaped or
    differenced. Run :func:`diagnose_panel` first to see how many
    duplicates exist, then call this function to resolve them before
    estimation.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    strategy : ``"first"`` | ``"last"`` | ``"mean"``
        How to resolve duplicates.  ``"mean"`` averages numeric columns and
        keeps the first value for non-numeric columns.

    Returns
    -------
    DataFrame
        Deduplicated panel in the same format as *data*.

    Raises
    ------
    ValueError
        If *strategy* is not one of ``"first"``, ``"last"``, ``"mean"``.

    Examples
    --------
    .. ipython::

        In [1]: import polars as pl
           ...: from moderndid import deduplicate_panel, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df_with_dups = pl.concat([df, df.head(5)])
           ...: deduped = deduplicate_panel(df_with_dups, idname="county", tname="year")
           ...: print(f"Before: {df_with_dups.shape[0]} rows, After: {deduped.shape[0]} rows")

    See Also
    --------
    diagnose_panel : Detect duplicates before removing them.
    """
    if strategy not in ("first", "last", "mean"):
        msg = f"strategy must be 'first', 'last', or 'mean', got {strategy!r}"
        raise ValueError(msg)

    df = to_polars(data)

    if strategy in ("first", "last"):
        result = df.unique(subset=[idname, tname], keep=strategy)
    else:
        numeric_cols = [c for c in df.columns if c not in (idname, tname) and df[c].dtype.is_numeric()]
        non_numeric_cols = [c for c in df.columns if c not in (idname, tname) and not df[c].dtype.is_numeric()]

        aggs: list[pl.Expr] = []
        for c in numeric_cols:
            aggs.append(pl.col(c).mean())
        for c in non_numeric_cols:
            aggs.append(pl.col(c).first())

        result = df.group_by([idname, tname]).agg(aggs)

    return from_polars(result, data)


def fill_panel_gaps(data: Any, idname: str, tname: str) -> Any:
    """Make the panel rectangular by inserting ``null`` rows for missing pairs.

    Unlike :func:`make_balanced_panel` (which drops incomplete units), this
    function *fills* gaps so that every unit appears in every period.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    DataFrame
        Rectangular panel in the same format as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import fill_panel_gaps, has_gaps, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: print(has_gaps(df, idname="county", tname="year"))

        In [2]: filled = fill_panel_gaps(df, idname="county", tname="year")
           ...: print(f"Before: {df.shape[0]} rows, After: {filled.shape[0]} rows")

    See Also
    --------
    scan_gaps : Inspect which pairs are missing before filling.
    make_balanced_panel : Drop incomplete units instead of filling gaps.
    """
    df = to_polars(data)

    ids = df.select(idname).unique()
    times = df.select(tname).unique()
    full = ids.join(times, how="cross")
    result = full.join(df, on=[idname, tname], how="left")

    return from_polars(result, data)


def complete_data(data: Any, idname: str, tname: str, min_periods: int | None = None) -> Any:
    """Keep units observed in at least *min_periods* time periods.

    Provides a flexible alternative to :func:`make_balanced_panel`. Rather
    than requiring every unit to appear in *all* periods, you can set a
    threshold so that units with a reasonable amount of data are retained.
    When *min_periods* is ``None`` the behaviour is identical to
    :func:`make_balanced_panel`.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    min_periods : int or None
        Minimum number of observed periods.  ``None`` (default) means *all*
        periods, equivalent to :func:`make_balanced_panel`.

    Returns
    -------
    DataFrame
        Filtered panel in the same format as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import complete_data, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: filtered = complete_data(df, idname="county", tname="year", min_periods=10)
           ...: print(f"Before: {df.shape[0]} rows, After: {filtered.shape[0]} rows")

    See Also
    --------
    make_balanced_panel : Strict balancing (all periods required).
    """
    df = to_polars(data)

    if df.is_empty():
        return from_polars(df, data)

    if min_periods is None:
        min_periods = df[tname].n_unique()

    counts = df.group_by(idname).agg(pl.col(tname).n_unique().alias("_n_periods"))
    keep_ids = counts.filter(pl.col("_n_periods") >= min_periods)[idname].to_list()
    result = df.filter(pl.col(idname).is_in(keep_ids))

    return from_polars(result, data)


def assign_rc_ids(data: Any) -> Any:
    """Add a unique ``"rowid"`` column for repeated cross-section data.

    In repeated cross-section designs each observation is a different
    individual, so there is no natural unit identifier to track over time.
    This function assigns a sequential integer ``"rowid"`` that can be
    passed as the ``idname`` argument to :func:`att_gt` with
    ``panel=False``.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.

    Returns
    -------
    DataFrame
        Original data plus an integer ``"rowid"`` column, in the same format
        as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import assign_rc_ids, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df = assign_rc_ids(df)
           ...: df.select("rowid", "county", "year").head(5)

    See Also
    --------
    att_gt : Pass ``panel=False`` for repeated cross-section estimation.
    """
    df = to_polars(data)
    result = df.with_row_index("rowid")
    return from_polars(result, data)


def are_varying(data: Any, idname: str, cols: list[str] | None = None) -> dict[str, bool]:
    """Check which columns vary within units over time.

    Difference-in-differences estimators distinguish between time-varying
    and time-invariant covariates. Time-invariant covariates (e.g.,
    baseline demographics) are appropriate for inclusion in the propensity
    score or outcome regression model, while time-varying covariates
    require additional assumptions. This function classifies columns so you
    can make informed covariate-selection decisions before estimation.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    cols : list[str] or None
        Columns to check.  Defaults to all columns except *idname*.

    Returns
    -------
    dict[str, bool]
        Mapping of column name to ``True`` if the column varies within any
        unit, ``False`` otherwise.

    See Also
    --------
    diagnose_panel : Full diagnostic battery including treatment variation.
    """
    df = to_polars(data)

    if cols is None:
        cols = [c for c in df.columns if c != idname]

    nuniq = df.group_by(idname).agg([pl.col(c).n_unique().alias(c) for c in cols])

    result: dict[str, bool] = {}
    for c in cols:
        result[c] = bool((nuniq[c] > 1).any())
    return result


def scan_gaps(data: Any, idname: str, tname: str) -> Any:
    """Identify missing unit-time combinations.

    Returns a DataFrame listing every unit-period pair that is absent from
    the data. Inspecting these gaps helps you decide whether to drop
    incomplete units with :func:`make_balanced_panel` or fill them with
    null rows using :func:`fill_panel_gaps`. For a quick Boolean check
    without materialising the gaps, use :func:`has_gaps`.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    DataFrame
        Rows with ``idname`` and ``tname`` columns for every *absent* pair,
        returned in the same format as *data*.

    See Also
    --------
    has_gaps : Quick Boolean check for missing pairs.
    fill_panel_gaps : Insert null rows for missing pairs.
    """
    df = to_polars(data)

    ids = df.select(idname).unique()
    times = df.select(tname).unique()
    full = ids.join(times, how="cross")
    gaps = full.join(df.select([idname, tname]).unique(), on=[idname, tname], how="anti")

    return from_polars(gaps, data)


def has_gaps(data: Any, idname: str, tname: str) -> bool:
    """Check whether the panel has any implicit missing unit-time pairs.

    A lightweight Boolean check that compares the number of observed
    unit-period pairs against the full cross-product. If this returns
    ``True``, call :func:`scan_gaps` to see which specific pairs are
    missing, then decide whether to fill them with :func:`fill_panel_gaps`
    or drop incomplete units with :func:`make_balanced_panel`.

    Parameters
    ----------
    data : DataFrame
        Panel data. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.

    Returns
    -------
    bool
        ``True`` if there are missing unit-time combinations.

    See Also
    --------
    scan_gaps : Materialise the missing unit-time pairs.
    fill_panel_gaps : Insert null rows for the missing pairs.
    """
    df = to_polars(data)

    n_units = df[idname].n_unique()
    n_periods = df[tname].n_unique()
    n_unique_pairs = df.select([idname, tname]).unique().height

    return n_unique_pairs < n_units * n_periods


def panel_to_wide(data: Any, idname: str, tname: str, separator: str = "_") -> Any:
    """Pivot a long panel to wide format.

    Reshapes the data so that each unit occupies a single row. Time-varying
    columns are spread into one column per period while time-invariant
    columns are kept as-is.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the
        Arrow PyCapsule Interface (``__arrow_c_stream__``), including
        polars, pandas, pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    tname : str
        Time period column.
    separator : str
        String inserted between the variable name and time label in the
        wide column names.  Default ``"_"``.

    Returns
    -------
    DataFrame
        Wide-format DataFrame with one row per unit, in the same format
        as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import make_balanced_panel, panel_to_wide, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df = make_balanced_panel(df, idname="county", tname="year")
           ...: wide = panel_to_wide(df, idname="county", tname="year")
           ...: wide.head(5)

    See Also
    --------
    wide_to_panel : Inverse operation (wide to long).
    """
    df = to_polars(data)

    other_cols = [c for c in df.columns if c not in (idname, tname)]
    if not other_cols:
        return from_polars(df.select(idname).unique(), data)

    nuniq = df.group_by(idname).agg([pl.col(c).n_unique().alias(c) for c in other_cols])
    constants = [c for c in other_cols if not bool((nuniq[c] > 1).any())]
    varying_cols = [c for c in other_cols if bool((nuniq[c] > 1).any())]

    if constants:
        result = df.group_by(idname).agg([pl.col(c).first() for c in constants])
    else:
        result = df.select(idname).unique()

    if varying_cols:
        pivoted = df.select([idname, tname, *varying_cols]).pivot(
            on=tname,
            index=idname,
            values=varying_cols,
            separator=separator,
        )
        if len(varying_cols) == 1:
            stub = varying_cols[0]
            rename_map = {c: f"{stub}{separator}{c}" for c in pivoted.columns if c != idname}
            pivoted = pivoted.rename(rename_map)
        result = result.join(pivoted, on=idname)

    return from_polars(result, data)


def wide_to_panel(
    data: Any,
    idname: str,
    stub_names: list[str],
    separator: str = "_",
    tname: str = "time",
) -> Any:
    """Unpivot wide-format data into a long panel.

    Gathers time-varying columns back into long format using the stub
    names and separator to identify which wide columns belong to each
    variable and period. All other columns (except *idname*) are treated
    as time-invariant and repeated for every period.

    Parameters
    ----------
    data : DataFrame
        Wide-format data. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars,
        pandas, pyarrow Table, and cudf DataFrames.
    idname : str
        Unit identifier column.
    stub_names : list[str]
        Variable-name prefixes that identify the time-varying columns.
        For example, ``["y", "x"]`` will match ``y_1``, ``y_2``,
        ``x_1``, ``x_2``, etc.
    separator : str
        Delimiter between the stub and the period label.  Default ``"_"``.
    tname : str
        Name for the created time column.  Default ``"time"``.

    Returns
    -------
    DataFrame
        Long-format panel in the same format as *data*.

    Examples
    --------
    .. ipython::

        In [1]: from moderndid import make_balanced_panel, panel_to_wide, wide_to_panel, load_favara_imbs
           ...:
           ...: df = load_favara_imbs()
           ...: df = make_balanced_panel(df, idname="county", tname="year")
           ...: wide = panel_to_wide(df, idname="county", tname="year")
           ...: long = wide_to_panel(wide, idname="county", stub_names=["Dl_vloans_b", "Dl_hpi"], tname="year")
           ...: long.head(10)

    See Also
    --------
    panel_to_wide : Inverse operation (long to wide).
    """
    df = to_polars(data)

    varying_map: dict[str, tuple[str, str]] = {}
    periods: set[str] = set()
    for col in df.columns:
        if col == idname:
            continue
        for stub in stub_names:
            prefix = f"{stub}{separator}"
            if col.startswith(prefix):
                period = col[len(prefix) :]
                varying_map[col] = (stub, period)
                periods.add(period)
                break

    try:
        periods_sorted = sorted(periods, key=int)
        cast_period = int
    except ValueError:
        try:
            periods_sorted = sorted(periods, key=float)
            cast_period = float
        except ValueError:
            periods_sorted = sorted(periods)
            cast_period = str

    constant_cols = [c for c in df.columns if c != idname and c not in varying_map]

    frames: list[pl.DataFrame] = []
    for period in periods_sorted:
        select_exprs: list[pl.Expr] = [pl.col(idname)]
        select_exprs.append(pl.lit(cast_period(period)).alias(tname))

        for const in constant_cols:
            select_exprs.append(pl.col(const))

        for stub in stub_names:
            col_name = f"{stub}{separator}{period}"
            if col_name in df.columns:
                select_exprs.append(pl.col(col_name).alias(stub))
            else:
                select_exprs.append(pl.lit(None).alias(stub))

        frames.append(df.select(select_exprs))

    result = pl.concat(frames).sort([idname, tname])
    return from_polars(result, data)


def _format_panel_diagnostics(diag: PanelDiagnostics) -> str:
    """Pretty-print a :class:`PanelDiagnostics` instance."""

    def _bool_str(val: bool | None) -> str:
        if val is None:
            return "N/A"
        return "Yes" if val else "No"

    def _bool_or_count(val: int | None) -> str:
        if val is None:
            return "N/A"
        return str(val)

    lines = format_title("Panel Diagnostics")

    rows = [
        ("Units", str(diag.n_units)),
        ("Periods", str(diag.n_periods)),
        ("Observations", str(diag.n_observations)),
        ("Balanced", _bool_str(diag.is_balanced)),
        ("Duplicate unit-time pairs", str(diag.n_duplicate_unit_time)),
        ("Unbalanced units", str(diag.n_unbalanced_units)),
        ("Gaps", str(diag.n_gaps)),
        ("Rows with missing values", str(diag.n_missing_rows)),
        ("Single-period units", str(diag.n_single_period_units)),
        ("Early-treated units", _bool_or_count(diag.n_early_treated)),
        ("Treatment time-varying", _bool_str(diag.treatment_time_varying)),
    ]

    table = _make_table(
        ["Metric", "Value"],
        rows,
        {"Metric": "l", "Value": "r"},
    )
    lines.extend(["", *table.split("\n")])

    if diag.suggestions:
        lines.extend(format_section_header("Suggestions"))
        for s in diag.suggestions:
            lines.append(f" {s}")
        lines.extend(format_footer())
    return "\n".join(adjust_separators(lines))
