"""Utility functions for preprocessing."""

import re
import warnings

import numpy as np
import polars as pl
import scipy.linalg

from ..dataframe import to_polars


def extract_unit_clusters(time_invariant_data, clustervars, idname):
    """Return each unit's cluster from the cluster variables in the order att_gt uses them."""
    if not clustervars:
        return None
    # Validation allows at most one cluster variable besides idname, and att_gt's bootstrap clusters
    # on that one whenever it's given, so the unit id decides the clusters only when it stands alone.
    others = [var for var in clustervars if var != idname]
    return time_invariant_data[(others or clustervars)[0]].to_numpy()


def map_to_idx(vals, time_map):
    """Map values to indices."""
    vals_arr = np.asarray(vals, dtype=float)
    if vals_arr.ndim == 0:
        val_item = vals_arr.item()
        if np.isinf(val_item):
            return val_item
        return time_map.get(val_item, val_item)

    result = np.empty(len(vals_arr), dtype=float)
    for i, v in enumerate(vals_arr):
        if np.isinf(v):
            result[i] = v
        else:
            result[i] = time_map.get(v, v)

    if not np.any(np.isinf(result)):
        return result.astype(int)
    return result


def nonfinite_to_null(data, keep_infinite=()):
    """Turn every NaN and infinity in the float columns into a null.

    The preprocessing steps treat a null as a missing value. A pandas NaN
    arrives in polars as a null. A NaN from polars or numpy arrives unchanged.
    So does an infinity such as the log of zero. Turning both into nulls makes
    the same data lose the same rows whatever library holds it.

    Since an infinity marks never-treated units in a cohort column, an
    infinity in a column of ``keep_infinite`` stays. A NaN there still turns
    into a null.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input data.
    keep_infinite : sequence of str, default ()
        Columns whose infinities stay.

    Returns
    -------
    pl.DataFrame
        The data with a null in place of every NaN and of every infinity
        outside ``keep_infinite``.
    """
    df = to_polars(data)
    float_columns = [name for name, dtype in df.schema.items() if dtype.is_float()]
    if not float_columns:
        return df
    return df.with_columns(
        pl.col(name).fill_nan(None) if name in keep_infinite else pl.when(pl.col(name).is_finite()).then(pl.col(name))
        for name in float_columns
    )


def make_balanced_panel(data, idname, tname):
    """Make balanced panel.

    A unit stays when it has exactly one row in every period. Counting rows
    alone would also keep a unit whose second row in one period stands in for
    a period it misses.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Input panel data.
    idname : str
        Name of the unit identifier column.
    tname : str
        Name of the time period column.

    Returns
    -------
    pl.DataFrame
        Balanced panel data containing only units observed once in every time period.
    """
    df = to_polars(data)
    if df.is_empty():
        return df

    n_periods = df[tname].n_unique()
    counts = df.group_by(idname).agg(pl.len().alias("n_rows"), pl.col(tname).n_unique().alias("n_periods"))
    complete = counts.filter((pl.col("n_rows") == n_periods) & (pl.col("n_periods") == n_periods))
    return df.filter(pl.col(idname).is_in(complete[idname].to_list()))


def get_first_difference(df, idname, yname, tname):
    """Get first difference.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data.
    idname : str
        Name of unit identifier column.
    yname : str
        Name of outcome column.
    tname : str
        Name of time column.

    Returns
    -------
    pl.DataFrame
        DataFrame with original columns plus 'dy' column containing first differences.
    """
    data = to_polars(df)
    data = data.sort([idname, tname])
    return data.with_columns((pl.col(yname) - pl.col(yname).shift(1).over(idname)).alias("dy"))


def get_group(df, idname, tname, treatname, treat_period=None):
    """Get group.

    A unit's group is the first period in which its treatment is positive.
    A unit that is never treated gets 0. A row whose period is missing never
    counts as that period. Null, NaN, and infinite periods are missing.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input data.
    idname : str
        Name of unit identifier column.
    tname : str
        Name of time column.
    treatname : str
        Name of treatment column.
    treat_period : int or None
        Known treatment onset period. When provided, units with any positive
        value of *treatname* are assigned ``G = treat_period`` and all others
        receive ``G = 0``.

    Returns
    -------
    pl.DataFrame
        The data with a ``G`` column added, or replaced if the data already has one.
    """
    data = to_polars(df)
    # A row without a unit id belongs to no unit. It must not pool with the other such rows.
    has_unit = pl.col(idname).is_not_null()

    # Since the windows below add no helper columns, no column of the data can clash with one.
    if treat_period is not None:
        ever_treated = (pl.col(treatname) > 0).any().over(idname)
        group = pl.when(ever_treated).then(pl.lit(treat_period, dtype=pl.Int64)).otherwise(pl.lit(0, dtype=pl.Int64))
        return data.with_columns(pl.when(has_unit).then(group).alias("G"))

    is_treated = pl.col(treatname) > 0
    # NaN and infinity are missing periods like null.
    # Only float columns can hold them. is_finite raises on strings, dates, booleans, and decimals.
    if data[tname].dtype.is_float():
        is_treated = is_treated & pl.col(tname).is_finite()
    first_period = pl.col(tname).filter(is_treated).min().over(idname)
    group = pl.when(has_unit).then(first_period).fill_null(0)
    return data.with_columns(group.cast(pl.Int64).alias("G"))


def two_by_two_subset(
    data,
    g,
    tp,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
):
    """Keep the units and periods that estimate one group-time effect.

    The subset holds period ``tp`` and its base period for the units of group
    ``g`` and their comparison units. A comparison unit is untreated, and not
    yet anticipating treatment, in both periods. Never-treated units qualify
    whether ``G`` codes them as ``inf`` or as 0. With
    ``control_group="nevertreated"`` they are the only comparison units.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Panel with the columns ``G``, ``period``, and ``id``.
    g : numeric
        Treatment group, the period in which its units start treatment.
    tp : numeric
        Time period of the effect.
    control_group : {"notyettreated", "nevertreated"}
        Which units serve as comparison units.
    anticipation : int
        Number of periods before treatment in which units may respond to it.
    base_period : {"varying", "universal"}
        Whether the base period before treatment is the period just before
        ``tp`` or always the period ``g - anticipation - 1``.

    Returns
    -------
    dict
        - **gt_data**: The subset with the period label ``name`` and the group indicator ``D``
        - **n1**: Number of units in the subset
        - **disidx**: Mask over the sorted unit ids that marks the units in the subset
    """
    df = to_polars(data)
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        base_period_val = tp - 1 if tp < (g - anticipation) else main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        # A comparison unit must be untreated, and not yet anticipating treatment, in both periods of the cell.
        latest_untreated = max(tp, base_period_val) + anticipation
        unit_mask = (pl.col("G") == g) | (pl.col("G") > latest_untreated) | (pl.col("G") == 0)
    else:
        unit_mask = (pl.col("G") == g) | pl.col("G").is_infinite() | (pl.col("G") == 0)

    this_data = df.filter(unit_mask)

    time_mask = (pl.col("period") == tp) | (pl.col("period") == base_period_val)
    this_data = this_data.filter(time_mask)

    this_data = this_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name"),
        (pl.col("G") == g).cast(pl.Int64).alias("D"),
    )

    if this_data["D"].n_unique() < 2:
        return {"gt_data": pl.DataFrame(), "n1": 0, "disidx": np.array([])}

    n1 = this_data["id"].n_unique()
    all_ids = np.unique(df["id"].to_numpy())
    subset_ids = this_data["id"].unique().to_numpy()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": this_data, "n1": n1, "disidx": disidx}


def choose_knots_quantile(x, num_knots):
    """Choose knots quantile."""
    if num_knots <= 0:
        return np.array([])

    x = np.asarray(x)
    if len(x) == 0:
        return np.array([])

    probs = np.linspace(0, 1, num_knots + 2)
    quantiles = np.quantile(x, probs)
    return quantiles[1:-1]


def create_dose_grid(dose_values, n_points=50):
    """Create dose grid."""
    dose_values = np.asarray(dose_values)
    positive_doses = dose_values[dose_values > 0]

    if len(positive_doses) == 0:
        return np.array([])

    return np.linspace(positive_doses.min(), positive_doses.max(), n_points)


def validate_dose_values(dose, treatment_group, never_treated_value=float("inf")):
    """Validate dose values."""
    dose = np.asarray(dose)
    treatment_group = np.asarray(treatment_group)

    errors = []
    warnings = []

    if (dose < 0).any():
        errors.append("Negative dose values detected")

    never_treated = treatment_group == never_treated_value
    never_treated_with_dose = never_treated & (dose > 0)
    if never_treated_with_dose.any():
        n_issues = never_treated_with_dose.sum()
        warnings.append(f"{n_issues} never-treated units have positive dose values")

    treated = (treatment_group != never_treated_value) & (treatment_group > 0)
    treated_no_dose = treated & (dose == 0)
    if treated_no_dose.any():
        n_issues = treated_no_dose.sum()
        warnings.append(f"{n_issues} treated units have zero dose values")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def parse_formula(formula):
    """Split a covariate formula into its outcome and covariate columns.

    The right-hand side lists data columns joined by ``+``. A column name may
    contain dots, as in ``log.pop``. A name in backticks, as in
    ``"~ `log pop` + x"``, may contain any character but a backtick. A ``1``
    stands for the intercept and adds no column.

    Since the estimators take each covariate as a plain column, any other term
    raises an error. A transformation such as ``I(x**2)``, ``log(x)``, or
    ``C(x)`` and an interaction such as ``x1:x2`` or ``x1*x2`` go into the data
    as columns of their own first.

    Parameters
    ----------
    formula : str
        Formula of the form ``"y ~ x1 + x2"`` or ``"~ x1 + x2"``.

    Returns
    -------
    dict
        - **outcome**: Name left of ``~``, or an empty string when there is none
        - **predictors**: Covariate column names in the order given, each once
        - **formula**: The formula as given
    """
    outcome, terms = _split_formula(formula)

    predictors = []
    for term in terms:
        name = _column_name(term)
        if name is None:
            _check_formula_term(term)
            raise ValueError(
                f"xformla term '{term}' is not a column name. xformla accepts column names joined by '+'. "
                "Add a transformed or interaction covariate to the data as its own column first. "
                "A name that holds spaces or symbols goes in backticks."
            )
        if name not in predictors:
            predictors.append(name)

    return {
        "outcome": outcome,
        "predictors": predictors,
        "formula": formula,
    }


def get_transformed_terms(formula):
    """List the terms of a covariate formula that are not column names.

    Since a term such as ``I(x**2)``, ``log(x)``, ``C(x)``, or ``x1:x2``
    transforms or combines columns, a formula engine has to evaluate it.
    Column names and the intercept ``1`` stay off the list. The formula
    otherwise follows the rules of :func:`extract_vars_from_formula`. A
    left-hand side, an empty term, or a term that drops the intercept raises
    an error.

    Parameters
    ----------
    formula : str
        Covariate formula such as ``"~ x1 + I(x1**2)"``.

    Returns
    -------
    list of str
        The transformed terms in the order given, empty when every term names
        a column.
    """
    outcome, terms = _split_formula(formula)
    _reject_outcome(formula, outcome)

    transformed = []
    for term in terms:
        if _column_name(term) is None:
            _check_formula_term(term)
            transformed.append(term)
    return transformed


def get_column_terms(formula):
    """List the columns that the plain terms of a covariate formula name.

    A plain term is one column name with or without backticks. A transformed
    term such as ``I(x**2)``, the intercept ``1``, and a left-hand side name
    no column here.

    Parameters
    ----------
    formula : str
        Covariate formula such as ``"~ x1 + I(x2**2)"``.

    Returns
    -------
    list of str
        The column names in the order given, each once.
    """
    _, terms = _split_formula(formula)
    return list(dict.fromkeys(name for name in map(_column_name, terms) if name is not None))


def get_formula_columns(formula, columns):
    """List the data columns that a formula refers to.

    The formula may hold transformations and interactions such as
    ``I(x**2)``, ``C(group)``, or ``x1:x2``. A name counts when it is a column
    of the data. Function names such as ``np.log`` drop out unless a column
    carries that name. When a dotted name such as ``x.clip`` is not a column,
    its longest leading part that is a column counts instead. A keyword
    argument name such as ``degree`` in ``poly(age, degree=2)`` never counts.

    Parameters
    ----------
    formula : str
        Formula with one ``~``, such as ``"~ x1 + I(x1**2) + C(group)"``.
    columns : list of str
        Column names of the data.

    Returns
    -------
    list of str
        The columns the formula names, in order of first appearance.
    """
    if len(_split_outside_quotes(formula, "~")) != 2:
        raise ValueError("Formula must be in the form '~ x1 + x2 + ...'")

    available = set(columns)
    found = []
    # Quoted strings are skipped so that a level such as 'a' in C(g, Treatment('a')) is not read as a column.
    # A keyword argument such as degree in poly(age, degree=2) names no column. The lookahead after the
    # name keeps a prefix such as d in df=3 from matching instead.
    pattern = r"`([^`]+)`|'[^']*'|\"[^\"]*\"|(?<![\w.])((?:[^\W\d]|\.+[^\W\d])[\w.]*)(?![\w.])(?!\s*=(?!=))"
    for match in re.finditer(pattern, formula):
        quoted, plain = match.group(1), match.group(2)
        if quoted is not None:
            candidates = [quoted]
        elif plain is not None:
            parts = plain.split(".")
            candidates = [".".join(parts[:k]) for k in range(len(parts), 0, -1)]
        else:
            continue
        name = next((c for c in candidates if c in available), None)
        if name is not None and name not in found:
            found.append(name)
    return found


def _split_outside_quotes(text, sep):
    """Split text at each separator outside quotes and brackets."""
    parts = []
    start = 0
    depth = 0
    quote = None
    for i, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "`'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _column_name(term):
    """Return the column a formula term names or None for any other term."""
    match = re.fullmatch(r"`([^`]+)`|((?:[^\W\d]|\.+[^\W\d])[\w.]*)", term)
    if match is None:
        return None
    return match.group(1) or match.group(2)


def _split_formula(formula):
    """Split a formula into its outcome and its right-hand terms other than the intercept."""
    sides = _split_outside_quotes(formula, "~")
    if len(sides) != 2:
        raise ValueError("Formula must be in the form '~ x1 + x2 + ...'")

    outcome = sides[0].strip()
    rhs = sides[1].strip()
    terms = [term.strip() for term in _split_outside_quotes(rhs, "+")] if rhs else []
    return _column_name(outcome) or outcome, [term for term in terms if term != "1"]


def _check_formula_term(term):
    """Raise an error for an empty term or a term that drops the intercept."""
    if not term:
        raise ValueError("xformla has an empty term. Remove the extra '+'.")
    if term == "0" or re.search(r"-\s*1$", term):
        raise ValueError(
            f"xformla term '{term}' drops the intercept. Since the estimators always include an "
            "intercept, remove the '0' or '-1' from xformla."
        )


def _reject_outcome(formula, outcome):
    """Raise an error when a covariate formula has a left-hand side."""
    if outcome:
        raise ValueError(
            f"xformla='{formula}' has a left-hand side. It lists only the covariates, as in '~ x1 + x2'. "
            "The outcome goes in yname."
        )


def extract_vars_from_formula(formula):
    """List the covariate columns that a formula names.

    The formula follows the grammar of :func:`parse_formula` and lists only
    covariates, as in ``"~ x1 + x2"``. Since the outcome goes in ``yname``, a
    left-hand side raises an error.

    Parameters
    ----------
    formula : str
        Covariate formula such as ``"~ x1 + x2"``.

    Returns
    -------
    list of str
        Covariate column names in the order given, each once.
    """
    parsed = parse_formula(formula)
    _reject_outcome(formula, parsed["outcome"])
    return parsed["predictors"]


def is_balanced_panel(data, tname, idname):
    """Check if the panel data is balanced.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        The input data.
    tname : str
        Name of time column.
    idname : str
        Name of id column.

    Returns
    -------
    bool
        True if panel is balanced (all units observed in all periods).
    """
    df = to_polars(data)
    n_periods = df[tname].n_unique()
    obs_per_unit = df.group_by(idname).agg(pl.col(tname).n_unique().alias("n_obs"))

    return (obs_per_unit["n_obs"] == n_periods).all()


def add_intercept(covariates):
    """Add intercept column to covariate matrix.

    Parameters
    ----------
    covariates : ndarray or None
        Covariate matrix.

    Returns
    -------
    ndarray or None
        Covariate matrix with intercept column prepended, or None if input is None.
    """
    if covariates is None or covariates.shape[1] == 0:
        return None

    intercept = np.ones((covariates.shape[0], 1))
    return np.hstack([intercept, covariates])


def extract_covariates(data, xformla):
    """Extract covariate matrix from DataFrame given a formula.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        The input data.
    xformla : str or None
        Formula for covariates in the form "~ x1 + x2 + x3".

    Returns
    -------
    ndarray or None
        Covariate matrix with intercept, or None if no covariates.
    """
    if xformla is None or xformla == "~1":
        return None

    df = to_polars(data)

    formula_str = xformla.strip()
    if formula_str.startswith("~"):
        formula_str = "y " + formula_str

    parsed = parse_formula(formula_str)
    covariate_names = parsed["predictors"]

    if not covariate_names or covariate_names == ["1"]:
        return None

    covariate_names = [c for c in covariate_names if c != "1"]

    missing_covs = [c for c in covariate_names if c not in df.columns]
    if missing_covs:
        raise ValueError(f"Covariates not found in data: {missing_covs}")

    X = df.select(covariate_names).to_numpy()
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def get_covariate_names_from_formula(xformla):
    """Extract covariate names from an xformla string, returning None if no covariates."""
    if not xformla or xformla == "~1":
        return None
    return extract_vars_from_formula(xformla)


def remove_collinear(cov_matrix, var_names, tol=1e-6):
    """Remove collinear columns from covariate matrix using QR decomposition."""
    if cov_matrix.shape[1] == 0:
        return cov_matrix, var_names

    _, r, pivot = scipy.linalg.qr(cov_matrix, mode="economic", pivoting=True)

    diag_r = np.abs(np.diag(r))
    rank = np.sum(diag_r > tol * diag_r[0]) if len(diag_r) > 0 else 0

    keep_indices = pivot[:rank]
    keep_indices = np.sort(keep_indices)

    kept_vars = [var_names[i] for i in keep_indices]
    return cov_matrix[:, keep_indices], kept_vars


def check_partition_collinearity(cov_matrix, subgroup, var_names, tol=1e-6):
    """Check for partition-specific collinearity in DDD comparisons."""
    if len(var_names) == 0:
        return {}, []

    comparison_groups = [3, 2, 1]
    partition_collinear: dict[str, list[str]] = {}

    for comp_group in comparison_groups:
        mask = (subgroup == 4) | (subgroup == comp_group)
        cov_subset = cov_matrix[mask]

        if cov_subset.shape[0] == 0:
            continue

        _, kept_vars = remove_collinear(cov_subset, list(var_names), tol=tol)
        kept_set = set(kept_vars)
        collinear_in_subset = [v for v in var_names if v not in kept_set]

        if collinear_in_subset:
            partition_name = f"subgroup 4 vs {comp_group}"
            for var in collinear_in_subset:
                if var not in partition_collinear:
                    partition_collinear[var] = []
                partition_collinear[var].append(partition_name)

    return partition_collinear, list(partition_collinear.keys())


def create_ddd_subgroups(treat, partition, treat_val):
    """Create subgroup assignments for DDD.

    Subgroup definitions:
    - 4: Treated AND Eligible (treat=g, partition=1)
    - 3: Treated BUT Ineligible (treat=g, partition=0)
    - 2: Eligible BUT Untreated (treat=0, partition=1)
    - 1: Untreated AND Ineligible (treat=0, partition=0)
    """
    is_treated = treat == treat_val
    is_eligible = partition == 1

    subgroup = np.where(
        is_treated & is_eligible,
        4,
        np.where(is_treated & ~is_eligible, 3, np.where(is_eligible, 2, 1)),
    )
    return subgroup


def validate_subgroup_sizes(subgroup_counts, min_size=5):
    """Validate that each subgroup has sufficient observations."""
    for sg, count in subgroup_counts.items():
        if count < min_size:
            raise ValueError(f"Subgroup {sg} has only {count} observations. Minimum required is {min_size}.")


def extract_ddd_covariates(df: pl.DataFrame, xformla, subgroup: np.ndarray | None = None):
    """Extract and process covariates from data for DDD."""
    if xformla == "~1":
        n_units = len(df.filter(pl.col("_post") == 0))
        return np.empty((n_units, 0)), []

    covariate_vars = extract_vars_from_formula(xformla)

    df_pre = df.filter(pl.col("_post") == 0)
    cov_matrix = df_pre.select(covariate_vars).to_numpy().astype(float)

    if np.any(np.isnan(cov_matrix)):
        warnings.warn(
            "Missing values in covariates. Rows with NaN will cause issues.",
            stacklevel=3,
        )

    cov_matrix, kept_vars = remove_collinear(cov_matrix, covariate_vars)

    if subgroup is not None and len(kept_vars) > 0:
        partition_collinear, all_collinear = check_partition_collinearity(cov_matrix, subgroup, kept_vars)

        if all_collinear:
            partition_warnings = [
                f"  - {var} (collinear in: {', '.join(partitions)})" for var, partitions in partition_collinear.items()
            ]
            warnings.warn(
                "The following covariates were dropped due to partition-specific collinearity:\n"
                + "\n".join(partition_warnings),
                stacklevel=3,
            )

            keep_mask = [var not in all_collinear for var in kept_vars]
            cov_matrix = cov_matrix[:, keep_mask]
            kept_vars = [v for v in kept_vars if v not in all_collinear]

    return cov_matrix, kept_vars
