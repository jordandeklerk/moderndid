"""Utility functions for preprocessing."""

import re

import numpy as np
import pandas as pd


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


def make_balanced_panel(data, idname, tname):
    """Make balanced panel."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    n_periods = data[tname].nunique()
    balanced = data.groupby(idname).filter(lambda x: len(x) == n_periods)
    return balanced.reset_index(drop=True)


def get_first_difference(df, idname, yname, tname):
    """Get first difference."""
    df = df.sort_values([idname, tname])
    lagged = df.groupby(idname)[yname].shift(1)
    return df[yname] - lagged


def get_group(df, idname, tname, treatname):
    """Get group."""
    df_sorted = df.sort_values([idname, tname])

    is_treated = df_sorted[treatname] > 0
    first_treat_mask = (is_treated.groupby(df_sorted[idname]).cumsum() == 1) & is_treated

    id_to_group = df_sorted[df_sorted[tname].where(first_treat_mask).notna()].groupby(idname)[tname].first()

    return df[idname].map(id_to_group).fillna(0).astype(int)


def two_by_two_subset(
    data,
    g,
    tp,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
):
    """Two by two subset for treatment DiD."""
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        base_period_val = tp - 1 if tp < (g - anticipation) else main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        unit_mask = (data["G"] == g) | (data["G"] > tp)
    else:
        unit_mask = (data["G"] == g) | np.isinf(data["G"])

    this_data = data.loc[unit_mask].copy()

    time_mask = (this_data["period"] == tp) | (this_data["period"] == base_period_val)
    this_data = this_data.loc[time_mask]

    this_data["name"] = np.where(this_data["period"] == tp, "post", "pre")
    this_data["D"] = 1 * (this_data["G"] == g)

    if this_data["D"].nunique() < 2:
        return {"gt_data": pd.DataFrame(), "n1": 0, "disidx": np.array([])}

    n1 = this_data["id"].nunique()
    all_ids = data["id"].unique()
    subset_ids = this_data["id"].unique()
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
    """Parse formula string to extract components."""
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError("Formula must be in the form 'y ~ x1 + x2 + ...'")

    outcome = parts[0].strip()
    predictors_str = parts[1].strip()

    var_pattern = r"\b[a-zA-Z_]\w*\b"
    all_vars = re.findall(var_pattern, predictors_str)

    exclude = {"C", "I", "Q", "bs", "ns", "log", "exp", "sqrt", "abs", "np"}
    predictors = [v for v in all_vars if v not in exclude]

    seen = set()
    predictors = [x for x in predictors if not (x in seen or seen.add(x))]

    return {
        "outcome": outcome,
        "predictors": predictors,
        "formula": formula,
    }


def extract_vars_from_formula(formula):
    """Extract all variable names from formula string."""
    parsed = parse_formula(formula)
    vars_list = []
    if parsed["outcome"]:
        vars_list.append(parsed["outcome"])
    vars_list.extend(parsed["predictors"])
    return vars_list


def is_balanced_panel(data, tname, idname):
    """Check if the panel data is balanced.

    Parameters
    ----------
    data : DataFrame
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
    n_periods = data[tname].nunique()
    obs_per_unit = data.groupby(idname)[tname].nunique()

    return (obs_per_unit == n_periods).all()


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
    data : DataFrame
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

    formula_str = xformla.strip()
    if formula_str.startswith("~"):
        formula_str = "y " + formula_str

    parsed = parse_formula(formula_str)
    covariate_names = parsed["predictors"]

    if not covariate_names or covariate_names == ["1"]:
        return None

    covariate_names = [c for c in covariate_names if c != "1"]

    missing_covs = [c for c in covariate_names if c not in data.columns]
    if missing_covs:
        raise ValueError(f"Covariates not found in data: {missing_covs}")

    X = data[covariate_names].values
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])
