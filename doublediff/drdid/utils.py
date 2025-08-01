"""Utility functions for drdid estimators."""

import warnings
from typing import Any, Literal

import formulaic as fml
import numpy as np
import pandas as pd


def preprocess_drdid(
    data: pd.DataFrame,
    y_col: str,
    time_col: str,
    id_col: str,
    treat_col: str,
    covariates_formula: str | None = None,
    panel: bool = True,
    normalized: bool = True,
    est_method: Literal["imp", "trad"] = "imp",
    weights_col: str | None = None,
    boot: bool = False,
    boot_type: Literal["weighted", "multiplier"] = "weighted",
    n_boot: int | None = None,
    inf_func: bool = False,
) -> dict[str, Any]:
    """Pre-processes data for DR DiD estimation.

    Validates input data, checks for required columns, handles missing values,
    balances panel data if requested, checks for time-invariant treatment/covariates
    in panel data, processes covariates using patsy, normalizes weights, and
    structures the output for downstream estimation functions.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing outcome, time, unit ID, treatment,
        and optionally covariates and weights. Must contain exactly two time periods.
    y_col : str
        Name of the column in `data` representing the outcome variable.
    time_col : str
        Name of the column in `data` representing the time period indicator.
        Must contain exactly two unique numeric values.
    id_col : str
        Name of the column in `data` representing the unique unit identifier.
        Required if `panel=True`.
    treat_col : str
        Name of the column in `data` representing the treatment indicator.
        Must contain only 0 (control) and 1 (treated). For panel data, this
        should indicate whether a unit is *ever* treated (time-invariant).
        For repeated cross-sections, it indicates treatment status in the post-period.
    covariates_formula : str or None, default None
        A patsy-style formula string for specifying covariates (e.g., "~ x1 + x2 + x1:x2").
        If None, only an intercept term (`~ 1`) is included. Covariates specified
        here must exist as columns in `data`. For panel data, covariates must be
        time-invariant.
    panel : bool, default True
        Indicates whether the data represents panel observations (True) or
        repeated cross-sections (False). If True, data is balanced, and
        treatment/covariates/weights are checked for time-invariance.
    normalized : bool, default True
        If True, the observation weights (`weights_col` or unit weights if None)
        are normalized to have a mean of 1.
    est_method : {"imp", "trad"}, default "imp"
        Specifies the estimation method context, potentially influencing future
        preprocessing steps (currently informational). "imp" for imputation-based,
        "trad" for traditional regression-based.
    weights_col : str or None, default None
        Name of the column in `data` containing observation weights. If None,
        unit weights (all 1.0) are assumed. For panel data, weights must be
        time-invariant.
    boot : bool, default False
        Flag indicating whether preprocessing is done in preparation for a
        bootstrap procedure (currently informational).
    boot_type : {"weighted", "multiplier"}, default "weighted"
        Specifies the type of bootstrap procedure if `boot=True` (currently informational).
    n_boot : int or None, default None
        Number of bootstrap replications if `boot=True` (currently informational).
    inf_func : bool, default False
        Flag indicating whether preprocessing is done for influence function
        calculations (currently informational).

    Returns
    -------
    dict[str, Any]
        A dictionary containing processed data elements.
    """
    if not isinstance(data, pd.DataFrame):
        warnings.warn("Input data is not a pandas DataFrame; converting...", UserWarning)
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    required_cols = [y_col, time_col, treat_col]
    if panel:
        required_cols.append(id_col)
    if weights_col:
        required_cols.append(weights_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    if est_method not in ["imp", "trad"]:
        warnings.warn(f"est_method='{est_method}' is not supported. Using 'imp'.", UserWarning)
        est_method = "imp"

    if boot and boot_type not in ["weighted", "multiplier"]:
        warnings.warn(f"boot_type='{boot_type}' is not supported. Using 'weighted'.", UserWarning)
        boot_type = "weighted"

    if not isinstance(normalized, bool):
        warnings.warn(f"normalized={normalized} is not supported. Using True.", UserWarning)
        normalized = True

    numeric_cols = [y_col, time_col, treat_col]
    if panel:
        numeric_cols.append(id_col)
    if weights_col:
        numeric_cols.append(weights_col)

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
                warnings.warn(f"Column '{col}' was not numeric; converted successfully.", UserWarning)
            except (ValueError, TypeError) as exc:
                raise TypeError(f"Column '{col}' must be numeric. Could not convert.") from exc

    time_periods = sorted(df[time_col].unique())
    if len(time_periods) != 2:
        raise ValueError("This package currently supports only two time periods (pre and post).")
    pre_period, post_period = time_periods

    groups = sorted(df[treat_col].unique())
    if len(groups) != 2 or not all(g in [0, 1] for g in groups):
        raise ValueError("Treatment indicator column must contain only 0 (control) and 1 (treated).")

    if covariates_formula is None:
        covariates_formula = "~ 1"

    try:
        model_matrix_result = fml.model_matrix(
            covariates_formula,
            df,
            output="pandas",
        )
        covariates_df = model_matrix_result
        if hasattr(model_matrix_result, "model_spec") and model_matrix_result.model_spec:
            original_cov_names = [var for var in model_matrix_result.model_spec.variables if var != "1"]
        else:
            original_cov_names = []
            warnings.warn("Could not retrieve model_spec from formulaic output. ", UserWarning)

    except Exception as e:
        raise ValueError(f"Error processing covariates_formula '{covariates_formula}' with formulaic: {e}") from e

    cols_to_drop = [name for name in original_cov_names if name in df.columns]
    df_processed = pd.concat([df.drop(columns=cols_to_drop), covariates_df], axis=1)

    if weights_col:
        df_processed["weights"] = df_processed[weights_col]
        if not pd.api.types.is_numeric_dtype(df_processed["weights"]):
            raise TypeError(f"Weights column '{weights_col}' must be numeric.")
        if (df_processed["weights"] < 0).any():
            warnings.warn("Some weights are negative. Ensure this is intended.", UserWarning)
    else:
        df_processed["weights"] = 1.0

    initial_rows = len(df_processed)
    cols_for_na_check_base = [y_col, time_col, "weights"] + list(covariates_df.columns)
    if pd.api.types.is_numeric_dtype(df_processed[treat_col]):
        cols_for_na_check_base.append(treat_col)
    if panel:
        cols_for_na_check = cols_for_na_check_base + [id_col]
    else:
        cols_for_na_check = cols_for_na_check_base

    na_counts = df_processed[cols_for_na_check].isna().sum()
    cols_with_na = na_counts[na_counts > 0].index.to_list()

    if cols_with_na:
        warnings.warn(
            f"Missing values found in columns: {', '.join(cols_with_na)}. "
            "Dropping rows with any missing values in relevant columns.",
            UserWarning,
        )
        df_processed = df_processed.dropna(subset=cols_for_na_check)

    if len(df_processed) < initial_rows:
        warnings.warn(f"Dropped {initial_rows - len(df_processed)} rows due to missing values.", UserWarning)
    if df_processed.empty:
        raise ValueError("DataFrame is empty after handling missing values.")

    unique_treat_values = df_processed[treat_col].unique()
    if len(unique_treat_values) < 2:
        raise ValueError(
            f"Data must contain both treated (1) and control (0) units in '{treat_col}'. "
            f"Found only: {unique_treat_values}. "
            "Ensure both groups are present after NA handling."
        )
    if not (np.any(unique_treat_values == 0) and np.any(unique_treat_values == 1)):
        raise ValueError(
            f"Treatment indicator column '{treat_col}' must contain both 0 and 1. "
            f"Found values: {unique_treat_values}. "
            "Ensure both groups are present after NA handling."
        )

    if panel:
        if df_processed.groupby([id_col, time_col]).size().max() > 1:
            raise ValueError(f"ID '{id_col}' is not unique within time period '{time_col}'.")

        _check_treatment_uniqueness(df_processed, id_col, treat_col)

        df_processed = _make_balanced_panel(df_processed, id_col, time_col)
        if df_processed.empty:
            raise ValueError("Balancing the panel resulted in an empty DataFrame. Check input data.")

        df_processed = df_processed.sort_values(by=[id_col, time_col])

        pre_df = df_processed[df_processed[time_col] == pre_period].set_index(id_col)
        post_df = df_processed[df_processed[time_col] == post_period].set_index(id_col)

        common_ids = pre_df.index.intersection(post_df.index)
        pre_df = pre_df.loc[common_ids]
        post_df = post_df.loc[common_ids]

        cov_cols_to_check = [col for col in covariates_df.columns if col != "Intercept"]
        if cov_cols_to_check:
            if not pre_df[cov_cols_to_check].equals(post_df[cov_cols_to_check]):
                diff_mask = (pre_df[cov_cols_to_check] != post_df[cov_cols_to_check]).any()
                diff_cols = diff_mask[diff_mask].index.to_list()
                raise ValueError(f"Covariates must be time-invariant in panel data. Differing columns: {diff_cols}")

        if not pre_df[treat_col].equals(post_df[treat_col]):
            raise ValueError(f"Treatment indicator ('{treat_col}') must be time-invariant in panel data.")

        if not pre_df["weights"].equals(post_df["weights"]):
            raise ValueError("Weights must be time-invariant in panel data.")

    covariates_final = df_processed[covariates_df.columns].values
    if covariates_final.shape[1] > 1:
        _, r = np.linalg.qr(covariates_final)
        diag_r = np.abs(np.diag(r))
        tol = diag_r.max() * 1e-6
        rank = np.sum(diag_r > tol)
        num_covariates = covariates_final.shape[1]

        if rank < num_covariates:
            warnings.warn(
                "Potential collinearity detected among covariates. "
                f"Rank ({rank}) is less than number of covariates ({num_covariates}). "
                "Results may be unstable.",
                UserWarning,
            )

    min_obs_per_group_period = df_processed.groupby([treat_col, time_col]).size().min()
    req_size = covariates_final.shape[1] + 5
    if min_obs_per_group_period < req_size:
        warnings.warn(
            "Small group size detected. Minimum observations in a treatment/period group is "
            f"{min_obs_per_group_period}, which might be less than recommended ({req_size}). "
            "Inference may be unreliable.",
            UserWarning,
        )

    rename_dict = {y_col: "y", treat_col: "D", time_col: "time"}
    if panel:
        rename_dict[id_col] = "id"
    df_processed = df_processed.rename(columns=rename_dict)

    if normalized and "weights" in df_processed.columns:
        mean_weight = df_processed["weights"].mean()
        if mean_weight > 0:
            df_processed["weights"] = df_processed["weights"] / mean_weight
        else:
            warnings.warn("Mean of weights is zero or negative. Cannot normalize.", UserWarning)

    output = {
        "panel": panel,
        "est_method": est_method,
        "normalized": normalized,
        "boot": boot,
        "boot_type": boot_type,
        "n_boot": n_boot,
        "inf_func": inf_func,
        "covariate_names": list(covariates_df.columns),
    }

    if panel:
        df_processed = df_processed.sort_values(by=["id", "time"])
        post_data = df_processed[df_processed["time"] == post_period].set_index("id")
        pre_data = df_processed[df_processed["time"] == pre_period].set_index("id")
        common_ids = post_data.index.intersection(pre_data.index)
        post_data = post_data.loc[common_ids]
        pre_data = pre_data.loc[common_ids]

        output["y1"] = post_data["y"].values
        output["y0"] = pre_data["y"].values
        output["D"] = post_data["D"].values
        output["covariates"] = post_data[output["covariate_names"]].values
        output["weights"] = post_data["weights"].values
        output["n_units"] = len(common_ids)

    else:
        output["y"] = df_processed["y"].values
        output["D"] = df_processed["D"].values
        output["post"] = (df_processed["time"] == post_period).astype(int).values
        output["covariates"] = df_processed[output["covariate_names"]].values
        output["weights"] = df_processed["weights"].values
        output["n_obs"] = len(df_processed)
    return output


def _check_treatment_uniqueness(df: pd.DataFrame, id_col: str, treat_col: str) -> None:
    """Check if treatment status is unique for each ID in panel data."""
    treat_counts = df.groupby(id_col)[treat_col].nunique()
    if (treat_counts > 1).any():
        invalid_ids = treat_counts[treat_counts > 1].index.to_list()
        raise ValueError(
            f"Treatment indicator ('{treat_col}') must be unique for each ID ('{id_col}'). "
            f"IDs with varying treatment: {invalid_ids}."
        )


def _make_balanced_panel(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """Convert an unbalanced panel DataFrame into a balanced one."""
    n_times = df[time_col].nunique()
    obs_counts = df.groupby(id_col).size()
    ids_to_keep = obs_counts[obs_counts == n_times].index
    if len(ids_to_keep) < len(obs_counts):
        warnings.warn(
            "Panel data is unbalanced. Dropping units with incomplete observations.",
            UserWarning,
        )
    return df[df[id_col].isin(ids_to_keep)].copy()


def _validate_inputs(arrays_dict, x, n_bootstrap, trim_level, check_intercept=False):
    """Validate inputs for bootstrap functions."""
    for name, arr in arrays_dict.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array.")

    for name, arr in arrays_dict.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional.")

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    first_array = next(iter(arrays_dict.values()))
    n_units = first_array.shape[0]

    for name, arr in arrays_dict.items():
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations.")

    if x.shape[0] != n_units:
        raise ValueError("All arrays must have the same number of observations.")

    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")

    if not 0 < trim_level < 1:
        raise ValueError("trim_level must be between 0 and 1.")

    if check_intercept and not np.all(x[:, 0] == 1.0):
        warnings.warn(
            "The first column of the covariate matrix 'x' does not appear to be an intercept (all ones). "
            "IPT propensity score estimation typically requires an intercept.",
            UserWarning,
        )

    return n_units


def _validate_wols_arrays(arrays_dict: dict[str, np.ndarray], x: np.ndarray, function_name: str = "wols") -> int:
    """Validate input arrays for WOLS functions."""
    all_arrays = list(arrays_dict.values()) + [x]
    if not all(isinstance(arr, np.ndarray) for arr in all_arrays):
        raise TypeError("All inputs must be NumPy arrays.")

    if function_name == "wols_panel":
        dim_error_msg = "delta_y, d, ps, and i_weights must be 1-dimensional."
    else:  # wols_rc
        dim_error_msg = "y, post, d, ps, and i_weights must be 1-dimensional."

    for arr in arrays_dict.values():
        if arr.ndim != 1:
            raise ValueError(dim_error_msg)

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    n_units = next(iter(arrays_dict.values())).shape[0]
    for arr in list(arrays_dict.values()) + [x]:
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations (first dimension).")

    return n_units


def _check_extreme_weights(weights: np.ndarray, threshold: float = 1e6) -> None:
    """Check for extreme weight ratios and warn if found."""
    if len(weights) > 1:
        positive_mask = weights > 0
        if np.any(positive_mask):
            min_positive = np.min(weights[positive_mask])
            max_weight = np.max(weights)
            if max_weight / min_positive > threshold:
                warnings.warn("Extreme weight ratios detected. Results may be numerically unstable.", UserWarning)


def _check_wls_condition_number(results, threshold_error: float = 1e15, threshold_warn: float = 1e10) -> None:
    """Check condition number of WLS results and handle accordingly."""
    try:
        condition_number = results.condition_number
        if condition_number > threshold_error:
            raise ValueError(
                f"Failed to solve linear system: The covariate matrix may be singular or ill-conditioned "
                f"(condition number: {condition_number:.2e})."
            )
        if condition_number > threshold_warn:
            warnings.warn(
                f"Potential multicollinearity detected (condition number: {condition_number:.2e}). "
                "Consider removing or combining covariates.",
                UserWarning,
            )
    except AttributeError:
        pass


def _check_coefficients_validity(coefficients: np.ndarray) -> None:
    """Check if coefficients contain invalid values."""
    if np.any(np.isnan(coefficients)) or np.any(np.isinf(coefficients)):
        raise ValueError(
            "Failed to solve linear system. Coefficients contain NaN/Inf values, "
            "likely due to multicollinearity or singular matrix."
        )


def _weighted_sum(term_val, weight_val, term_name):
    sum_w = np.sum(weight_val)
    if sum_w == 0 or not np.isfinite(sum_w):
        warnings.warn(f"Sum of weights for {term_name} is {sum_w}. Term will be NaN.", UserWarning)
        return np.nan

    weighted_sum_term = np.sum(weight_val * term_val)
    if not np.isfinite(weighted_sum_term):
        warnings.warn(
            f"Weighted sum for {term_name} is not finite ({weighted_sum_term}). Term will be NaN.", UserWarning
        )
        return np.nan

    return weighted_sum_term / sum_w
