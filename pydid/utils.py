"""Preprocessing functions for doubly robust DiD estimators."""

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

    df_processed = df_processed.rename(columns={y_col: "y", treat_col: "D", time_col: "time", id_col: "id"})
    if panel:
        df_processed = df_processed.rename(columns={id_col: "id"})

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


def preprocess_synth(
    data: pd.DataFrame,
    y_col: str,
    time_col: str,
    id_col: str,
    treat_col: str,
    treatment_period: Any,
    covariates_formula: str | None = None,
    l_outcome_periods: list[Any] | None = None,
    weights_col: str | None = None,
    normalized: bool = True,
    post_periods_of_interest: list[Any] | None = None,
) -> dict[str, Any]:
    """Pre-processes data for synthetic control estimators.

    Validates inputs, identifies treated/control units, balances panel data
    for relevant pre- and post-treatment periods, processes covariates (patsy)
    and lagged outcomes, normalizes weights, and structures output matrices
    (Y_pre, Y_post, Z) for treated and control groups.

    Parameters
    ----------
    data : pd.DataFrame
        Input data. Must be panel data.
    y_col : str
        Outcome variable column name.
    time_col : str
        Time period indicator column name.
    id_col : str
        Unit identifier column name.
    treat_col : str
        Treatment indicator column name (0 for control, 1 for treated).
        Assumed to be time-invariant for a unit.
    treatment_period : Any
        The time period when treatment begins for the 'treated' group.
    covariates_formula : str or None, default None
        Patsy formula for covariates (e.g., "~ x1 + x2"). Covariates are
        taken from the last pre-treatment period or assumed time-invariant.
    l_outcome_periods : list[Any] or None, default None
        List of specific pre-treatment time periods whose outcome (y_col)
        values should be used as predictors.
    weights_col : str or None, default None
        Observation weights column name. Assumed time-invariant per unit.
    normalized : bool, default True
        If True, unit weights are normalized to mean 1.
    post_periods_of_interest : list[Any] or None, default None
        Specific post-treatment periods for which outcomes should be reported.
        If None, only `treatment_period` is used.

    Returns
    -------
    dict[str, Any]
        Dictionary with processed data:

        - Y_pre_treat, Y_post_treat, Y_pre_control, Y_post_control: Outcome matrices.
        - Z_treat, Z_control: Predictor matrices.
        - weights_treat, weights_control: Unit weights.
        - n_treated, n_control, n_pre_periods, n_post_periods: Counts.
        - pre_periods, post_periods: Lists of period values.
        - predictor_names: List of predictor names for Z matrices.
    """
    if not isinstance(data, pd.DataFrame):
        warnings.warn("Input data is not a pandas DataFrame; converting...", UserWarning)
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    base_required_cols = [y_col, time_col, id_col, treat_col]
    if weights_col:
        base_required_cols.append(weights_col)
    missing_cols = [col for col in base_required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    for col in [y_col, time_col, id_col, treat_col]:
        if not pd.api.types.is_numeric_dtype(df[col]) and col != id_col:
            if pd.api.types.is_numeric_dtype(df[col]):
                pass
            elif col == time_col:
                try:
                    df[col] = pd.to_numeric(df[col])
                    warnings.warn(f"Column '{col}' was not numeric; converted successfully.", UserWarning)
                except (ValueError, TypeError) as exc:
                    if not np.issubdtype(df[col].dtype, np.datetime64):
                        raise TypeError(f"Column '{col}' must be numeric or datetime. Could not convert.") from exc
            else:
                try:
                    df[col] = pd.to_numeric(df[col])
                    warnings.warn(f"Column '{col}' was not numeric; converted successfully.", UserWarning)
                except (ValueError, TypeError) as exc:
                    raise TypeError(f"Column '{col}' must be numeric. Could not convert.") from exc

    all_time_periods = sorted(df[time_col].unique())
    if treatment_period not in all_time_periods:
        raise ValueError(f"treatment_period ({treatment_period}) not found in time column '{time_col}'.")
    if treatment_period == all_time_periods[0]:
        raise ValueError("treatment_period cannot be the first period. Need pre-treatment periods.")

    pre_periods = [t for t in all_time_periods if t < treatment_period]
    if not pre_periods:
        raise ValueError("No pre-treatment periods found before treatment_period.")

    if l_outcome_periods:
        if not all(p in pre_periods for p in l_outcome_periods):
            raise ValueError("All l_outcome_periods must be in the pre-treatment period.")

    default_post_periods = [treatment_period]
    if post_periods_of_interest is None:
        post_periods_of_interest = default_post_periods
    else:
        if not all(p >= treatment_period for p in post_periods_of_interest):
            raise ValueError("All post_periods_of_interest must be >= treatment_period.")
        if not all(p in all_time_periods for p in post_periods_of_interest):
            raise ValueError("Not all post_periods_of_interest found in time column.")
    post_periods_of_interest = sorted(list(set(post_periods_of_interest)))

    relevant_periods = sorted(list(set(pre_periods + post_periods_of_interest)))
    _check_treatment_uniqueness(df, id_col, treat_col)

    treated_ids = df[df[treat_col] == 1][id_col].unique()
    control_ids = df[df[treat_col] == 0][id_col].unique()

    if len(treated_ids) == 0:
        raise ValueError("No treated units found (treat_col == 1).")
    if len(control_ids) == 0:
        raise ValueError("No control units found (treat_col == 0) to form a donor pool.")

    df_subset = df[df[id_col].isin(np.concatenate([treated_ids, control_ids]))]
    df_balanced = _filter_and_balance_for_periods(df_subset, id_col, time_col, relevant_periods)
    if df_balanced.empty:
        raise ValueError("DataFrame empty after balancing for relevant periods. Check data coverage.")

    final_all_ids = df_balanced[id_col].unique()
    final_treated_ids = sorted(list(set(treated_ids).intersection(final_all_ids)))
    final_control_ids = sorted(list(set(control_ids).intersection(final_all_ids)))

    if not final_treated_ids:
        raise ValueError("No treated units remain after balancing panel for all relevant periods.")
    if not final_control_ids:
        raise ValueError("No control units remain after balancing panel for all relevant periods.")

    df_balanced = df_balanced.sort_values(by=[id_col, time_col])

    na_check_cols = [y_col]
    if weights_col:
        na_check_cols.append(weights_col)

    ids_with_na = df_balanced[df_balanced[time_col].isin(relevant_periods)][na_check_cols + [id_col]]
    ids_with_na = ids_with_na.groupby(id_col).filter(lambda x: x[na_check_cols].isna().any().any())
    ids_with_na = ids_with_na[id_col].unique()

    if len(ids_with_na) > 0:
        msg = f"Dropping {len(ids_with_na)} units due to NA in '{y_col}' or '{weights_col}' within relevant periods."
        warnings.warn(msg, UserWarning)
        df_balanced = df_balanced[~df_balanced[id_col].isin(ids_with_na)]
        final_treated_ids = sorted(list(set(final_treated_ids) - set(ids_with_na)))
        final_control_ids = sorted(list(set(final_control_ids) - set(ids_with_na)))

    if not final_treated_ids:
        raise ValueError("No treated units remain after NA handling.")
    if not final_control_ids:
        raise ValueError("No control units remain after NA handling.")

    if weights_col:
        df_weights = df_balanced[df_balanced[time_col] == relevant_periods[0]][[id_col, weights_col]].copy()
        df_weights = df_weights.rename(columns={weights_col: "_unit_weights"})
        if not pd.api.types.is_numeric_dtype(df_weights["_unit_weights"]):
            raise TypeError(f"Weights column '{weights_col}' must be numeric.")
        if (df_weights["_unit_weights"] < 0).any():
            warnings.warn("Some weights are negative. Ensure this is intended.", UserWarning)
    else:
        df_weights = pd.DataFrame({id_col: df_balanced[id_col].unique(), "_unit_weights": 1.0})

    weights_map = df_weights.set_index(id_col)["_unit_weights"]
    weights_treat_arr = weights_map.loc[final_treated_ids].values.astype(float)
    weights_control_arr = weights_map.loc[final_control_ids].values.astype(float)

    if normalized:
        overall_mean_weight = weights_map.loc[np.concatenate([final_treated_ids, final_control_ids])].mean()
        if overall_mean_weight > 0:
            weights_treat_arr /= overall_mean_weight
            weights_control_arr /= overall_mean_weight
        else:
            warnings.warn("Mean of weights is zero or negative. Cannot normalize.", UserWarning)

    df_pivot_y = df_balanced.pivot_table(index=id_col, columns=time_col, values=y_col)

    y_pre_treat_df = df_pivot_y.loc[final_treated_ids, pre_periods]
    y_post_treat_df = df_pivot_y.loc[final_treated_ids, post_periods_of_interest]
    y_pre_control_df = df_pivot_y.loc[final_control_ids, pre_periods]
    y_post_control_df = df_pivot_y.loc[final_control_ids, post_periods_of_interest]

    predictor_names = []
    z_parts_treat = []
    z_parts_control = []

    if l_outcome_periods:
        lagged_outcomes_df = df_pivot_y[l_outcome_periods]
        z_parts_treat.append(lagged_outcomes_df.loc[final_treated_ids].T.values)
        z_parts_control.append(lagged_outcomes_df.loc[final_control_ids].T.values)
        predictor_names.extend([f"{y_col}_lag_{p}" for p in l_outcome_periods])

    generated_cov_names = []

    if covariates_formula:
        last_pre_period = pre_periods[-1]
        df_for_formulaic = df_balanced[
            (df_balanced[time_col] == last_pre_period)
            & (df_balanced[id_col].isin(np.concatenate([final_treated_ids, final_control_ids])))
        ].set_index(id_col)

        try:
            model_matrix_full = fml.model_matrix(covariates_formula, df_for_formulaic, output="pandas")
            generated_cov_names = [col for col in model_matrix_full.columns if col != "Intercept"]

            if generated_cov_names:
                cov_matrix_treat = model_matrix_full.loc[final_treated_ids, generated_cov_names].T.values
                cov_matrix_control = model_matrix_full.loc[final_control_ids, generated_cov_names].T.values
                z_parts_treat.append(cov_matrix_treat)
                z_parts_control.append(cov_matrix_control)
                predictor_names.extend(generated_cov_names)
        except Exception as e:
            raise ValueError(
                f"Error processing covariates_formula '{covariates_formula}' for synth with formulaic: {e}"
            ) from e

    z_treat = np.vstack(z_parts_treat) if z_parts_treat else np.empty((0, len(final_treated_ids)))
    z_control = np.vstack(z_parts_control) if z_parts_control else np.empty((0, len(final_control_ids)))

    if z_treat.shape[0] == 0:
        msg = "No predictors specified (l_outcome_periods or covariates_formula). "
        msg += "Z matrices will be empty."
        warnings.warn(msg, UserWarning)

    if z_control.shape[0] > 0 and z_control.shape[1] > 0:
        if z_control.shape[0] < z_control.shape[1]:
            matrix_for_rank_check = z_control.T
            if matrix_for_rank_check.shape[0] > 0 and matrix_for_rank_check.shape[1] > 0:
                _, r = np.linalg.qr(matrix_for_rank_check)
                diag_r = np.abs(np.diag(r))
                tol = diag_r.max() * 1e-7 if diag_r.size > 0 else 1e-7
                rank = np.sum(diag_r > tol)
                num_predictors = matrix_for_rank_check.shape[1]
                if rank < num_predictors:
                    msg = "Potential collinearity detected among predictors for control units. "
                    msg += f"Rank ({rank}) is less than number of predictors ({num_predictors}). "
                    msg += "Synthetic control weights may be unstable or not unique."
                    warnings.warn(msg, UserWarning)
        else:
            msg = f"Number of predictors ({z_control.shape[0]}) >= "
            msg += f"number of control units ({z_control.shape[1]}). "
            msg += "This can lead to issues in synthetic control estimation."
            warnings.warn(msg, UserWarning)

    output = {
        "Y_pre_treat": y_pre_treat_df.T.values,
        "Y_post_treat": y_post_treat_df.T.values,
        "Y_pre_control": y_pre_control_df.T.values,
        "Y_post_control": y_post_control_df.T.values,
        "Z_treat": z_treat,
        "Z_control": z_control,
        "weights_treat": weights_treat_arr,
        "weights_control": weights_control_arr,
        "n_treated": len(final_treated_ids),
        "n_control": len(final_control_ids),
        "n_pre_periods": len(pre_periods),
        "n_post_periods": len(post_periods_of_interest),
        "pre_periods": pre_periods,
        "post_periods_of_interest": post_periods_of_interest,
        "treatment_period": treatment_period,
        "predictor_names": predictor_names,
        "treated_ids": final_treated_ids,
        "control_ids": final_control_ids,
        "normalized_weights": normalized,
    }
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


def _filter_and_balance_for_periods(
    df: pd.DataFrame, id_col: str, time_col: str, required_periods: list[Any]
) -> pd.DataFrame:
    """Filter dataframe for required_periods and keeps only units observed in all of them."""
    df_filtered = df[df[time_col].isin(required_periods)].copy()
    n_required_periods = len(set(required_periods))

    obs_counts = df_filtered.groupby(id_col)[time_col].nunique()
    ids_to_keep = obs_counts[obs_counts == n_required_periods].index

    if len(ids_to_keep) < df_filtered[id_col].nunique():
        msg = (
            "Balancing panel for specified periods. Dropping units not observed in all "
            f"{n_required_periods} required periods: {sorted(list(set(required_periods)))}."
        )
        warnings.warn(msg, UserWarning)
    return df_filtered[df_filtered[id_col].isin(ids_to_keep)].copy()
