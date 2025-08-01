"""Multi-period difference-in-differences group-time ATT computation."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from causaldid.drdid.estimators.drdid_panel import drdid_panel
from causaldid.drdid.estimators.drdid_rc import drdid_rc
from causaldid.drdid.estimators.reg_did_panel import reg_did_panel
from causaldid.drdid.estimators.reg_did_rc import reg_did_rc
from causaldid.drdid.estimators.std_ipw_did_panel import std_ipw_did_panel
from causaldid.drdid.estimators.std_ipw_did_rc import std_ipw_did_rc

from .preprocess.models import DIDData


class ATTgtResult(NamedTuple):
    """Result from group-time ATT estimation."""

    att: float
    group: float
    year: float
    post: int


class ComputeATTgtResult(NamedTuple):
    """Result from compute_att_gt function."""

    attgt_list: list[ATTgtResult]
    influence_functions: sp.csr_matrix


def compute_att_gt(data: DIDData):
    """Compute group-time average treatment effects.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object containing all necessary data and configuration.

    Returns
    -------
    ComputeATTgtResult
        NamedTuple containing list of ATT results and influence functions
    """
    n_units = data.config.id_count
    time_periods = data.config.time_periods

    n_time_periods = len(time_periods) - 1 if data.config.base_period != "universal" else len(time_periods)
    time_factor = 1 if data.config.base_period != "universal" else 0

    group_time_pairs = [(g, t) for g in range(data.config.treated_groups_count) for t in range(n_time_periods)]

    att_results = []
    influence_func_list = []

    for group_idx, time_idx in group_time_pairs:
        estimation_result = run_att_gt_estimation(group_idx, time_idx, data)

        is_post_treatment = int(
            data.config.treated_groups[group_idx] <= data.config.time_periods[time_idx + time_factor]
        )

        if estimation_result is None or estimation_result["att"] is None:
            if data.config.base_period == "universal":
                att_results.append(
                    ATTgtResult(
                        att=0.0,
                        group=data.config.treated_groups[group_idx],
                        year=data.config.time_periods[time_idx + time_factor],
                        post=is_post_treatment,
                    )
                )
                influence_func_list.append(np.zeros(n_units))
        else:
            att_estimate = estimation_result["att"]
            influence_func = estimation_result["inf_func"]

            if np.isnan(att_estimate):
                att_estimate = 0.0
                influence_func = np.zeros(n_units)

            att_results.append(
                ATTgtResult(
                    att=att_estimate,
                    group=data.config.treated_groups[group_idx],
                    year=data.config.time_periods[time_idx + time_factor],
                    post=is_post_treatment,
                )
            )
            influence_func_list.append(influence_func)

    if influence_func_list:
        influence_matrix = np.column_stack(influence_func_list)
        sparse_influence_funcs = sp.csr_matrix(influence_matrix)
    else:
        sparse_influence_funcs = sp.csr_matrix((n_units, 0))

    return ComputeATTgtResult(attgt_list=att_results, influence_functions=sparse_influence_funcs)


def run_att_gt_estimation(
    group_idx,
    time_idx,
    data,
):
    """Run ATT estimation for a given group-time pair.

    Parameters
    ----------
    group_idx : int
        Index of the treated group.
    time_idx : int
        Index of the time period.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    dict or None
        Dictionary with ATT and influence function, or None if estimation not feasible.
    """
    time_factor = 1 if data.config.base_period != "universal" else 0

    if data.config.base_period == "universal":
        pre_periods = np.where(
            data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
        )[0]
        if len(pre_periods) > 0:
            pre_treatment_idx = pre_periods[-1]
        else:
            pre_treatment_idx = None
    else:
        pre_treatment_idx = time_idx

    is_post_treatment = data.config.treated_groups[group_idx] <= data.config.time_periods[time_idx + time_factor]

    if is_post_treatment and data.config.base_period != "universal":
        pre_periods = np.where(
            data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
        )[0]

        if len(pre_periods) == 0:
            warnings.warn(
                f"No pre-treatment periods for group first treated at {data.config.treated_groups[group_idx]}. "
                "Units from this group are dropped.",
                UserWarning,
            )
            return None

        pre_treatment_idx = pre_periods[-1]

    if (
        data.config.base_period == "universal"
        and pre_treatment_idx is not None
        and data.config.time_periods[pre_treatment_idx] == data.config.time_periods[time_idx + time_factor]
    ):
        return None

    cohort_index = get_did_cohort_index(group_idx, time_idx, time_factor, pre_treatment_idx, data)

    has_treated = np.any(cohort_index == 1)
    has_control = np.any(cohort_index == 0)

    if not (has_treated and has_control):
        return None

    if data.config.panel:
        cohort_data = {
            "D": cohort_index,
            "y1": data.outcomes_tensor[time_idx + time_factor],
            "y0": data.outcomes_tensor[pre_treatment_idx],
            "weights": data.weights,
        }
        covariates = data.covariates_tensor[min(pre_treatment_idx, time_idx)]
    else:
        post_mask = data.data[data.config.tname] == data.config.time_periods[time_idx + time_factor]
        cohort_data = {
            "D": cohort_index,
            "y": data.data[data.config.yname].values,
            "post": post_mask.astype(int).values,
            "weights": data.data["weights"].values,
        }
        if data.config.allow_unbalanced_panel:
            cohort_data["rowid"] = data.data[".rowid"].values
        covariates = data.covariates_matrix

    try:
        return run_drdid(cohort_data, covariates, data)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        warnings.warn(
            f"Error in computing 2x2 DiD for (g,t) = ({data.config.treated_groups[group_idx]},"
            f"{data.config.time_periods[time_idx]}): {e}",
            UserWarning,
        )
        return None


def get_did_cohort_index(
    group_idx,
    time_idx,
    time_factor,
    pre_treatment_idx,
    data,
):
    """Get cohort indices for current group-time pair.

    Parameters
    ----------
    group_idx : int
        Index of the treated group.
    time_idx : int
        Index of the time period.
    time_factor : int
        Time factor (1 for varying base period, 0 for universal).
    pre_treatment_idx : int
        Index of the pre-treatment period.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    np.ndarray
        Array of 1s (treated), 0s (control), and NaNs indicating cohort membership.
    """
    if data.config.panel:
        # Determine control group boundaries
        if data.config.control_group == "notyettreated":
            # Find first cohort treated after the relevant period
            relevant_period = data.config.time_periods[
                max(time_idx, pre_treatment_idx) + time_factor + data.config.anticipation
            ]
            future_cohorts = data.cohort_counts[data.cohort_counts["cohort"] > relevant_period]
            if len(future_cohorts) > 0:
                min_control = future_cohorts["cohort"].iloc[0]
            else:
                min_control = np.inf
        else:  # nevertreated
            min_control = np.inf

        max_control = np.inf

        n_units = len(data.time_invariant_data) if data.config.allow_unbalanced_panel else data.config.id_count
        cohort_index = np.full(n_units, np.nan)

        if max_control not in data.cohort_counts["cohort"].values:
            max_control = data.cohort_counts["cohort"].iloc[-1]

        # Control group indices
        control_mask = (data.cohort_counts["cohort"] >= min_control) & (data.cohort_counts["cohort"] <= max_control)
        if control_mask.any():
            control_idx = control_mask.idxmax()
            start_control = data.cohort_counts.loc[: control_idx - 1, "cohort_size"].sum() if control_idx > 0 else 0
            end_control = data.cohort_counts.loc[:control_idx, "cohort_size"].sum()
            cohort_index[start_control:end_control] = 0

        # Treated group indices
        treated_mask = data.cohort_counts["cohort"] == data.config.treated_groups[group_idx]
        if treated_mask.any():
            treat_idx = treated_mask.idxmax()
            start_treat = data.cohort_counts.iloc[:treat_idx]["cohort_size"].sum() if treat_idx > 0 else 0
            end_treat = data.cohort_counts.iloc[: treat_idx + 1]["cohort_size"].sum()
            cohort_index[start_treat:end_treat] = 1

    else:
        n_units = len(data.data)
        cohort_index = np.full(n_units, np.nan)

        treated_flag = data.data[data.config.gname] == data.config.treated_groups[group_idx]

        if data.config.control_group == "nevertreated":
            control_flag = data.data[data.config.gname] == np.inf
        else:  # notyettreated
            relevant_period = data.config.time_periods[
                max(time_idx, pre_treatment_idx) + time_factor + data.config.anticipation
            ]
            control_flag = (data.data[data.config.gname] == np.inf) | (
                (data.data[data.config.gname] > relevant_period)
                & (data.data[data.config.gname] != data.config.treated_groups[group_idx])
            )

        keep_periods = data.data[data.config.tname].isin(
            [data.config.time_periods[time_idx + time_factor], data.config.time_periods[pre_treatment_idx]]
        )

        cohort_index[keep_periods & control_flag] = 0
        cohort_index[keep_periods & treated_flag] = 1

    return cohort_index


def run_drdid(
    cohort_data,
    covariates,
    data,
):
    """Run DR-DiD estimation for current group-time pair.

    Parameters
    ----------
    cohort_data : dict
        Dictionary containing outcome and treatment data for the cohort.
    covariates : ndarray
        Covariate matrix for the estimation.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    dict
        Dictionary with ATT estimate and influence function.
    """
    n = len(cohort_data["D"])
    est_method = data.config.est_method
    valid_obs = ~np.isnan(cohort_data["D"])

    if valid_obs.sum() == 0:
        return {"att": np.nan, "inf_func": np.zeros(n)}

    if data.config.panel:
        y1 = cohort_data["y1"][valid_obs]
        y0 = cohort_data["y0"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates[valid_obs] if len(covariates) > 1 else np.ones(valid_obs.sum())

        if callable(est_method):
            result = est_method(y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == "ipw":
            result = std_ipw_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == "reg":
            result = reg_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:  # doubly robust (default)
            result = drdid_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )

        influence_func = np.zeros(n)
        influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    else:
        y = cohort_data["y"][valid_obs]
        post = cohort_data["post"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates if len(covariates) == n else covariates[valid_obs]

        if callable(est_method):
            result = est_method(y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == "ipw":
            result = std_ipw_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == "reg":
            result = reg_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:  # doubly robust (default)
            result = drdid_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )

        # Handle influence function for unbalanced panel
        if data.config.allow_unbalanced_panel and "rowid" in cohort_data:
            inf_func_long = np.zeros(n)
            inf_func_long[valid_obs] = (data.config.id_count / valid_obs.sum()) * result.att_inf_func

            unique_ids = np.unique(cohort_data["rowid"])
            influence_func = np.zeros(len(unique_ids))
            for i, uid in enumerate(unique_ids):
                mask = cohort_data["rowid"] == uid
                influence_func[i] = inf_func_long[mask].sum()
        else:
            influence_func = np.zeros(n)
            influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    return {"att": result.att, "inf_func": influence_func}
