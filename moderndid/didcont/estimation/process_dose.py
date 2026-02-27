"""Processing functions for continuous treatment dose-response results."""

import warnings

import numpy as np
import scipy.stats as st

from moderndid.did.mboot import mboot

from ...cupy.backend import to_numpy
from ..spline import BSpline
from .container import DoseResult
from .process_aggte import (
    check_critical_value,
    get_se,
    overall_weights,
)
from .process_attgt import process_att_gt


def process_dose_gt(
    gt_results, pte_params, balance_event=None, min_event_time=-np.inf, max_event_time=np.inf, rng=None
):
    """Process group-time results for continuous treatment dose-response.

    Parameters
    ----------
    gt_results : dict
        Dictionary containing group-time specific results with keys:

        - **attgt_list**: list of ATT(g,t) estimates
        - **influence_func**: influence function matrix
        - **extra_gt_returns**: list of extra returns with dose-specific results

    pte_params : PTEParams
        Parameters object containing estimation settings including dose values.
    balance_event : int, optional
        Relevant for dynamic aggregation but not used in dose processing.
    min_event_time : float, default=-np.inf
        Minimum event time for filtering.
    max_event_time : float, default=np.inf
        Maximum event time for filtering.
    rng : numpy.random.Generator, optional
        Random number generator for bootstrap. If None, a new generator is created.

    Returns
    -------
    DoseResult
        NamedTuple containing dose-response results.
    """
    if rng is None:
        rng = np.random.default_rng()

    att_gt = process_att_gt(gt_results, pte_params, rng=rng)
    all_extra_gt_returns = att_gt.extra_gt_returns

    if not all_extra_gt_returns:
        raise ValueError("No dose-specific results found in extra_gt_returns")

    groups = np.array([item["group"] for item in all_extra_gt_returns])
    time_periods = np.array([item["time_period"] for item in all_extra_gt_returns])

    if not np.array_equal(
        np.column_stack([groups, time_periods]),
        np.column_stack([att_gt.groups, att_gt.times]),
    ):
        raise ValueError("Mismatch between order of groups and time periods in processing dose results")

    weights_dict = overall_weights(att_gt, balance_event, min_event_time, max_event_time)

    inner_extra_gt_returns = [item.get("extra_gt_returns") for item in all_extra_gt_returns]

    att_d_by_group = [(item.get("att_d") if item else None) for item in inner_extra_gt_returns]
    acrt_d_by_group = [(item.get("acrt_d") if item else None) for item in inner_extra_gt_returns]
    att_overall_by_group = np.array(
        [(item.get("att_overall", np.nan) if item else np.nan) for item in inner_extra_gt_returns]
    )
    acrt_overall_by_group = np.array(
        [(item.get("acrt_overall", np.nan) if item else np.nan) for item in inner_extra_gt_returns]
    )

    bread_matrices = [(item.get("bread") if item else None) for item in inner_extra_gt_returns]
    x_expanded_by_group = [(item.get("x_expanded") if item else None) for item in inner_extra_gt_returns]

    acrt_influence_matrix = gt_results["influence_func"]
    n_obs = acrt_influence_matrix.shape[0]
    bootstrap_iterations = pte_params.biters
    alpha = pte_params.alp
    confidence_band = pte_params.cband

    overall_att = float(np.nansum(att_overall_by_group * weights_dict["weights"]))

    att_influence_matrix = att_gt.influence_func
    overall_att_inf_func = _compute_overall_att_inf_func(weights_dict["weights"], att_influence_matrix)
    overall_att_se = float(
        get_se(
            overall_att_inf_func[:, None],
            bootstrap=True,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            rng=rng,
        )
    )

    overall_acrt = float(np.nansum(acrt_overall_by_group * weights_dict["weights"]))
    overall_acrt_inf_func = np.sum(acrt_influence_matrix * weights_dict["weights"][np.newaxis, :], axis=1)
    overall_acrt_se = float(
        get_se(
            overall_acrt_inf_func[:, None],
            bootstrap=True,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            rng=rng,
        )
    )

    dose_values = pte_params.dvals
    if dose_values is None or len(dose_values) == 0:
        warnings.warn("No dose values provided, returning overall results only")
        return DoseResult(
            dose=np.array([]),
            overall_att=overall_att,
            overall_att_se=overall_att_se,
            overall_att_inf_func=overall_att_inf_func,
            overall_acrt=overall_acrt,
            overall_acrt_se=overall_acrt_se,
            overall_acrt_inf_func=overall_acrt_inf_func,
            pte_params=pte_params,
        )

    degree = pte_params.degree if pte_params.degree is not None else 1
    knots = pte_params.knots if pte_params.knots is not None else np.array([])

    if len(knots) > 0:
        bspline = BSpline(x=dose_values, internal_knots=knots, degree=degree)
    else:
        bspline = BSpline(x=dose_values, degree=degree)

    basis_matrix = to_numpy(bspline.basis(complete_basis=False))
    basis_matrix = np.column_stack([np.ones(len(dose_values)), basis_matrix])

    if degree > 0:
        derivative_matrix = to_numpy(bspline.derivative(derivs=1, complete_basis=False))
        derivative_matrix = np.column_stack([np.zeros(len(dose_values)), derivative_matrix])
    else:
        derivative_matrix = np.zeros((len(dose_values), basis_matrix.shape[1]))

    if att_d_by_group and any(x is not None for x in att_d_by_group):
        att_d = _weighted_combine_arrays(att_d_by_group, weights_dict["weights"])
    else:
        att_d = np.full(len(dose_values), np.nan)

    if acrt_d_by_group and any(x is not None for x in acrt_d_by_group):
        acrt_d = _weighted_combine_arrays(acrt_d_by_group, weights_dict["weights"])
    else:
        acrt_d = np.full(len(dose_values), np.nan)

    att_d_inf_func, acrt_d_inf_func = _compute_dose_influence_functions(
        x_expanded_by_group,
        bread_matrices,
        basis_matrix,
        derivative_matrix,
        acrt_influence_matrix,
        att_influence_matrix,
        weights_dict["weights"],
        n_obs,
    )

    if att_d_inf_func is not None:
        boot_res = mboot(att_d_inf_func, n_units=n_obs, biters=bootstrap_iterations, alp=alpha, random_state=rng)
        att_d_se = boot_res["se"]
        att_d_crit_val = boot_res["crit_val"] if confidence_band else st.norm.ppf(1 - alpha / 2)
        att_d_crit_val = check_critical_value(att_d_crit_val, alpha)
    else:
        att_d_se = np.full(len(dose_values), np.nan)
        att_d_crit_val = st.norm.ppf(1 - alpha / 2)

    if acrt_d_inf_func is not None:
        acrt_boot_res = mboot(acrt_d_inf_func, n_units=n_obs, biters=bootstrap_iterations, alp=alpha, random_state=rng)
        acrt_d_se = acrt_boot_res["se"]
        acrt_d_crit_val = acrt_boot_res["crit_val"] if confidence_band else st.norm.ppf(1 - alpha / 2)
        acrt_d_crit_val = check_critical_value(acrt_d_crit_val, alpha)
    else:
        acrt_d_se = np.full(len(dose_values), np.nan)
        acrt_d_crit_val = st.norm.ppf(1 - alpha / 2)

    return DoseResult(
        dose=dose_values,
        overall_att=overall_att,
        overall_att_se=overall_att_se,
        overall_att_inf_func=overall_att_inf_func,
        overall_acrt=overall_acrt,
        overall_acrt_se=overall_acrt_se,
        overall_acrt_inf_func=overall_acrt_inf_func,
        att_d=att_d,
        att_d_se=att_d_se,
        att_d_crit_val=att_d_crit_val,
        att_d_inf_func=att_d_inf_func,
        acrt_d=acrt_d,
        acrt_d_se=acrt_d_se,
        acrt_d_crit_val=acrt_d_crit_val,
        acrt_d_inf_func=acrt_d_inf_func,
        pte_params=pte_params,
    )


def _compute_dose_influence_functions(
    x_expanded_by_group,
    bread_matrices,
    basis_matrix,
    derivative_matrix,
    acrt_influence_matrix,
    att_influence_matrix,
    weights,
    n_obs,
):
    """Compute influence functions for dose-specific treatment effects.

    Parameters
    ----------
    x_expanded_by_group : list
        List of expanded design matrices for each group.
    bread_matrices : list
        List of bread matrices from sandwich estimator for each group.
    basis_matrix : ndarray
        B-spline basis matrix evaluated at dose values.
    derivative_matrix : ndarray
        Derivative of B-spline basis matrix.
    acrt_influence_matrix : ndarray
        Influence function matrix for ACRT from group-time estimation.
    att_influence_matrix : ndarray
        Influence function matrix for ATT from group-time estimation.
    weights : ndarray
        Weights for aggregating across groups.
    n_obs : int
        Number of observations.

    Returns
    -------
    tuple of ndarray
        (att_d_influence, acrt_d_influence) - Influence functions for ATT(d) and ACRT(d).
    """
    n_doses = basis_matrix.shape[0]
    n_groups = acrt_influence_matrix.shape[1]

    att_d_influence = np.zeros((n_obs, n_doses))
    acrt_d_influence = np.zeros((n_obs, n_doses))

    treated_mask = acrt_influence_matrix != 0
    all_involved_mask = att_influence_matrix != 0
    comparison_mask = all_involved_mask & ~treated_mask

    n_treated_by_group = np.sum(treated_mask, axis=0)

    for group_idx in range(n_groups):
        if weights[group_idx] == 0:
            continue

        x_expanded = x_expanded_by_group[group_idx]
        bread = bread_matrices[group_idx]

        if x_expanded is not None and bread is not None and n_treated_by_group[group_idx] > 0:
            treated_contribution = x_expanded @ bread @ basis_matrix.T
            treated_indices = np.where(treated_mask[:, group_idx])[0]

            n_rows_x_expanded = treated_contribution.shape[0]
            n_treated_to_use = min(len(treated_indices), n_rows_x_expanded)

            for i in range(n_treated_to_use):
                idx = treated_indices[i]
                att_d_influence[idx, :] += (
                    weights[group_idx] * (n_obs / n_treated_by_group[group_idx]) * treated_contribution[i, :]
                )

            comparison_influence = att_influence_matrix[comparison_mask[:, group_idx], group_idx]
            att_d_influence[comparison_mask[:, group_idx], :] -= weights[group_idx] * np.tile(
                comparison_influence[:, np.newaxis], (1, n_doses)
            )

            acrt_contribution = x_expanded @ bread @ derivative_matrix.T

            for i in range(n_treated_to_use):
                idx = treated_indices[i]
                acrt_d_influence[idx, :] += (
                    weights[group_idx] * (n_obs / n_treated_by_group[group_idx]) * acrt_contribution[i, :]
                )

    return att_d_influence, acrt_d_influence


def _weighted_combine_arrays(array_list, weights):
    """Combine list of arrays with weights."""
    if not array_list:
        return np.array([])

    arrays = []
    valid_weights = []

    for i, arr in enumerate(array_list):
        if arr is not None:
            arrays.append(np.asarray(arr))
            valid_weights.append(weights[i])

    if not arrays:
        return np.array([])

    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / np.sum(valid_weights)

    result = np.zeros_like(arrays[0], dtype=np.float64)

    for arr, w in zip(arrays, valid_weights, strict=False):
        result += w * arr

    return result


def _compute_overall_att_inf_func(weights, att_influence_matrix):
    """Compute influence function for overall ATT by aggregating group-time influence functions."""
    if att_influence_matrix is None:
        return None

    overall_influence = np.sum(att_influence_matrix * weights[np.newaxis, :], axis=1)

    return overall_influence


def _summary_dose_result(dose_result):
    """Create summary of dose-response results."""
    summary = {
        "dose": dose_result.dose,
        "overall_att": dose_result.overall_att,
        "overall_att_se": dose_result.overall_att_se,
        "overall_acrt": dose_result.overall_acrt,
        "overall_acrt_se": dose_result.overall_acrt_se,
        "att_d": dose_result.att_d,
        "att_d_se": dose_result.att_d_se,
        "att_d_crit_val": dose_result.att_d_crit_val,
        "acrt_d": dose_result.acrt_d,
        "acrt_d_se": dose_result.acrt_d_se,
        "acrt_d_crit_val": dose_result.acrt_d_crit_val,
    }

    if dose_result.pte_params:
        summary.update(
            {
                "alpha": dose_result.pte_params.alp,
                "cband": dose_result.pte_params.cband,
                "biters": dose_result.pte_params.biters,
            }
        )

    return summary
