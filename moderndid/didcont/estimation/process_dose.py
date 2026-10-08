"""Processing functions for continuous treatment dose-response results."""

import warnings

import numpy as np
import scipy.stats as st

from moderndid.did.mboot import _mboot

from ...cupy.backend import to_numpy
from ..container import DoseResult
from ..spline import BSpline
from .process_aggte import (
    check_critical_value,
    get_se,
    overall_weights,
    set_small_se_to_nan,
    weight_influence_function_from_cells,
)
from .process_attgt import process_att_gt


def process_dose_gt(
    gt_results, pte_params, balance_event=None, min_event_time=-np.inf, max_event_time=np.inf, rng=None
):
    """Process group-time results for continuous treatment dose-response.

    Every cell enters with the weights of the group aggregation. These give each cohort its share of the
    treated units and split that share evenly over the cohort's post-treatment cells. The overall ATT
    takes its influence function from the binary ATT of each cell. Each aggregate also carries the term
    that estimating the cohort shares adds to its influence function.

    Parameters
    ----------
    gt_results : dict
        Dictionary containing group-time specific results with keys:

        - **attgt_list**: list of ATT(g,t) estimates
        - **influence_func**: influence function matrix of each cell's overall ACRT
        - **extra_gt_returns**: list of extra returns with dose-specific results and the cell's
          ``att_inf_func``, ``rows``, and ``treated`` entries

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

    # Since only the cells are used below, a cell-level band would add a warning about a value nobody reads.
    att_gt = process_att_gt(gt_results, pte_params._replace(cband=False), rng=rng)
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
    weights = weights_dict["weights"]

    inner_extra_gt_returns = [item.get("extra_gt_returns") for item in all_extra_gt_returns]

    att_d_by_group = [(item.get("att_d") if item else None) for item in inner_extra_gt_returns]
    acrt_d_by_group = [(item.get("acrt_d") if item else None) for item in inner_extra_gt_returns]
    att_overall_by_group = np.array(
        [(item.get("att_overall", np.nan) if item else np.nan) for item in inner_extra_gt_returns]
    )
    acrt_overall_by_group = np.array(
        [(item.get("acrt_overall", np.nan) if item else np.nan) for item in inner_extra_gt_returns]
    )

    acrt_influence_matrix = gt_results["influence_func"]
    n_obs = acrt_influence_matrix.shape[0]
    bootstrap_iterations = pte_params.biters
    alpha = pte_params.alp
    confidence_band = pte_params.cband

    att_influence_matrix = _binary_att_influence_matrix(inner_extra_gt_returns, groups, time_periods, n_obs)
    weight_inf_func = weight_influence_function_from_cells(att_gt, weights)

    overall_att = float(np.nansum(att_overall_by_group * weights))
    overall_att_inf_func = _compute_overall_att_inf_func(
        weights, att_influence_matrix
    ) + weight_inf_func @ np.nan_to_num(att_overall_by_group)
    overall_att_se = float(
        get_se(
            overall_att_inf_func[:, None],
            bootstrap=True,
            bootstrap_iterations=bootstrap_iterations,
            alpha=alpha,
            rng=rng,
        )
    )
    overall_att_se = set_small_se_to_nan(overall_att_se)

    overall_acrt = float(np.nansum(acrt_overall_by_group * weights))
    overall_acrt_inf_func = _compute_overall_att_inf_func(
        weights, acrt_influence_matrix
    ) + weight_inf_func @ np.nan_to_num(acrt_overall_by_group)
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

    if att_d_by_group and any(x is not None for x in att_d_by_group):
        att_d = _weighted_combine_arrays(att_d_by_group, weights)
    else:
        att_d = np.full(len(dose_values), np.nan)

    if acrt_d_by_group and any(x is not None for x in acrt_d_by_group):
        acrt_d = _weighted_combine_arrays(acrt_d_by_group, weights)
    else:
        acrt_d = np.full(len(dose_values), np.nan)

    att_d_inf_func, acrt_d_inf_func = _compute_dose_influence_functions(
        inner_extra_gt_returns,
        dose_values,
        degree,
        knots,
        weights,
        n_obs,
    )
    att_d_inf_func = att_d_inf_func + weight_inf_func @ _stack_cell_curves(att_d_by_group, len(dose_values))
    acrt_d_inf_func = acrt_d_inf_func + weight_inf_func @ _stack_cell_curves(acrt_d_by_group, len(dose_values))

    # Since dropping the draws that move a column with zero bootstrap scale would condition the band on that column
    # staying at zero, they stay in the sample.
    boot_res = _mboot(
        att_d_inf_func,
        n_units=n_obs,
        biters=bootstrap_iterations,
        alp=alpha,
        random_state=rng,
        keep_infinite_draws=True,
    )
    att_d_se = boot_res["se"]
    att_d_crit_val = boot_res["crit_val"] if confidence_band else st.norm.ppf(1 - alpha / 2)
    att_d_crit_val = check_critical_value(att_d_crit_val, alpha)

    acrt_boot_res = _mboot(
        acrt_d_inf_func,
        n_units=n_obs,
        biters=bootstrap_iterations,
        alp=alpha,
        random_state=rng,
        keep_infinite_draws=True,
    )
    acrt_d_se = acrt_boot_res["se"]
    acrt_d_crit_val = acrt_boot_res["crit_val"] if confidence_band else st.norm.ppf(1 - alpha / 2)
    acrt_d_crit_val = check_critical_value(acrt_d_crit_val, alpha)

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
    cell_returns,
    dose_values,
    degree,
    knots,
    weights,
    n_obs,
):
    """Compute influence functions for dose-specific treatment effects.

    Within a cell the estimated spline coefficients drive the treated units' part of both influence
    functions. Since ATT(d) also subtracts the comparison mean, the comparison units enter it with the
    same value at every dose. Each cell builds its basis with the boundary knots that its point
    estimates use.

    Parameters
    ----------
    cell_returns : list
        Dose-specific results of each cell with ``x_expanded``, ``bread``, ``boundary_knots``,
        ``att_inf_func``, ``rows``, and ``treated`` entries, or None for cells without results.
    dose_values : ndarray
        Doses at which the effects are evaluated.
    degree : int
        Degree of the B-spline basis.
    knots : ndarray
        Interior knots of the B-spline basis.
    weights : ndarray
        Aggregation weight of each cell.
    n_obs : int
        Number of units.

    Returns
    -------
    tuple of ndarray
        Tuple containing:

        - **att_d_influence**: Influence function of ATT(d) at each dose
        - **acrt_d_influence**: Influence function of ACRT(d) at each dose
    """
    n_doses = len(dose_values)

    att_d_influence = np.zeros((n_obs, n_doses))
    acrt_d_influence = np.zeros((n_obs, n_doses))

    for weight, cell in zip(weights, cell_returns, strict=True):
        if weight == 0 or not cell:
            continue

        rows = np.asarray(cell["rows"])
        treated = np.asarray(cell["treated"], dtype=bool)
        treated_rows = rows[treated]
        basis_matrix, derivative_matrix = _dose_basis(dose_values, degree, knots, cell.get("boundary_knots"))

        score_bread = cell["x_expanded"] @ cell["bread"]
        scale = weight * n_obs / len(treated_rows)
        att_d_influence[treated_rows, :] += scale * (score_bread @ basis_matrix.T)
        acrt_d_influence[treated_rows, :] += scale * (score_bread @ derivative_matrix.T)
        att_d_influence[rows[~treated], :] += weight * np.asarray(cell["att_inf_func"])[~treated][:, None]

    return att_d_influence, acrt_d_influence


def _dose_basis(dose_values, degree, knots, boundary_knots):
    """Evaluate the intercept-augmented B-spline basis and its derivative at the doses."""
    bspline = BSpline(x=dose_values, degree=degree, internal_knots=knots, boundary_knots=boundary_knots)

    basis_matrix = to_numpy(bspline.basis(complete_basis=False))
    basis_matrix = np.column_stack([np.ones(len(dose_values)), basis_matrix])

    if degree > 0:
        derivative_matrix = to_numpy(bspline.derivative(derivs=1, complete_basis=False))
        derivative_matrix = np.column_stack([np.zeros(len(dose_values)), derivative_matrix])
    else:
        derivative_matrix = np.zeros_like(basis_matrix)

    return basis_matrix, derivative_matrix


def _binary_att_influence_matrix(cell_returns, groups, time_periods, n_obs):
    """Place each cell's binary ATT influence function on the full sample."""
    att_influence_matrix = np.zeros((n_obs, len(cell_returns)))

    for k, cell in enumerate(cell_returns):
        if not cell:
            continue
        if "att_inf_func" not in cell or "rows" not in cell:
            raise ValueError(
                f"Dose results for group {groups[k]} and period {time_periods[k]} lack the binary ATT "
                "influence function ('att_inf_func') or its rows ('rows')."
            )
        att_influence_matrix[np.asarray(cell["rows"]), k] = cell["att_inf_func"]

    return att_influence_matrix


def _stack_cell_curves(curves, n_doses):
    """Stack per-cell dose curves into a matrix with zeros for missing cells."""
    stacked = np.zeros((len(curves), n_doses))
    for k, curve in enumerate(curves):
        if curve is not None:
            stacked[k] = np.nan_to_num(np.asarray(curve, dtype=float))
    return stacked


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
