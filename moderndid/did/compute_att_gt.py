"""Multi-period difference-in-differences group-time ATT computation."""

import warnings

import numpy as np
import polars as pl
import scipy.sparse as sp
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from moderndid.core.parallel import parallel_map
from moderndid.core.preprocess import WEIGHTS_COLUMN, ControlGroup, EstimationMethod
from moderndid.cupy.backend import get_backend, to_numpy
from moderndid.drdid.estimators.drdid_panel import drdid_panel
from moderndid.drdid.estimators.drdid_rc import drdid_rc
from moderndid.drdid.estimators.reg_did_panel import reg_did_panel
from moderndid.drdid.estimators.reg_did_rc import reg_did_rc
from moderndid.drdid.estimators.std_ipw_did_panel import std_ipw_did_panel
from moderndid.drdid.estimators.std_ipw_did_rc import std_ipw_did_rc

from .container import ATTgtResult, ComputeATTgtResult


def compute_att_gt(data, n_jobs=1):
    """Compute group-time average treatment effects.

    Each 2x2 comparison on a balanced panel uses the weights from the earlier
    of its two periods. When a unit's weight changes over time, a warning
    names this rule.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object containing all necessary data and configuration.
    n_jobs : int, default=1
        Number of parallel jobs. 1 = sequential, -1 = all cores, >1 = that many workers.

    Returns
    -------
    ComputeATTgtResult
        - **attgt_list**: One ATTgtResult for each group-time cell in the result
        - **influence_functions**: Sparse matrix of the cells' influence functions with one row per unit
    """
    n_units = data.config.id_count
    time_periods = data.config.time_periods

    n_time_periods = len(time_periods) - 1 if data.config.base_period != "universal" else len(time_periods)
    group_time_pairs = [(g, t) for g in range(data.config.treated_groups_count) for t in range(n_time_periods)]

    if _weights_vary_within_units(data):
        warnings.warn(
            "Time-varying weights detected. For balanced panel data, each 2x2 comparison uses the weights "
            "from the earlier of its two periods, the base period for post-treatment cells.",
            UserWarning,
        )

    # Since every cell of an unbalanced panel maps rows to the same units, the mapping is built once here.
    unit_index = _unit_index(data) if _is_unbalanced_panel(data) else None
    args_list = [(g_idx, t_idx, data, unit_index) for g_idx, t_idx in group_time_pairs]
    cell_results = parallel_map(_process_gt_cell_did, args_list, n_jobs=n_jobs)

    att_results = []
    influence_func_list = []

    for att_result, inf_func in cell_results:
        if att_result is not None:
            att_results.append(att_result)
            influence_func_list.append(inf_func)

    if influence_func_list:
        xp = get_backend()
        if xp is not np:
            try:
                import cupyx.scipy.sparse as cusp

                influence_matrix = xp.column_stack([xp.asarray(f) for f in influence_func_list])
                sparse_influence_funcs = cusp.csr_matrix(influence_matrix)
            except ImportError:
                influence_matrix = np.column_stack(influence_func_list)
                sparse_influence_funcs = sp.csr_matrix(influence_matrix)
        else:
            influence_matrix = np.column_stack(influence_func_list)
            sparse_influence_funcs = sp.csr_matrix(influence_matrix)
    else:
        sparse_influence_funcs = sp.csr_matrix((n_units, 0))

    return ComputeATTgtResult(attgt_list=att_results, influence_functions=sparse_influence_funcs)


def run_att_gt_estimation(
    group_idx,
    time_idx,
    data,
    unit_index=None,
):
    """Run ATT estimation for a given group-time pair.

    A built-in method screens each cell before it estimates the cell. The
    screen skips the cell with a warning when the fitted propensity score
    reaches 0.999 for a unit with positive weight, or when the Gram matrix of
    the comparison units' covariates is numerically singular. Both checks
    weight the units with the sampling weights that the estimators use.

    A cell whose estimator fails or returns no estimate is skipped with a
    warning that names the cell.

    Parameters
    ----------
    group_idx : int
        Index of the treated group.
    time_idx : int
        Index of the time period.
    data : DIDData
        Preprocessed DiD data object.
    unit_index : ndarray, optional
        Position of each row's unit in ``data.time_invariant_data`` for an
        unbalanced panel. It's built from the data when not given.

    Returns
    -------
    dict or None
        The cell's estimate, or None when the cell is skipped.

        - **att**: Estimated ATT of the cell
        - **inf_func**: Influence function of the estimate with one entry per unit
    """
    time_factor = 1 if data.config.base_period != "universal" else 0
    g_val = data.config.treated_groups[group_idx]
    t_val = data.config.time_periods[time_idx + time_factor]
    g_label = _period_label(g_val)
    t_label = _period_label(t_val)

    if data.config.base_period == "universal":
        pre_periods = np.where(data.config.time_periods < (g_val - data.config.anticipation))[0]
        pre_treatment_idx = pre_periods[-1] if len(pre_periods) > 0 else None
    else:
        pre_treatment_idx = time_idx

    is_post_treatment = g_val <= t_val

    if is_post_treatment and data.config.base_period != "universal":
        pre_periods = np.where(data.config.time_periods < (g_val - data.config.anticipation))[0]

        if len(pre_periods) == 0:
            warnings.warn(
                f"No pre-treatment periods for group first treated at {g_label}. Units from this group are dropped.",
                UserWarning,
            )
            return None

        pre_treatment_idx = pre_periods[-1]

    if (
        data.config.base_period == "universal"
        and pre_treatment_idx is not None
        and data.config.time_periods[pre_treatment_idx] == t_val
    ):
        return None

    cohort_index = get_did_cohort_index(group_idx, time_idx, time_factor, pre_treatment_idx, data)

    has_treated = np.any(cohort_index == 1)
    has_control = np.any(cohort_index == 0)

    if not (has_treated and has_control):
        return None

    if data.config.panel:
        # min(pre_treatment_idx, time_idx) is the earlier of the cell's two periods. Weights that vary over
        # time enter through that period, as the covariates do.
        earlier_idx = min(pre_treatment_idx, time_idx)
        cohort_data = {
            "D": cohort_index,
            "y1": data.outcomes_tensor[time_idx + time_factor],
            "y0": data.outcomes_tensor[pre_treatment_idx],
            "weights": data.weights_tensor[earlier_idx],
        }
        covariates = data.covariates_tensor[earlier_idx]
    else:
        periods = data.data[data.config.tname].to_numpy()
        cohort_data = {
            "D": cohort_index,
            "y": data.data[data.config.yname].to_numpy(),
            "post": (periods == t_val).astype(int),
            "period": periods,
            "weights": data.data[WEIGHTS_COLUMN].to_numpy(),
        }
        if _is_unbalanced_panel(data):
            cohort_data["unit_index"] = _unit_index(data) if unit_index is None else unit_index
        covariates = data.covariates_matrix

    if not callable(data.config.est_method):
        valid_obs = ~np.isnan(cohort_index)
        d_valid = cohort_index[valid_obs]

        if not data.config.panel:
            post = cohort_data["post"][valid_obs]
            G = d_valid == 1
            C = d_valid == 0
            skip = False
            if np.sum(G & (post == 1)) == 0:
                warnings.warn(f"No treated units in group {g_label} in time period {t_label}", UserWarning)
                skip = True
            if np.sum(G & (post == 0)) == 0:
                warnings.warn(f"No treated units in group {g_label} in pre-treatment period", UserWarning)
                skip = True
            if np.sum(C & (post == 1)) == 0:
                warnings.warn(f"No control units for group {g_label} in time period {t_label}", UserWarning)
                skip = True
            if np.sum(C & (post == 0)) == 0:
                warnings.warn(f"No control units for group {g_label} in pre-treatment period", UserWarning)
                skip = True
            if skip:
                return None

        cov_valid = to_numpy(covariates)[valid_obs] if covariates.ndim > 1 else np.ones((valid_obs.sum(), 1))
        weights_valid = to_numpy(cohort_data["weights"])[valid_obs]
        if _cell_fails_screen(d_valid == 1, cov_valid, weights_valid, data.config.est_method, g_label, t_label):
            return None

    try:
        return run_drdid(cohort_data, covariates, data)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        warnings.warn(f"The 2x2 comparison for group {g_label} in time period {t_label} failed. {e}", UserWarning)
        return None


def _cell_fails_screen(treated, covariates, weights, est_method, g_label, t_label):
    """Tell whether a cell fails the overlap or the regression screen.

    The overlap screen fits the propensity score by a logit with the sampling
    weights as frequency weights. It flags the cell when the fitted score of a
    unit with positive weight reaches 0.999. The regression screen flags the
    cell when the weighted Gram matrix of the comparison units' covariates is
    numerically singular. Units with zero weight leave both fits, as they
    leave the estimators' fits.

    Parameters
    ----------
    treated : ndarray
        Boolean indicator of the treated units among the cell's units.
    covariates : ndarray
        Covariates of the cell's units, the intercept included.
    weights : ndarray
        Sampling weights of the cell's units.
    est_method : EstimationMethod
        Built-in estimation method of the cell.
    g_label : str
        Treatment group of the cell as the warnings print it.
    t_label : str
        Time period of the cell as the warnings print it.

    Returns
    -------
    bool
        True when the cell fails a screen. A warning then names the cell.
    """
    # The estimators divide the cell's weights by their mean before they fit. Since the logit fit stops once the
    # deviance changes by less than a fixed amount and the deviance grows with the weights, the screen does the same.
    if weights.sum() > 0:
        weights = weights / weights.mean()

    fails = False

    if est_method in (EstimationMethod.DOUBLY_ROBUST, EstimationMethod.IPW):
        try:
            logit = sm.GLM(treated.astype(float), covariates, family=sm.families.Binomial(), freq_weights=weights)
            pscores = logit.fit(maxiter=100, disp=False).fittedvalues
            if np.max(pscores, initial=0.0, where=weights > 0) >= 0.999:
                fails = True
                warnings.warn(f"Overlap condition violated for {g_label} in time period {t_label}", UserWarning)
        except (np.linalg.LinAlgError, PerfectSeparationError):
            pass

    if est_method in (EstimationMethod.DOUBLY_ROBUST, EstimationMethod.REGRESSION):
        comparison_covariates = covariates[~treated]
        if comparison_covariates.shape[0] > 0:
            gram = comparison_covariates.T @ (weights[~treated][:, None] * comparison_covariates)
            try:
                if np.linalg.cond(gram) > 1.0 / np.finfo(float).eps:
                    fails = True
                    warnings.warn(
                        f"Not enough control units for group {g_label} in time period {t_label} "
                        "to run specified regression",
                        UserWarning,
                    )
            except np.linalg.LinAlgError:
                fails = True
                warnings.warn(f"Singular covariate matrix for group {g_label} in time period {t_label}", UserWarning)

    return fails


def _period_label(value):
    """Format a cohort or period without a trailing decimal zero.

    Preprocessing stores cohorts as floats so that infinity can mark the
    never-treated units. Messages print 2004.0 as 2004, as the result table
    does. A fractional period such as 2004.5 keeps its decimals.

    Parameters
    ----------
    value : float
        Cohort or time period.

    Returns
    -------
    str
        The value as messages print it.
    """
    return np.format_float_positional(float(value), trim="-")


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
        if data.config.control_group == ControlGroup.NOT_YET_TREATED:
            relevant_period = (
                data.config.time_periods[max(time_idx, pre_treatment_idx) + time_factor] + data.config.anticipation
            )
            future_cohorts = data.cohort_counts.filter(data.cohort_counts["cohort"] > relevant_period)
            min_control = future_cohorts["cohort"][0] if len(future_cohorts) > 0 else np.inf
        else:
            min_control = np.inf

        max_control = np.inf

        n_units = len(data.time_invariant_data) if data.config.allow_unbalanced_panel else data.config.id_count
        cohort_index = np.full(n_units, np.nan)

        cohort_values = data.cohort_counts["cohort"].to_numpy()
        if max_control not in cohort_values:
            max_control = cohort_values[-1]

        control_mask = (data.cohort_counts["cohort"] >= min_control) & (data.cohort_counts["cohort"] <= max_control)
        control_mask_np = control_mask.to_numpy()
        cohort_sizes = data.cohort_counts["cohort_size"].to_numpy()
        for idx in np.where(control_mask_np)[0]:
            start_control = int(cohort_sizes[:idx].sum()) if idx > 0 else 0
            end_control = int(cohort_sizes[: idx + 1].sum())
            cohort_index[start_control:end_control] = 0

        treated_mask = data.cohort_counts["cohort"] == data.config.treated_groups[group_idx]
        treated_mask_np = treated_mask.to_numpy()
        if treated_mask_np.any():
            treat_idx = int(np.argmax(treated_mask_np))
            cohort_sizes = data.cohort_counts["cohort_size"].to_numpy()
            start_treat = int(cohort_sizes[:treat_idx].sum()) if treat_idx > 0 else 0
            end_treat = int(cohort_sizes[: treat_idx + 1].sum())
            cohort_index[start_treat:end_treat] = 1

    else:
        n_units = len(data.data)
        cohort_index = np.full(n_units, np.nan)

        treated_flag = (data.data[data.config.gname] == data.config.treated_groups[group_idx]).to_numpy()

        if data.config.control_group == ControlGroup.NEVER_TREATED:
            control_flag = (data.data[data.config.gname] == np.inf).to_numpy()
        else:
            relevant_period = (
                data.config.time_periods[max(time_idx, pre_treatment_idx) + time_factor] + data.config.anticipation
            )
            control_flag = (
                (data.data[data.config.gname] == np.inf)
                | (
                    (data.data[data.config.gname] > relevant_period)
                    & (data.data[data.config.gname] != data.config.treated_groups[group_idx])
                )
            ).to_numpy()

        keep_periods = (
            data.data[data.config.tname]
            .is_in([data.config.time_periods[time_idx + time_factor], data.config.time_periods[pre_treatment_idx]])
            .to_numpy()
        )

        cohort_index[keep_periods & control_flag] = 0
        cohort_index[keep_periods & treated_flag] = 1

    return cohort_index


def _is_unbalanced_panel(data):
    """Tell whether the rows belong to the units of an unbalanced panel.

    Preprocessing hands an unbalanced panel to the repeated cross section estimators and keys its
    rows by unit. Each unit's influence function then adds up over its rows. With ``panel=False``
    every row is its own observation whatever ``allow_unbalanced_panel`` says.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    bool
        True for an unbalanced panel and False for a balanced panel or repeated cross sections.
    """
    return not data.config.panel and not data.config.true_repeated_cross_sections


def _unit_index(data):
    """Map each row of the data to its unit's row in the time-invariant data.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    ndarray
        Position of each row's unit in ``data.time_invariant_data``.
    """
    unit_ids = data.time_invariant_data[data.config.idname]
    return data.data[data.config.idname].replace_strict(unit_ids, np.arange(len(unit_ids))).to_numpy()


def _weights_vary_within_units(data):
    """Tell whether some unit carries different weights in different periods.

    Repeated cross sections have no units to follow. A spread of at most the
    square root of machine precision counts as no change.

    Parameters
    ----------
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    bool
        True when the normalized weights of some unit differ across its rows.
    """
    if data.config.weightsname is None or data.config.true_repeated_cross_sections:
        return False
    weights = pl.col(WEIGHTS_COLUMN)
    spread = data.data.group_by(data.config.idname).agg((weights.max() - weights.min()).alias("spread"))
    return bool((spread["spread"] > np.sqrt(np.finfo(float).eps)).any())


def run_drdid(
    cohort_data,
    covariates,
    data,
):
    """Run DR-DiD estimation for current group-time pair.

    The estimate must be a number with an influence function. When the
    treated or comparison units all have zero weight, or when the estimator
    returns a NaN ATT or no influence function, a ValueError says so. In
    repeated cross sections the weight check also runs within each of the
    cell's two periods. After such an error :func:`run_att_gt_estimation`
    skips the cell with a warning, as it does for every comparison that
    fails. A custom estimator follows the same rule.

    Parameters
    ----------
    cohort_data : dict
        Dictionary containing outcome and treatment data for the cohort. For
        repeated cross sections an optional ``period`` entry gives each row's
        time period to the weight check.
    covariates : ndarray
        Covariate matrix for the estimation.
    data : DIDData
        Preprocessed DiD data object.

    Returns
    -------
    dict
        - **att**: Estimated ATT of the cell
        - **inf_func**: Influence function of the estimate with one entry per unit
    """
    n = len(cohort_data["D"])
    est_method = data.config.est_method
    valid_obs = ~np.isnan(cohort_data["D"])

    if valid_obs.sum() == 0:
        return {"att": np.nan, "inf_func": np.full(n, np.nan)}

    if data.config.panel:
        y1 = cohort_data["y1"][valid_obs]
        y0 = cohort_data["y0"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]
        _check_cell_weights(d, weights)

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates[valid_obs] if len(covariates) > 1 else np.ones(valid_obs.sum())

        if callable(est_method):
            result = est_method(y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == EstimationMethod.IPW:
            result = std_ipw_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == EstimationMethod.REGRESSION:
            result = reg_did_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:
            result = drdid_panel(
                y1=y1, y0=y0, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        _check_estimate(result)

        influence_func = np.zeros(n)
        influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    else:
        y = cohort_data["y"][valid_obs]
        post = cohort_data["post"][valid_obs]
        d = cohort_data["D"][valid_obs]
        weights = cohort_data["weights"][valid_obs]
        periods = cohort_data.get("period")
        _check_cell_weights(d, weights, None if periods is None else periods[valid_obs])

        if covariates.ndim > 1:
            cov_valid = covariates[valid_obs]
        else:
            cov_valid = covariates if len(covariates) == n else covariates[valid_obs]

        if callable(est_method):
            result = est_method(y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, influence_func=True)
        elif est_method == EstimationMethod.IPW:
            result = std_ipw_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        elif est_method == EstimationMethod.REGRESSION:
            result = reg_did_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        else:
            result = drdid_rc(
                y=y, post=post, d=d, covariates=cov_valid, i_weights=weights, boot=False, influence_func=True
            )
        _check_estimate(result)

        if _is_unbalanced_panel(data) and "unit_index" in cohort_data:
            inf_func_long = np.zeros(n)
            inf_func_long[valid_obs] = (data.config.id_count / valid_obs.sum()) * result.att_inf_func
            # Since aggte pairs these rows with the groups and weights in time_invariant_data, each unit's
            # observations add up in that unit's row there.
            influence_func = np.bincount(
                cohort_data["unit_index"], weights=inf_func_long, minlength=data.config.id_count
            )
        else:
            influence_func = np.zeros(n)
            influence_func[valid_obs] = (n / valid_obs.sum()) * result.att_inf_func

    return {"att": result.att, "inf_func": influence_func}


def _check_cell_weights(d, weights, periods=None):
    """Raise a ValueError when the treated or comparison units all have zero weight.

    Repeated cross sections draw new units in each period. For them the check
    also runs within each of the cell's two periods.

    Parameters
    ----------
    d : ndarray
        Treatment indicator of the cell's units.
    weights : ndarray
        Sampling weights of the cell's units.
    periods : ndarray, optional
        Time period of each row in repeated cross sections.
    """
    weights = to_numpy(weights)
    sides = [(d == 1, "treated units"), (d == 0, "comparison units")]
    for units, name in sides:
        if not np.sum(weights[units]) > 0:
            raise ValueError(f"The {name} all have zero weight.")
    if periods is None:
        return
    for period in np.unique(periods):
        for units, name in sides:
            in_period = units & (periods == period)
            # Since an empty side has no weights to check, the presence checks and the estimator handle it.
            if np.any(in_period) and not np.sum(weights[in_period]) > 0:
                raise ValueError(f"The {name} in time period {_period_label(period)} all have zero weight.")


def _check_estimate(result):
    """Raise a ValueError when an estimator returns no estimate to report.

    Parameters
    ----------
    result : NamedTuple
        Result of a 2x2 estimator that holds ``att`` and ``att_inf_func``.
    """
    if result.att is None or np.isnan(result.att):
        raise ValueError("The estimator returned NaN for the ATT.")
    if result.att_inf_func is None:
        raise ValueError("The estimator returned no influence function.")


def _process_gt_cell_did(group_idx, time_idx, data, unit_index=None):
    """Process a single (group, time) cell for DiD estimation.

    Returns
    -------
    tuple of (ATTgtResult or None, ndarray or None)
    """
    n_units = data.config.id_count
    time_factor = 1 if data.config.base_period != "universal" else 0

    estimation_result = run_att_gt_estimation(group_idx, time_idx, data, unit_index)

    is_post_treatment = int(data.config.treated_groups[group_idx] <= data.config.time_periods[time_idx + time_factor])

    if estimation_result is None or estimation_result["att"] is None:
        if data.config.base_period == "universal":
            pre_periods = np.where(
                data.config.time_periods < (data.config.treated_groups[group_idx] - data.config.anticipation)
            )[0]
            is_reference = (
                len(pre_periods) > 0
                and data.config.time_periods[pre_periods[-1]] == data.config.time_periods[time_idx + time_factor]
            )
            att_val = 0.0 if is_reference else np.nan
            inf_func = np.zeros(n_units) if is_reference else np.full(n_units, np.nan)
            return (
                ATTgtResult(
                    att=att_val,
                    group=data.config.treated_groups[group_idx],
                    year=data.config.time_periods[time_idx + time_factor],
                    post=is_post_treatment,
                ),
                inf_func,
            )
        return (None, None)

    att_estimate = estimation_result["att"]
    influence_func = estimation_result["inf_func"]

    if np.isnan(att_estimate):
        influence_func = np.full(n_units, np.nan)

    return (
        ATTgtResult(
            att=att_estimate,
            group=data.config.treated_groups[group_idx],
            year=data.config.time_periods[time_idx + time_factor],
            post=is_post_treatment,
        ),
        influence_func,
    )
