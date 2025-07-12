"""Sun and Abraham interaction-weighted estimator for staggered DiD."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import statsmodels.api as sm
from numba import njit

from .utils import bin_factor, create_interactions


class SunAbrahamResult(NamedTuple):
    """Result from Sun and Abraham estimator.

    Contains event-study estimates, standard errors, and diagnostic information
    from the interaction-weighted estimator for staggered treatment designs.

    Attributes
    ----------
    att_by_event : ndarray
        Average treatment effect for each event time.
    se_by_event : ndarray
        Standard errors for each event time.
    event_times : ndarray
        Event times (relative periods).
    vcov : ndarray
        Variance-covariance matrix of event-time estimates.
    cohort_shares : ndarray
        Share of each cohort in the sample.
    influence_func : ndarray, optional
        Influence function for event-time estimates.
    att : float, optional
        Overall average treatment effect on the treated.
    se_att : float, optional
        Standard error of overall ATT.
    n_cohorts : int
        Number of treatment cohorts.
    n_periods : int
        Number of time periods.
    estimation_params : dict
        Additional estimation parameters and diagnostics.
    """

    att_by_event: np.ndarray
    se_by_event: np.ndarray
    event_times: np.ndarray
    vcov: np.ndarray
    cohort_shares: np.ndarray
    influence_func: np.ndarray | None
    att: float | None
    se_att: float | None
    n_cohorts: int
    n_periods: int
    estimation_params: dict

    @property
    def aggregation_type(self) -> str:
        """Return aggregation type for EventStudyProtocol."""
        return "dynamic"


def sunab(
    cohort,
    period,
    outcome=None,
    covariates=None,
    weights=None,
    ref_cohort=None,
    ref_period=-1,
    bin=None,
    bin_rel=None,
    bin_c=None,
    bin_p=None,
    att=False,
    no_agg=False,
    return_interactions=False,
):
    """Compute Sun and Abraham interaction-weighted DiD estimator.

    Implements the interaction-weighted estimator for staggered adoption
    designs that addresses negative weighting issues in two-way fixed effects
    models. Creates cohort and relative period interactions and estimates
    treatment effects that are robust to heterogeneous treatment effects.

    Parameters
    ----------
    cohort : array-like
        Treatment cohort indicating when units received treatment. For never-treated
        units, use a value outside the range of periods.
    period : array-like
        Time period. Can be either:

        - Calendar periods: Relative periods will be computed as period - cohort
        - Relative periods: Must contain negative, zero, and positive values

    outcome : array-like, optional
        Outcome variable. If provided with covariates, estimates are computed directly.
    covariates : array-like, optional
        Covariate matrix for regression. Should include fixed effects if desired.
    weights : array-like, optional
        Observation weights. Normalized to sum to number of observations.
    ref_cohort : array-like, optional
        Additional reference cohorts to exclude from interactions.
    ref_period : int or array-like, default=-1
        Reference period(s). Default is -1 (one period before treatment).
        Must specify at least one reference period. Special values '.F' and '.L'
        can be used to reference first and last periods.
    bin : str, list, or dict, optional
        Binning specification for both cohorts and periods. Options:

        - "bin::n": Group every n consecutive values
        - List of values to group together
        - Dict mapping new names to lists of old values
        - Regex patterns (prefix with "@")

    bin_rel : str, list, or dict, optional
        Binning for relative periods only (after computation).
    bin_c : str, list, or dict, optional
        Binning for cohorts only.
    bin_p : str, list, or dict, optional
        Binning for periods only.
    att : bool, default=False
        If True, compute overall ATT instead of event-study coefficients.
    no_agg : bool, default=False
        If True, return all cohort-period interactions without aggregation.
    return_interactions : bool, default=False
        If True, return interaction matrix instead of estimates.

    Returns
    -------
    SunAbrahamResult or ndarray
        If outcome and covariates provided: SunAbrahamResult with estimates
        If return_interactions=True: Interaction matrix
        Otherwise: SunAbrahamResult with placeholder values

    Notes
    -----
    The estimator creates interactions between treatment cohorts and relative
    time periods, ensuring that comparisons are only made between units
    treated at the same time. Never-treated units serve as the control group,
    while always-treated units are excluded from the analysis.

    The aggregation uses cohort shares as weights, computed from the
    interaction matrix to ensure proper weighting of cohort-specific effects.

    References
    ----------

    .. [1] Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects
           in event studies with heterogeneous treatment effects. Journal of
           Econometrics, 225(2), 175-199.
    """
    cohort = np.asarray(cohort)
    period = np.asarray(period)

    if cohort.shape[0] != period.shape[0]:
        raise ValueError("cohort and period must have the same length")

    n_total = len(cohort)

    if bin is not None and (bin_c is not None or bin_p is not None):
        raise ValueError("Cannot use 'bin' with 'bin_c' or 'bin_p'. Use only the latter.")

    valid_obs = ~(np.isnan(cohort) | np.isnan(period))
    if not np.all(valid_obs):
        cohort = cohort[valid_obs]
        period = period[valid_obs]
        if outcome is not None:
            outcome = outcome[valid_obs]
        if covariates is not None:
            covariates = covariates[valid_obs]
        if weights is not None:
            weights = weights[valid_obs]

    n = len(cohort)

    if bin_c is not None:
        cohort = bin_factor(bin_c, cohort, "cohort")

    if bin_p is not None:
        period = bin_factor(bin_p, period, "period")

    period_unique = np.unique(period)
    cohort_unique = np.unique(cohort)

    is_relative = (
        np.issubdtype(period.dtype, np.number)
        and 0 in period_unique
        and np.min(period_unique) < 0 < np.max(period_unique)
    )

    if is_relative:
        if bin is not None:
            raise ValueError(
                "Cannot use 'bin' when 'period' contains relative periods. Use 'bin_rel' to bin relative periods."
            )
        # For relative periods, cohorts are just identifiers
        # No need to compute relative periods or identify never-treated
    else:
        if bin is not None:
            period = bin_factor(bin, period, "period")
            cohort = bin_factor(bin, cohort, "cohort")
            period_unique = np.unique(period)
            cohort_unique = np.unique(cohort)

        # Create relative periods from calendar periods
        # Never-treated are cohorts not in period values
        never_treated_cohorts = np.setdiff1d(cohort_unique, period_unique)
        is_never_treated = np.isin(cohort, never_treated_cohorts)

        if np.all(is_never_treated):
            raise ValueError("Problem creating relative time periods. No cohort values found in period values.")

        rel_period = np.full(n, -999, dtype=np.float64)

        if np.issubdtype(period.dtype, np.number) and np.issubdtype(cohort.dtype, np.number):
            treated_mask = ~is_never_treated
            rel_period[treated_mask] = period[treated_mask] - cohort[treated_mask]
        else:
            all_values = np.unique(np.concatenate([period_unique, cohort_unique]))
            value_map = {v: i for i, v in enumerate(np.sort(all_values))}

            period_numeric = np.array([value_map.get(p, -1) for p in period])
            cohort_numeric = np.array([value_map.get(c, -1) for c in cohort])

            treated_mask = ~is_never_treated & (period_numeric >= 0) & (cohort_numeric >= 0)
            rel_period[treated_mask] = period_numeric[treated_mask] - cohort_numeric[treated_mask]

        rel_period[is_never_treated] = -1

        period = rel_period

    if bin_rel is not None:
        period = bin_factor(bin_rel, period, "relative period")

    period_min = np.min(period[np.isfinite(period)])
    period_max = np.max(period[np.isfinite(period)])

    if np.isscalar(ref_period):
        ref_period = [ref_period]

    ref_period_list = []
    for rp in ref_period:
        if isinstance(rp, str):
            if rp == ".F":
                ref_period_list.append(period_min)
            elif rp == ".L":
                ref_period_list.append(period_max)
            else:
                raise ValueError(f"Unknown special reference period: {rp}")
        else:
            ref_period_list.append(rp)

    ref_period = np.array(ref_period_list)

    # Find never/always treated based on relative periods
    cohort_unique = np.unique(cohort)
    cohort_int = np.searchsorted(cohort_unique, cohort)
    n_cohorts = len(cohort_unique)

    sort_idx = np.argsort(cohort_int)
    never_treated_cohorts, always_treated_idx = _find_never_always_treated(
        cohort_int[sort_idx], period[sort_idx], n_cohorts
    )

    is_ref_cohort = np.isin(cohort_int, never_treated_cohorts)
    if ref_cohort is not None:
        is_ref_cohort |= np.isin(cohort, ref_cohort)

    is_ref_period = np.isin(period, ref_period)
    is_always_treated = np.zeros(n, dtype=bool)
    is_always_treated[sort_idx[always_treated_idx]] = True
    keep_mask = ~(is_ref_cohort | is_ref_period) & ~is_always_treated

    if not np.any(keep_mask):
        warnings.warn("No observations remain after removing references", UserWarning)
        return _empty_result()

    cohort_active = cohort[keep_mask]
    period_active = period[keep_mask]

    cohort_values = np.unique(cohort_active)
    period_values = np.unique(period_active)

    if no_agg:
        inter_result = create_interactions(period_active, cohort_active, name="period", return_dict=True)

        interaction_matrix = np.zeros((n_total, inter_result["matrix"].shape[1]))

        if np.all(valid_obs):
            interaction_matrix[keep_mask] = inter_result["matrix"]
        else:
            original_idx = np.where(valid_obs)[0]
            interaction_matrix[original_idx[keep_mask]] = inter_result["matrix"]

        if np.any(is_always_treated):
            if not np.all(valid_obs):
                original_idx = np.where(valid_obs)[0]
                interaction_matrix[original_idx[is_always_treated]] = np.nan
            else:
                interaction_matrix[is_always_treated] = np.nan

    else:
        # Create period-specific interactions for aggregation
        interaction_matrix = create_period_interactions(
            period,
            period_values,
            is_ref_cohort,
            is_always_treated,
            n_total,
            valid_obs if not np.all(valid_obs) else None,
        )

    if return_interactions:
        return interaction_matrix

    if outcome is None or covariates is None:
        return _create_interaction_result(period_values, cohort_values, interaction_matrix, cohort, is_ref_cohort, att)

    result_dict = estimate_sunab_model(
        outcome, covariates, interaction_matrix, cohort, period, period_values, weights, att, no_agg
    )

    if result_dict is None:
        return _empty_result()

    if no_agg:
        return _create_disaggregated_result(result_dict["coefs"], result_dict["vcov"], cohort, period_values)

    return SunAbrahamResult(
        att_by_event=result_dict["att_by_event"],
        se_by_event=result_dict["se_by_event"],
        event_times=period_values,
        vcov=result_dict["vcov_event"],
        cohort_shares=result_dict["cohort_shares"],
        influence_func=None,
        att=result_dict["att"],
        se_att=result_dict["se_att"],
        n_cohorts=len(np.unique(cohort)),
        n_periods=len(period_values),
        estimation_params={
            "aggregated": result_dict["aggregated"],
            "n_obs": result_dict["n_obs"],
            "n_treated_cohorts": result_dict["n_treated_cohorts"],
        },
    )


def estimate_sunab_model(outcome, covariates, interaction_matrix, cohort, period, period_values, weights, att, no_agg):
    """Estimate Sun & Abraham model and compute aggregated effects.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable.
    covariates : ndarray
        Covariate matrix.
    interaction_matrix : ndarray
        Cohort by period interaction matrix.
    cohort : ndarray
        Treatment cohort for each observation.
    period : ndarray
        Relative period for each observation.
    period_values : ndarray
        Unique relative period values.
    weights : ndarray, optional
        Observation weights.
    att : bool
        Whether to compute overall ATT.
    no_agg : bool
        If True, return disaggregated coefficients.

    Returns
    -------
    dict
        Dictionary with estimation results.
    """
    X = np.column_stack([covariates, interaction_matrix])
    y = outcome

    if weights is None:
        weights = np.ones(len(y))
    weights = weights / np.mean(weights)

    valid_rows = ~np.any(np.isnan(interaction_matrix), axis=1)
    if not np.all(valid_rows):
        X = X[valid_rows]
        y = y[valid_rows]
        weights = weights[valid_rows]
        cohort = cohort[valid_rows]
        period = period[valid_rows]

    if X.shape[0] == 0 or X.shape[0] < X.shape[1]:
        warnings.warn("Insufficient observations for estimation", UserWarning)
        return None

    try:
        wls_model = sm.WLS(y, X, weights=weights)
        results = wls_model.fit()

        beta = results.params
        vcov_full = results.cov_params()

        if np.linalg.cond(results.normalized_cov_params) > 1e10:
            warnings.warn("Design matrix is near-singular", UserWarning)

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(f"Failed to fit weighted least squares model: {e}", UserWarning)
        return None

    n_covariates = covariates.shape[1]
    interaction_coefs = beta[n_covariates:]
    interaction_vcov = vcov_full[n_covariates:, n_covariates:]

    if no_agg:
        return {
            "coefs": interaction_coefs,
            "vcov": interaction_vcov,
            "se": np.sqrt(np.diag(interaction_vcov)),
            "aggregated": False,
        }

    agg_results = aggregate_to_event_study(
        interaction_coefs,
        interaction_vcov,
        interaction_matrix[valid_rows] if not np.all(valid_rows) else interaction_matrix,
        cohort,
        period_values,
        weights,
        att,
    )

    return {
        **agg_results,
        "aggregated": True,
        "n_obs": len(cohort),
        "n_treated_cohorts": len(np.unique(cohort[~np.isinf(cohort)])),
    }


def sunab_att(cohort, period, **kwargs):
    """Estimate overall average treatment effect on the treated.

    Convenience wrapper for computing the overall ATT instead of
    event-study coefficients. Aggregates post-treatment effects.

    Parameters
    ----------
    cohort : array-like
        Treatment cohort.
    period : array-like
        Time period.
    **kwargs : dict
        Additional arguments passed to sunab().

    Returns
    -------
    SunAbrahamResult
        Result with overall ATT estimate.
    """
    return sunab(cohort, period, att=True, **kwargs)


def aggregate_sunab(result, cohort_coefs, cohort_vcov):
    """Aggregate cohort-specific coefficients to event-study estimates.

    Computes weighted averages of cohort-specific treatment effects
    using cohort shares as weights to obtain proper event-study coefficients.

    Parameters
    ----------
    result : SunAbrahamResult
        Result object containing event times and cohort information.
    cohort_coefs : dict
        Dictionary mapping (period, cohort) tuples to coefficient estimates.
    cohort_vcov : ndarray
        Variance-covariance matrix for cohort-specific coefficients.

    Returns
    -------
    SunAbrahamResult
        Updated result with aggregated event-study coefficients.
    """
    event_times = result.event_times
    n_events = len(event_times)

    att_by_event = np.zeros(n_events)
    vcov_aggregated = np.zeros((n_events, n_events))

    coef_names = list(cohort_coefs.keys())

    for e_idx, event_time in enumerate(event_times):
        event_coef_idx = []
        event_shares = []

        for idx, (period, _) in enumerate(coef_names):
            if period == event_time:
                event_coef_idx.append(idx)
                event_shares.append(1.0)

        if not event_coef_idx:
            continue

        event_shares = np.array(event_shares)
        event_shares = event_shares / event_shares.sum()

        event_coefs = np.array([cohort_coefs[coef_names[idx]] for idx in event_coef_idx])
        att_by_event[e_idx] = np.sum(event_shares * event_coefs)

        shares_matrix = np.outer(event_shares, event_shares)

        vcov_subset = cohort_vcov[np.ix_(event_coef_idx, event_coef_idx)]

        vcov_aggregated[e_idx, e_idx] = np.sum(shares_matrix * vcov_subset)

    return result._replace(
        att_by_event=att_by_event, se_by_event=np.sqrt(np.diag(vcov_aggregated)), vcov=vcov_aggregated
    )


def create_period_interactions(period, period_values, is_ref_cohort, is_always_treated, n_total, valid_obs_mask):
    """Create period-specific interactions for aggregation.

    Parameters
    ----------
    period : ndarray
        Relative period for each observation.
    period_values : ndarray
        Unique relative period values.
    is_ref_cohort : ndarray
        Boolean mask for reference cohort observations.
    is_always_treated : ndarray
        Boolean mask for always-treated observations.
    n_total : int
        Total number of observations.
    valid_obs_mask : ndarray, optional
        Mask indicating valid observations.

    Returns
    -------
    ndarray
        Interaction matrix with NaN for always-treated units.
    """
    n_periods = len(period_values)
    interaction_matrix = np.zeros((n_total, n_periods))

    for p_idx, p_val in enumerate(period_values):
        period_mask = (period == p_val) & ~is_ref_cohort & ~is_always_treated

        if valid_obs_mask is not None:
            original_idx = np.where(valid_obs_mask)[0]
            interaction_matrix[original_idx[period_mask], p_idx] = 1
        else:
            interaction_matrix[period_mask, p_idx] = 1

    if np.any(is_always_treated):
        if valid_obs_mask is not None:
            original_idx = np.where(valid_obs_mask)[0]
            interaction_matrix[original_idx[is_always_treated]] = np.nan
        else:
            interaction_matrix[is_always_treated] = np.nan

    return interaction_matrix


def aggregate_to_event_study(coefs, vcov, interactions, cohort, period_values, weights, compute_att):
    """Aggregate cohort-specific effects to event-study coefficients.

    Uses cohort shares computed from the interaction matrix to
    properly weight cohort-specific effects into event-study estimates.

    Parameters
    ----------
    coefs : ndarray
        Cohort-specific coefficient estimates.
    vcov : ndarray
        Variance-covariance matrix of coefficients.
    interactions : ndarray
        Interaction matrix from estimation.
    cohort : ndarray
        Treatment cohort for each observation.
    period_values : ndarray
        Unique relative period values.
    weights : ndarray
        Observation weights.
    compute_att : bool
        Whether to compute overall ATT.

    Returns
    -------
    dict
        Dictionary with aggregated results including:

        - att_by_event: Event-study coefficients
        - se_by_event: Standard errors
        - vcov_event: Variance-covariance matrix
        - att: Overall ATT (if requested)
        - se_att: Standard error of ATT (if requested)
        - cohort_shares: Share of each cohort
    """
    n_periods = len(period_values)
    att_by_event = np.zeros(n_periods)
    vcov_event = np.zeros((n_periods, n_periods))

    if interactions.shape[1] == n_periods:
        for p_idx in range(n_periods):
            col_mask = interactions[:, p_idx] != 0
            if not np.any(col_mask):
                continue

            share = np.sum(weights[col_mask] * interactions[col_mask, p_idx])
            share = share / np.sum(weights[col_mask])

            att_by_event[p_idx] = coefs[p_idx] * share
            vcov_event[p_idx, p_idx] = vcov[p_idx, p_idx] * (share**2)
    else:
        n_cols = interactions.shape[1]
        col_to_period = np.zeros(n_cols, dtype=int)
        col_to_cohort = np.zeros(n_cols, dtype=int)

        for col_idx in range(n_cols):
            col_to_cohort[col_idx] = col_idx // n_periods
            col_to_period[col_idx] = col_idx % n_periods

        for p_idx, _ in enumerate(period_values):
            period_cols = np.where(col_to_period == p_idx)[0]

            if len(period_cols) == 0:
                continue

            cohort_shares = np.zeros(len(period_cols))

            for i, col in enumerate(period_cols):
                col_mask = interactions[:, col] != 0
                if np.any(col_mask):
                    cohort_shares[i] = np.sum(weights[col_mask] * interactions[col_mask, col])

            if cohort_shares.sum() > 0:
                cohort_shares = cohort_shares / cohort_shares.sum()
            else:
                cohort_shares = np.ones(len(period_cols)) / len(period_cols)

            att_by_event[p_idx] = np.sum(cohort_shares * coefs[period_cols])

            for i, col_i in enumerate(period_cols):
                for j, col_j in enumerate(period_cols):
                    vcov_event[p_idx, p_idx] += cohort_shares[i] * cohort_shares[j] * vcov[col_i, col_j]

    se_by_event = np.sqrt(np.diag(vcov_event))

    if compute_att:
        post_mask = period_values >= 0
        if np.any(post_mask):
            post_weights = np.ones(np.sum(post_mask)) / np.sum(post_mask)
            att = np.sum(att_by_event[post_mask] * post_weights)

            var_att = post_weights @ vcov_event[post_mask][:, post_mask] @ post_weights
            se_att = np.sqrt(var_att)
        else:
            att = np.nan
            se_att = np.nan
    else:
        att = None
        se_att = None

    cohort_unique = np.unique(cohort)
    cohort_counts = np.array([np.sum(weights[cohort == c]) for c in cohort_unique])
    cohort_shares = cohort_counts / cohort_counts.sum()

    return {
        "att_by_event": att_by_event,
        "se_by_event": se_by_event,
        "vcov_event": vcov_event,
        "att": att,
        "se_att": se_att,
        "cohort_shares": cohort_shares,
    }


def _create_interaction_result(period_values, cohort_values, interaction_matrix, cohort, is_ref_cohort, att):
    """Create result object with interaction information.

    Parameters
    ----------
    period_values : ndarray
        Unique relative period values.
    cohort_values : ndarray
        Unique cohort values.
    interaction_matrix : ndarray
        Computed interaction matrix.
    cohort : ndarray
        Original cohort array.
    is_ref_cohort : ndarray
        Reference cohort indicator.
    att : bool
        Whether ATT was requested.

    Returns
    -------
    SunAbrahamResult
        Result with interaction information.
    """
    cohort_counts = np.bincount(np.searchsorted(np.unique(cohort[~is_ref_cohort]), cohort[~is_ref_cohort]))
    cohort_shares = cohort_counts / cohort_counts.sum()

    return SunAbrahamResult(
        att_by_event=np.zeros(len(period_values)),
        se_by_event=np.zeros(len(period_values)),
        event_times=np.sort(period_values),
        vcov=np.eye(len(period_values)),
        cohort_shares=cohort_shares,
        influence_func=None,
        att=0.0 if att else None,
        se_att=0.0 if att else None,
        n_cohorts=len(cohort_values),
        n_periods=len(period_values),
        estimation_params={"interaction_matrix_shape": interaction_matrix.shape, "status": "interactions_only"},
    )


def _create_disaggregated_result(coefs, vcov, cohort, period_values):
    """Create result with disaggregated cohort-period coefficients."""
    se = np.sqrt(np.diag(vcov))

    return SunAbrahamResult(
        att_by_event=coefs,
        se_by_event=se,
        event_times=period_values,
        vcov=vcov,
        cohort_shares=np.ones(len(coefs)) / len(coefs),
        influence_func=None,
        att=None,
        se_att=None,
        n_cohorts=len(np.unique(cohort)),
        n_periods=len(period_values),
        estimation_params={"aggregated": False},
    )


@njit
def _find_never_always_treated(cohort_int, period, n_cohorts):
    """Find never-treated cohorts and always-treated units.

    Parameters
    ----------
    cohort_int : ndarray
        Integer cohort identifiers (sorted).
    period : ndarray
        Relative period for each observation.
    n_cohorts : int
        Number of unique cohorts.

    Returns
    -------
    never_treated_cohorts : ndarray
        Cohort indices that are never treated.
    always_treated_idx : ndarray
        Observation indices that are always treated.
    """
    cohort_min = np.full(n_cohorts, np.inf)
    cohort_max = np.full(n_cohorts, -np.inf)

    n = len(cohort_int)
    i = 0

    while i < n:
        current_cohort = cohort_int[i]

        while i < n and cohort_int[i] == current_cohort:
            cohort_min[current_cohort] = min(cohort_min[current_cohort], period[i])
            cohort_max[current_cohort] = max(cohort_max[current_cohort], period[i])
            i += 1

    # Never-treated: only have negative relative periods
    never_treated = []
    for c in range(n_cohorts):
        if cohort_max[c] < 0:
            never_treated.append(c)

    # Always-treated: only have non-negative relative periods
    always_treated_idx = []
    i = 0
    while i < n:
        current_cohort = cohort_int[i]
        if cohort_min[current_cohort] >= 0:
            while i < n and cohort_int[i] == current_cohort:
                always_treated_idx.append(i)
                i += 1
        else:
            while i < n and cohort_int[i] == current_cohort:
                i += 1

    return np.array(never_treated, dtype=np.int32), np.array(always_treated_idx, dtype=np.int32)


def _empty_result():
    """Return empty result when estimation is not possible.

    Returns
    -------
    SunAbrahamResult
        Result with NaN values and empty arrays.
    """
    return SunAbrahamResult(
        att_by_event=np.array([]),
        se_by_event=np.array([]),
        event_times=np.array([]),
        vcov=np.empty((0, 0)),
        cohort_shares=np.array([]),
        influence_func=None,
        att=np.nan,
        se_att=np.nan,
        n_cohorts=0,
        n_periods=0,
        estimation_params={"status": "no_valid_observations"},
    )
