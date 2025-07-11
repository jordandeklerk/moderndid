"""Internal computational functions for Sun and Abraham estimator."""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from numba import njit


def estimate_sunab_model(outcome, covariates, interaction_matrix, cohort, period, period_values, weights, att, no_agg):
    """Estimate Sun & Abraham model and compute aggregated effects.

    Performs weighted least squares estimation and aggregates
    cohort-specific effects if requested.

    Parameters
    ----------
    outcome : ndarray
        Outcome variable.
    covariates : ndarray
        Covariate matrix.
    interaction_matrix : ndarray
        Cohort × period interaction matrix.
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


@njit
def find_never_always_treated(cohort_int, period, n_cohorts):
    """Find never-treated cohorts and always-treated units.

    Identifies control and problematic treatment groups based on
    the pattern of relative time periods for each cohort.

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


def create_period_interactions(period, period_values, is_ref_cohort, is_always_treated, n_total, valid_obs_mask):
    """Create period-specific interactions for aggregation.

    Builds the interaction matrix with proper handling of reference
    groups and always-treated units.

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
