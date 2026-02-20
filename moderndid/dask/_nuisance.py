"""Distributed nuisance estimation for DDD estimators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from moderndid.distributed._nuisance import (
    DistOutcomeRegResult,
    DistPScoreResult,
    _build_partitions_for_subset,
    _compute_outcome_regression_null,
    _compute_pscore_null,
)

from ._regression import distributed_logistic_irls, distributed_wls
from ._utils import get_default_partitions


def compute_all_nuisances_distributed(
    client,
    y1,
    y0,
    subgroup,
    covariates,
    weights,
    est_method="dr",
    trim_level=0.995,
    n_partitions=None,
):
    r"""Compute all nuisance parameters using distributed regression.

    Estimates propensity scores and outcome regressions for the three
    subgroup comparisons :math:`(4, 3)`, :math:`(4, 2)`, and
    :math:`(4, 1)` required by the DDD estimator. Propensity scores are
    estimated via distributed logistic IRLS and outcome regressions via
    distributed WLS, both using tree-reduced Gram matrices.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    y1 : ndarray
        Post-treatment outcomes.
    y0 : ndarray
        Pre-treatment outcomes.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates including intercept, shape :math:`(n, k)`.
    weights : ndarray
        Observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method. ``"dr"`` estimates both propensity scores and
        outcome regressions. ``"reg"`` skips propensity scores.
        ``"ipw"`` skips outcome regressions.
    trim_level : float, default 0.995
        Trimming level for propensity scores.
    n_partitions : int or None
        Number of partitions for distributed computation. If None, uses
        the number of threads across all workers.

    Returns
    -------
    pscores : list of DistPScoreResult
        Propensity score results for comparisons [3, 2, 1].
    or_results : list of DistOutcomeRegResult
        Outcome regression results for comparisons [3, 2, 1].
    """
    if n_partitions is None:
        n_partitions = get_default_partitions(client)
    with ThreadPoolExecutor(max_workers=6) as pool:
        ps_futures = {}
        or_futures = {}

        for comp_subgroup in [3, 2, 1]:
            if est_method == "reg":
                ps_futures[comp_subgroup] = pool.submit(_compute_pscore_null, subgroup, comp_subgroup)
            else:
                ps_futures[comp_subgroup] = pool.submit(
                    _compute_pscore_distributed,
                    client,
                    subgroup,
                    covariates,
                    weights,
                    comp_subgroup,
                    trim_level,
                    n_partitions,
                )

            if est_method == "ipw":
                or_futures[comp_subgroup] = pool.submit(
                    _compute_outcome_regression_null,
                    y1,
                    y0,
                    subgroup,
                    comp_subgroup,
                )
            else:
                or_futures[comp_subgroup] = pool.submit(
                    _compute_outcome_regression_distributed,
                    client,
                    y1,
                    y0,
                    subgroup,
                    covariates,
                    weights,
                    comp_subgroup,
                    n_partitions,
                )

        pscores = [ps_futures[sg].result() for sg in [3, 2, 1]]
        or_results = [or_futures[sg].result() for sg in [3, 2, 1]]

    return pscores, or_results


def _compute_pscore_distributed(client, subgroup, covariates, weights, comp_subgroup, trim_level, n_partitions):
    """Compute propensity scores using distributed logistic IRLS."""
    mask = (subgroup == 4) | (subgroup == comp_subgroup)
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]
    sub_subgroup = subgroup[mask]

    pa4 = (sub_subgroup == 4).astype(np.float64)

    partitions = _build_partitions_for_subset(sub_covariates, sub_weights, pa4, n_partitions)

    try:
        beta = distributed_logistic_irls(client, partitions)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError(
            f"Failed to estimate propensity scores for subgroup {comp_subgroup} due to singular matrix."
        ) from e

    if np.any(np.isnan(beta)):
        raise ValueError(f"Propensity score model has NA coefficients for comparison with subgroup {comp_subgroup}.")

    ps_fit = 1.0 / (1.0 + np.exp(-(sub_covariates @ beta)))
    ps_fit = np.clip(ps_fit, 1e-10, 1 - 1e-10)
    ps_fit = np.minimum(ps_fit, 1 - 1e-6)

    keep_ps = np.ones(len(pa4), dtype=bool)
    keep_ps[pa4 == 0] = ps_fit[pa4 == 0] < trim_level

    n_sub = len(sub_weights)
    W = sub_weights * ps_fit * (1 - ps_fit)
    info_matrix = sub_covariates.T @ (W[:, None] * sub_covariates)
    hessian_matrix = np.linalg.inv(info_matrix) * n_sub

    return DistPScoreResult(propensity_scores=ps_fit, hessian_matrix=hessian_matrix, keep_ps=keep_ps)


def _compute_outcome_regression_distributed(client, y1, y0, subgroup, covariates, weights, comp_subgroup, n_partitions):
    """Compute outcome regression using distributed WLS."""
    mask = (subgroup == 4) | (subgroup == comp_subgroup)
    control_mask = subgroup == comp_subgroup

    sub_y1 = y1[mask]
    sub_y0 = y0[mask]
    sub_covariates = covariates[mask]
    delta_y = sub_y1 - sub_y0

    control_delta_y = (y1 - y0)[control_mask]
    control_covariates = covariates[control_mask]
    control_weights = weights[control_mask]

    partitions = _build_partitions_for_subset(control_covariates, control_weights, control_delta_y, n_partitions)

    try:
        reg_coeff = distributed_wls(client, partitions)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError(f"Failed to estimate outcome regression for subgroup {comp_subgroup}.") from e

    if np.any(np.isnan(reg_coeff)):
        raise ValueError(f"Outcome regression model has NA coefficients for comparison with subgroup {comp_subgroup}.")

    or_delta = sub_covariates @ reg_coeff

    return DistOutcomeRegResult(delta_y=delta_y, or_delta=or_delta, reg_coeff=reg_coeff)
