"""Shared 2-period panel DDD estimator logic for distributed backends."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult

from ._inf_func import compute_did_distributed
from ._validate import _validate_inputs


def ddd_panel_core(
    nuisance_fn,
    variance_fn,
    bootstrap_fn,
    y1,
    y0,
    subgroup,
    covariates,
    i_weights=None,
    est_method="dr",
    boot=False,
    biters=1000,
    influence_func=False,
    alpha=0.05,
    random_state=None,
    n_partitions=8,
):
    r"""Shared 2-period doubly robust DDD estimator for panel data.

    Computes the triple-difference ATT for a two-period panel using
    distributed nuisance estimation and influence-function-based
    inference. The DDD estimand is:

    .. math::

        \text{ATT}^{DDD} = \text{DiD}(4, 3) + \text{DiD}(4, 2) - \text{DiD}(4, 1)

    where subgroup 4 is treated-eligible, 3 is treated-ineligible,
    2 is control-eligible, and 1 is control-ineligible.

    Parameters
    ----------
    nuisance_fn : callable
        Function with signature
        ``fn(y1, y0, subgroup, covariates, weights, est_method, n_partitions)``
        returning ``(pscores, or_results)``.
    variance_fn : callable
        Function with signature
        ``fn(inf_func_partitions, n_total, k) -> se``.
    bootstrap_fn : callable
        Function with signature
        ``fn(inf_func_partitions, n_total, biters, alpha, random_state)``
        returning ``(bres, se, crit_val)``.
    y1 : ndarray
        Post-treatment outcomes.
    y0 : ndarray
        Pre-treatment outcomes.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates including intercept, shape :math:`(n, k)`.
    i_weights : ndarray or None
        Observation weights.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    boot : bool, default False
        Whether to use multiplier bootstrap for inference.
    biters : int, default 1000
        Number of bootstrap iterations.
    influence_func : bool, default False
        Whether to return the influence function.
    alpha : float, default 0.05
        Significance level.
    random_state : int or None
        Random seed for bootstrap.
    n_partitions : int, default 8
        Number of partitions for distributed computation.

    Returns
    -------
    DDDPanelResult
        Result containing ATT, standard error, confidence intervals,
        bootstrap draws (if requested), and influence function.
    """
    y1, y0, subgroup, covariates, i_weights, n_units = _validate_inputs(y1, y0, subgroup, covariates, i_weights)

    subgroup_counts = {
        "subgroup_1": int(np.sum(subgroup == 1)),
        "subgroup_2": int(np.sum(subgroup == 2)),
        "subgroup_3": int(np.sum(subgroup == 3)),
        "subgroup_4": int(np.sum(subgroup == 4)),
    }

    pscores, or_results = nuisance_fn(
        y1=y1,
        y0=y0,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        est_method=est_method,
        n_partitions=n_partitions,
    )

    did_results, ddd_att, inf_func_arr = compute_did_distributed(
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        pscores=pscores,
        or_results=or_results,
        est_method=est_method,
        n_total=n_units,
    )

    did_atts = {
        "att_4v3": did_results[0][0],
        "att_4v2": did_results[1][0],
        "att_4v1": did_results[2][0],
    }

    dr_boot = None
    z_val = stats.norm.ppf(1 - alpha / 2)

    if not boot:
        se_ddd = np.std(inf_func_arr, ddof=1) / np.sqrt(n_units)
        uci = ddd_att + z_val * se_ddd
        lci = ddd_att - z_val * se_ddd
    else:
        splits = np.array_split(np.arange(n_units), n_partitions)
        inf_partitions = [inf_func_arr[idx].reshape(-1, 1) for idx in splits if len(idx) > 0]

        bres, se_arr, crit_val = bootstrap_fn(
            inf_func_partitions=inf_partitions,
            n_total=n_units,
            biters=biters,
            alpha=alpha,
            random_state=random_state,
        )
        dr_boot = bres.flatten()
        se_ddd = se_arr[0]
        cv = crit_val if np.isfinite(crit_val) else z_val
        if np.isfinite(se_ddd) and se_ddd > 0:
            uci = ddd_att + cv * se_ddd
            lci = ddd_att - cv * se_ddd
        else:
            uci = lci = ddd_att
            warnings.warn("Bootstrap standard error is zero or NaN.", UserWarning)

    if not influence_func:
        inf_func_arr = None

    args = {
        "panel": True,
        "est_method": est_method,
        "boot": boot,
        "boot_type": "multiplier",
        "biters": biters,
        "alpha": alpha,
    }

    return DDDPanelResult(
        att=ddd_att,
        se=se_ddd,
        uci=uci,
        lci=lci,
        boots=dr_boot,
        att_inf_func=inf_func_arr,
        did_atts=did_atts,
        subgroup_counts=subgroup_counts,
        args=args,
    )
