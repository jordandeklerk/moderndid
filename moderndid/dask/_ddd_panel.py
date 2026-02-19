"""Distributed 2-period panel DDD estimator."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

# Re-use the result type from the local estimator
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult

from ._bootstrap import distributed_mboot_ddd
from ._inf_func import compute_did_distributed
from ._nuisance import compute_all_nuisances_distributed
from ._utils import get_default_partitions


def dask_ddd_panel(
    client,
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
    n_partitions=None,
):
    r"""Distributed 2-period doubly robust DDD estimator for panel data.

    Computes the triple-difference ATT for a two-period panel using
    distributed nuisance estimation and influence-function-based
    inference. The DDD estimand is:

    .. math::

        \text{ATT}^{DDD} = \text{DiD}(4, 3) + \text{DiD}(4, 2) - \text{DiD}(4, 1)

    where subgroup 4 is treated-eligible, 3 is treated-ineligible,
    2 is control-eligible, and 1 is control-ineligible.

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
    n_partitions : int or None
        Number of partitions. Defaults to number of threads across
        all workers.

    Returns
    -------
    DDDPanelResult
        Result containing ATT, standard error, confidence intervals,
        bootstrap draws (if requested), and influence function.
    """
    y1, y0, subgroup, covariates, i_weights, n_units = _validate_inputs(y1, y0, subgroup, covariates, i_weights)

    if n_partitions is None:
        n_partitions = get_default_partitions(client)

    subgroup_counts = {
        "subgroup_1": int(np.sum(subgroup == 1)),
        "subgroup_2": int(np.sum(subgroup == 2)),
        "subgroup_3": int(np.sum(subgroup == 3)),
        "subgroup_4": int(np.sum(subgroup == 4)),
    }

    pscores, or_results = compute_all_nuisances_distributed(
        client=client,
        y1=y1,
        y0=y0,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        est_method=est_method,
        n_partitions=n_partitions,
    )

    did_results, ddd_att, inf_func_arr = compute_did_distributed(
        client=client,
        subgroup=subgroup,
        covariates=covariates,
        weights=i_weights,
        pscores=pscores,
        or_results=or_results,
        est_method=est_method,
        n_total=n_units,
        n_partitions=n_partitions,
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
        # Partition influence function for distributed bootstrap
        splits = np.array_split(np.arange(n_units), n_partitions)
        inf_partitions = [inf_func_arr[idx].reshape(-1, 1) for idx in splits if len(idx) > 0]

        bres, se_arr, crit_val = distributed_mboot_ddd(
            client=client,
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


def _validate_inputs(y1, y0, subgroup, covariates, i_weights):
    """Validate and preprocess input arrays."""
    y1 = np.asarray(y1).flatten()
    y0 = np.asarray(y0).flatten()
    subgroup = np.asarray(subgroup).flatten()
    n_units = len(subgroup)

    if len(y1) != n_units or len(y0) != n_units:
        raise ValueError("y1, y0, and subgroup must have the same length.")

    if covariates is None:
        covariates = np.ones((n_units, 1))
    else:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    if covariates.shape[0] != n_units:
        raise ValueError("covariates must have the same number of rows as subgroup.")

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if len(i_weights) != n_units:
            raise ValueError("i_weights must have the same length as subgroup.")
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights = i_weights / np.mean(i_weights)

    unique_subgroups = set(int(v) for v in np.unique(subgroup))
    if not unique_subgroups.issubset({1, 2, 3, 4}):
        raise ValueError(f"subgroup must contain only values 1, 2, 3, 4. Got {unique_subgroups}.")
    if 4 not in unique_subgroups:
        raise ValueError("subgroup must contain at least one unit in subgroup 4.")

    return y1, y0, subgroup, covariates, i_weights, n_units
