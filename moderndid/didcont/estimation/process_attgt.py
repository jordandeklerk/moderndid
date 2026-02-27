"""Processing functions for ATT(g,t) results."""

import warnings

import numpy as np
import scipy.stats

from moderndid.did.mboot import mboot

from ...cupy.backend import get_backend, to_numpy
from .container import GroupTimeATTResult


def process_att_gt(att_gt_results, pte_params, rng=None):
    """Process ATT(g,t) results.

    Parameters
    ----------
    att_gt_results : dict
        Dictionary containing:

        - **attgt_list**: list of ATT(g,t) estimates
        - **influence_func**: influence function matrix
        - **extra_gt_returns**: list of extra returns from gt-specific calculations

    pte_params : PTEParams
        Parameters object containing estimation settings.
    rng : numpy.random.Generator, optional
        Random number generator for bootstrap. If None, a new generator is created.

    Returns
    -------
    GroupTimeATTResult
        NamedTuple containing processed ATT(g,t) results.
    """
    attgt_list = att_gt_results["attgt_list"]
    influence_func = att_gt_results["influence_func"]

    att = np.array([item["att"] for item in attgt_list])
    groups = np.array([item["group"] for item in attgt_list])
    times = np.array([item["time_period"] for item in attgt_list])
    extra_gt_returns = att_gt_results.get("extra_gt_returns", [])

    n_units = influence_func.shape[0]
    vcov_analytical = influence_func.T @ influence_func / n_units

    cband = pte_params.cband
    alpha = pte_params.alp

    critical_value = scipy.stats.norm.ppf(1 - alpha / 2)
    boot_results = mboot(
        influence_func,
        n_units=n_units,
        biters=int(pte_params.biters) if pte_params.biters else 1000,
        alp=alpha,
        random_state=rng,
    )

    if cband:
        critical_value = boot_results["crit_val"]

    se = boot_results["se"]
    pre_indices = np.where(groups > times)[0]
    pre_att = att[pre_indices]
    pre_vcov = vcov_analytical[np.ix_(pre_indices, pre_indices)]

    wald_stat = None
    wald_pvalue = None

    if len(pre_indices) == 0:
        if len(attgt_list) > 0:
            warnings.warn("No pre-treatment periods to test", UserWarning)
    elif np.any(np.isnan(to_numpy(pre_vcov))):
        warnings.warn("Not returning pre-test Wald statistic due to NA pre-treatment values", UserWarning)
    elif np.linalg.matrix_rank(to_numpy(pre_vcov)) < pre_vcov.shape[0]:
        warnings.warn("Not returning pre-test Wald statistic due to singular covariance matrix", UserWarning)
    else:
        try:
            xp = get_backend()
            wald_stat = float(n_units * pre_att.T @ xp.linalg.solve(pre_vcov, pre_att))
            n_restrictions = len(pre_indices)
            wald_pvalue = 1 - scipy.stats.chi2.cdf(wald_stat, n_restrictions)
        except (np.linalg.LinAlgError, Exception):  # noqa: BLE001
            warnings.warn("Could not compute Wald statistic due to numerical issues", UserWarning)

    if hasattr(pte_params, "data") and hasattr(pte_params, "tname"):
        original_time_periods = np.sort(np.unique(pte_params.data[pte_params.tname]))

        if hasattr(pte_params, "t_list") and not np.all(np.isin(pte_params.t_list, original_time_periods)):
            time_map = {i + 1: orig for i, orig in enumerate(original_time_periods)}

            groups = np.array([time_map.get(g, g) for g in groups])
            times = np.array([time_map.get(t, t) for t in times])

            if extra_gt_returns:
                for egr in extra_gt_returns:
                    if "group" in egr:
                        egr["group"] = time_map.get(egr["group"], egr["group"])
                    if "time_period" in egr:
                        egr["time_period"] = time_map.get(egr["time_period"], egr["time_period"])

    return GroupTimeATTResult(
        groups=groups,
        times=times,
        att=att,
        vcov_analytical=vcov_analytical,
        se=se,
        critical_value=critical_value,
        influence_func=influence_func,
        n_units=n_units,
        wald_stat=wald_stat,
        wald_pvalue=wald_pvalue,
        cband=cband,
        alpha=alpha,
        pte_params=pte_params,
        extra_gt_returns=extra_gt_returns,
    )
