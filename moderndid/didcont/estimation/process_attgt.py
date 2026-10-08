"""Processing functions for ATT(g,t) results."""

import warnings

import numpy as np
import scipy.stats

from moderndid.did.mboot import _mboot

from ...cupy.backend import get_backend, to_numpy
from ..container import GroupTimeATTResult


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

    pointwise_value = scipy.stats.norm.ppf(1 - alpha / 2)
    critical_value = pointwise_value
    # Since dropping the draws that move a column with zero bootstrap scale would condition the band on that column
    # staying at zero, they stay in the sample.
    boot_results = _mboot(
        influence_func,
        n_units=n_units,
        biters=int(pte_params.biters) if pte_params.biters else 1000,
        alp=alpha,
        random_state=rng,
        keep_infinite_draws=True,
    )

    if cband:
        critical_value = boot_results["crit_val"]
        # A band that covers every cell at once can't be narrower than the pointwise intervals.
        if critical_value < pointwise_value:
            warnings.warn("Simultaneous band smaller than pointwise; using pointwise intervals.")
            critical_value = pointwise_value

    se = boot_results["se"]
    # Since the reference period of a universal base has no estimate, its standard error stays undefined.
    if pte_params.base_period == "universal":
        se = np.where(times == groups - 1 - pte_params.anticipation, np.nan, se)
    # A cell without variance, such as the reference period under a universal base, carries nothing for the
    # pre-test and would make its covariance singular.
    analytic_se = np.sqrt(np.diag(to_numpy(vcov_analytical)) / n_units)
    zero_variance = analytic_se <= np.sqrt(np.finfo(float).eps) * 10
    pre_indices = np.where((groups > times) & ~zero_variance)[0]
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

    time_map = _period_labels(pte_params)
    if time_map:
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


def _period_labels(pte_params):
    """Map the period positions that index the cells to the periods as the data codes them."""
    data = getattr(pte_params, "data", None)
    tname = getattr(pte_params, "tname", None)
    t_list = getattr(pte_params, "t_list", None)
    if data is None or tname is None or t_list is None:
        return {}

    # Cells index periods by their position in the "period" column. Reading each position's label from the
    # data keeps a coding such as 2, 3, 4, 5 from passing for positions.
    columns = getattr(data, "columns", [])
    if "period" in columns and tname in columns and tname != "period":
        pairs = data.select("period", tname).unique()
        positions = pairs["period"].to_numpy()
        if not np.all(np.isin(t_list, positions)):
            return {}
        time_map = dict(zip(positions.tolist(), pairs[tname].to_list(), strict=True))
        return {} if all(position == label for position, label in time_map.items()) else time_map

    original_time_periods = np.sort(np.unique(data[tname]))
    if np.all(np.isin(t_list, original_time_periods)):
        return {}
    return {i + 1: orig for i, orig in enumerate(original_time_periods)}
