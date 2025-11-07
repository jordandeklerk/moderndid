"""Processing functions for ATT(g,t) results."""

import warnings

import numpy as np
import scipy.stats

from .container import GroupTimeATTResult


def process_att_gt(att_gt_results, pte_params):
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
    boot_results = multiplier_bootstrap(influence_func, alpha=alpha)

    if cband:
        critical_value = boot_results["critical_value"]

    se = boot_results["se"]
    pre_indices = np.where(groups > times)[0]
    pre_att = att[pre_indices]
    pre_vcov = vcov_analytical[np.ix_(pre_indices, pre_indices)]

    wald_stat = None
    wald_pvalue = None

    if len(pre_indices) == 0:
        if len(attgt_list) > 0:
            warnings.warn("No pre-treatment periods to test", UserWarning)
    elif np.any(np.isnan(pre_vcov)):
        warnings.warn("Not returning pre-test Wald statistic due to NA pre-treatment values", UserWarning)
    elif np.linalg.matrix_rank(pre_vcov) < pre_vcov.shape[0]:
        warnings.warn("Not returning pre-test Wald statistic due to singular covariance matrix", UserWarning)
    else:
        try:
            wald_stat = n_units * pre_att.T @ np.linalg.solve(pre_vcov, pre_att)
            n_restrictions = len(pre_indices)
            wald_pvalue = 1 - scipy.stats.chi2.cdf(wald_stat, n_restrictions)
        except np.linalg.LinAlgError:
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


def multiplier_bootstrap(influence_func, biters=1000, alpha=0.05, rng=None):
    """Multiplier bootstrap for inference.

    Parameters
    ----------
    influence_func : ndarray
        Influence function matrix of shape (n, k) where n is the number of
        observations and k is the number of parameters.
    biters : int, default=1000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    rng : numpy.random.Generator, optional
        Generator used for resampling. If omitted, a fresh generator from
        ``np.random.default_rng`` is created.

    Returns
    -------
    dict
        Dictionary containing:

        - **se**: Bootstrap standard errors
        - **critical_value**: Critical value for uniform confidence bands
    """
    n_obs = influence_func.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    bootstrap_results = []

    for _ in range(biters):
        weights = rng.choice([-1, 1], size=n_obs, replace=True)
        bootstrap_draw = np.sqrt(n_obs) * np.mean(weights[:, np.newaxis] * influence_func, axis=0)
        bootstrap_results.append(bootstrap_draw)

    bootstrap_results = np.array(bootstrap_results)

    def compute_se(bootstrap_column):
        q75 = np.percentile(bootstrap_column, 75, method="lower")
        q25 = np.percentile(bootstrap_column, 25, method="lower")
        iqr_se = (q75 - q25) / (scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25))
        return iqr_se

    se = np.array([compute_se(bootstrap_results[:, j]) for j in range(bootstrap_results.shape[1])])
    se = se / np.sqrt(n_obs)

    t_stats = []
    for b_result in bootstrap_results:
        se_safe = np.where(se > 1e-8, se, 1.0)
        t_stat = np.max(np.abs(b_result / se_safe) / np.sqrt(n_obs))
        t_stats.append(t_stat)

    t_stats = np.array(t_stats)

    critical_value = np.percentile(t_stats, (1 - alpha) * 100, method="lower")
    pointwise_crit = scipy.stats.norm.ppf(1 - alpha / 2)

    if np.isnan(critical_value) or np.isinf(critical_value):
        warnings.warn(
            "Simultaneous critical value is NA or infinite. This can happen if standard errors are zero. "
            "Reporting pointwise confidence intervals instead.",
            UserWarning,
        )
        critical_value = pointwise_crit
    elif critical_value < pointwise_crit:
        warnings.warn(
            "Simultaneous confidence band is smaller than pointwise one. "
            "Reporting pointwise confidence intervals instead.",
            UserWarning,
        )
        critical_value = pointwise_crit
    elif critical_value >= 7:
        warnings.warn(
            "Simultaneous critical value is arguably 'too large' to be reliable. "
            "This usually happens when the number of observations per group is small "
            "or there is not much variation in outcomes.",
            UserWarning,
        )

    return {"se": se, "critical_value": critical_value}
