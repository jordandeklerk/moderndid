"""Multi-Period Objects for Group-Time Average Treatment Effects."""

from typing import NamedTuple

import numpy as np


class MPResult(NamedTuple):
    """Container for group-time average treatment effect results.

    Attributes
    ----------
    groups : ndarray
        Which group (defined by period first treated) each group-time ATT is for.
    times : ndarray
        Which time period each group-time ATT is for.
    att_gt : ndarray
        The group-time average treatment effects for each group-time combination.
    vcov_analytical : ndarray
        Analytical estimator for the asymptotic variance-covariance matrix.
    se_gt : ndarray
        Standard errors for group-time ATTs. If bootstrap used, provides bootstrap-based SE.
    critical_value : float
        Critical value - simultaneous if obtaining simultaneous confidence bands,
        otherwise based on pointwise normal approximation.
    influence_func : ndarray
        The influence function for estimating group-time average treatment effects.
    n_units : int, optional
        The number of unique cross-sectional units.
    wald_stat : float, optional
        The Wald statistic for pre-testing the common trends assumption.
    wald_pvalue : float, optional
        The p-value of the Wald statistic for pre-testing common trends.
    aggregate_effects : object, optional
        An aggregate treatment effects object.
    alpha : float
        The significance level (default 0.05).
    estimation_params : dict
        Dictionary containing DID estimation parameters including:

        - call_info: original function call information
        - control_group: 'nevertreated' or 'notyettreated'
        - anticipation_periods: number of anticipation periods
        - estimation_method: estimation method used
        - bootstrap: whether bootstrap was used
        - uniform_bands: whether simultaneous confidence bands were computed
        - G: unit-level group assignments
        - weights_ind: unit-level sampling weights
    """

    groups: np.ndarray
    times: np.ndarray
    att_gt: np.ndarray
    vcov_analytical: np.ndarray
    se_gt: np.ndarray
    critical_value: float
    influence_func: np.ndarray
    n_units: int | None = None
    wald_stat: float | None = None
    wald_pvalue: float | None = None
    aggregate_effects: object | None = None
    alpha: float = 0.05
    estimation_params: dict = {}
    G: np.ndarray | None = None
    weights_ind: np.ndarray | None = None


def mp(
    groups,
    times,
    att_gt,
    vcov_analytical,
    se_gt,
    critical_value,
    influence_func,
    n_units=None,
    wald_stat=None,
    wald_pvalue=None,
    aggregate_effects=None,
    alpha=0.05,
    estimation_params=None,
    G=None,
    weights_ind=None,
):
    """Create a multi-period result object for group-time ATTs.

    Parameters
    ----------
    groups : ndarray
        Group indicators (defined by period first treated).
    times : ndarray
        Time period indicators.
    att_gt : ndarray
        Group-time average treatment effects.
    vcov_analytical : ndarray
        Analytical variance-covariance matrix estimator.
    se_gt : ndarray
        Standard errors for group-time ATTs.
    critical_value : float
        Critical value for confidence intervals.
    influence_func : ndarray
        Influence function for group-time ATTs.
    n_units : int, optional
        Number of unique cross-sectional units.
    wald_stat : float, optional
        Wald statistic for common trends test.
    wald_pvalue : float, optional
        P-value for common trends test.
    aggregate_effects : object, optional
        Aggregate treatment effects object.
    alpha : float, default=0.05
        Significance level.
    estimation_params : dict, optional
        DID estimation parameters.
    G : ndarray, optional
        Unit-level group assignments (length n, where n is number of units).
    weights_ind : ndarray, optional
        Unit-level sampling weights (length n, where n is number of units).

    Returns
    -------
    MPResult
        NamedTuple containing multi-period results.
    """
    groups = np.asarray(groups)
    times = np.asarray(times)
    att_gt = np.asarray(att_gt)
    se_gt = np.asarray(se_gt)

    n_gt = len(groups)
    if len(times) != n_gt:
        raise ValueError("groups and times must have the same length.")
    if len(att_gt) != n_gt:
        raise ValueError("att_gt must have same length as groups and times.")
    if len(se_gt) != n_gt:
        raise ValueError("se_gt must have same length as groups and times.")

    if estimation_params is None:
        estimation_params = {}

    return MPResult(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov_analytical,
        se_gt=se_gt,
        critical_value=critical_value,
        influence_func=influence_func,
        n_units=n_units,
        wald_stat=wald_stat,
        wald_pvalue=wald_pvalue,
        aggregate_effects=aggregate_effects,
        alpha=alpha,
        estimation_params=estimation_params,
        G=G,
        weights_ind=weights_ind,
    )


def format_mp_result(result):
    """Format multi-period results.

    Parameters
    ----------
    result : MPResult
        The multi-period result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []

    lines.append("")

    if result.estimation_params.get("call_info"):
        lines.append("Call:")
        lines.append(f"{result.estimation_params['call_info']}")
        lines.append("")

    lines.append("Reference: Callaway and Sant'Anna (2021)")
    lines.append("")

    lines.append("Group-Time Average Treatment Effects:")

    conf_level = int((1 - result.alpha) * 100)
    bootstrap = result.estimation_params.get("bootstrap", False)
    uniform_bands = result.estimation_params.get("uniform_bands", False)

    if bootstrap and uniform_bands:
        band_type = "Simult."
    else:
        band_type = "Pointwise"

    band_text = f"[{conf_level}% {band_type} Conf. Band]"

    conf_lower = result.att_gt - result.critical_value * result.se_gt
    conf_upper = result.att_gt + result.critical_value * result.se_gt

    sig = (conf_upper < 0) | (conf_lower > 0)
    sig[np.isnan(sig)] = False

    lines.append(f" {'Group':>6} {'Time':>6} {'ATT(g,t)':>10} {'Std. Error':>12}  {band_text}")

    for i, group in enumerate(result.groups):
        sig_marker = "*" if sig[i] else " "
        lines.append(
            f" {group:>6.0f} {result.times[i]:>6.0f} "
            f"{result.att_gt[i]:>10.4f} {result.se_gt[i]:>12.4f}  "
            f"[{conf_lower[i]:>8.4f}, {conf_upper[i]:>8.4f}] {sig_marker}"
        )

    lines.append("---")
    lines.append("Signif. codes: '*' confidence band does not cover 0")
    lines.append("")

    if result.wald_pvalue is not None:
        lines.append(f"P-value for pre-test of parallel trends assumption:  {result.wald_pvalue:.4f}")
        lines.append("")

    control_group = result.estimation_params.get("control_group")
    if control_group:
        control_text = {"nevertreated": "Never Treated", "notyettreated": "Not Yet Treated"}.get(
            control_group, control_group
        )
        lines.append(f"Control Group:  {control_text},  ")

    anticipation = result.estimation_params.get("anticipation_periods", 0)
    lines.append(f"Anticipation Periods:  {anticipation}")

    est_method = result.estimation_params.get("estimation_method")
    if est_method:
        method_text = {"dr": "Doubly Robust", "ipw": "Inverse Probability Weighting", "reg": "Outcome Regression"}.get(
            est_method, est_method
        )
        lines.append(f"Estimation Method:  {method_text}")

    lines.append("")

    return "\n".join(lines)


def summary_mp(result):
    """Print summary of a multi-period result.

    Parameters
    ----------
    result : MPResult
        The multi-period result to summarize.

    Returns
    -------
    str
        Formatted summary string.
    """
    return format_mp_result(result)


def _mp_repr(self):
    return format_mp_result(self)


def _mp_str(self):
    return format_mp_result(self)


MPResult.__repr__ = _mp_repr
MPResult.__str__ = _mp_str


class MPPretestResult(NamedTuple):
    """Container for pre-test results of conditional parallel trends assumption.

    Attributes
    ----------
    cvm_stat : float
        Cramer von Mises test statistic.
    cvm_boots : ndarray, optional
        Vector of bootstrapped Cramer von Mises test statistics.
    cvm_critval : float
        Cramer von Mises critical value.
    cvm_pval : float
        P-value for Cramer von Mises test.
    ks_stat : float
        Kolmogorov-Smirnov test statistic.
    ks_boots : ndarray, optional
        Vector of bootstrapped Kolmogorov-Smirnov test statistics.
    ks_critval : float
        Kolmogorov-Smirnov critical value.
    ks_pval : float
        P-value for Kolmogorov-Smirnov test.
    cluster_vars : list[str], optional
        Variables that were clustered on for the test.
    x_formula : str, optional
        Formula for the X variables used in the test.
    """

    cvm_stat: float
    cvm_boots: np.ndarray | None
    cvm_critval: float
    cvm_pval: float
    ks_stat: float
    ks_boots: np.ndarray | None
    ks_critval: float
    ks_pval: float
    cluster_vars: list[str] | None = None
    x_formula: str | None = None


def mp_pretest(
    cvm_stat,
    cvm_critval,
    cvm_pval,
    ks_stat,
    ks_critval,
    ks_pval,
    cvm_boots=None,
    ks_boots=None,
    cluster_vars=None,
    x_formula=None,
):
    """Create a pre-test result object for conditional parallel trends assumption.

    Parameters
    ----------
    cvm_stat : float
        Cramer von Mises test statistic.
    cvm_critval : float
        Cramer von Mises critical value.
    cvm_pval : float
        P-value for Cramer von Mises test.
    ks_stat : float
        Kolmogorov-Smirnov test statistic.
    ks_critval : float
        Kolmogorov-Smirnov critical value.
    ks_pval : float
        P-value for Kolmogorov-Smirnov test.
    cvm_boots : ndarray, optional
        Vector of bootstrapped Cramer von Mises test statistics.
    ks_boots : ndarray, optional
        Vector of bootstrapped Kolmogorov-Smirnov test statistics.
    cluster_vars : list[str], optional
        Variables that were clustered on for the test.
    x_formula : str, optional
        Formula for the X variables used in the test.

    Returns
    -------
    MPPretestResult
        NamedTuple containing pre-test results.
    """
    if cvm_boots is not None:
        cvm_boots = np.asarray(cvm_boots)
    if ks_boots is not None:
        ks_boots = np.asarray(ks_boots)

    return MPPretestResult(
        cvm_stat=cvm_stat,
        cvm_boots=cvm_boots,
        cvm_critval=cvm_critval,
        cvm_pval=cvm_pval,
        ks_stat=ks_stat,
        ks_boots=ks_boots,
        ks_critval=ks_critval,
        ks_pval=ks_pval,
        cluster_vars=cluster_vars,
        x_formula=x_formula,
    )


def format_mp_pretest_result(result):
    """Format pre-test results.

    Parameters
    ----------
    result : MPPretestResult
        The pre-test result to format.

    Returns
    -------
    str
        Formatted string representation of the results.
    """
    lines = []

    lines.append("")
    lines.append("Pre-test of Conditional Parallel Trends Assumption")
    lines.append("=" * 50)
    lines.append("")

    lines.append("Cramer von Mises Test:")
    lines.append(f"  Test Statistic: {result.cvm_stat:.4f}")
    lines.append(f"  Critical Value: {result.cvm_critval:.4f}")
    lines.append(f"  P-value       : {result.cvm_pval:.4f}")
    lines.append("")

    lines.append("Kolmogorov-Smirnov Test:")
    lines.append(f"  Test Statistic: {result.ks_stat:.4f}")
    lines.append(f"  Critical Value: {result.ks_critval:.4f}")
    lines.append(f"  P-value       : {result.ks_pval:.4f}")
    lines.append("")

    if result.cluster_vars:
        cluster_str = ", ".join(result.cluster_vars)
        lines.append(f"Clustering on: {cluster_str}")

    if result.x_formula:
        lines.append(f"X formula: {result.x_formula}")

    lines.append("")

    return "\n".join(lines)


def summary_mp_pretest(result):
    """Print summary of a pre-test result.

    Parameters
    ----------
    result : MPPretestResult
        The pre-test result to summarize.

    Returns
    -------
    str
        Formatted summary string.
    """
    return format_mp_pretest_result(result)


def _mp_pretest_repr(self):
    return format_mp_pretest_result(self)


def _mp_pretest_str(self):
    return format_mp_pretest_result(self)


MPPretestResult.__repr__ = _mp_pretest_repr
MPPretestResult.__str__ = _mp_pretest_str
