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
    return str(result)


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
    return str(result)
