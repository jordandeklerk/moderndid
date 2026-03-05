"""Containers for panel treatment effects."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import polars as pl


class PTEParams(NamedTuple):
    """Container for panel treatment effect parameters.

    Attributes
    ----------
    yname : str
        Name of the outcome variable.
    gname : str
        Name of the group variable (first treatment period).
    tname : str
        Name of the time period variable.
    idname : str
        Name of the id variable.
    data : pl.DataFrame
        Panel data as a pandas DataFrame.
    g_list : np.ndarray
        Array of unique group identifiers.
    t_list : np.ndarray
        Array of unique time period identifiers.
    cband : bool
        Whether to compute a uniform confidence band.
    alp : float
        Significance level for confidence intervals.
    boot_type : str
        Method for bootstrapping.
    anticipation : int
        Number of periods of anticipation.
    base_period : str
        Base period for computing ATT(g,t).
    weightsname : str
        Name of the weights variable.
    control_group : str
        Which units to use as the control group.
    gt_type : str
        Type of group-time average treatment effect.
    ret_quantile : float
        Quantile to return for conditional distribution.
    biters : int
        Number of bootstrap iterations.
    dname : str
        Name of the continuous treatment variable.
    degree : int
        Degree of the spline for continuous treatment.
    num_knots : int
        Number of knots for the spline.
    knots : np.ndarray
        Array of knot locations for the spline.
    dvals : np.ndarray
        Values of the dose to evaluate the dose-response function.
    target_parameter : str
        The target parameter of interest.
    aggregation : str
        Type of aggregation for results.
    treatment_type : str
        Type of treatment (e.g., 'continuous').
    xformula : str
        Formula for covariates.
    dose_est_method : str
        Method for estimating dose-specific effects ('parametric' or 'cck').
    """

    #: Name of the outcome variable.
    yname: str
    #: Name of the group variable (first treatment period).
    gname: str
    #: Name of the time period variable.
    tname: str
    #: Name of the id variable.
    idname: str
    #: Panel data as a Polars DataFrame.
    data: pl.DataFrame
    #: Array of unique group identifiers.
    g_list: np.ndarray
    #: Array of unique time period identifiers.
    t_list: np.ndarray
    #: Whether to compute a uniform confidence band.
    cband: bool
    #: Significance level for confidence intervals.
    alp: float
    #: Method for bootstrapping.
    boot_type: str
    #: Number of periods of anticipation.
    anticipation: int
    #: Base period for computing ATT(g,t).
    base_period: str
    #: Name of the weights variable.
    weightsname: str
    #: Which units to use as the control group.
    control_group: str
    #: Type of group-time average treatment effect.
    gt_type: str
    #: Quantile to return for conditional distribution.
    ret_quantile: float
    #: Number of bootstrap iterations.
    biters: int
    #: Name of the continuous treatment variable.
    dname: str
    #: Degree of the spline for continuous treatment.
    degree: int
    #: Number of knots for the spline.
    num_knots: int
    #: Array of knot locations for the spline.
    knots: np.ndarray
    #: Values of the dose to evaluate the dose-response function.
    dvals: np.ndarray
    #: The target parameter of interest.
    target_parameter: str
    #: Type of aggregation for results.
    aggregation: str
    #: Type of treatment.
    treatment_type: str
    #: Formula for covariates.
    xformula: str
    #: Method for estimating dose-specific effects.
    dose_est_method: str = "parametric"


class AttgtResult(NamedTuple):
    """Container for a single ATT(g,t) result with influence function."""

    #: ATT estimate for this (g,t) cell.
    attgt: float
    #: Influence function for this estimate.
    inf_func: np.ndarray | None
    #: Extra returns from group-time calculations.
    extra_gt_returns: dict | None


class PTEResult(NamedTuple):
    """Container for panel treatment effects results."""

    #: Group-time ATT results.
    att_gt: object
    #: Overall ATT estimate.
    overall_att: object | None
    #: Event study results.
    event_study: object | None
    #: Panel treatment effect parameters.
    ptep: PTEParams


class PTEAggteResult(NamedTuple):
    """Container for aggregated panel treatment effect parameters.

    Attributes
    ----------
    overall_att : float
        The estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'overall', 'dynamic', 'group'}
        Type of aggregation performed.
    event_times : np.ndarray, optional
        Event/group values depending on aggregation type:

        - For dynamic effects: length of exposure (event time)
        - For group effects: treatment group indicators
    att_by_event : np.ndarray, optional
        ATT estimates specific to each event time value.
    se_by_event : np.ndarray, optional
        Standard errors specific to each event time value.
    critical_value : float, optional
        Critical value for uniform confidence bands.
    influence_func : dict, optional
        Dictionary containing influence functions:

        - **overall**: Overall ATT influence function
        - **by_event**: Event-specific influence functions
    min_event_time : int, optional
        Minimum event time (for dynamic effects).
    max_event_time : int, optional
        Maximum event time (for dynamic effects).
    balance_event : int, optional
        Balanced event time threshold.
    att_gt_result : object
        Original group-time ATT result object.
    """

    #: Estimated overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for overall ATT.
    overall_se: float
    #: Type of aggregation performed.
    aggregation_type: Literal["overall", "dynamic", "group"]
    #: Event/group values depending on aggregation type.
    event_times: np.ndarray | None = None
    #: ATT estimates specific to each event time value.
    att_by_event: np.ndarray | None = None
    #: Standard errors specific to each event time value.
    se_by_event: np.ndarray | None = None
    #: Critical value for uniform confidence bands.
    critical_value: float | None = None
    #: Influence functions for overall and event-specific ATTs.
    influence_func: dict | None = None
    #: Minimum event time.
    min_event_time: int | None = None
    #: Maximum event time.
    max_event_time: int | None = None
    #: Balanced event time threshold.
    balance_event: int | None = None
    #: Original group-time ATT result object.
    att_gt_result: object | None = None


@dataclass
class GroupTimeATTResult:
    """Container for group-time average treatment effect results.

    Attributes
    ----------
    groups : np.ndarray
        Which group (defined by period first treated) each group-time ATT is for.
    times : np.ndarray
        Which time period each group-time ATT is for.
    att : np.ndarray
        The group-time average treatment effects for each group-time combination.
    vcov_analytical : np.ndarray
        Analytical estimator for the asymptotic variance-covariance matrix.
    se : np.ndarray
        Standard errors for group-time ATTs. If bootstrap used, provides bootstrap-based SE.
    critical_value : float
        Critical value - simultaneous if obtaining simultaneous confidence bands,
        otherwise based on pointwise normal approximation.
    influence_func : np.ndarray
        The influence function for estimating group-time average treatment effects.
    n_units : int
        The number of unique cross-sectional units.
    wald_stat : float, optional
        The Wald statistic for pre-testing the common trends assumption.
    wald_pvalue : float, optional
        The p-value of the Wald statistic for pre-testing common trends.
    cband : bool
        Whether uniform confidence band was computed.
    alpha : float
        The significance level.
    pte_params : object
        The PTE parameters object containing estimation settings.
    extra_gt_returns : list
        List of extra returns from gt-specific calculations.
    """

    #: Which group (defined by period first treated) each group-time ATT is for.
    groups: np.ndarray
    #: Which time period each group-time ATT is for.
    times: np.ndarray
    #: Group-time average treatment effects.
    att: np.ndarray
    #: Analytical estimator for the asymptotic variance-covariance matrix.
    vcov_analytical: np.ndarray
    #: Standard errors for group-time ATTs.
    se: np.ndarray
    #: Critical value for confidence intervals.
    critical_value: float
    #: Influence function for estimating group-time average treatment effects.
    influence_func: np.ndarray
    #: Number of unique cross-sectional units.
    n_units: int
    #: Wald statistic for pre-testing common trends.
    wald_stat: float | None = None
    #: P-value of the Wald statistic for pre-testing common trends.
    wald_pvalue: float | None = None
    #: Whether uniform confidence band was computed.
    cband: bool = True
    #: Significance level.
    alpha: float = 0.05
    #: PTE parameters object containing estimation settings.
    pte_params: object | None = None
    #: Extra returns from group-time calculations.
    extra_gt_returns: list | None = None

    @property
    def att_gt(self):
        """Alias for att field to maintain compatibility with aggte."""
        return self.att

    @property
    def se_gt(self):
        """Alias for se field to maintain compatibility with aggte."""
        return self.se

    @property
    def estimation_params(self):
        """Return estimation parameters for aggte compatibility."""
        return {
            "bootstrap": True,
            "biters": 999,
            "uniform_bands": self.cband,
            "alpha": self.alpha,
        }

    @property
    def G(self):
        """Unit-level group assignments (not tracked in continuous DiD)."""
        return None


class PteEmpBootResult(NamedTuple):
    """Container for empirical bootstrap results.

    Attributes
    ----------
    attgt_results : pl.DataFrame
        ATT(g,t) estimates with standard errors.
    overall_results : dict
        Overall ATT estimate and standard error.
    group_results : pl.DataFrame | None
        Group-specific ATT estimates and standard errors.
    dyn_results : pl.DataFrame | None
        Dynamic (event-time) ATT estimates and standard errors.
    extra_gt_returns : list | None
        Extra returns from group-time calculations.
    """

    #: ATT(g,t) estimates with standard errors.
    attgt_results: pl.DataFrame
    #: Overall ATT estimate and standard error.
    overall_results: dict
    #: Group-specific ATT estimates and standard errors.
    group_results: pl.DataFrame | None = None
    #: Dynamic (event-time) ATT estimates and standard errors.
    dyn_results: pl.DataFrame | None = None
    #: Extra returns from group-time calculations.
    extra_gt_returns: list | None = None


class DoseResult(NamedTuple):
    """Container for continuous treatment dose-response results.

    Attributes
    ----------
    dose : np.ndarray
        Vector containing the values of the dose used in estimation.
    overall_att : float
        Estimate of the overall ATT, the mean of ATT(D) given D > 0.
    overall_att_se : float
        The standard error of the estimate of overall_att.
    overall_att_inf_func : np.ndarray
        The influence function for estimating overall_att.
    overall_acrt : float
        Estimate of the overall ACRT, the mean of ACRT(D|D) given D > 0.
    overall_acrt_se : float
        The standard error for the estimate of overall_acrt.
    overall_acrt_inf_func : np.ndarray
        The influence function for estimating overall_acrt.
    att_d : np.ndarray
        Estimates of ATT(d) for each value of dose.
    att_d_se : np.ndarray
        Standard error of ATT(d) for each value of dose.
    att_d_crit_val : float
        Critical value to produce pointwise or uniform confidence interval for ATT(d).
    att_d_inf_func : np.ndarray
        Matrix containing the influence function from estimating ATT(d).
    acrt_d : np.ndarray
        Estimates of ACRT(d) for each value of dose.
    acrt_d_se : np.ndarray
        Standard error of ACRT(d) for each value of dose.
    acrt_d_crit_val : float
        Critical value to produce pointwise or uniform confidence interval for ACRT(d).
    acrt_d_inf_func : np.ndarray
        Matrix containing the influence function from estimating ACRT(d).
    pte_params : object
        A PTEParams object containing other parameters passed to the function.
    """

    #: Values of the dose used in estimation.
    dose: np.ndarray
    #: Estimate of the overall ATT.
    overall_att: float | None = None
    #: Standard error of the overall ATT estimate.
    overall_att_se: float | None = None
    #: Influence function for estimating overall ATT.
    overall_att_inf_func: np.ndarray | None = None
    #: Estimate of the overall ACRT.
    overall_acrt: float | None = None
    #: Standard error of the overall ACRT estimate.
    overall_acrt_se: float | None = None
    #: Influence function for estimating overall ACRT.
    overall_acrt_inf_func: np.ndarray | None = None
    #: Estimates of ATT(d) for each value of dose.
    att_d: np.ndarray | None = None
    #: Standard errors of ATT(d) for each value of dose.
    att_d_se: np.ndarray | None = None
    #: Critical value for ATT(d) confidence intervals.
    att_d_crit_val: float | None = None
    #: Influence function matrix for ATT(d).
    att_d_inf_func: np.ndarray | None = None
    #: Estimates of ACRT(d) for each value of dose.
    acrt_d: np.ndarray | None = None
    #: Standard errors of ACRT(d) for each value of dose.
    acrt_d_se: np.ndarray | None = None
    #: Critical value for ACRT(d) confidence intervals.
    acrt_d_crit_val: float | None = None
    #: Influence function matrix for ACRT(d).
    acrt_d_inf_func: np.ndarray | None = None
    #: PTEParams object containing estimation settings.
    pte_params: object | None = None
