"""Result containers for multi-period DiD estimators."""

from typing import Literal, NamedTuple

import numpy as np
from scipy import stats

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    control_group_label,
    est_method_label,
    make_effect_names,
    make_group_time_names,
    se_type_label,
)
from moderndid.core.result import extract_n_obs, extract_vcov_info


class AGGTEResult(NamedTuple):
    """Container for aggregated treatment effect parameters.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    overall_att : float
        The estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}
        Type of aggregation performed.
    event_times : np.ndarray, optional
        Event/group/time values depending on aggregation type:

        - For dynamic effects: length of exposure
        - For group effects: treatment group indicators
        - For calendar effects: time periods
    att_by_event : ndarray, optional
        ATT estimates specific to each event time value.
    se_by_event : ndarray, optional
        Standard errors specific to each event time value.
    critical_values : ndarray, optional
        Critical values for uniform confidence bands.
    influence_func : ndarray, optional
        Influence function of the aggregated parameters.

        - For overall ATT: 1D array of length n_units
        - For dynamic/group/calendar: 2D array of shape (n_units, n_events) containing
          influence functions for each event-specific ATT
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D array of length n_units).
        This is stored separately for compatibility with both aggregation types.
    min_event_time : int, optional
        Minimum event time (for dynamic effects).
    max_event_time : int, optional
        Maximum event time (for dynamic effects).
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict
        Dictionary containing DID estimation parameters including:

        - alpha: significance level
        - bootstrap: whether bootstrap was used
        - uniform_bands: whether uniform confidence bands were computed
        - control_group: 'nevertreated' or 'notyettreated'
        - anticipation_periods: number of anticipation periods
        - estimation_method: estimation method used
    call_info : dict
        Information about the function call that created this object.
    """

    #: Estimated overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for overall ATT.
    overall_se: float
    #: Type of aggregation performed.
    aggregation_type: Literal["simple", "dynamic", "group", "calendar"]
    #: Event/group/time values depending on aggregation type.
    event_times: np.ndarray | None = None
    #: ATT estimates specific to each event time value.
    att_by_event: np.ndarray | None = None
    #: Standard errors specific to each event time value.
    se_by_event: np.ndarray | None = None
    #: Critical values for uniform confidence bands.
    critical_values: np.ndarray | None = None
    #: Influence function of the aggregated parameters.
    influence_func: np.ndarray | None = None
    #: Influence function for the overall ATT.
    influence_func_overall: np.ndarray | None = None
    #: Minimum event time.
    min_event_time: int | None = None
    #: Maximum event time.
    max_event_time: int | None = None
    #: Balanced event time threshold.
    balanced_event_threshold: int | None = None
    #: DID estimation parameters.
    estimation_params: dict = {}
    #: Information about the function call that created this object.
    call_info: dict = {}

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        alpha = float(self.estimation_params.get("alpha", 0.05))
        names = ["Overall ATT"]
        estimates = [self.overall_att]
        se = [self.overall_se]

        crit = None
        if self.event_times is not None and self.att_by_event is not None and self.se_by_event is not None:
            z_crit = stats.norm.ppf(1 - alpha / 2)
            prefix = {"dynamic": "Event", "group": "Group", "calendar": "Time"}.get(self.aggregation_type, "Effect")
            names.extend(make_effect_names(self.event_times, prefix=prefix))
            estimates.extend(np.asarray(self.att_by_event, dtype=float).tolist())
            se.extend(np.asarray(self.se_by_event, dtype=float).tolist())

            if self.critical_values is not None:
                event_crit = np.asarray(self.critical_values, dtype=float)
            else:
                event_crit = np.full(len(self.event_times), z_crit)
            crit = np.concatenate([[z_crit], event_crit])

        return build_coef_table_with_ci(names, estimates, se, alpha=alpha, critical_values=crit)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return extract_n_obs(
                self.influence_func_overall,
                self.influence_func,
                params=self.estimation_params,
            )
        if key == "n_units":
            return extract_n_obs(
                self.influence_func_overall,
                self.influence_func,
                params=self.estimation_params,
                keys=("n_units",),
            )
        if key == "aggregation":
            return self.aggregation_type
        if key == "se_type":
            return se_type_label(bool(self.estimation_params.get("bootstrap", False)))
        if key == "control_group":
            return control_group_label(self.estimation_params.get("control_group"))
        if key == "estimation_method":
            return est_method_label(self.estimation_params.get("estimation_method"))
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "Aggregated ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """AGGTE output does not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.estimation_params)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "n_units": "Units",
            "aggregation": "Aggregation",
            "control_group": "Control Group",
            "estimation_method": "Estimation Method",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["aggregation", "se_type", "control_group"]
        if self.__maketables_stat__("N") is not None:
            keys.insert(0, "N")
        if self.__maketables_stat__("n_units") is not None:
            idx = keys.index("N") + 1 if "N" in keys else 0
            keys.insert(idx, "n_units")
        if self.estimation_params.get("estimation_method") is not None:
            keys.append("estimation_method")
        return keys


class MPResult(NamedTuple):
    """Container for group-time average treatment effect results.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

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

    #: Which group (defined by period first treated) each group-time ATT is for.
    groups: np.ndarray
    #: Which time period each group-time ATT is for.
    times: np.ndarray
    #: Group-time average treatment effects.
    att_gt: np.ndarray
    #: Analytical estimator for the asymptotic variance-covariance matrix.
    vcov_analytical: np.ndarray
    #: Standard errors for group-time ATTs.
    se_gt: np.ndarray
    #: Critical value for confidence intervals.
    critical_value: float
    #: Influence function for estimating group-time average treatment effects.
    influence_func: np.ndarray
    #: Number of unique cross-sectional units.
    n_units: int | None = None
    #: Wald statistic for pre-testing common trends.
    wald_stat: float | None = None
    #: P-value of the Wald statistic for pre-testing common trends.
    wald_pvalue: float | None = None
    #: Aggregate treatment effects object.
    aggregate_effects: object | None = None
    #: Significance level.
    alpha: float = 0.05
    #: DID estimation parameters.
    estimation_params: dict = {}
    #: Unit-level group assignments.
    G: np.ndarray | None = None
    #: Unit-level sampling weights.
    weights_ind: np.ndarray | None = None

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        names = make_group_time_names(self.groups, self.times, prefix="ATT")
        crit = self.critical_value if self.critical_value is not None else None
        return build_coef_table_with_ci(names, self.att_gt, self.se_gt, alpha=float(self.alpha), critical_values=crit)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return int(self.n_units) if self.n_units is not None else None
        if key == "wald_pvalue":
            return self.wald_pvalue
        if key == "se_type":
            return se_type_label(bool(self.estimation_params.get("bootstrap", False)))
        if key == "control_group":
            return control_group_label(self.estimation_params.get("control_group"))
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "ATT(g,t)"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Group-time ATT results do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.estimation_params)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {"wald_pvalue": "Pre-trends p-value", "control_group": "Control Group"}

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["N", "se_type", "control_group"]
        if self.wald_pvalue is not None:
            keys.insert(1, "wald_pvalue")
        return keys


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

    #: Cramer von Mises test statistic.
    cvm_stat: float
    #: Bootstrapped Cramer von Mises test statistics.
    cvm_boots: np.ndarray | None
    #: Cramer von Mises critical value.
    cvm_critval: float
    #: P-value for Cramer von Mises test.
    cvm_pval: float
    #: Kolmogorov-Smirnov test statistic.
    ks_stat: float
    #: Bootstrapped Kolmogorov-Smirnov test statistics.
    ks_boots: np.ndarray | None
    #: Kolmogorov-Smirnov critical value.
    ks_critval: float
    #: P-value for Kolmogorov-Smirnov test.
    ks_pval: float
    #: Variables that were clustered on for the test.
    cluster_vars: list[str] | None = None
    #: Formula for the X variables used in the test.
    x_formula: str | None = None


def aggte(
    overall_att,
    overall_se,
    aggregation_type="simple",
    event_times=None,
    att_by_event=None,
    se_by_event=None,
    critical_values=None,
    influence_func=None,
    influence_func_overall=None,
    min_event_time=None,
    max_event_time=None,
    balanced_event_threshold=None,
    estimation_params=None,
    call_info=None,
):
    """Create an aggregate treatment effect result object.

    Parameters
    ----------
    overall_att : float
        The estimated overall ATT.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}, default='simple'
        Type of aggregation performed.
    event_times : ndarray, optional
        Event/group/time values for disaggregated effects.
    att_by_event : ndarray, optional
        ATT estimates for each event time value.
    se_by_event : ndarray, optional
        Standard errors for each event time value.
    critical_values : ndarray, optional
        Critical values for confidence bands.
    influence_func : ndarray, optional
        Influence function of aggregated parameters.

        - For dynamic/group/calendar: 2D array of shape (n_units, n_events)
        - For simple: 1D array of length n_units
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D array).
    min_event_time : int, optional
        Minimum event time.
    max_event_time : int, optional
        Maximum event time.
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict, optional
        DID estimation parameters.
    call_info : dict, optional
        Information about the function call.

    Returns
    -------
    AGGTEResult
        NamedTuple containing aggregated treatment effect parameters.
    """
    if aggregation_type not in ["simple", "dynamic", "group", "calendar"]:
        raise ValueError(
            f"Invalid aggregation_type: {aggregation_type}. Must be one of 'simple', 'dynamic', 'group', 'calendar'."
        )

    if event_times is not None:
        n_events = len(event_times)
        if att_by_event is not None and len(att_by_event) != n_events:
            raise ValueError("att_by_event must have same length as event_times.")
        if se_by_event is not None and len(se_by_event) != n_events:
            raise ValueError("se_by_event must have same length as event_times.")
        if critical_values is not None and len(critical_values) != n_events:
            raise ValueError("critical_values must have same length as event_times.")

    if estimation_params is None:
        estimation_params = {}
    if call_info is None:
        call_info = {}

    return AGGTEResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type=aggregation_type,
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        critical_values=critical_values,
        influence_func=influence_func,
        influence_func_overall=influence_func_overall,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
        balanced_event_threshold=balanced_event_threshold,
        estimation_params=estimation_params,
        call_info=call_info,
    )


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
