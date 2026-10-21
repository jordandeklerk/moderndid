"""Result containers for multi-period DiD estimators."""

from typing import Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy import stats

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    control_group_label,
    est_method_label,
    make_effect_names,
    make_group_time_names,
    se_type_label,
)
from moderndid.core.result import extract_clustervar, extract_n_obs, extract_vcov_info


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
        Event times for the dynamic aggregation, treatment groups for the group
        aggregation, and periods for the calendar aggregation.
    att_by_event : ndarray, optional
        ATT estimates specific to each event time value.
    se_by_event : ndarray, optional
        Standard errors specific to each event time value.
    critical_values : ndarray, optional
        Critical values for uniform confidence bands.
    influence_func : ndarray, optional
        Influence function of the aggregated parameters. The simple
        aggregation stores the overall effect's function as a vector of length
        n_units. The other aggregations store one column for each event time,
        group, or period in a matrix of shape (n_units, n_events).
        It is None for a fit on a Spark DataFrame.
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D array of length n_units).
        This is stored separately for compatibility with both aggregation types.
        Like ``influence_func``, it is None for a fit on a Spark DataFrame.
    min_event_time : int, optional
        Minimum event time (for dynamic effects).
    max_event_time : int, optional
        Maximum event time (for dynamic effects).
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict
        Estimation settings, such as ``alpha``, ``bootstrap``,
        ``uniform_bands``, ``control_group``, ``anticipation_periods``, and
        ``estimation_method``.
    call_info : dict
        Information about the function call that created this object.
    vcov_analytical : ndarray, optional
        Analytical estimator for the asymptotic variance-covariance matrix of
        the disaggregated effects, or of the overall effect for the simple
        aggregation. Dividing it by the number of units gives the covariance
        matrix of the estimates. With a cluster variable besides the unit
        identifier it sums the influence functions within clusters.
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
    #: Analytical estimator for the asymptotic variance-covariance matrix of the disaggregated effects.
    vcov_analytical: np.ndarray | None = None

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
            return se_type_label(
                bool(self.estimation_params.get("bootstrap", False)), extract_clustervar(self.estimation_params)
            )
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


class InfluenceMoments(NamedTuple):
    r"""Container for sums that stand in for the influence functions of a fit on a Spark DataFrame.

    Unit :math:`i` contributes the row

    .. math::

        z_i = \left(\psi_i', \ \omega_i \mathbf{1}\{G_i = g_1\}, \ \ldots, \
        \omega_i \mathbf{1}\{G_i = g_C\}, \ 1\right)'.

    The row stacks the unit's influence function :math:`\psi_i` for every
    group-time effect, its first-period sampling weight :math:`\omega_i` in the
    column of its treated cohort :math:`G_i`, and a constant. From these sums
    :func:`~moderndid.aggte` computes the standard errors and bootstrap draws of
    every aggregation.

    Attributes
    ----------
    cohorts : ndarray
        Treated cohorts :math:`g_1 < \cdots < g_C` of the weight columns.
    gram : ndarray
        Sum of :math:`z_i z_i'` over the units. Its shape is (L, L) for L the
        length of :math:`z_i`.
    cluster_gram : ndarray, optional
        Sum of :math:`s_c s_c'` over the clusters for :math:`s_c` the sum of
        :math:`z_i` over cluster :math:`c`, or None without a cluster variable
        besides the unit identifier.
    n_clusters : int, optional
        Number of clusters, or None without a cluster variable besides the unit identifier.
    bootstrap_sums : ndarray, optional
        Sums :math:`\sum_i v_{bi} z_i` of shape (biters, L) for :math:`v_{bi}` the
        multiplier of unit :math:`i` in draw :math:`b`, or None without bootstrap
        draws. The units of a cluster share one multiplier.
    random_state : int, optional
        Seed of the bootstrap multipliers, or None without bootstrap draws.
    """

    #: Treated cohort of each weight column.
    cohorts: np.ndarray
    #: Sum of the outer products of the unit rows.
    gram: np.ndarray
    #: Sum of the outer products of the cluster sums of the unit rows.
    cluster_gram: np.ndarray | None = None
    #: Number of clusters.
    n_clusters: int | None = None
    #: Multiplier bootstrap sums of the unit rows.
    bootstrap_sums: np.ndarray | None = None
    #: Seed of the bootstrap multipliers.
    random_state: int | None = None


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
        Analytical estimator for the asymptotic variance-covariance matrix. With
        a cluster variable besides the unit identifier it sums the influence
        function within clusters.
    se_gt : ndarray
        Standard errors for group-time ATTs. If bootstrap used, provides bootstrap-based SE.
    critical_value : float
        Critical value of the confidence intervals. It is simultaneous with a
        uniform confidence band and pointwise from the normal distribution
        otherwise.
    influence_func : ndarray
        The influence function for estimating group-time average treatment effects.
        It is None for a fit on a Spark DataFrame.
    n_units : int, optional
        The number of unique cross-sectional units.
    wald_stat : float, optional
        The Wald statistic for pre-testing the common trends assumption. A fit
        with a cluster variable besides the unit identifier has none.
    wald_pvalue : float, optional
        The p-value of the Wald statistic for pre-testing common trends.
    aggregate_effects : object, optional
        An aggregate treatment effects object.
    alpha : float
        The significance level (default 0.05).
    estimation_params : dict
        Estimation settings, such as ``control_group``,
        ``anticipation_periods``, ``estimation_method``, ``bootstrap``,
        ``uniform_bands``, and ``base_period``.
    G : ndarray, optional
        Group of each unit.
    weights_ind : ndarray, optional
        Sampling weight of each unit.
    influence_moments : InfluenceMoments, optional
        Sums over units that replace ``influence_func``, ``G``, and ``weights_ind``
        for a fit on a Spark DataFrame.
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
    #: Sums over units that replace the influence function of a fit on a Spark DataFrame.
    influence_moments: InfluenceMoments | None = None

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
            return se_type_label(
                bool(self.estimation_params.get("bootstrap", False)), extract_clustervar(self.estimation_params)
            )
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


class ATTgtResult(NamedTuple):
    """Container for a single group-time ATT estimate.

    Attributes
    ----------
    att : float
        The estimated ATT for this group-time cell.
    group : float
        The treatment group identifier (first treatment period).
    year : float
        The time period for this estimate.
    post : int
        Indicator for whether this is a post-treatment period (1) or
        pre-treatment period (0).
    """

    #: Estimated ATT for this group-time cell.
    att: float
    #: Treatment group identifier (first treatment period).
    group: float
    #: Time period for this estimate.
    year: float
    #: Indicator for post-treatment (1) or pre-treatment (0).
    post: int


class CellSpec(NamedTuple):
    """Container for the periods and comparison cohorts of one group-time cell.

    Attributes
    ----------
    group : float
        The treatment group identifier (first treatment period).
    time : float
        The time period of the cell.
    time_index : int
        Position of ``time`` in ``config.time_periods``.
    base_index : int, optional
        Position of the cell's base period in ``config.time_periods``, or None
        for a cell skipped with a warning.
    earlier_index : int, optional
        Position of the earlier of the cell's two periods, or None for a cell
        skipped with a warning. In a balanced panel the cell takes its
        covariates and sampling weights from that period.
    post : int
        Indicator for whether this is a post-treatment period (1) or
        pre-treatment period (0).
    reference : bool
        Whether the cell compares the universal base period with itself. Its
        effect and its influence function are zero.
    skip_warning : str, optional
        Warning to emit before skipping the cell, or None for a cell to keep.
    control_cohorts : ndarray
        Cohorts from ``cohort_counts`` whose units form the comparison group.
        It never holds ``group``.
    """

    #: Treatment group identifier (first treatment period).
    group: float
    #: Time period of the cell.
    time: float
    #: Position of the time period in the configured periods.
    time_index: int
    #: Position of the base period in the configured periods.
    base_index: int | None
    #: Position of the earlier of the cell's two periods.
    earlier_index: int | None
    #: Indicator for post-treatment (1) or pre-treatment (0).
    post: int
    #: Whether the cell compares the universal base period with itself.
    reference: bool
    #: Warning to emit before skipping the cell.
    skip_warning: str | None
    #: Cohorts whose units form the comparison group.
    control_cohorts: np.ndarray


class ComputeATTgtResult(NamedTuple):
    """Container for compute_att_gt results.

    Attributes
    ----------
    attgt_list : list[ATTgtResult]
        List of group-time ATT results, one per (group, time) cell.
    influence_functions : scipy.sparse.csr_matrix
        Sparse matrix of influence functions with shape
        (n_units, n_group_time_cells).
    """

    #: List of group-time ATT results, one per (group, time) cell.
    attgt_list: list[ATTgtResult]
    #: Sparse matrix of influence functions.
    influence_functions: sp.csr_matrix


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
    vcov_analytical=None,
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
        Influence function of the aggregated parameters. The simple
        aggregation passes the overall effect's function as a vector of length
        n_units. The other aggregations pass one column for each event time,
        group, or period in a matrix of shape (n_units, n_events).
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
    vcov_analytical : ndarray, optional
        Analytical asymptotic variance-covariance matrix of the disaggregated
        effects, or of the overall effect for the simple aggregation.

    Returns
    -------
    AGGTEResult
        NamedTuple containing:

        - **overall_att**: Overall average treatment effect on the treated
        - **overall_se**: Standard error of the overall effect
        - **aggregation_type**: The aggregation performed
        - **event_times**: Event times, groups, or periods of the disaggregated effects
        - **att_by_event**: Effect at each event time, group, or period
        - **se_by_event**: Standard error of each disaggregated effect
        - **critical_values**: Critical value of the confidence band for each disaggregated effect
        - **influence_func**: Influence functions of the disaggregated effects, or of the overall effect for 'simple'
        - **influence_func_overall**: Influence function of the overall effect
        - **min_event_time**: Lower end of an event study's window, or None
        - **max_event_time**: Upper end of an event study's window, or None
        - **balanced_event_threshold**: Event time that balanced an event study, or None
        - **estimation_params**: Dictionary of the estimation settings, empty when none are given
        - **call_info**: Dictionary describing the call, empty when none is given
        - **vcov_analytical**: Asymptotic variance-covariance matrix of the disaggregated effects, or of the overall
          effect for 'simple'
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
        vcov_analytical=vcov_analytical,
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
    influence_moments=None,
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
    influence_moments : InfluenceMoments, optional
        Sums over units that replace the influence function of a fit on a Spark DataFrame.

    Returns
    -------
    MPResult
        NamedTuple containing:

        - **groups**: Group of each group-time effect as an array
        - **times**: Time period of each group-time effect as an array
        - **att_gt**: Group-time average treatment effects as an array
        - **vcov_analytical**: Analytical variance-covariance matrix
        - **se_gt**: Standard error of each group-time effect as an array
        - **critical_value**: Critical value of the confidence intervals
        - **influence_func**: Influence functions of the group-time effects
        - **n_units**: Number of cross-sectional units, or None
        - **wald_stat**: Wald statistic of the pre-test, or None
        - **wald_pvalue**: P-value of the Wald pre-test, or None
        - **aggregate_effects**: Aggregate treatment effects object, or None
        - **alpha**: Significance level
        - **estimation_params**: Dictionary of the estimation settings, empty when none are given
        - **G**: Group of each unit, or None
        - **weights_ind**: Sampling weight of each unit, or None
        - **influence_moments**: Sums over units for a fit on a Spark DataFrame, or None
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
        influence_moments=influence_moments,
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
        NamedTuple containing:

        - **cvm_stat**: Cramer von Mises test statistic
        - **cvm_boots**: Bootstrapped Cramer von Mises statistics as an array, or None
        - **cvm_critval**: Cramer von Mises critical value
        - **cvm_pval**: P-value of the Cramer von Mises test
        - **ks_stat**: Kolmogorov-Smirnov test statistic
        - **ks_boots**: Bootstrapped Kolmogorov-Smirnov statistics as an array, or None
        - **ks_critval**: Kolmogorov-Smirnov critical value
        - **ks_pval**: P-value of the Kolmogorov-Smirnov test
        - **cluster_vars**: Variables the test clustered on, or None
        - **x_formula**: Formula of the covariates in the test, or None
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
    """Return the formatted summary of an ``MPResult``.

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
    """Return the formatted summary of an ``MPPretestResult``.

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
