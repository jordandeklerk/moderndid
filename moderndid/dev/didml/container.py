"""Result containers for ML-based DiD estimators."""

from typing import Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy import stats

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    control_group_label,
    make_effect_names,
    make_group_time_names,
    se_type_label,
)
from moderndid.core.result import extract_n_obs, extract_vcov_info


class DIDMLResult(NamedTuple):
    """Container for group-time ML ATT and CATT results.

    Holds doubly-robust ML estimates of group-time average treatment effects on
    the treated together with individual-level conditional treatment effects
    (CATTs), influence functions, and minimax weights for each group-time cell.
    Implements the ``maketables`` plug-in interface for publication-quality
    tables.

    Attributes
    ----------
    groups : ndarray
        Treatment cohort (period first treated) for each group-time cell.
    times : ndarray
        Time period for each group-time cell.
    att_gt : ndarray
        Group-time average treatment effects on the treated (TAU_hat).
    se_gt : ndarray
        Standard errors for the group-time ATTs.
    critical_value : float
        Critical value for confidence intervals (pointwise normal or simultaneous).
    influence_func : ndarray
        Influence-function matrix of shape (n_units, n_cells) used for variance
        estimation and downstream aggregation.
    cates : scipy.sparse.csr_matrix
        Sparse matrix of shape (n_obs, n_cells) holding individual CATT estimates
        for the rows participating in each group-time cell.
    scores : scipy.sparse.csr_matrix
        Sparse matrix of shape (n_obs, n_cells) holding doubly-robust score
        contributions per observation per cell.
    gammas : scipy.sparse.csr_matrix
        Sparse matrix of shape (n_obs, n_cells) holding the cell-specific
        minimax weights from the CVXR-equivalent solver.
    unit_ids : ndarray
        Unit identifier for each row of the sparse CATT/score/gamma matrices.
    unit_periods : ndarray
        Time period for each row of the sparse CATT/score/gamma matrices.
    drdid_benchmark : ndarray, optional
        Doubly-robust ATT benchmark estimates for each cell, used for
        comparison when ``compute_drdid_benchmark`` is enabled.
    drdid_benchmark_se : ndarray, optional
        Standard errors for the doubly-robust benchmark estimates.
    n_units : int, optional
        Number of unique cross-sectional units.
    wald_stat : float, optional
        Wald statistic for pre-testing parallel trends from pre-treatment cells.
    wald_pvalue : float, optional
        P-value of the pre-trends Wald statistic.
    aggregate_effects : object, optional
        Aggregate treatment effects object populated by ``aggte_didml``.
    alpha : float
        Significance level (default 0.05).
    estimation_params : dict
        Dictionary of estimation metadata. Standard keys include ``yname``,
        ``control_group``, ``anticipation_periods``, ``estimation_method``,
        ``bootstrap``, ``uniform_bands``, ``base_period``, ``panel``,
        ``clustervars``, ``cluster``, ``biters``, ``random_state``, ``n_units``,
        ``n_obs``, ``alpha``, plus ML-specific entries ``nu_model``,
        ``sigma_model``, ``delta_model``, ``k_folds``, ``tune_penalty``,
        ``t_func``, ``use_gamma``, and ``zeta``.
    """

    #: Treatment cohort for each group-time cell.
    groups: np.ndarray
    #: Time period for each group-time cell.
    times: np.ndarray
    #: Group-time average treatment effects on the treated.
    att_gt: np.ndarray
    #: Standard errors for the group-time ATTs.
    se_gt: np.ndarray
    #: Critical value for confidence intervals.
    critical_value: float
    #: Influence-function matrix (n_units, n_cells).
    influence_func: np.ndarray
    #: Sparse matrix of individual CATT estimates per cell.
    cates: sp.csr_matrix
    #: Sparse matrix of doubly-robust score contributions per cell.
    scores: sp.csr_matrix
    #: Sparse matrix of minimax weights per cell.
    gammas: sp.csr_matrix
    #: Unit identifier for each sparse-matrix row.
    unit_ids: np.ndarray
    #: Time period for each sparse-matrix row.
    unit_periods: np.ndarray
    #: DRDID benchmark estimates per cell.
    drdid_benchmark: np.ndarray | None = None
    #: Standard errors for the DRDID benchmark.
    drdid_benchmark_se: np.ndarray | None = None
    #: Number of unique cross-sectional units.
    n_units: int | None = None
    #: Wald statistic for pre-trends.
    wald_stat: float | None = None
    #: P-value for the pre-trends Wald statistic.
    wald_pvalue: float | None = None
    #: Aggregate treatment effects object.
    aggregate_effects: object | None = None
    #: Significance level.
    alpha: float = 0.05
    #: Estimation parameters dictionary.
    estimation_params: dict = {}

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
        if key == "nu_model":
            return self.estimation_params.get("nu_model")
        if key == "sigma_model":
            return self.estimation_params.get("sigma_model")
        if key == "delta_model":
            return self.estimation_params.get("delta_model")
        if key == "k_folds":
            return self.estimation_params.get("k_folds")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "ATT(g,t)"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Group-time ML results do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.estimation_params)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "wald_pvalue": "Pre-trends p-value",
            "control_group": "Control Group",
            "nu_model": "Nu model",
            "sigma_model": "Sigma model",
            "delta_model": "Delta model",
            "k_folds": "Cross-fit folds",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["N", "se_type", "control_group", "nu_model", "sigma_model", "delta_model"]
        if self.wald_pvalue is not None:
            keys.insert(1, "wald_pvalue")
        return keys


class DIDMLAggResult(NamedTuple):
    """Container for aggregated ML treatment effect parameters.

    Mirrors the shape of ``AGGTEResult`` so that downstream tooling (plotting,
    maketables) can dispatch on the same field structure. Adds an optional
    doubly-robust event-time benchmark column.

    Attributes
    ----------
    overall_att : float
        Estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for the overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}
        Type of aggregation performed.
    event_times : ndarray, optional
        Event/group/time values depending on aggregation type.
    att_by_event : ndarray, optional
        ATT estimates specific to each event time value.
    se_by_event : ndarray, optional
        Standard errors specific to each event time value.
    critical_values : ndarray, optional
        Critical values for uniform confidence bands.
    influence_func : ndarray, optional
        Influence functions of the aggregated parameters.

        - For overall ATT: 1D array of length n_units.
        - For dynamic/group/calendar: 2D array of shape (n_units, n_events).
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT (1D, length n_units).
    drdid_benchmark_by_event : ndarray, optional
        Doubly-robust benchmark ATTs at each event time.
    min_event_time : int, optional
        Minimum event time for dynamic effects.
    max_event_time : int, optional
        Maximum event time for dynamic effects.
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict
        Estimation metadata. See ``DIDMLResult.estimation_params``.
    call_info : dict
        Information about the function call that created this object.
    """

    #: Estimated overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for the overall ATT.
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
    #: Influence functions of the aggregated parameters.
    influence_func: np.ndarray | None = None
    #: Influence function for the overall ATT.
    influence_func_overall: np.ndarray | None = None
    #: Doubly-robust benchmark ATTs at each event time.
    drdid_benchmark_by_event: np.ndarray | None = None
    #: Minimum event time.
    min_event_time: int | None = None
    #: Maximum event time.
    max_event_time: int | None = None
    #: Balanced event time threshold.
    balanced_event_threshold: int | None = None
    #: Estimation parameters dictionary.
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
        if key == "nu_model":
            return self.estimation_params.get("nu_model")
        if key == "sigma_model":
            return self.estimation_params.get("sigma_model")
        if key == "delta_model":
            return self.estimation_params.get("delta_model")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "Aggregated ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Aggregated ML results do not report fixed-effects formulas."""
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
            "nu_model": "Nu model",
            "sigma_model": "Sigma model",
            "delta_model": "Delta model",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["aggregation", "se_type", "control_group", "nu_model", "sigma_model", "delta_model"]
        if self.__maketables_stat__("N") is not None:
            keys.insert(0, "N")
        if self.__maketables_stat__("n_units") is not None:
            idx = keys.index("N") + 1 if "N" in keys else 0
            keys.insert(idx, "n_units")
        return keys


class BLPResult(NamedTuple):
    """Container for Best Linear Predictor (BLP) heterogeneity results.

    Stores per-event-time linear projections of dynamic CATEs onto observed
    covariates, used to characterize which features predict heterogeneous
    treatment effects.

    Attributes
    ----------
    event_times : ndarray
        Event times at which BLP regressions were fit.
    coefs : dict[str, ndarray]
        Per-covariate arrays of coefficients indexed by event time.
    ses : dict[str, ndarray]
        Per-covariate arrays of standard errors indexed by event time.
    pvalues : dict[str, ndarray]
        Per-covariate arrays of p-values indexed by event time.
    rhs_formula : str
        Right-hand side formula used for the BLP regression.
    """

    #: Event times for the BLP fits.
    event_times: np.ndarray
    #: Coefficients per covariate, per event time.
    coefs: dict[str, np.ndarray]
    #: Standard errors per covariate, per event time.
    ses: dict[str, np.ndarray]
    #: P-values per covariate, per event time.
    pvalues: dict[str, np.ndarray]
    #: Right-hand side formula used for the BLP regression.
    rhs_formula: str


class CLANResult(NamedTuple):
    """Container for Classification Analysis (CLAN) heterogeneity results.

    Stores covariate means in the top and bottom CATE quantile groups along with
    test statistics for whether high-effect and low-effect groups differ in
    their observed characteristics.

    Attributes
    ----------
    affected : list[str]
        Covariate names tested for heterogeneity.
    threshold : float
        Quantile threshold used to define top/bottom groups (e.g. 0.2 for top
        and bottom quintile).
    high_means : ndarray
        Mean of each affected covariate within the top-CATE group.
    low_means : ndarray
        Mean of each affected covariate within the bottom-CATE group.
    diffs : ndarray
        Estimated mean differences (high minus low) per covariate.
    pvalues : ndarray
        P-values for the high-vs-low mean differences.
    test_type : {'glh', 't'}
        Test variant. ``'glh'`` is the multivariate generalized linear
        hypothesis test; ``'t'`` is a Welch two-sample t-test per covariate.
    """

    #: Covariate names tested.
    affected: list[str]
    #: Quantile threshold defining top and bottom groups.
    threshold: float
    #: Mean of each affected covariate in the top-CATE group.
    high_means: np.ndarray
    #: Mean of each affected covariate in the bottom-CATE group.
    low_means: np.ndarray
    #: High-minus-low mean differences per covariate.
    diffs: np.ndarray
    #: P-values for the high-vs-low mean differences.
    pvalues: np.ndarray
    #: Test variant.
    test_type: Literal["glh", "t"]


def didml_result(
    groups,
    times,
    att_gt,
    se_gt,
    critical_value,
    influence_func,
    cates,
    scores,
    gammas,
    unit_ids,
    unit_periods,
    drdid_benchmark=None,
    drdid_benchmark_se=None,
    n_units=None,
    wald_stat=None,
    wald_pvalue=None,
    aggregate_effects=None,
    alpha=0.05,
    estimation_params=None,
):
    """Construct a ``DIDMLResult``.

    Parameters
    ----------
    groups : array_like
        Treatment cohort for each group-time cell.
    times : array_like
        Time period for each group-time cell.
    att_gt : array_like
        Group-time ATT estimates.
    se_gt : array_like
        Standard errors for the group-time ATTs.
    critical_value : float
        Critical value for confidence intervals.
    influence_func : ndarray
        Influence-function matrix of shape (n_units, n_cells).
    cates : scipy.sparse matrix
        Sparse matrix of individual CATT estimates per cell.
    scores : scipy.sparse matrix
        Sparse matrix of doubly-robust scores per cell.
    gammas : scipy.sparse matrix
        Sparse matrix of minimax weights per cell.
    unit_ids : array_like
        Unit identifier for each sparse-matrix row.
    unit_periods : array_like
        Time period for each sparse-matrix row.
    drdid_benchmark : array_like, optional
        Doubly-robust ATT benchmark estimates per cell.
    drdid_benchmark_se : array_like, optional
        Standard errors for the DRDID benchmark estimates.
    n_units : int, optional
        Number of unique cross-sectional units.
    wald_stat : float, optional
        Wald statistic for pre-trends.
    wald_pvalue : float, optional
        P-value of the pre-trends Wald statistic.
    aggregate_effects : object, optional
        Aggregate treatment effects object.
    alpha : float, default=0.05
        Significance level.
    estimation_params : dict, optional
        Estimation metadata dictionary.

    Returns
    -------
    DIDMLResult
        NamedTuple holding the ML group-time ATT/CATT result.
    """
    groups = np.asarray(groups)
    times = np.asarray(times)
    att_gt = np.asarray(att_gt)
    se_gt = np.asarray(se_gt)

    n_gt = len(groups)
    if len(times) != n_gt:
        raise ValueError("groups and times must have the same length.")
    if len(att_gt) != n_gt:
        raise ValueError("att_gt must have the same length as groups and times.")
    if len(se_gt) != n_gt:
        raise ValueError("se_gt must have the same length as groups and times.")

    for matrix, name in ((cates, "cates"), (scores, "scores"), (gammas, "gammas")):
        if not sp.issparse(matrix):
            raise TypeError(f"{name} must be a scipy.sparse matrix.")
        if matrix.shape[1] != n_gt:
            raise ValueError(f"{name} must have {n_gt} columns (one per group-time cell), got {matrix.shape[1]}.")

    unit_ids = np.asarray(unit_ids)
    unit_periods = np.asarray(unit_periods)
    if len(unit_ids) != cates.shape[0]:
        raise ValueError("unit_ids length must match the number of rows in cates.")
    if len(unit_periods) != cates.shape[0]:
        raise ValueError("unit_periods length must match the number of rows in cates.")

    if drdid_benchmark is not None:
        drdid_benchmark = np.asarray(drdid_benchmark)
        if len(drdid_benchmark) != n_gt:
            raise ValueError("drdid_benchmark must have the same length as att_gt.")
    if drdid_benchmark_se is not None:
        drdid_benchmark_se = np.asarray(drdid_benchmark_se)
        if len(drdid_benchmark_se) != n_gt:
            raise ValueError("drdid_benchmark_se must have the same length as att_gt.")

    if estimation_params is None:
        estimation_params = {}

    return DIDMLResult(
        groups=groups,
        times=times,
        att_gt=att_gt,
        se_gt=se_gt,
        critical_value=critical_value,
        influence_func=influence_func,
        cates=sp.csr_matrix(cates),
        scores=sp.csr_matrix(scores),
        gammas=sp.csr_matrix(gammas),
        unit_ids=unit_ids,
        unit_periods=unit_periods,
        drdid_benchmark=drdid_benchmark,
        drdid_benchmark_se=drdid_benchmark_se,
        n_units=n_units,
        wald_stat=wald_stat,
        wald_pvalue=wald_pvalue,
        aggregate_effects=aggregate_effects,
        alpha=alpha,
        estimation_params=estimation_params,
    )


def didml_agg(
    overall_att,
    overall_se,
    aggregation_type="simple",
    event_times=None,
    att_by_event=None,
    se_by_event=None,
    critical_values=None,
    influence_func=None,
    influence_func_overall=None,
    drdid_benchmark_by_event=None,
    min_event_time=None,
    max_event_time=None,
    balanced_event_threshold=None,
    estimation_params=None,
    call_info=None,
):
    """Construct a ``DIDMLAggResult`` from aggregation outputs.

    Parameters
    ----------
    overall_att : float
        Estimated overall ATT.
    overall_se : float
        Standard error for the overall ATT.
    aggregation_type : {'simple', 'dynamic', 'group', 'calendar'}, default='simple'
        Type of aggregation performed.
    event_times : array_like, optional
        Event/group/time values for disaggregated effects.
    att_by_event : array_like, optional
        ATT estimates for each event-time value.
    se_by_event : array_like, optional
        Standard errors for each event-time value.
    critical_values : array_like, optional
        Critical values for confidence bands.
    influence_func : ndarray, optional
        Influence function of the aggregated parameters.
    influence_func_overall : ndarray, optional
        Influence function for the overall ATT.
    drdid_benchmark_by_event : array_like, optional
        DRDID benchmark ATT at each event time.
    min_event_time : int, optional
        Minimum event time.
    max_event_time : int, optional
        Maximum event time.
    balanced_event_threshold : int, optional
        Balanced event time threshold.
    estimation_params : dict, optional
        Estimation metadata.
    call_info : dict, optional
        Information about the function call.

    Returns
    -------
    DIDMLAggResult
        NamedTuple holding the aggregated ML treatment effect.
    """
    if aggregation_type not in ("simple", "dynamic", "group", "calendar"):
        raise ValueError(
            f"Invalid aggregation_type: {aggregation_type!r}. Must be one of 'simple', 'dynamic', 'group', 'calendar'."
        )

    if event_times is not None:
        n_events = len(event_times)
        if att_by_event is not None and len(att_by_event) != n_events:
            raise ValueError("att_by_event must have same length as event_times.")
        if se_by_event is not None and len(se_by_event) != n_events:
            raise ValueError("se_by_event must have same length as event_times.")
        if critical_values is not None and len(critical_values) != n_events:
            raise ValueError("critical_values must have same length as event_times.")
        if drdid_benchmark_by_event is not None and len(drdid_benchmark_by_event) != n_events:
            raise ValueError("drdid_benchmark_by_event must have same length as event_times.")

    if estimation_params is None:
        estimation_params = {}
    if call_info is None:
        call_info = {}

    return DIDMLAggResult(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type=aggregation_type,
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        critical_values=critical_values,
        influence_func=influence_func,
        influence_func_overall=influence_func_overall,
        drdid_benchmark_by_event=drdid_benchmark_by_event,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
        balanced_event_threshold=balanced_event_threshold,
        estimation_params=estimation_params,
        call_info=call_info,
    )


def blp_result(event_times, coefs, ses, pvalues, rhs_formula):
    """Construct a ``BLPResult``.

    Parameters
    ----------
    event_times : array_like
        Event times at which BLP regressions were fit.
    coefs : dict[str, array_like]
        Coefficient arrays per covariate, one entry per event time.
    ses : dict[str, array_like]
        Standard error arrays per covariate.
    pvalues : dict[str, array_like]
        P-value arrays per covariate.
    rhs_formula : str
        Right-hand side formula used for the BLP regression.

    Returns
    -------
    BLPResult
        NamedTuple holding BLP heterogeneity output.
    """
    event_times = np.asarray(event_times)
    keys = set(coefs)
    if set(ses) != keys or set(pvalues) != keys:
        raise ValueError("coefs, ses, and pvalues must share the same keys.")

    coefs_out = {k: np.asarray(v, dtype=float) for k, v in coefs.items()}
    ses_out = {k: np.asarray(v, dtype=float) for k, v in ses.items()}
    pvalues_out = {k: np.asarray(v, dtype=float) for k, v in pvalues.items()}

    expected_len = len(event_times)
    for k in keys:
        for label, arr in (("coefs", coefs_out[k]), ("ses", ses_out[k]), ("pvalues", pvalues_out[k])):
            if len(arr) != expected_len:
                raise ValueError(f"{label}[{k!r}] must have the same length as event_times.")

    return BLPResult(
        event_times=event_times,
        coefs=coefs_out,
        ses=ses_out,
        pvalues=pvalues_out,
        rhs_formula=rhs_formula,
    )


def clan_result(
    affected,
    threshold,
    high_means,
    low_means,
    diffs,
    pvalues,
    test_type="glh",
):
    """Construct a ``CLANResult``.

    Parameters
    ----------
    affected : list[str]
        Covariate names tested.
    threshold : float
        Quantile threshold defining top and bottom CATE groups (in (0, 0.5]).
    high_means : array_like
        Means of each covariate in the top-CATE group.
    low_means : array_like
        Means of each covariate in the bottom-CATE group.
    diffs : array_like
        High-minus-low mean differences per covariate.
    pvalues : array_like
        P-values for the high-vs-low mean differences.
    test_type : {'glh', 't'}, default='glh'
        Test variant.

    Returns
    -------
    CLANResult
        NamedTuple holding CLAN heterogeneity output.
    """
    if test_type not in ("glh", "t"):
        raise ValueError(f"test_type must be 'glh' or 't', got {test_type!r}.")
    if not 0 < threshold <= 0.5:
        raise ValueError(f"threshold must be in (0, 0.5], got {threshold!r}.")

    affected = list(affected)
    high_means = np.asarray(high_means, dtype=float)
    low_means = np.asarray(low_means, dtype=float)
    diffs = np.asarray(diffs, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)

    n = len(affected)
    for arr, label in ((high_means, "high_means"), (low_means, "low_means"), (diffs, "diffs"), (pvalues, "pvalues")):
        if len(arr) != n:
            raise ValueError(f"{label} must have the same length as affected.")

    return CLANResult(
        affected=affected,
        threshold=threshold,
        high_means=high_means,
        low_means=low_means,
        diffs=diffs,
        pvalues=pvalues,
        test_type=test_type,
    )


def summary_didml(result: DIDMLResult) -> str:
    """Print summary of a DIDMLResult.

    Parameters
    ----------
    result : DIDMLResult
        The ML group-time result to summarize.

    Returns
    -------
    str
        Formatted summary string.
    """
    return str(result)


def summary_didml_agg(result: DIDMLAggResult) -> str:
    """Print summary of a DIDMLAggResult.

    Parameters
    ----------
    result : DIDMLAggResult
        The aggregated ML result to summarize.

    Returns
    -------
    str
        Formatted summary string.
    """
    return str(result)
