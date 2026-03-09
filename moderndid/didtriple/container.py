"""Result containers for triple difference-in-differences estimators."""

from typing import Literal, NamedTuple

import numpy as np
from scipy import stats

from moderndid.core.maketables import (
    build_coef_table,
    build_coef_table_with_ci,
    build_single_coef_table,
    ci_from_se,
    control_group_label,
    make_effect_names,
    make_group_time_names,
    n_from_first_dim,
    se_type_label,
    vcov_info_from_bootstrap,
)
from moderndid.core.result import extract_vcov_info


class DDDAggResult(NamedTuple):
    """Container for aggregated DDD treatment effect parameters.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    overall_att : float
        The estimated overall average treatment effect on the treated.
    overall_se : float
        Standard error for overall ATT.
    aggregation_type : {'simple', 'eventstudy', 'group', 'calendar'}
        Type of aggregation performed.
    egt : ndarray, optional
        Event times, groups, or calendar times depending on aggregation type.
    att_egt : ndarray, optional
        ATT estimates for each element in egt.
    se_egt : ndarray, optional
        Standard errors for each element in egt.
    crit_val : float
        Critical value for confidence intervals.
    inf_func : ndarray, optional
        Influence function matrix for disaggregated effects.
    inf_func_overall : ndarray, optional
        Influence function for the overall ATT.
    args : dict
        Arguments used for aggregation.
    """

    #: Estimated overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for overall ATT.
    overall_se: float
    #: Type of aggregation performed.
    aggregation_type: Literal["simple", "eventstudy", "group", "calendar"]
    #: Event times, groups, or calendar times depending on aggregation type.
    egt: np.ndarray | None = None
    #: ATT estimates for each element in egt.
    att_egt: np.ndarray | None = None
    #: Standard errors for each element in egt.
    se_egt: np.ndarray | None = None
    #: Critical value for confidence intervals.
    crit_val: float = 1.96
    #: Influence function matrix for disaggregated effects.
    inf_func: np.ndarray | None = None
    #: Influence function for the overall ATT.
    inf_func_overall: np.ndarray | None = None
    #: Arguments used for aggregation.
    args: dict = {}

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        alpha = float(self.args.get("alpha", 0.05))
        names = ["Overall ATT"]
        estimates = [self.overall_att]
        se = [self.overall_se]

        crit = None
        if self.egt is not None and self.att_egt is not None and self.se_egt is not None:
            z_crit = stats.norm.ppf(1 - alpha / 2)
            prefix = {"eventstudy": "Event", "group": "Group", "calendar": "Time"}.get(
                self.aggregation_type,
                "Effect",
            )
            names.extend(make_effect_names(self.egt, prefix=prefix))
            estimates.extend(np.asarray(self.att_egt, dtype=float).tolist())
            se.extend(np.asarray(self.se_egt, dtype=float).tolist())

            event_crit = np.full(len(self.egt), self.crit_val)
            crit = np.concatenate([[z_crit], event_crit])

        return build_coef_table_with_ci(names, estimates, se, alpha=alpha, critical_values=crit)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            n_obs = n_from_first_dim(self.inf_func_overall)
            if n_obs is not None:
                return n_obs
            return n_from_first_dim(self.inf_func)
        if key == "aggregation":
            return self.aggregation_type
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        if key == "control_group":
            return control_group_label(self.args.get("control_group"))
        if key == "est_method":
            return self.args.get("est_method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.args.get("yname", "Aggregated DDD ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DDD aggregation output does not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.args, bootstrap_key="boot")

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "aggregation": "Aggregation",
            "control_group": "Control Group",
            "est_method": "Estimation Method",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["aggregation", "se_type", "control_group"]
        if self.__maketables_stat__("N") is not None:
            keys.insert(0, "N")
        if self.args.get("est_method") is not None:
            keys.append("est_method")
        return keys


class DDDPanelResult(NamedTuple):
    """Container for DDD panel estimation results.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    att : float
        The DDD point estimate for the ATT.
    se : float
        Standard error of the ATT estimate.
    uci : float
        Upper bound of the 95% confidence interval.
    lci : float
        Lower bound of the 95% confidence interval.
    boots : ndarray or None
        Bootstrap draws if bootstrap inference was used.
    att_inf_func : ndarray or None
        Influence function if requested.
    did_atts : dict
        Individual DiD ATT estimates for each comparison.
    subgroup_counts : dict
        Number of units in each subgroup.
    args : dict
        Arguments used for estimation.
    """

    #: DDD point estimate for the ATT.
    att: float
    #: Standard error of the ATT estimate.
    se: float
    #: Upper bound of the 95% confidence interval.
    uci: float
    #: Lower bound of the 95% confidence interval.
    lci: float
    #: Bootstrap draws if bootstrap inference was used.
    boots: np.ndarray | None
    #: Influence function if requested.
    att_inf_func: np.ndarray | None
    #: Individual DiD ATT estimates for each comparison.
    did_atts: dict
    #: Number of units in each subgroup.
    subgroup_counts: dict
    #: Arguments used for estimation.
    args: dict

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATT", self.att, self.se, ci95l=self.lci, ci95u=self.uci)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return int(sum(self.subgroup_counts.values()))
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        if key == "est_method":
            return self.args.get("est_method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.args.get("yname", "DDD ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DDD results do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return vcov_info_from_bootstrap(is_bootstrap=bool(self.args.get("boot", False)))

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {"est_method": "Estimation Method"}

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type", "est_method"]


class DDDRCResult(NamedTuple):
    """Container for DDD repeated cross-section estimation results.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    att : float
        The DDD point estimate for the ATT.
    se : float
        Standard error of the ATT estimate.
    uci : float
        Upper bound of the 95% confidence interval.
    lci : float
        Lower bound of the 95% confidence interval.
    boots : ndarray or None
        Bootstrap draws if bootstrap inference was used.
    att_inf_func : ndarray or None
        Influence function if requested.
    did_atts : dict
        Individual DiD ATT estimates for each comparison.
    subgroup_counts : dict
        Number of observations in each subgroup.
    args : dict
        Arguments used for estimation.
    """

    #: DDD point estimate for the ATT.
    att: float
    #: Standard error of the ATT estimate.
    se: float
    #: Upper bound of the 95% confidence interval.
    uci: float
    #: Lower bound of the 95% confidence interval.
    lci: float
    #: Bootstrap draws if bootstrap inference was used.
    boots: np.ndarray | None
    #: Influence function if requested.
    att_inf_func: np.ndarray | None
    #: Individual DiD ATT estimates for each comparison.
    did_atts: dict
    #: Number of observations in each subgroup.
    subgroup_counts: dict
    #: Arguments used for estimation.
    args: dict

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATT", self.att, self.se, ci95l=self.lci, ci95u=self.uci)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return int(sum(self.subgroup_counts.values()))
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        if key == "est_method":
            return self.args.get("est_method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.args.get("yname", "DDD ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DDD results do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return vcov_info_from_bootstrap(is_bootstrap=bool(self.args.get("boot", False)))

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {"est_method": "Estimation Method"}

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type", "est_method"]


class ATTgtResult(NamedTuple):
    """Container for a single (g,t) cell from the multi-period DDD estimator."""

    #: DDD point estimate for this (g,t) cell.
    att: float
    #: Group identifier (first treatment period).
    group: int
    #: Time period.
    time: int
    #: Whether this is a post-treatment period (1) or not (0).
    post: int


class DDDMultiPeriodResult(NamedTuple):
    """Container for multi-period DDD estimation results.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    att : ndarray
        Array of ATT(g,t) point estimates.
    se : ndarray
        Array of standard errors for each ATT(g,t).
    uci : ndarray
        Array of upper confidence interval bounds.
    lci : ndarray
        Array of lower confidence interval bounds.
    groups : ndarray
        Array of treatment cohort identifiers for each estimate.
    times : ndarray
        Array of time period identifiers for each estimate.
    glist : ndarray
        Unique treatment cohorts.
    tlist : ndarray
        Unique time periods.
    inf_func_mat : ndarray
        Matrix of influence functions (n_units x n_estimates).
    n : int
        Number of units.
    args : dict
        Arguments used for estimation.
    unit_groups : ndarray
        Array of treatment group for each unit (length n).
    """

    #: Array of ATT(g,t) point estimates.
    att: np.ndarray
    #: Array of standard errors for each ATT(g,t).
    se: np.ndarray
    #: Array of upper confidence interval bounds.
    uci: np.ndarray
    #: Array of lower confidence interval bounds.
    lci: np.ndarray
    #: Array of treatment cohort identifiers for each estimate.
    groups: np.ndarray
    #: Array of time period identifiers for each estimate.
    times: np.ndarray
    #: Unique treatment cohorts.
    glist: np.ndarray
    #: Unique time periods.
    tlist: np.ndarray
    #: Matrix of influence functions (n_units x n_estimates).
    inf_func_mat: np.ndarray
    #: Number of units.
    n: int
    #: Arguments used for estimation.
    args: dict
    #: Array of treatment group for each unit.
    unit_groups: np.ndarray

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        names = make_group_time_names(self.groups, self.times, prefix="ATT")
        ci90l, ci90u = ci_from_se(self.att, self.se, alpha=0.10)
        return build_coef_table(names, self.att, self.se, ci95l=self.lci, ci95u=self.uci, ci90l=ci90l, ci90u=ci90u)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return int(self.n)
        if key == "n_cohorts":
            return len(self.glist)
        if key == "n_periods":
            return len(self.tlist)
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        if key == "control_group":
            return control_group_label(self.args.get("control_group"))
        if key == "base_period":
            return self.args.get("base_period")
        if key == "est_method":
            return self.args.get("est_method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.args.get("yname", "DDD ATT(g,t)"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DDD group-time outputs do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.args, bootstrap_key="boot")

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "n_cohorts": "Treatment Cohorts",
            "n_periods": "Time Periods",
            "control_group": "Control Group",
            "base_period": "Base Period",
            "est_method": "Estimation Method",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "n_cohorts", "n_periods", "se_type", "control_group", "base_period", "est_method"]


class ATTgtRCResult(NamedTuple):
    """Container for a single (g,t) cell from the multi-period DDD RCS estimator."""

    #: DDD point estimate for this (g,t) cell.
    att: float
    #: Group identifier (first treatment period).
    group: int
    #: Time period.
    time: int
    #: Whether this is a post-treatment period (1) or not (0).
    post: int


class DDDMultiPeriodRCResult(NamedTuple):
    """Container for multi-period DDD repeated cross-section estimation results.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Attributes
    ----------
    att : ndarray
        Array of ATT(g,t) point estimates.
    se : ndarray
        Array of standard errors for each ATT(g,t).
    uci : ndarray
        Array of upper confidence interval bounds.
    lci : ndarray
        Array of lower confidence interval bounds.
    groups : ndarray
        Array of treatment cohort identifiers for each estimate.
    times : ndarray
        Array of time period identifiers for each estimate.
    glist : ndarray
        Unique treatment cohorts.
    tlist : ndarray
        Unique time periods.
    inf_func_mat : ndarray
        Matrix of influence functions (n_obs x n_estimates).
    n : int
        Number of observations (not units, since this is RCS).
    args : dict
        Arguments used for estimation.
    unit_groups : ndarray
        Array of treatment group for each observation (length n).
    """

    #: Array of ATT(g,t) point estimates.
    att: np.ndarray
    #: Array of standard errors for each ATT(g,t).
    se: np.ndarray
    #: Array of upper confidence interval bounds.
    uci: np.ndarray
    #: Array of lower confidence interval bounds.
    lci: np.ndarray
    #: Array of treatment cohort identifiers for each estimate.
    groups: np.ndarray
    #: Array of time period identifiers for each estimate.
    times: np.ndarray
    #: Unique treatment cohorts.
    glist: np.ndarray
    #: Unique time periods.
    tlist: np.ndarray
    #: Matrix of influence functions (n_obs x n_estimates).
    inf_func_mat: np.ndarray
    #: Number of observations.
    n: int
    #: Arguments used for estimation.
    args: dict
    #: Array of treatment group for each observation.
    unit_groups: np.ndarray

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        names = make_group_time_names(self.groups, self.times, prefix="ATT")
        ci90l, ci90u = ci_from_se(self.att, self.se, alpha=0.10)
        return build_coef_table(names, self.att, self.se, ci95l=self.lci, ci95u=self.uci, ci90l=ci90l, ci90u=ci90u)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return int(self.n)
        if key == "n_cohorts":
            return len(self.glist)
        if key == "n_periods":
            return len(self.tlist)
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        if key == "control_group":
            return control_group_label(self.args.get("control_group"))
        if key == "base_period":
            return self.args.get("base_period")
        if key == "est_method":
            return self.args.get("est_method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.args.get("yname", "DDD ATT(g,t)"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DDD group-time outputs do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.args, bootstrap_key="boot")

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "n_cohorts": "Treatment Cohorts",
            "n_periods": "Time Periods",
            "control_group": "Control Group",
            "base_period": "Base Period",
            "est_method": "Estimation Method",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "n_cohorts", "n_periods", "se_type", "control_group", "base_period", "est_method"]
