"""Aggregate Treatment Effect Parameters Object for DDD."""

from typing import Literal, NamedTuple

import numpy as np

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    control_group_label,
    make_effect_names,
    n_from_first_dim,
    se_type_label,
    vcov_info_from_bootstrap,
)


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
        names = ["Overall ATT"]
        estimates = [self.overall_att]
        se = [self.overall_se]

        if self.egt is not None and self.att_egt is not None and self.se_egt is not None:
            prefix = {"eventstudy": "Event", "group": "Group", "calendar": "Time"}.get(
                self.aggregation_type,
                "Effect",
            )
            names.extend(make_effect_names(self.egt, prefix=prefix))
            estimates.extend(np.asarray(self.att_egt, dtype=float).tolist())
            se.extend(np.asarray(self.se_egt, dtype=float).tolist())

        return build_coef_table_with_ci(names, estimates, se, alpha=float(self.args.get("alpha", 0.05)))

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
        return vcov_info_from_bootstrap(
            is_bootstrap=bool(self.args.get("boot", False)),
            cluster=self.args.get("cluster"),
        )

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
