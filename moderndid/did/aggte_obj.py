"""Aggregate Treatment Effect Parameters Object."""

from typing import Literal, NamedTuple

import numpy as np

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    control_group_label,
    est_method_label,
    make_effect_names,
    n_from_first_dim,
    se_type_label,
    vcov_info_from_bootstrap,
)


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
        names = ["Overall ATT"]
        estimates = [self.overall_att]
        se = [self.overall_se]

        if self.event_times is not None and self.att_by_event is not None and self.se_by_event is not None:
            prefix = {"dynamic": "Event", "group": "Group", "calendar": "Time"}.get(self.aggregation_type, "Effect")
            names.extend(make_effect_names(self.event_times, prefix=prefix))
            estimates.extend(np.asarray(self.att_by_event, dtype=float).tolist())
            se.extend(np.asarray(self.se_by_event, dtype=float).tolist())

        return build_coef_table_with_ci(names, estimates, se, alpha=float(self.estimation_params.get("alpha", 0.05)))

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            n_obs = self.estimation_params.get("n_obs")
            if n_obs is not None:
                return int(n_obs)
            n = n_from_first_dim(self.influence_func_overall)
            if n is not None:
                return n
            return n_from_first_dim(self.influence_func)
        if key == "n_units":
            n_units = self.estimation_params.get("n_units")
            if n_units is not None:
                return int(n_units)
            return n_from_first_dim(self.influence_func_overall) or n_from_first_dim(self.influence_func)
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
        cluster = self.estimation_params.get("cluster")
        if cluster is None:
            cluster = self.estimation_params.get("clustervars")
        return vcov_info_from_bootstrap(
            is_bootstrap=bool(self.estimation_params.get("bootstrap", False)),
            cluster=cluster,
        )

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
