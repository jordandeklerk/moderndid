"""Aggregate Treatment Effect Parameters Object for DDD."""

from typing import Literal, NamedTuple

import numpy as np


class DDDAggResult(NamedTuple):
    """Container for aggregated DDD treatment effect parameters.

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
