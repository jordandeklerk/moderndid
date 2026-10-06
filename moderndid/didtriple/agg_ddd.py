"""Aggregate Group-Time Average Treatment Effects for Triple Differences."""

from __future__ import annotations

import numpy as np

from .compute_agg_ddd import compute_agg_ddd
from .container import DDDAggResult


def agg_ddd(
    ddd_result,
    type="eventstudy",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    dropna=False,
    boot=True,
    biters=1000,
    cband=True,
    alpha=0.05,
    random_state=None,
) -> DDDAggResult:
    r"""Aggregate group-time average treatment effects for triple differences.

    Takes the full set of group-time average treatment effects from ``ddd`` and
    aggregates them into interpretable summary measures, following [1]_ and [2]_.
    Different aggregation schemes answer different policy questions about
    treatment effect heterogeneity.

    Let :math:`\mathcal{G}_{\mathrm{trt}}` denote the set of treatment cohorts,
    :math:`T` the final time period, and :math:`ATT(g,t)` the group-time average
    treatment effect for cohort :math:`g` at time :math:`t`.

    The event-study aggregation reveals how effects evolve with exposure time
    :math:`e = t - g`

    .. math::

        ES(e) = \sum_{g \in \mathcal{G}_{\mathrm{trt}}}
        \mathbb{P}(G=g \mid G+e \in [2, T]) \, ATT(g, g+e).

    The simple aggregation computes an overall summary by averaging event-study
    coefficients across post-treatment periods

    .. math::

        ES_{\mathrm{avg}} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} ES(e),

    where :math:`\mathcal{E}` is the support of post-treatment event times.

    Group-specific aggregation averages effects over time for each treatment
    cohort :math:`g`

    .. math::

        \theta_g = \frac{1}{T - g + 1} \sum_{t=g}^{T} ATT(g, t).

    Calendar-time aggregation averages across treated cohorts within each period
    :math:`t`

    .. math::

        \theta_t = \sum_{g \leq t} \mathbb{P}(G=g \mid G \leq t) \, ATT(g, t).

    When :func:`ddd` used sampling weights, each unit counts toward these cohort
    shares with its weight.

    See the :ref:`triple differences example <example_triple_did>` for an event study
    and an overall effect from the crop insurance data.

    Parameters
    ----------
    ddd_result : DDDMultiPeriodResult
        Result from :func:`ddd_mp` containing group-time ATTs.
    type : {"simple", "eventstudy", "group", "calendar"}, default="eventstudy"
        Type of aggregation to perform:

        - 'simple': Weighted average of all post-treatment ATT(g,t) with weights
          proportional to group size.
        - 'eventstudy': Event-study aggregation showing effects at different lengths
          of exposure to treatment.
        - 'group': Average treatment effects across different treatment cohorts.
        - 'calendar': Average treatment effects across different calendar time periods.
    balance_e : int, optional
        If set (and type="eventstudy"), balances the sample with respect
        to event time. For example, if balance_e=2, groups not exposed for at least
        3 periods (e=0, 1, 2) are dropped.
    min_e : float, default=-inf
        Minimum event time to include in eventstudy aggregation.
    max_e : float, default=inf
        Maximum event time to include in aggregation.
    dropna : bool, default=False
        Whether to remove NA values before aggregation.
    boot : bool, default=True
        Whether to compute standard errors using the multiplier bootstrap.
    biters : int, default=1000
        Number of bootstrap iterations.
    cband : bool, default=True
        Whether to compute uniform confidence bands. Requires boot=True.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    random_state : int, Generator, optional
        Controls randomness of the bootstrap.

    Returns
    -------
    DDDAggResult
        Aggregated treatment effect results containing:

        - overall_att: Overall aggregated ATT
        - overall_se: Standard error for overall ATT
        - aggregation_type: Type of aggregation performed
        - egt: Event times, groups, or calendar times
        - att_egt: ATT estimates for each element in egt
        - se_egt: Standard errors for each element in egt
        - crit_val: Critical value for confidence intervals
        - inf_func: Influence function matrix
        - inf_func_overall: Influence function for overall ATT

    See Also
    --------
    ddd : Compute group-time average treatment effects for triple differences.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. C. (2021).
           *Difference-in-differences with multiple time periods.*
           Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001

    .. [2] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
           *Better Understanding Triple Differences Estimators.*
           arXiv preprint arXiv:2505.09942.
           https://arxiv.org/abs/2505.09942
    """
    valid_types = ("simple", "eventstudy", "group", "calendar")
    if type not in valid_types:
        raise ValueError(f"type='{type}' is not valid. Must be one of: 'simple', 'eventstudy', 'group', 'calendar'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if balance_e is not None and (not isinstance(balance_e, int) or balance_e < 0):
        raise ValueError(f"balance_e={balance_e} is not valid. Must be a non-negative integer.")
    if min_e > max_e:
        raise ValueError(f"min_e={min_e} must be less than or equal to max_e={max_e}.")

    return compute_agg_ddd(
        ddd_result=ddd_result,
        aggregation_type=type,
        balance_e=balance_e,
        min_e=min_e,
        max_e=max_e,
        dropna=dropna,
        boot=boot,
        biters=biters,
        cband=cband,
        alpha=alpha,
        random_state=random_state,
    )
