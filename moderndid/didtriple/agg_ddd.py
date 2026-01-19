"""Aggregate Group-Time Average Treatment Effects for Triple Differences."""

from __future__ import annotations

import numpy as np

from .agg_ddd_obj import DDDAggResult
from .compute_agg_ddd import compute_agg_ddd


def agg_ddd(
    ddd_result,
    aggregation_type="eventstudy",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    dropna=False,
    boot=True,
    nboot=999,
    cband=True,
    alpha=0.05,
    random_state=None,
) -> DDDAggResult:
    r"""Aggregate group-time average treatment effects for triple differences.

    Takes group-time average treatment effects from :func:`ddd_mp` and aggregates
    them into summary parameters following [2]_. The event study aggregation computes
    effects at each event time :math:`e = t - g`

    .. math::
        ES(e) = \sum_{g \in \mathcal{G}_{\mathrm{trt}}}
        \mathbb{P}(G=g \mid G+e \in [2, T]) \, ATT(g, g+e).

    The simple aggregation computes a scalar summary by averaging all
    post-treatment event study coefficients

    .. math::
        ES_{\mathrm{avg}} = \frac{1}{N_E} \sum_{e \in \mathcal{E}} ES(e),

    where :math:`\mathcal{E}` is the support of post-treatment event time
    and :math:`N_E` is its cardinality.

    The group aggregation computes average effects for each treatment cohort
    :math:`g` by averaging over all post-treatment periods for that cohort [1]_

    .. math::
        \theta_g = \frac{1}{T - g + 1} \sum_{t=g}^{T} ATT(g, t).

    The overall group effect is then a weighted average across cohorts with
    weights proportional to cohort size.

    The calendar aggregation computes average effects for each calendar
    time period :math:`t` by averaging across all cohorts treated by time :math:`t` [1]_

    .. math::
        \theta_t = \sum_{g \leq t} \mathbb{P}(G=g \mid G \leq t) \, ATT(g, t).

    Parameters
    ----------
    ddd_result : DDDMultiPeriodResult
        Result from :func:`ddd_mp` containing group-time ATTs.
    aggregation_type : {"simple", "eventstudy", "group", "calendar"}, default="eventstudy"
        Type of aggregation to perform:

        - 'simple': Weighted average of all post-treatment ATT(g,t) with weights
          proportional to group size.
        - 'eventstudy': Event-study aggregation showing effects at different lengths
          of exposure to treatment.
        - 'group': Average treatment effects across different treatment cohorts.
        - 'calendar': Average treatment effects across different calendar time periods.
    balance_e : int, optional
        If set (and aggregation_type="eventstudy"), balances the sample with respect
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
    nboot : int, default=999
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
    return compute_agg_ddd(
        ddd_result=ddd_result,
        aggregation_type=aggregation_type,
        balance_e=balance_e,
        min_e=min_e,
        max_e=max_e,
        dropna=dropna,
        boot=boot,
        nboot=nboot,
        cband=cband,
        alpha=alpha,
        random_state=random_state,
    )
