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

    Examples
    --------
    First, we compute group-time average treatment effects using the ``ddd`` function
    on multi-period data with staggered treatment adoption:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: from moderndid import ddd, agg_ddd, gen_dgp_mult_periods
           ...:
           ...: dgp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
           ...: df = dgp["data"]
           ...:
           ...: ddd_result = ddd(
           ...:     data=df,
           ...:     yname="y",
           ...:     tname="time",
           ...:     idname="id",
           ...:     gname="group",
           ...:     pname="partition",
           ...:     control_group="nevertreated",
           ...:     est_method="dr",
           ...: )
           ...: ddd_result

    Now we can aggregate these group-time effects in different ways. The "simple" aggregation
    computes an overall ATT by taking a weighted average of all post-treatment group-time ATTs:

    .. ipython::
        :okwarning:

        In [2]: simple_agg = agg_ddd(ddd_result, aggregation_type="simple")
           ...: print(simple_agg)

    The "group" aggregation computes average treatment effects separately for each treatment
    cohort (units first treated in the same period):

    .. ipython::
        :okwarning:

        In [3]: group_agg = agg_ddd(ddd_result, aggregation_type="group")
           ...: print(group_agg)

    The "eventstudy" aggregation (default) creates an event study, showing how treatment
    effects evolve relative to the treatment start date:

    .. ipython::
        :okwarning:

        In [4]: es_agg = agg_ddd(ddd_result, aggregation_type="eventstudy")
           ...: print(es_agg)

    We can also limit the event study to specific event times:

    .. ipython::
        :okwarning:

        In [5]: es_limited = agg_ddd(
           ...:     ddd_result,
           ...:     aggregation_type="eventstudy",
           ...:     min_e=-2,
           ...:     max_e=2
           ...: )
           ...: print(es_limited)

    The "calendar" aggregation computes average treatment effects by calendar time period:

    .. ipython::
        :okwarning:

        In [6]: calendar_agg = agg_ddd(ddd_result, aggregation_type="calendar")
           ...: print(calendar_agg)

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
