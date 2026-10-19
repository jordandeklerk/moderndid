"""Aggregate Group-Time Average Treatment Effects for Triple Differences."""

import numpy as np

from .compute_agg_ddd import compute_agg_ddd


def agg_ddd(
    ddd_result,
    type="eventstudy",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    na_rm=False,
    boot=True,
    biters=1000,
    cband=True,
    alpha=0.05,
    random_state=None,
):
    r"""Aggregate group-time average treatment effects for triple differences.

    Averages the group-time average treatment effects of :func:`ddd` into an event
    study, cohort or calendar summaries, or one overall effect [1]_, [2]_. Since the
    ineligible units of an enabling cohort are never treated, each summary weights a
    cohort by its share of the eligible treated units. The shares and the standard
    errors carry over the sampling weights and clusters of the :func:`ddd` fit. The
    standard errors also account for the estimated shares.

    See the :ref:`triple differences example <example_triple_did>` for an event study
    and an overall effect from the crop insurance data.

    Parameters
    ----------
    ddd_result : DDDMultiPeriodResult or DDDMultiPeriodRCResult
        Multi-period result of :func:`ddd`, :func:`ddd_mp`, or :func:`ddd_mp_rc`.
    type : {"simple", "eventstudy", "group", "calendar"}, default="eventstudy"
        Aggregation to compute.
    balance_e : int, optional
        Event time through which every cohort in an event study must be observed.
    min_e : float, default=-inf
        Earliest event time in an event study.
    max_e : float, default=inf
        Latest event time in the simple, group, and event-study aggregations.
    na_rm : bool, default=False
        Whether to drop group-time effects that are NaN before aggregating. Without it,
        a NaN effect stops the aggregation with an error.
    boot : bool, default=True
        Whether the multiplier bootstrap computes the standard errors.
    biters : int, default=1000
        Number of bootstrap draws.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands. Requires boot=True.
    alpha : float, default=0.05
        Significance level of the confidence intervals.
    random_state : int, Generator, optional
        Seed or generator for the bootstrap.

    Returns
    -------
    DDDAggResult
        A NamedTuple containing:

        - **overall_att**: Overall effect of the aggregation
        - **overall_se**: Standard error of the overall effect
        - **aggregation_type**: Aggregation that was computed
        - **egt**: Event times, cohorts, or periods of the effects in att_egt
        - **att_egt**: Effect at each entry of egt
        - **se_egt**: Standard error of each effect in att_egt
        - **crit_val**: Critical value of the confidence intervals or bands
        - **inf_func**: Influence functions of the effects in att_egt
        - **inf_func_overall**: Influence function of the overall effect
        - **args**: Arguments of the aggregation

        The simple aggregation leaves egt, att_egt, se_egt, and inf_func as None.

    See Also
    --------
    ddd : Compute group-time average treatment effects for triple differences.

    Notes
    -----
    Let :math:`G` be the period in which a unit is first treated. An eligible unit in
    enabling cohort :math:`S` has :math:`G = S`. An ineligible unit is never treated and
    has :math:`G = \infty`. The aggregations weight cohort :math:`g` by
    :math:`\mathbb{P}(G = g) = \mathbb{P}(S = g, Q = 1)`. Since the shares are
    estimated, the influence function of every summary that weights cohorts by them
    includes a share term, as in Corollary 2 of [1]_ and Corollary 4.2 of [2]_.

    The event study of [2]_ averages the cohorts :math:`\mathcal{G}_e` with an estimate
    at event time :math:`e = t - g`,

    .. math::

        ES(e) = \sum_{g \in \mathcal{G}_e}
        \frac{\mathbb{P}(G = g)}{\sum_{h \in \mathcal{G}_e} \mathbb{P}(G = h)}
        \, ATT(g, g + e).

    With ``balance_e`` it keeps only the cohorts observed through that event time and
    stops there. Its overall effect averages :math:`ES(e)` over the post-treatment event
    times in :math:`\mathcal{E}`,

    .. math::

        ES_{\mathrm{avg}} = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} ES(e).

    For cohort :math:`g` and the final period :math:`T`, the group aggregation is

    .. math::

        \theta_g = \frac{1}{T - g + 1} \sum_{t = g}^{T} ATT(g, t).

    Its overall effect weights :math:`\theta_g` by
    :math:`\mathbb{P}(G = g \mid G \leq T)`. The calendar aggregation in period
    :math:`t` is

    .. math::

        \theta_t = \sum_{g \leq t} \mathbb{P}(G = g \mid G \leq t) \, ATT(g, t).

    Its overall effect averages :math:`\theta_t` over the periods that have at least one
    :math:`ATT(g, t)`. The simple aggregation of [1]_ is

    .. math::

        \theta_{\mathrm{simple}} =
        \frac{\sum_{g} \sum_{t \geq g} \mathbb{P}(G = g) \, ATT(g, t)}
        {\sum_{g} \sum_{t \geq g} \mathbb{P}(G = g)}.

    A cohort treated for longer contributes more of these effects. The simple
    aggregation therefore leans toward early cohorts more than the group aggregation
    does.

    Each overall effect averages post-treatment group-time effects. When none is left,
    because those cells failed to estimate or ``max_e`` leaves them out, the overall
    effect is NaN. A warning names the cause. The group and calendar aggregations report
    only the cohorts and periods with an estimated post-treatment effect. An event study
    without one still reports its pre-treatment event times. A result whose ATT(g,t) are
    all NaN, or an event-study window without event times, stops with an error.

    Since :func:`ddd` warns when the clusters are few, agg_ddd does not warn again. The
    :ref:`triple differences background <background-tripledid>` derives the event study
    and the influence function of the estimated shares.

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
        na_rm=na_rm,
        boot=boot,
        biters=biters,
        cband=cband,
        alpha=alpha,
        random_state=random_state,
    )
