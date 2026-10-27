"""Aggregate Group-Time Average Treatment Effects."""

import numpy as np

from .compute_aggte import compute_aggte


def aggte(
    MP,
    type="group",
    balance_e=None,
    min_e=-np.inf,
    max_e=np.inf,
    na_rm=False,
    boot=None,
    biters=None,
    cband=None,
    alpha=None,
    clustervars=None,
    random_state=None,
):
    r"""Aggregate group-time average treatment effects.

    Averages the group-time average treatment effects of :func:`~moderndid.att_gt` into
    the summaries of Callaway and Sant'Anna (2021) [1]_. An event study follows the
    effect over the time since treatment. The group and calendar aggregations average it
    within each group or period. Every aggregation also reports an overall effect. The
    standard errors combine the influence functions of the group-time effects and
    account for the estimated weights.

    A fit on a Spark DataFrame keeps sums over its units in place of their influence
    functions. From those sums aggte computes the same analytic standard errors as for
    an in-memory fit but returns no influence functions. The bootstrap reuses the draws
    that the fit stored. A call that asks for more draws, another seed, or draws by unit
    from a clustered fit therefore raises an error.

    See the :ref:`staggered DiD example <example_staggered_did>` for event study,
    cohort, and overall aggregations of the minimum wage effects.

    Parameters
    ----------
    MP : MPResult
        The result of :func:`~moderndid.att_gt`.
    type : {'simple', 'dynamic', 'group', 'calendar'}, default='group'
        The aggregation. 'dynamic' averages the effects at each event time, 'group'
        within each treatment group, and 'calendar' within each period. 'simple' averages
        all post-treatment effects with weights proportional to group size.
    balance_e : int, optional
        The event time to balance an event study on. The event study then keeps only the
        groups observed through that event time and stops there.
    min_e : float, default=-inf
        The earliest event time in an event study.
    max_e : float, default=inf
        The latest event time in an event study and in the post-treatment effects that
        the simple and group aggregations average.
    na_rm : bool, default=False
        Whether to drop group-time effects that are NaN before aggregating. Without it,
        a NaN effect stops the aggregation with an error.
    boot : bool, optional
        Whether to compute the standard errors with the multiplier bootstrap. Defaults
        to the setting of the att_gt fit.
    biters : int, optional
        The number of bootstrap draws. Defaults to the setting of the att_gt fit.
    cband : bool, optional
        Whether to report a uniform confidence band over the disaggregated effects. It
        does not require ``boot=True``. Defaults to the setting of the att_gt fit.
    alpha : float, optional
        The significance level. Defaults to the level of the att_gt fit.
    clustervars : list[str], optional
        Variables to cluster the standard errors on, each the idname or one of the
        clustervars of the att_gt fit. An empty list, or the idname alone, gives
        unit-level inference. Defaults to the clustervars of the att_gt fit.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible results
        across calls.

    Returns
    -------
    AGGTEResult
        Object containing the aggregated treatment effect results:

        - **overall_att**: Overall average treatment effect on the treated
        - **overall_se**: Standard error of the overall effect
        - **aggregation_type**: The aggregation that ``type`` selected
        - **event_times**: Event times, groups, or periods of the disaggregated effects
        - **att_by_event**: Effect at each event time, group, or period
        - **se_by_event**: Standard error of each disaggregated effect
        - **critical_values**: Critical value of the confidence band for each disaggregated effect
        - **influence_func**: Influence functions of the disaggregated effects, or of the overall effect for 'simple'
        - **influence_func_overall**: Influence function of the overall effect
        - **min_event_time**: Lower end of an event study's window, or None without one
        - **max_event_time**: Upper end of an event study's window, or None without one
        - **balanced_event_threshold**: The ``balance_e`` that balanced an event study, or None
        - **estimation_params**: Dictionary of the estimation settings
        - **call_info**: Dictionary describing the call that produced the result
        - **vcov_analytical**: Asymptotic variance-covariance matrix of the disaggregated effects, or of the overall
          effect for 'simple'

    See Also
    --------
    att_gt : Compute group-time average treatment effects.

    Notes
    -----
    Let :math:`\mathcal{G}` be the set of treatment groups, :math:`\mathcal{T}` the
    final period, and :math:`G` the period in which a unit is first treated. The dynamic
    aggregation, or event study, averages the groups observed :math:`e = t - g` periods
    after treatment,

    .. math::

        \theta_{es}(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \le \mathcal{T}\}
        P(G = g \mid G + e \le \mathcal{T}) \, ATT(g, g + e).

    The group aggregation averages each group :math:`\tilde{g}` over its
    post-treatment periods,

    .. math::

        \theta_{sel}(\tilde{g}) = \frac{1}{\mathcal{T} - \tilde{g} + 1}
        \sum_{t=\tilde{g}}^{\mathcal{T}} ATT(\tilde{g}, t).

    Its overall effect weights these group effects by the distribution of treatment
    timing,

    .. math::

        \theta_{sel}^O = \sum_{g \in \mathcal{G}} \theta_{sel}(g) \,
        P(G = g \mid G \le \mathcal{T}).

    The calendar aggregation averages the groups treated by each period
    :math:`\tilde{t}`,

    .. math::

        \theta_{c}(\tilde{t}) = \sum_{g \in \mathcal{G}} \mathbf{1}\{\tilde{t} \ge g\}
        P(G = g \mid G \le \tilde{t}) \, ATT(g, \tilde{t}).

    The overall effects of the dynamic and calendar aggregations give every
    post-treatment event time or period the same weight. The simple aggregation
    averages every post-treatment group-time effect with weights proportional to group
    size,

    .. math::

        \theta_{W}^O = \frac{1}{\kappa} \sum_{g \in \mathcal{G}} \sum_{t=2}^{\mathcal{T}}
        \mathbf{1}\{g \le t\} \, ATT(g, t) \, P(G = g \mid G \le \mathcal{T}),

    where :math:`\kappa` makes the weights sum to one. A group contributes one term for
    each period after it is treated. Groups treated earlier therefore receive more weight
    than in :math:`\theta_{sel}^O`. The :ref:`staggered DiD background
    <background-did>` develops these weights and the influence function of an
    aggregate.

    Each overall effect averages post-treatment group-time effects. When none is left,
    because those cells failed to estimate or ``max_e`` leaves them out, the overall
    effect is NaN. A warning names the cause. The group and calendar aggregations report
    only the groups and periods with an estimated post-treatment effect. An event study
    without one still reports its pre-treatment event times. An att_gt result without
    group-time effects, or an event-study window without event times, stops with an
    error.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    valid_types = ("simple", "dynamic", "group", "calendar")
    if type not in valid_types:
        raise ValueError(f"type='{type}' is not valid. Must be one of: 'simple', 'dynamic', 'group', 'calendar'.")
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if biters is not None and (not isinstance(biters, int) or biters < 1):
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if balance_e is not None and (not isinstance(balance_e, int) or balance_e < 0):
        raise ValueError(f"balance_e={balance_e} is not valid. Must be a non-negative integer.")
    if min_e > max_e:
        raise ValueError(f"min_e={min_e} must be less than or equal to max_e={max_e}.")
    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")

    call_info = {
        "function": f"aggte(MP, type='{type}')",
    }

    result = compute_aggte(
        multi_period_result=MP,
        aggregation_type=type,
        balance_e=balance_e,
        min_e=min_e,
        max_e=max_e,
        na_rm=na_rm,
        bootstrap=boot,
        biters=biters,
        confidence_band=cband,
        alpha=alpha,
        clustervars=clustervars,
        random_state=random_state,
    )

    result.call_info.update(call_info)

    return result
