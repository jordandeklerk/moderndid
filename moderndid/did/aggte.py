"""Aggregate Group-Time Average Treatment Effects."""

from __future__ import annotations

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
    alp=None,
    clustervars=None,
    random_state=None,
):
    r"""Aggregate group-time average treatment effects.

    Takes the full set of group-time average treatment effects from ``att_gt``
    and aggregates them into interpretable summary measures, following Callaway
    and Sant'Anna (2021) [1]_. Different aggregation schemes answer different
    policy questions about treatment effect heterogeneity.

    Let :math:`\mathcal{G}` denote the set of treatment groups, :math:`\mathcal{T}`
    the final time period, :math:`G` the random variable for treatment timing,
    and :math:`ATT(g,t)` the group-time average treatment effect for group
    :math:`g` at time :math:`t`.

    The event-study or dynamic aggregation reveals how effects evolve with
    exposure time :math:`e = t - g`. The event-study parameter averages effects
    across groups observed :math:`e` periods after treatment

    .. math::

        \theta_{es}(e) = \sum_{g \in \mathcal{G}} \mathbf{1}\{g + e \le \mathcal{T}\}
        P(G = g \mid G + e \le \mathcal{T}) \, ATT(g, g + e).

    Group-specific aggregation averages effects over time for each treatment
    cohort :math:`\tilde{g}`, so it shows whether early and late adopters
    experience different effects

    .. math::

        \theta_{sel}(\tilde{g}) = \frac{1}{\mathcal{T} - \tilde{g} + 1}
        \sum_{t=\tilde{g}}^{\mathcal{T}} ATT(\tilde{g}, t).

    The overall effect the group aggregation reports weights these cohort
    effects by the distribution of treatment timing

    .. math::

        \theta_{sel}^O = \sum_{g \in \mathcal{G}} \theta_{sel}(g) \,
        P(G = g \mid G \le \mathcal{T}).

    Calendar-time aggregation averages across treated groups within each period
    :math:`\tilde{t}`, so it shows how effects vary with time-specific factors

    .. math::

        \theta_{c}(\tilde{t}) = \sum_{g \in \mathcal{G}} \mathbf{1}\{\tilde{t} \ge g\}
        P(G = g \mid G \le \tilde{t}) \, ATT(g, \tilde{t}).

    The simple aggregation averages every post-treatment group-time effect with
    weights proportional to group size

    .. math::

        \theta_{W}^O = \frac{1}{\kappa} \sum_{g \in \mathcal{G}} \sum_{t=2}^{\mathcal{T}}
        \mathbf{1}\{g \le t\} \, ATT(g, t) \, P(G = g \mid G \le \mathcal{T}),

    where :math:`\kappa` makes the weights sum to one. A group contributes one
    term for each period after it is treated, so groups treated earlier receive
    more weight than in :math:`\theta_{sel}^O`.

    See the :ref:`staggered DiD example <example_staggered_did>` for event study,
    cohort, and overall aggregations of the minimum wage effects.

    Parameters
    ----------
    MP : MPResult
        An MP object (i.e., the results of the att_gt() method).
    type : {'simple', 'dynamic', 'group', 'calendar'}, default='group'
        Which type of aggregated treatment effect parameter to compute:

        - 'simple': Computes a weighted average of all group-time average
          treatment effects with weights proportional to group size.
        - 'dynamic': Computes average effects across different lengths of
          exposure to the treatment (similar to an event study).
        - 'group': Computes average treatment effects across different groups.
        - 'calendar': Computes average treatment effects across different
          time periods.
    balance_e : int, optional
        If set (and if one computes dynamic effects), it balances the sample
        with respect to event time. For example, if balance_e=2, aggte will
        drop groups that are not exposed to treatment for at least three
        periods (the initial period when e=0 as well as the next two periods
        when e=1 and e=2). This ensures that the composition of groups does
        not change when event time changes.
    min_e : float, default=-inf
        For event studies, this is the smallest event time to compute dynamic
        effects for. By default, min_e = -Inf so that effects at all lengths
        of exposure are computed.
    max_e : float, default=inf
        For event studies, this is the largest event time to compute dynamic
        effects for. By default, max_e = Inf so that effects at all lengths
        of exposure are computed.
    na_rm : bool, default=False
        Logical value if we are to remove missing values from analyses.
    boot : bool, optional
        Boolean for whether or not to compute standard errors using the
        multiplier bootstrap. If standard errors are clustered, then one must
        set boot=True. Default is value set in the MP object. If boot is
        False, then analytical standard errors are reported.
    biters : int, optional
        The number of bootstrap iterations to use. The default is the value
        set in the MP object, and this is only applicable if boot=True.
    cband : bool, optional
        Boolean for whether or not to compute a uniform confidence band that
        covers all of the group-time average treatment effects with fixed
        probability 1-alp. In order to compute uniform confidence bands,
        boot must also be set to True. The default is the value set in
        the MP object.
    alp : float, optional
        The significance level, default is value set in the MP object.
    clustervars : list[str], optional
        A vector of variables to cluster on. At most, there can be two
        variables (otherwise will throw an error) and one of these must be
        the same as idname which allows for clustering at the individual
        level. Default is the variables set in the MP object.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.

    Returns
    -------
    AGGTEResult
        An AGGTEobj object that holds the results from the aggregation.

    See Also
    --------
    att_gt : Compute group-time average treatment effects.

    References
    ----------
    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    valid_types = ("simple", "dynamic", "group", "calendar")
    if type not in valid_types:
        raise ValueError(f"type='{type}' is not valid. Must be one of: 'simple', 'dynamic', 'group', 'calendar'.")
    if alp is not None and not 0 < alp < 1:
        raise ValueError(f"alp={alp} is not valid. Must be between 0 and 1 (exclusive).")
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
        dropna=na_rm,
        bootstrap=boot,
        bootstrap_iterations=biters,
        confidence_band=cband,
        alpha=alp,
        clustervars=clustervars,
        random_state=random_state,
    )

    result.call_info.update(call_info)

    return result
