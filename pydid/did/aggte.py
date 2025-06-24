"""Aggregate Group-Time Average Treatment Effects."""

from __future__ import annotations

from .compute_aggte import compute_aggte


def aggte(
    MP,
    type="group",
    balance_e=None,
    min_e=-float("inf"),
    max_e=float("inf"),
    na_rm=False,
    bstrap=None,
    biters=None,
    cband=None,
    alp=None,
    clustervars=None,
):
    """Aggregate Group-Time Average Treatment Effects.

    Takes group-time average treatment effects and aggregates them into a smaller
    number of parameters. There are several possible aggregations including
    "simple", "dynamic", "group", and "calendar."

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
    bstrap : bool, optional
        Boolean for whether or not to compute standard errors using the
        multiplier bootstrap. If standard errors are clustered, then one must
        set bstrap=True. Default is value set in the MP object. If bstrap is
        False, then analytical standard errors are reported.
    biters : int, optional
        The number of bootstrap iterations to use. The default is the value
        set in the MP object, and this is only applicable if bstrap=True.
    cband : bool, optional
        Boolean for whether or not to compute a uniform confidence band that
        covers all of the group-time average treatment effects with fixed
        probability 1-alp. In order to compute uniform confidence bands,
        bstrap must also be set to True. The default is the value set in
        the MP object.
    alp : float, optional
        The significance level, default is value set in the MP object.
    clustervars : list[str], optional
        A vector of variables to cluster on. At most, there can be two
        variables (otherwise will throw an error) and one of these must be
        the same as idname which allows for clustering at the individual
        level. Default is the variables set in the MP object.

    Returns
    -------
    AGGTEResult
        An AGGTEobj object that holds the results from the aggregation.

    References
    ----------
    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences
           with multiple time periods. Journal of Econometrics, 225(2), 200-230.
    """
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
        bootstrap=bstrap,
        bootstrap_iterations=biters,
        confidence_band=cband,
        alpha=alp,
        clustervars=clustervars,
    )

    result.call_info.update(call_info)

    return result
