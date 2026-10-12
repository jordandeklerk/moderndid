"""Event-study aggregation of group-time ATTs and per-unit CATTs."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats as _stats

from .compute import bootstrap_se, simultaneous_critical_value
from .container import didml_agg


def aggte_didml(result, type="dynamic", alpha=None):
    r"""Aggregate ML group-time ATTs to a dynamic event-study summary.

    Implements the event-study aggregation of [1]_. It extends the aggregation
    scheme of Callaway and Sant'Anna [2]_ to the doubly robust ML setting.

    The effect at each event time averages the group-time ATTs that sit the
    same number of periods from treatment. Cohorts enter in proportion to
    their shares of the units. A fit with sampling weights uses weighted
    shares. The overall summary is the simple mean of the post-treatment
    event-time effects.

    The standard errors account for the estimated cohort shares. A fit with
    bootstrap draws aggregates each draw with that draw's own shares and
    takes the spread of the aggregated draws. A fit without them adds the
    shares' influence functions to the cells' influence functions.

    A bootstrap draw can miss a small cohort entirely. That cohort then
    drops out of the draw's average at each event time. When a draw holds
    none of the cohorts at an event time, the standard errors of that event
    time and of the overall summary leave the draw out.

    Parameters
    ----------
    result : DIDMLResult
        Output of :func:`didml`.
    type : {'dynamic'}, default='dynamic'
        Aggregation scheme. Only ``'dynamic'`` is supported in the current
        release; ``'simple'``, ``'group'``, and ``'calendar'`` raise
        :class:`NotImplementedError`.
    alpha : float, optional
        Significance level for confidence bands. Defaults to
        ``result.alpha``.

    Returns
    -------
    DIDMLAggResult
        Object containing aggregated event-study results:

        - **overall_att**: Simple mean of post-treatment event-time ATTs
        - **overall_se**: Standard error for ``overall_att``
        - **aggregation_type**: Type of aggregation performed (always ``'dynamic'``)
        - **event_times**: Array of integer event times :math:`e = t - g`
        - **att_by_event**: Per-event-time aggregate :math:`\theta(e)`
        - **se_by_event**: Standard error per event time
        - **critical_values**: Critical values per event time, simultaneous across event times for a bootstrap
          fit with uniform bands and the standard normal quantile at :math:`1 - \alpha/2` otherwise
        - **influence_func**: Aggregate influence-function matrix of shape ``(n_units, n_event_times)``
        - **influence_func_overall**: Influence-function vector of length ``n_units`` for ``overall_att``
        - **drdid_benchmark_by_event**: Doubly robust benchmark aggregated to event time when
          ``result.drdid_benchmark`` is present
        - **min_event_time**: Smallest event time in ``event_times``
        - **max_event_time**: Largest event time in ``event_times``
        - **balanced_event_threshold**: Balanced event-time threshold (``None`` for this aggregation)
        - **estimation_params**: Dictionary with estimation details carried over from ``result``
        - **call_info**: Empty dictionary that keeps the field layout of ``AGGTEResult``

    Examples
    --------
    Run :func:`didml` and aggregate the group-time ATTs to event time:

    .. ipython::
        :okwarning:

        In [1]: from moderndid import didml, aggte_didml, gen_did_scalable
           ...:
           ...: data = gen_did_scalable(
           ...:     n=300, n_periods=4, n_cohorts=2,
           ...:     n_covariates=4, random_state=0,
           ...: )["data"].to_pandas()
           ...:
           ...: res = didml(
           ...:     data, yname="y", tname="time", idname="id", gname="group",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4",
           ...:     k_folds=5, boot=False, random_state=0,
           ...: )
           ...: agg = aggte_didml(res, type="dynamic")
           ...: print(agg)

    Notes
    -----
    The general aggregation has the form

    .. math::

        \theta = \sum_{g \in \mathcal{G}} \sum_{t=2}^{T} w(g, t)\,
            \mathrm{ATT}(g, t),

    where the weight function :math:`w(g, t)` selects which form of dynamic
    treatment effect the parameter targets. The event-study weights are

    .. math::

        w(g, t) = \mathbb{1}\{g + e \le T\}\,
            \mathbb{1}\{t - g = e\}\,
            P(G = g \mid G + e \le T),

    where :math:`e = t - g` is the length of exposure to treatment. Plugging
    the weights back in collapses the double sum to the per-event-time
    aggregate in ``att_by_event``

    .. math::

        \theta(e) = \sum_{g \in \mathcal{G}}
            \mathbb{1}\{g + e \le T\}\,
            P(G = g \mid G + e \le T)\,
            \mathrm{ATT}(g, g + e).

    The overall summary in ``overall_att`` is the simple mean of the
    post-treatment event-time effects

    .. math::

        \bar\theta = \frac{1}{|\mathcal{E}_+|} \sum_{e \ge 0} \theta(e).

    This summary follows the overall event-study convention of [2]_.

    The weights use the estimated shares
    :math:`\hat p_g = \sum_i w_i \mathbb{1}\{G_i = g\} / \sum_i w_i` with
    sampling weights :math:`w_i`. Without bootstrap draws, the influence
    function of :math:`\hat\theta(e)` for unit :math:`i` is

    .. math::

        \psi_i^{\theta(e)} = \sum_{g \in \mathcal{G}_e} \hat w_g(e)\,
            \psi_i^{(g, g + e)}
            + \sum_{g \in \mathcal{G}_e} \widehat{\mathrm{ATT}}(g, g + e)\,
            \psi_i^{w_g(e)},

    where :math:`\mathcal{G}_e` holds the cohorts observed at event time
    :math:`e`, :math:`\psi_i^{(g, t)}` is the cell influence function in
    ``result.influence_func``, and the share term is

    .. math::

        \psi_i^{w_g(e)} = \frac{\psi_i^{p_g}}{P_e}
            - \frac{\hat p_g}{P_e^2} \sum_{g' \in \mathcal{G}_e} \psi_i^{p_{g'}},
        \qquad
        \psi_i^{p_g} = \frac{w_i}{\bar w}
            \bigl(\mathbb{1}\{G_i = g\} - \hat p_g\bigr),
        \qquad
        P_e = \sum_{g \in \mathcal{G}_e} \hat p_g.

    See Also
    --------
    didml : Compute group-time ATTs and CATTs with cross-fitted ML nuisances.
    dynamic_cates : Aggregate per-unit CATTs or doubly robust scores to event time.

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962

    .. [2] Callaway, B., & Sant'Anna, P. H. C. (2021).
           "Difference-in-differences with multiple time periods."
           Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    if type != "dynamic":
        raise NotImplementedError(f"Only type='dynamic' is supported; got {type!r}.")

    shares = cohort_shares(result)
    indices, weights = _event_time_weights(result.groups, result.times, shares)
    event_times = np.array(sorted(indices.keys()), dtype=int)

    att_by_event = _aggregate_to_event_time(result.att_gt, indices, weights)
    inf_by_event = _aggregate_influence_func(result.influence_func, indices, weights)
    if result.G is not None:
        inf_by_event += _share_influence_func(result, shares, indices)

    n_units = int(result.n_units)
    post_mask = event_times >= 0
    overall_att = float(att_by_event[post_mask].mean()) if post_mask.any() else float("nan")
    overall_inf = inf_by_event[:, post_mask].mean(axis=1) if post_mask.any() else np.zeros(n_units)

    alpha = float(alpha if alpha is not None else result.alpha)
    z_crit = float(_stats.norm.ppf(1 - alpha / 2))
    critical_values = np.full(len(event_times), z_crit)

    overall_se = float("nan")
    if result.att_gt_draws is not None:
        event_draws = _aggregate_draws(result, indices)
        se_by_event = bootstrap_se(event_draws)
        if post_mask.any():
            overall_se = float(bootstrap_se(event_draws[:, post_mask].mean(axis=1)[:, None])[0])
        if result.estimation_params.get("uniform_bands", False):
            band = simultaneous_critical_value(event_draws, att_by_event, se_by_event, alpha)
            critical_values = np.full(len(event_times), band)
    else:
        se_by_event = np.sqrt((inf_by_event**2).sum(axis=0)) / n_units
        if post_mask.any():
            overall_se = float(np.sqrt(overall_inf @ overall_inf) / n_units)

    drdid_benchmark_by_event = None
    if result.drdid_benchmark is not None:
        drdid_benchmark_by_event = _aggregate_to_event_time(result.drdid_benchmark, indices, weights)

    estimation_params = dict(result.estimation_params)
    estimation_params["alpha"] = alpha

    return didml_agg(
        overall_att=overall_att,
        overall_se=overall_se,
        aggregation_type="dynamic",
        event_times=event_times,
        att_by_event=att_by_event,
        se_by_event=se_by_event,
        critical_values=critical_values,
        influence_func=inf_by_event,
        influence_func_overall=overall_inf,
        drdid_benchmark_by_event=drdid_benchmark_by_event,
        min_event_time=int(event_times.min()) if len(event_times) else None,
        max_event_time=int(event_times.max()) if len(event_times) else None,
        estimation_params=estimation_params,
    )


def dynamic_cates(result, type="cates"):
    r"""Aggregate per-unit CATTs or doubly robust scores to event time.

    Extends the event-study aggregation of [1]_ to the per-unit conditional
    treatment effects from :func:`didml`. For each unit and event time
    :math:`e = t - g`, the function combines the cell-level CATT predictions
    in ``result.cates`` across cohorts with the cohort weights of
    :func:`aggte_didml`. With ``type="scores"``, it combines the
    doubly robust scores in ``result.scores`` instead. The per-unit
    aggregates then share the event-time scale of :func:`aggte_didml`.

    Parameters
    ----------
    result : DIDMLResult
        Output of :func:`didml`.
    type : {'cates', 'scores'}, default='cates'
        Which per-unit matrix to aggregate. ``'cates'`` uses the ML CATT
        predictions; ``'scores'`` uses the doubly robust score
        contributions.

    Returns
    -------
    polars.DataFrame
        Long-format data with columns:

        - **id**: Unit identifier from ``result.unit_ids``
        - **event_time**: Integer event time :math:`e = t - g`
        - **cate**: Aggregated per-unit CATT at that event time, present when ``type="cates"``
        - **score**: Aggregated per-unit DR score at that event time, present when ``type="scores"``

    Examples
    --------
    Run :func:`didml` and pivot per-unit CATTs to event time:

    .. ipython::
        :okwarning:

        In [1]: from moderndid import didml, dynamic_cates, gen_did_scalable
           ...:
           ...: data = gen_did_scalable(
           ...:     n=300, n_periods=4, n_cohorts=2,
           ...:     n_covariates=4, random_state=0,
           ...: )["data"].to_pandas()
           ...:
           ...: res = didml(
           ...:     data, yname="y", tname="time", idname="id", gname="group",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4",
           ...:     k_folds=5, boot=False, random_state=0,
           ...: )
           ...: dyn_cates = dynamic_cates(res, type="cates")
           ...: print(dyn_cates.head())

    See Also
    --------
    aggte_didml : Aggregate ML group-time ATTs to a dynamic event-study summary.

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962
    """
    if type not in ("cates", "scores"):
        raise ValueError(f"type must be 'cates' or 'scores', got {type!r}.")

    matrix = result.cates if type == "cates" else result.scores
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    indices, weights = _event_time_weights(result.groups, result.times, cohort_shares(result))
    event_times = np.array(sorted(indices.keys()), dtype=int)

    n_units = dense.shape[0]
    aggregated = np.zeros((n_units, len(event_times)))

    for k, e in enumerate(event_times):
        aggregated[:, k] = dense[:, indices[int(e)]] @ weights[int(e)]

    unit_ids = np.asarray(result.unit_ids)

    if unit_ids.shape[0] != n_units:
        unit_ids = np.arange(n_units)

    n_events = len(event_times)
    long_id = np.repeat(unit_ids, n_events)
    long_event = np.tile(event_times, n_units)
    long_value = aggregated.reshape(-1)

    value_col = "cate" if type == "cates" else "score"
    return pl.DataFrame({"id": long_id, "event_time": long_event, value_col: long_value})


def cohort_shares(result):
    """Return the weighted share of each cohort among all units keyed by cohort.

    A result without unit-level cohorts falls back to the unweighted counts in
    ``estimation_params["cohort_counts"]``.
    """
    cohorts = np.unique(np.asarray(result.groups, dtype=float))
    if result.G is None:
        counts = result.estimation_params["cohort_counts"]
        return {float(g): float(counts[float(g)]) for g in cohorts}

    weights = _unit_weights(result)
    groups = np.asarray(result.G, dtype=float)
    return {float(g): float(np.sum(weights * (groups == g)) / np.sum(weights)) for g in cohorts}


def _unit_weights(result):
    """Return each unit's sampling weight or ones when the result holds none."""
    if result.weights_ind is None:
        return np.ones(len(result.G))
    return np.asarray(result.weights_ind, dtype=float)


def _share_influence_func(result, shares, indices):
    """Return the influence-function term that the estimated cohort shares add at each event time."""
    weights = _unit_weights(result)
    groups = np.asarray(result.G, dtype=float)
    weight_ratio = weights / np.mean(weights)
    att = np.asarray(result.att_gt, dtype=float)
    cohorts = np.asarray(result.groups, dtype=float)

    out = np.zeros((len(groups), len(indices)))
    for k, e in enumerate(sorted(indices.keys())):
        cells = indices[e]
        p = np.array([shares[float(g)] for g in cohorts[cells]])
        total = p.sum()
        share_if = np.column_stack([weight_ratio * ((groups == g) - shares[float(g)]) for g in cohorts[cells]])
        weight_if = share_if / total - np.outer(share_if.sum(axis=1), p) / total**2
        out[:, k] = weight_if @ att[cells]
    return out


def _aggregate_draws(result, indices):
    """Aggregate every bootstrap draw to event time with that draw's cohort shares."""
    draws = np.asarray(result.att_gt_draws, dtype=float)
    share_draws = np.asarray(result.cohort_share_draws, dtype=float)
    columns = {float(g): j for j, g in enumerate(np.unique(np.asarray(result.groups, dtype=float)))}
    cohorts = np.asarray(result.groups, dtype=float)

    out = np.zeros((draws.shape[0], len(indices)))
    for k, e in enumerate(sorted(indices.keys())):
        cells = indices[e]
        p = share_draws[:, [columns[float(g)] for g in cohorts[cells]]]
        total = p.sum(axis=1, keepdims=True)
        # Since a cohort that a draw does not hold has a zero share and no refit, it leaves that draw's average.
        # An event time whose cohorts are all missing from the draw stays NaN there.
        weights = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
        terms = np.where(weights > 0, weights * draws[:, cells], 0.0)
        out[:, k] = np.where(total[:, 0] > 0, terms.sum(axis=1), np.nan)
    return out


def _event_time_weights(groups, times, shares):
    """Bucket cells by event time and return normalized cohort-share weights per event time."""
    event_times = (np.asarray(times, dtype=float) - np.asarray(groups, dtype=float)).astype(int)
    unique_event_times = np.unique(event_times)

    pg = np.array([shares[float(g)] for g in groups], dtype=float)

    indices = {}
    weights = {}

    for e in unique_event_times:
        mask = event_times == e
        idx = np.flatnonzero(mask)
        w = pg[idx]
        total = w.sum()

        w = np.full_like(w, 1.0 / len(w)) if total <= 0 else w / total

        indices[int(e)] = idx
        weights[int(e)] = w

    return indices, weights


def _aggregate_to_event_time(values, indices, weights):
    """Form the per-event-time weighted average of a per-cell vector."""
    out = np.zeros(len(indices))

    for k, e in enumerate(sorted(indices.keys())):
        out[k] = float(np.dot(weights[e], values[indices[e]]))

    return out


def _aggregate_influence_func(inf_func, indices, weights):
    """Apply the per-event-time weights to the per-cell influence-function matrix."""
    n_units = inf_func.shape[0]
    out = np.zeros((n_units, len(indices)))

    for k, e in enumerate(sorted(indices.keys())):
        out[:, k] = inf_func[:, indices[e]] @ weights[e]

    return out
