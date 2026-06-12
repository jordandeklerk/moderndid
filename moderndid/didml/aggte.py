"""Event-study aggregation of group-time ATTs and per-unit CATTs."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats as _stats

from .container import didml_agg


def aggte_didml(result, type="dynamic", alpha=None):
    r"""Aggregate ML group-time ATTs to a dynamic event-study summary.

    Implements the event-study aggregation of [1]_, which extends the
    Callaway and Sant'Anna [2]_ aggregation scheme to the doubly-robust ML
    setting. The general aggregation has the form

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
    the weights back in collapses the double sum to a per-event-time
    aggregate

    .. math::

        \theta(e) = \sum_{g \in \mathcal{G}}
            \mathbb{1}\{g + e \le T\}\,
            P(G = g \mid G + e \le T)\,
            \mathrm{ATT}(g, t),

    which this function returns in ``att_by_event``.

    The overall summary, returned as ``overall_att``, is the simple mean of
    the post-treatment event-time effects

    .. math::

        \bar\theta = \frac{1}{|\mathcal{E}_+|} \sum_{e \ge 0} \theta(e),

    which follows the overall event-study summary convention of [2]_.

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
        - **overall_se**: Influence-function-based standard error for ``overall_att``
        - **aggregation_type**: Type of aggregation performed (always ``'dynamic'``)
        - **event_times**: Array of integer event times :math:`e = t - g`
        - **att_by_event**: Per-event-time aggregate :math:`\theta(e)`
        - **se_by_event**: Influence-function-based standard error per event time
        - **critical_values**: Pointwise critical values given by the standard normal quantile at :math:`1 - \alpha/2`
        - **influence_func**: Aggregate influence-function matrix of shape ``(n_units, n_event_times)``
        - **influence_func_overall**: Influence-function vector of length ``n_units`` for ``overall_att``
        - **drdid_benchmark_by_event**: Doubly-robust benchmark aggregated to event time when
          ``result.drdid_benchmark`` is present
        - **min_event_time**: Smallest event time in ``event_times``
        - **max_event_time**: Largest event time in ``event_times``
        - **balanced_event_threshold**: Balanced event-time threshold (``None`` for this aggregation)
        - **estimation_params**: Dictionary with estimation details carried over from ``result``
        - **call_info**: Information about the function call that created the result

    Examples
    --------
    We can run :func:`didml` and then aggregate to event time:

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
           ...:     k_folds=5, tune_penalty=False, random_state=0,
           ...: )
           ...: agg = aggte_didml(res, type="dynamic")
           ...: print(agg)

    See Also
    --------
    didml : Compute group-time ATTs and CATTs with cross-fitted ML nuisances.
    dynamic_cates : Aggregate per-unit CATTs or doubly-robust scores to event time.

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

    cohort_counts = result.estimation_params["cohort_counts"]
    indices, weights = _event_time_weights(result.groups, result.times, cohort_counts)
    event_times = np.array(sorted(indices.keys()), dtype=int)

    att_by_event = _aggregate_to_event_time(result.att_gt, indices, weights)
    inf_by_event = _aggregate_influence_func(result.influence_func, indices, weights)

    n_units = int(result.n_units)
    var_by_event = (inf_by_event**2).sum(axis=0) / n_units
    se_by_event = np.sqrt(var_by_event / n_units)

    drdid_benchmark_by_event = None
    if result.drdid_benchmark is not None:
        drdid_benchmark_by_event = _aggregate_to_event_time(result.drdid_benchmark, indices, weights)

    post_mask = event_times >= 0

    if post_mask.any():
        overall_att = float(att_by_event[post_mask].mean())
        overall_inf = inf_by_event[:, post_mask].mean(axis=1)
        overall_var = float(overall_inf @ overall_inf) / n_units
        overall_se = float(np.sqrt(overall_var / n_units))
    else:
        overall_att = float("nan")
        overall_inf = np.zeros(n_units)
        overall_se = float("nan")

    alpha = float(alpha if alpha is not None else result.alpha)
    z_crit = float(_stats.norm.ppf(1 - alpha / 2))
    critical_values = np.full(len(event_times), z_crit)

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
    r"""Aggregate per-unit CATTs or doubly-robust scores to event time.

    Extends the event-study aggregation of [1]_ to the per-unit conditional
    treatment effects from :func:`didml`. For each unit and event time
    :math:`e = t - g`, the cell-level CATT predictions stored in
    ``result.cates`` (or the doubly-robust scores in ``result.scores`` when
    ``type="scores"``) are combined across cohorts with the same
    cohort-probability weights used by :func:`aggte_didml`, placing the
    per-unit aggregates on the same event-time scale.

    Parameters
    ----------
    result : DIDMLResult
        Output of :func:`didml`.
    type : {'cates', 'scores'}, default='cates'
        Which per-unit matrix to aggregate. ``'cates'`` uses the ML CATT
        predictions; ``'scores'`` uses the doubly-robust score
        contributions.

    Returns
    -------
    polars.DataFrame
        Long-format frame with columns:

        - **id**: Unit identifier from ``result.unit_ids``
        - **event_time**: Integer event time :math:`e = t - g`
        - **cate**: Aggregated per-unit CATT at that event time, present when ``type="cates"``
        - **score**: Aggregated per-unit DR score at that event time, present when ``type="scores"``

    Examples
    --------
    Run :func:`didml` and pivot per-unit CATEs to event time:

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
           ...:     k_folds=5, tune_penalty=False, random_state=0,
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

    cohort_counts = result.estimation_params["cohort_counts"]
    matrix = result.cates if type == "cates" else result.scores
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    indices, weights = _event_time_weights(result.groups, result.times, cohort_counts)
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


def _event_time_weights(groups, times, cohort_counts):
    """Bucket cells by event time and return normalized cohort-size weights per event time."""
    event_times = (np.asarray(times, dtype=float) - np.asarray(groups, dtype=float)).astype(int)
    unique_event_times = np.unique(event_times)

    pg = np.array([cohort_counts[float(g)] for g in groups], dtype=float)

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
