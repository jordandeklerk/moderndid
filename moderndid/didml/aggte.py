"""Event-study aggregation of group-time ATTs."""

from __future__ import annotations

import numpy as np
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

    The overall summary is the simple mean of post-treatment event-time
    effects, :math:`\bar\theta = (1/|\mathcal{E}_+|) \sum_{e \ge 0} \theta(e)`,
    returned as ``overall_att``.

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
        Aggregated result. Key fields:

        - **event_times**: Array of integer event times :math:`e = t - g`.
        - **att_by_event**: Per-event-time aggregate :math:`\theta(e)`.
        - **se_by_event**: Influence-function-based standard error per event time.
        - **critical_values**: Pointwise normal critical values at level :math:`1 - \alpha/2`.
        - **overall_att**: Simple mean of post-treatment event-time ATTs.
        - **overall_se**: Influence-function-based standard error for ``overall_att``.
        - **influence_func**: Aggregate influence-function matrix of shape ``(n_units, n_event_times)``.
        - **influence_func_overall**: Influence-function vector of length ``n_units`` for ``overall_att``.
        - **drdid_benchmark_by_event**: Doubly-robust benchmark aggregated to
          event time when ``result.drdid_benchmark`` is present.

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
