"""Kernels that compute the statistics of one pass over the rows of repeated cross-sections or an unbalanced panel."""

import numpy as np

from .container import Partial
from .kernels import (
    _add_keyed_draws,
    _cohort_codes,
    _deviance_terms,
    _factor,
    _irls_weights,
    _logit_derivative,
    _logit_link,
)


def row_summary_kernel(shard, state):
    r"""Sum what every cell of rows needs before its first iteration.

    A cell takes the rows of its two periods from its treated cohort and its
    comparison cohorts. Its first propensity score iteration is that of
    :func:`~moderndid.did.shards.kernels.summary_kernel` on these rows. The
    outcome regressions with the sampling weights cover the four groups of
    :class:`~moderndid.did.shards.container.RowCellSumsState` for the doubly
    robust estimator and the two comparison groups for the outcome regression
    estimator.

    Parameters
    ----------
    shard : RowShard
        The rows of one shard.
    state : RowSummaryState
        The cells to describe and their estimator.

    Returns
    -------
    Partial
        Statistics of the shard with no keyed sums:

        - **sums**: Number of the shard's units and, for each cell, the counts of the rows and of the rows with
          positive weight in each group, the weight sums of all its rows and of each group, the weighted outcome
          sum of each group, the column sums of squares of its design over all its rows and over each group, the
          Gram matrices of its comparison rows and of each group, and the starting deviance
        - **maxima**: Largest range of a unit's row weights, whether the shard holds rows without keys, and for each
          cell and group the largest weight and the negated smallest positive weight
        - **factors**: QR factors of each cell's first iteration and of the weighted regressions of the outcome
          of its groups on their covariates
    """
    fits, cells = state.fits, state.cells
    n_cells = len(fits.group)
    n_coefficients = _n_coefficients(shard)
    codes = _cohort_codes(shard, fits.cohorts)
    n_regressions = {"dr": 4, "reg": 2}.get(state.method, 0)

    counts = np.zeros((n_cells, 4))
    positives = np.zeros((n_cells, 4))
    weight_sums = np.zeros((n_cells, 5))
    outcome_sums = np.zeros((n_cells, 4))
    squares = np.zeros((n_cells, 5, n_coefficients))
    screen_gram = np.zeros((n_cells, n_coefficients, n_coefficients))
    grams = np.zeros((n_cells, 4, n_coefficients, n_coefficients))
    deviance = np.zeros(n_cells)
    largest_weight = np.full((n_cells, 4), -np.inf)
    smallest_weight = np.full((n_cells, 4), -np.inf)
    factors = {}

    for c in range(n_cells):
        rows, treated, post = _cell_rows(shard, codes, fits, cells, c)
        if len(rows) == 0:
            continue
        x = _design(shard, rows)
        w = _weights(shard, rows)
        y = shard.y[rows]
        groups = 2 * treated + post

        weight_sums[c, 0] = w.sum()
        squares[c, 0] = (x * x).sum(axis=0)
        comparison = ~treated
        screen_gram[c] = x[comparison].T @ (w[comparison][:, None] * x[comparison])
        for g in range(4):
            members = groups == g
            xg, wg, yg = x[members], w[members], y[members]
            counts[c, g] = members.sum()
            positives[c, g] = (wg > 0).sum()
            weight_sums[c, 1 + g] = wg.sum()
            outcome_sums[c, g] = np.sum(wg * yg)
            squares[c, 1 + g] = (xg * xg).sum(axis=0)
            grams[c, g] = xg.T @ (wg[:, None] * xg)
            largest_weight[c, g] = np.max(wg, initial=-np.inf)
            smallest_weight[c, g] = np.max(-wg, initial=-np.inf, where=wg > 0)
            if g < n_regressions and members.any():
                root = np.sqrt(wg)
                factors[f"regression{c}_{g}"] = _factor(root[:, None] * xg, (root * yg)[:, None])

        if state.method != "reg":
            d = treated.astype(float)
            mu = (d + 0.5) / 2
            root = np.sqrt(w * _irls_weights(mu))
            response = _logit_link(mu) + _logit_derivative(mu) * (d - mu)
            factors[f"logit{c}"] = _factor(root[:, None] * x, (root * response)[:, None])
            deviance[c] = np.sum(_deviance_terms(d, mu) * w)

    spread = 0.0
    if shard.units is not None and shard.weights is not None and len(shard.weights) > 0:
        starts = _unit_starts(shard.units)
        spread = np.max(np.maximum.reduceat(shard.weights, starts) - np.minimum.reduceat(shard.weights, starts))
    unkeyed = float(len(shard.cohort) > 0 and shard.unit_keys is None and shard.cluster_keys is None)

    return Partial(
        sums={
            "units": np.array(float(_n_units(shard))),
            "counts": counts,
            "positives": positives,
            "weight_sums": weight_sums,
            "outcome_sums": outcome_sums,
            "squares": squares,
            "screen_gram": screen_gram,
            "grams": grams,
            "deviance": deviance,
        },
        maxima={
            "weight_spread": np.array(spread),
            "unkeyed": np.array(unkeyed),
            "largest_weight": largest_weight,
            "smallest_weight": smallest_weight,
        },
        factors=factors,
        keyed={},
    )


def row_irls_kernel(shard, state):
    """Evaluate the propensity score fit of each cell of rows at its current coefficients.

    The deviance, the separation, and the factor of the Newton step are those
    of :func:`~moderndid.did.shards.kernels.irls_kernel` on the rows of each
    cell.

    Parameters
    ----------
    shard : RowShard
        The rows of one shard.
    state : RowFitState
        The cells and their current coefficients.

    Returns
    -------
    Partial
        Statistics of the shard with no keyed sums:

        - **sums**: Deviance of each cell's fit
        - **maxima**: Largest distance between a row's score and its treatment indicator in each cell and whether
          the exponential in some row's score overflows
        - **factors**: QR factor of each cell's Newton step
    """
    fits, cells = state.fits, state.cells
    n_cells = len(fits.group)
    codes = _cohort_codes(shard, fits.cohorts)
    deviance = np.zeros(n_cells)
    separation = np.zeros(n_cells)
    overflow = np.zeros(n_cells)
    factors = {}

    for j in range(n_cells):
        rows, treated, _ = _cell_rows(shard, codes, fits, cells, j)
        if len(rows) == 0:
            continue
        x = _design(shard, rows) / state.scale[j]
        frequency = _weights(shard, rows) / state.normalizer[j]
        d = treated.astype(float)
        eta = x @ state.coefficients[j]
        mu = _logistic(eta)
        root = np.sqrt(frequency * _irls_weights(mu))
        factors[f"logit{j}"] = _factor(root[:, None] * x, (root * (_logit_derivative(mu) * (d - mu)))[:, None])
        deviance[j] = np.sum(_deviance_terms(d, mu) * frequency)
        separation[j] = np.max(np.abs(mu - d))
        overflow[j] = _overflows(eta)

    return Partial(
        sums={"deviance": deviance},
        maxima={"separation": separation, "overflow": overflow},
        factors=factors,
        keyed={},
    )


def row_cell_sums_kernel(shard, state):
    r"""Sum the parts of every cell's effect and influence function over rows at the final fits.

    The weights and terms are those of :func:`~moderndid.drdid_rc`,
    :func:`~moderndid.std_ipw_did_rc`, and :func:`~moderndid.reg_did_rc` on
    the rows of each cell.

    Parameters
    ----------
    shard : RowShard
        The rows of one shard.
    state : RowCellSumsState
        The cells at their final coefficients.

    Returns
    -------
    Partial
        Statistics of the shard with no factors or keyed sums:

        - **sums**: For each cell, the sums of the weights of its four groups and of its treated rows, of the
          weights times the effect terms, of the weights times the covariates, of the comparison weights times the
          effect terms less their centers times the covariates, and the logit Hessian
        - **maxima**: Largest propensity score of a row with positive weight in each cell

    Notes
    -----
    Let :math:`\omega` denote a row's sampling weight over its mean in the
    cell, :math:`D` its treatment indicator, and :math:`P` the indicator of the
    cell's own period. The treated group of period :math:`t` takes the weights
    :math:`a_{1t} = \omega D \mathbf{1}\{P = t\}` and the treated rows of both
    periods :math:`a_D = \omega D`. With a propensity score :math:`p` clipped
    to :math:`[10^{-6}, 1 - 10^{-6}]`, the comparison group of period
    :math:`t` takes

    .. math::

        a_{0t} = \mathbf{1}\{p < 0.995\} \, \omega \frac{p}{1 - p} (1 - D) \mathbf{1}\{P = t\}.
    """
    fits = state.fits
    table, cells = fits.fits, fits.cells
    n_cells = len(table.group)
    n_coefficients = _n_coefficients(shard)
    codes = _cohort_codes(shard, table.cohorts)

    weights = np.zeros((n_cells, 5))
    terms = np.zeros((n_cells, 8))
    weighted_x = np.zeros((n_cells, 5, n_coefficients))
    terms_x = np.zeros((n_cells, 2, n_coefficients))
    hessian = np.zeros((n_cells, n_coefficients, n_coefficients))
    pscore_max = np.zeros(n_cells)

    for j in range(n_cells):
        rows, treated, post = _cell_rows(shard, codes, table, cells, j)
        if len(rows) == 0:
            continue
        x = _design(shard, rows)
        y = shard.y[rows]
        wn = _weights(shard, rows) / fits.normalizer[j]
        d = treated.astype(float)
        p = post.astype(float)
        fitted = x @ state.regression[j].T
        ps = None
        if state.method != "reg":
            score = _logistic((x / fits.scale[j]) @ fits.coefficients[j])
            ps = np.clip(score, 1e-6, 1 - 1e-6)
            pscore_max[j] = np.max(score, initial=0.0, where=wn > 0)
            hessian[j] = x.T @ ((ps * (1 - ps) * wn)[:, None] * x)
        group_weights = _group_weights(wn, d, p, treated, ps)
        weights[j] = [np.sum(a) for a in group_weights]
        weighted_x[j] = [np.sum(a[:, None] * x, axis=0) for a in group_weights]

        a_tb, a_tp, a_d = group_weights[2:]
        if state.method == "reg":
            terms[j, 2:5] = (np.sum(a_tb * y), np.sum(a_tp * y), np.sum(a_d * (fitted[:, 1] - fitted[:, 0])))
            continue

        outcome = y if state.method == "ipw" else y - (p * fitted[:, 1] + (1 - p) * fitted[:, 0])
        terms[j, :4] = [np.sum(a * outcome) for a in group_weights[:4]]
        # The driver subtracts each comparison term times the summed covariates from these products. Centering near that
        # term keeps the subtraction from cancelling the digits the influence function needs.
        terms_x[j] = [
            np.sum((a * (outcome - center))[:, None] * x, axis=0)
            for a, center in zip(group_weights[:2], state.centers[j], strict=True)
        ]
        if state.method == "dr":
            post_change = fitted[:, 3] - fitted[:, 1]
            pre_change = fitted[:, 2] - fitted[:, 0]
            terms[j, 4:] = (
                np.sum(a_d * post_change),
                np.sum(a_tp * post_change),
                np.sum(a_d * pre_change),
                np.sum(a_tb * pre_change),
            )

    return Partial(
        sums={"weights": weights, "terms": terms, "weighted_x": weighted_x, "terms_x": terms_x, "hessian": hessian},
        maxima={"pscore_max": pscore_max},
        factors={},
        keyed={},
    )


def row_moments_kernel(shard, state):
    r"""Sum the outer products and bootstrap draws of each unit's influence function row over rows.

    Each row of a kept cell contributes the influence function of
    :func:`~moderndid.drdid_rc`, :func:`~moderndid.std_ipw_did_rc`, or
    :func:`~moderndid.reg_did_rc` times :math:`n / n_c` for :math:`n` the
    number of units of the data and :math:`n_c` the number of the cell's rows.
    A unit of an unbalanced panel sums the values of its rows. The units'
    rows :math:`z_i`, their multipliers, and their cluster sums follow
    :func:`~moderndid.did.shards.kernels.moments_kernel`.

    Parameters
    ----------
    shard : RowShard
        The rows of one shard.
    state : RowMomentsState
        The kept cells and their constants.

    Returns
    -------
    Partial
        Statistics of the shard with no maxima or factors:

        - **sums**: Sum of :math:`z_i z_i'` and, with bootstrap draws, of :math:`v_{bi} z_i` for each draw
        - **keyed**: Sums of :math:`z_i` within each cluster under the name ``cluster`` in a fit with a cluster
          variable
    """
    n_cells = len(state.kind)
    n_columns = n_cells + len(state.cohorts) + 1
    rows_of_units = _unit_rows(shard, state, n_cells)
    n_units = _n_units(shard)
    gram = np.zeros((n_columns, n_columns))
    draws = np.zeros((state.biters, n_columns)) if state.biters > 0 else None

    clustered = shard.cluster_keys is not None
    if clustered:
        cluster_keys, cluster_codes = np.unique(shard.cluster_keys, return_inverse=True)
        cluster_codes = cluster_codes.ravel()
        cluster_sums = np.zeros((len(cluster_keys), n_columns))
        # Sorting puts each cluster of the block in one run of rows. Each run then sums with one reduction.
        order = np.argsort(cluster_codes, kind="stable")
    else:
        order = np.arange(n_units)

    for start in range(0, n_units, 2048):
        block = order[start : start + 2048]
        rows = rows_of_units[block]
        gram += rows.T @ rows
        if clustered:
            block_codes = cluster_codes[block]
            runs = np.flatnonzero(np.r_[True, block_codes[1:] != block_codes[:-1]])
            run_sums = np.add.reduceat(rows, runs, axis=0)
            cluster_sums[block_codes[runs]] += run_sums
            if draws is not None:
                _add_keyed_draws(draws, cluster_keys[block_codes[runs]], state.random_state, run_sums)
        elif draws is not None:
            _add_keyed_draws(draws, shard.unit_keys[block], state.random_state, rows)

    sums = {"gram": gram}
    if draws is not None:
        sums["bootstrap"] = draws
    keyed = {"cluster": (cluster_keys, cluster_sums)} if clustered else {}
    return Partial(sums=sums, maxima={}, factors={}, keyed=keyed)


def _unit_rows(shard, state, n_cells):
    """Return the influence function row of each unit of a shard with its weight columns and constant."""
    n_units = _n_units(shard)
    rows_of_units = np.zeros((n_units, n_cells + len(state.cohorts) + 1))
    fits = state.fits
    codes = _cohort_codes(shard, fits.fits.cohorts)
    for i in np.flatnonzero(state.kind == 0):
        rows, treated, post = _cell_rows(shard, codes, fits.fits, fits.cells, i)
        if len(rows) == 0:
            continue
        values = state.ratio[i] * _influence(shard, state, i, rows, treated, post)
        if shard.units is None:
            rows_of_units[rows, i] = values
        else:
            # A unit's influence function sums the values of its rows, as the in-memory fit does.
            rows_of_units[:, i] = np.bincount(shard.units[rows], weights=values, minlength=n_units)

    weights = shard.weights if shard.weights is not None else np.ones(len(shard.cohort))
    if shard.units is None:
        unit_weights, unit_cohorts = weights, shard.cohort
    else:
        sizes = np.bincount(shard.units, minlength=n_units)
        unit_weights = np.bincount(shard.units, weights=weights, minlength=n_units) / sizes
        unit_cohorts = shard.cohort[np.cumsum(sizes) - sizes]

    for j, cohort in enumerate(state.cohorts):
        rows_of_units[:, n_cells + j] = unit_weights * (unit_cohorts == cohort)
    rows_of_units[:, -1] = 1.0
    rows_of_units[:, np.flatnonzero(state.kind == 2)] = np.nan
    return rows_of_units


def _influence(shard, state, i, rows, treated, post):
    """Return the influence function of one kept cell at its rows in a shard before the ratio of units to rows."""
    fits = state.fits
    x = _design(shard, rows)
    y = shard.y[rows]
    wn = _weights(shard, rows) / fits.normalizer[i]
    d = treated.astype(float)
    p = post.astype(float)
    terms, means, directions = state.terms[i], state.means[i], state.directions[i]
    fitted = x @ state.regression[i].T
    ps = None
    if state.method != "reg":
        ps = np.clip(_logistic((x / fits.scale[i]) @ fits.coefficients[i]), 1e-6, 1 - 1e-6)
    a_cb, a_cp, a_tb, a_tp, a_d = _group_weights(wn, d, p, treated, ps)
    m_cb, m_cp, m_tb, m_tp, m_d = means

    if state.method == "reg":
        eta_tb, eta_tp, eta_cont = terms[2:5]
        residual_cb = wn * (1 - d) * (1 - p) * (y - fitted[:, 0])
        residual_cp = wn * (1 - d) * p * (y - fitted[:, 1])
        inf_treat = (a_tp * y - a_tp * eta_tp) / m_tp - (a_tb * y - a_tb * eta_tb) / m_tb
        inf_control = (
            a_d * (fitted[:, 1] - fitted[:, 0])
            - a_d * eta_cont
            + residual_cp * (x @ directions[1])
            - residual_cb * (x @ directions[0])
        ) / m_d
        return inf_treat - inf_control

    pscore_term = (wn * (d - ps)) * (x @ directions[4])
    if state.method == "ipw":
        att_cb, att_cp, att_tb, att_tp = terms[:4]
        inf_treat = (a_tp * y / m_tp - a_tp * att_tp / m_tp) - (a_tb * y / m_tb - a_tb * att_tb / m_tb)
        inf_cont = (a_cp * y / m_cp - a_cp * att_cp / m_cp) - (a_cb * y / m_cb - a_cb * att_cb / m_cb) + pscore_term
        return inf_treat - inf_cont

    att_cb, att_cp, att_tb, att_tp, att_d_post, att_d1, att_d_pre, att_d0 = terms
    residual = y - (p * fitted[:, 1] + (1 - p) * fitted[:, 0])
    post_change = fitted[:, 3] - fitted[:, 1]
    pre_change = fitted[:, 2] - fitted[:, 0]
    inf_treat = (a_tp * residual / m_tp - a_tp * att_tp / m_tp) - (a_tb * residual / m_tb - a_tb * att_tb / m_tb)
    inf_cont = (a_cp * residual / m_cp - a_cp * att_cp / m_cp) - (a_cb * residual / m_cb - a_cb * att_cb / m_cb)
    inf_eff = (
        (a_d * post_change / m_d - a_d * att_d_post / m_d) - (a_tp * post_change / m_tp - a_tp * att_d1 / m_tp)
    ) - ((a_d * pre_change / m_d - a_d * att_d_pre / m_d) - (a_tb * pre_change / m_tb - a_tb * att_d0 / m_tb))
    # With each row in exactly one group, the four regressions' estimation effects need one term per row.
    inf_regressions = 0.0
    for g, (treat, after) in enumerate(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))):
        in_group = (d == treat) & (p == after)
        inf_regressions = inf_regressions + (wn * in_group * (y - fitted[:, g])) * (x @ directions[g])
    return inf_treat - inf_cont + inf_eff + inf_regressions - pscore_term


def _group_weights(wn, d, p, treated, ps):
    """Return the weights of a cell's four groups and treated rows with zero comparison weights without scores."""
    a_tb = wn * d * (1 - p)
    a_tp = wn * d * p
    a_d = wn * d
    if ps is None:
        zeros = np.zeros(len(wn))
        return zeros, zeros, a_tb, a_tp, a_d
    trim = treated | (ps < 0.995)
    a_cb = trim * wn * ps * (1 - d) * (1 - p) / (1 - ps)
    a_cp = trim * wn * ps * (1 - d) * p / (1 - ps)
    return a_cb, a_cp, a_tb, a_tp, a_d


def _logistic(eta):
    """Return the logistic function of a linear predictor without overflowing."""
    # Predictors past the log of the largest double skip the exponential since their score is zero either way. Otherwise
    # its overflow warning, once per shard and pass, would vary with the split of the rows.
    exponent = -eta
    limit = np.log(np.finfo(float).max)
    t = np.exp(np.minimum(exponent, limit))
    t[exponent > limit] = np.inf
    return 1.0 / (1.0 + t)


def _overflows(eta):
    """Tell whether the exponential in the logistic function of some finite linear predictor overflows."""
    return float(np.any(np.isfinite(eta) & (-eta > np.log(np.finfo(float).max))))


def _cell_rows(shard, codes, fits, cells, j):
    """Return the positions of one cell's rows with whether each is treated and whether it is in the cell's period."""
    group = fits.group[j]
    period = shard.period
    member = ((period == cells.time[j]) | (period == cells.base[j])) & ((codes == group) | fits.controls[j][codes])
    rows = np.flatnonzero(member)
    return rows, codes[rows] == group, period[rows] == cells.time[j]


def _design(shard, rows):
    """Return the covariates of some rows after a column of ones."""
    if shard.covariates is None:
        return np.ones((len(rows), 1))
    return np.column_stack([np.ones(len(rows)), shard.covariates[rows]])


def _weights(shard, rows):
    """Return the sampling weights of some rows or ones in an unweighted fit."""
    if shard.weights is None:
        return np.ones(len(rows))
    return shard.weights[rows]


def _n_coefficients(shard):
    """Return the number of coefficients of a design with an intercept."""
    return 1 if shard.covariates is None else shard.covariates.shape[1] + 1


def _n_units(shard):
    """Return the number of units of a shard whose rows may each be a unit of their own."""
    return len(shard.cohort) if shard.units is None else len(shard.unit_keys)


def _unit_starts(units):
    """Return the position of each unit's first row among rows that keep each unit's rows together."""
    return np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
