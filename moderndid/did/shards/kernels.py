"""Kernels that compute the statistics of one pass over the units of a shard."""

import numpy as np

from moderndid.core.bootstrap import _keyed_sign_blocks

from .container import Partial


def summary_kernel(shard, state):
    r"""Sum what every fit needs before its first iteration.

    The first iteration of a propensity score fit regresses the working
    response :math:`z_0` on the design with weights
    :math:`w \mu_0 (1 - \mu_0)` for

    .. math::

        \mu_0 = \frac{D + 1/2}{2}, \qquad
        z_0 = \log\frac{\mu_0}{1 - \mu_0} + \frac{D - \mu_0}{\mu_0 (1 - \mu_0)}.

    The factors and the starting deviance leave out the fit's mean weight and
    column scales.

    Parameters
    ----------
    shard : Shard
        The units of one shard.
    state : SummaryState
        The fits to describe and their cells.

    Returns
    -------
    Partial
        Statistics of the shard with no keyed sums:

        - **sums**: Number of the shard's units and, for each fit, the counts of its units, treated units,
          comparison units, and comparison units with positive weight, the weight sums of the first three groups,
          the column sums of squares of its design over its units and over its comparison units, the comparison
          Gram matrix, and the starting deviance
        - **maxima**: Largest range of a unit's weights across periods and, for each fit, the largest comparison
          weight and the negated smallest positive comparison weight
        - **factors**: QR factors of each fit's first iteration and of the weighted regressions of its cells'
          outcome changes on the comparison units' covariates
    """
    fits = state.fits
    n_fits = len(fits.group)
    n_coefficients = _n_coefficients(shard)
    codes = _cohort_codes(shard, fits.cohorts)

    counts = np.zeros((n_fits, 4))
    weight_sums = np.zeros((n_fits, 3))
    squares = np.zeros((n_fits, 2, n_coefficients))
    comparison_gram = np.zeros((n_fits, n_coefficients, n_coefficients))
    deviance = np.zeros(n_fits)
    largest = np.full(n_fits, -np.inf)
    smallest = np.full(n_fits, -np.inf)
    factors = {}

    for f in range(n_fits):
        rows, treated = _fit_rows(fits, codes, f)
        if len(rows) == 0:
            continue
        period = fits.earlier[f]
        x = _design(shard, rows, period)
        w = _weights(shard, rows, period)
        comparison = ~treated
        xc, wc = x[comparison], w[comparison]

        counts[f] = (len(rows), treated.sum(), comparison.sum(), (wc > 0).sum())
        weight_sums[f] = (w.sum(), w[treated].sum(), wc.sum())
        squares[f, 0] = (x * x).sum(axis=0)
        squares[f, 1] = (xc * xc).sum(axis=0)
        comparison_gram[f] = xc.T @ (wc[:, None] * xc)
        largest[f] = np.max(wc, initial=-np.inf)
        smallest[f] = np.max(-wc, initial=-np.inf, where=wc > 0)

        if state.logit:
            d = treated.astype(float)
            mu = (d + 0.5) / 2
            root = np.sqrt(w * _irls_weights(mu))
            response = _logit_link(mu) + _logit_derivative(mu) * (d - mu)
            factors[f"logit{f}"] = _factor(root[:, None] * x, (root * response)[:, None])
            deviance[f] = np.sum(_deviance_terms(d, mu) * w)

        if state.regression and comparison.any():
            fit_cells = np.flatnonzero(state.cells.fit == f)
            outcomes = shard.y[rows[comparison]]
            changes = outcomes[:, state.cells.time[fit_cells]] - outcomes[:, state.cells.base[fit_cells]]
            root = np.sqrt(wc)
            factors[f"regression{f}"] = _factor(root[:, None] * xc, root[:, None] * changes)

    spread = 0.0
    if shard.weights is not None and len(shard.weights) > 0:
        spread = np.max(shard.weights.max(axis=1) - shard.weights.min(axis=1))

    return Partial(
        sums={
            "units": np.array(float(len(shard.cohort))),
            "counts": counts,
            "weight_sums": weight_sums,
            "squares": squares,
            "comparison_gram": comparison_gram,
            "deviance": deviance,
        },
        maxima={
            "largest_weight": largest,
            "smallest_weight": smallest,
            "weight_spread": np.array(spread),
        },
        factors=factors,
        keyed={},
    )


def irls_kernel(shard, state):
    r"""Evaluate each propensity score fit at its current coefficients.

    A fit with weights :math:`w` that average one over its units has scores
    :math:`\mu = 1 / (1 + e^{-X\gamma})` on its scaled design. Its Newton step
    comes from the regression of :math:`(D - \mu) / (\mu (1 - \mu))` on the
    design with weights :math:`w \mu (1 - \mu)`.

    Parameters
    ----------
    shard : Shard
        The units of one shard.
    state : FitState
        The fits and their current coefficients.

    Returns
    -------
    Partial
        Statistics of the shard with no keyed sums:

        - **sums**: Deviance of each fit
        - **maxima**: Largest distance between a unit's score and its treatment indicator in each fit
        - **factors**: QR factor of each fit's Newton step

    Notes
    -----
    The binomial deviance of a fit is

    .. math::

        2 \sum_i w_i \left(D_i \log\frac{D_i}{\mu_i} + (1 - D_i)
        \log\frac{1 - D_i}{1 - \mu_i}\right).
    """
    table = state.table
    n_fits = len(table.group)
    codes = _cohort_codes(shard, table.cohorts)
    deviance = np.zeros(n_fits)
    separation = np.zeros(n_fits)
    factors = {}

    for f in range(n_fits):
        rows, treated = _fit_rows(table, codes, f)
        if len(rows) == 0:
            continue
        period = table.earlier[f]
        x = _design(shard, rows, period) / state.scale[f]
        frequency = _weights(shard, rows, period) / state.normalizer[f]
        d = treated.astype(float)
        mu = _probability(x @ state.coefficients[f])
        root = np.sqrt(frequency * _irls_weights(mu))
        factors[f"logit{f}"] = _factor(root[:, None] * x, (root * (_logit_derivative(mu) * (d - mu)))[:, None])
        deviance[f] = np.sum(_deviance_terms(d, mu) * frequency)
        separation[f] = np.max(np.abs(mu - d))

    return Partial(sums={"deviance": deviance}, maxima={"separation": separation}, factors=factors, keyed={})


def cell_sums_kernel(shard, state):
    """Sum the parts of every cell's effect and influence function at the final fits.

    The weights and terms are those of :func:`~moderndid.drdid_panel`,
    :func:`~moderndid.std_ipw_did_panel`, and :func:`~moderndid.reg_did_panel`.

    Parameters
    ----------
    shard : Shard
        The units of one shard.
    state : CellSumsState
        The fits at their final coefficients and the cells to sum.

    Returns
    -------
    Partial
        Statistics of the shard with no factors or keyed sums:

        - **sums**: Each fit's treated and comparison weights with their covariate products, logit Hessian, and
          comparison Gram matrix, and each cell's treated and comparison terms of the effect with the covariate
          products of the comparison term
        - **maxima**: Largest propensity score of a unit with positive weight in each fit
    """
    fits = state.fits
    table = fits.table
    cells = state.cells
    n_fits = len(table.group)
    n_cells = len(cells.fit)
    n_coefficients = _n_coefficients(shard)
    codes = _cohort_codes(shard, table.cohorts)

    weights = np.zeros((n_fits, 2))
    weighted_x = np.zeros((n_fits, 2, n_coefficients))
    hessian = np.zeros((n_fits, n_coefficients, n_coefficients))
    comparison_gram = np.zeros((n_fits, n_coefficients, n_coefficients))
    pscore_max = np.zeros(n_fits)
    terms = np.zeros((n_cells, 2))
    terms_x = np.zeros((n_cells, n_coefficients))

    for f in range(n_fits):
        rows, treated = _fit_rows(table, codes, f)
        if len(rows) == 0:
            continue
        period = table.earlier[f]
        x = _design(shard, rows, period)
        wn = _weights(shard, rows, period) / fits.normalizer[f]
        d = treated.astype(float)
        w_treat, w_cont, score, ps = _unit_weights(x, wn, d, treated, fits, f, state.method)

        weights[f] = (np.sum(w_treat), np.sum(w_cont))
        weighted_x[f, 0] = np.sum(w_treat[:, None] * x, axis=0)
        weighted_x[f, 1] = np.sum(w_cont[:, None] * x, axis=0)
        if state.method != "reg":
            pscore_max[f] = np.max(score, initial=0.0, where=wn > 0)
            hessian[f] = x.T @ ((ps * (1 - ps) * wn)[:, None] * x)
        if state.method != "ipw":
            comparison_gram[f] = x.T @ ((wn * (1 - d))[:, None] * x)

        for c in np.flatnonzero(cells.fit == f):
            dy = shard.y[rows, cells.time[c]] - shard.y[rows, cells.base[c]]
            if state.method == "reg":
                terms[c] = (np.sum(w_treat * dy), np.sum(w_cont * (x @ state.regression[c])))
                continue
            term = dy - x @ state.regression[c] if state.method == "dr" else dy
            terms[c] = (np.sum(w_treat * term), np.sum(w_cont * term))
            terms_x[c] = np.sum((w_cont * term)[:, None] * x, axis=0)

    return Partial(
        sums={
            "weights": weights,
            "weighted_x": weighted_x,
            "hessian": hessian,
            "comparison_gram": comparison_gram,
            "terms": terms,
            "terms_x": terms_x,
        },
        maxima={"pscore_max": pscore_max},
        factors={},
        keyed={},
    )


def moments_kernel(shard, state):
    r"""Sum the outer products and bootstrap draws of each unit's influence function row.

    Unit :math:`i` contributes its row :math:`z_i` of
    :class:`~moderndid.InfluenceMoments`. Its multipliers :math:`v_{bi}` are
    the draws of :func:`~moderndid.core.bootstrap.keyed_rademacher` for the key
    of its cluster in a fit with a cluster variable and for its own key
    otherwise.

    Parameters
    ----------
    shard : Shard
        The units of one shard.
    state : MomentsState
        The fits, the kept cells, and their constants.

    Returns
    -------
    Partial
        Statistics of the shard with no maxima or factors:

        - **sums**: Sum of :math:`z_i z_i'` and, with bootstrap draws, of :math:`v_{bi} z_i` for each draw
        - **keyed**: Sums of :math:`z_i` within each cluster under the name ``cluster`` in a fit with a cluster
          variable
    """
    n_columns = len(state.kind) + len(state.cohorts) + 1
    n_units = len(shard.cohort)
    codes = _cohort_codes(shard, state.fits.table.cohorts)
    gram = np.zeros((n_columns, n_columns))
    draws = np.zeros((state.biters, n_columns)) if state.biters > 0 else None

    clustered = shard.cluster_keys is not None
    if clustered:
        cluster_keys, cluster_codes = np.unique(shard.cluster_keys, return_inverse=True)
        cluster_codes = cluster_codes.ravel()
        cluster_sums = np.zeros((len(cluster_keys), n_columns))
        # Sorting puts each cluster of a block in one run of rows. Each run then sums with one reduction.
        order = np.argsort(cluster_codes, kind="stable")
    else:
        order = np.arange(n_units)

    for start in range(0, n_units, 2048):
        block = order[start : start + 2048]
        rows = _moment_rows(shard, state, codes, block)
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


def _add_keyed_draws(draws, keys, seed, rows):
    """Add the keyed multipliers of each bootstrap draw times the rows to its sums in blocks of draws."""
    # A block of 2048 rows and 1024 draws holds 2**21 signs. The multipliers take about 40 MB whatever the number of
    # draws. Every block keeps the fit's width because a product with a single draw rounds differently.
    width = 128 * min(8, -(-len(draws) // 128))
    for start in range(0, len(draws), width):
        stop = min(start + width, len(draws))
        signs = _keyed_sign_blocks(keys, seed, start // 128, width // 128)
        draws[start:stop] += (signs.T @ rows)[: stop - start]


def _unit_weights(x, wn, d, treated, fits, f, method):
    """Return the treated and comparison weights of a fit's units with their raw and clipped propensity scores."""
    if method == "reg":
        return wn * d, wn * d, None, None
    score = _probability((x / fits.scale[f]) @ fits.coefficients[f])
    ps = np.clip(score, 1e-6, 1 - 1e-6)
    trim = treated | (ps < 0.995)
    return trim * wn * d, trim * wn * ps * (1 - d) / (1 - ps), score, ps


def _moment_rows(shard, state, codes, block):
    """Return the influence function rows of a block of a shard's units."""
    n_cells = len(state.kind)
    rows = np.zeros((len(block), n_cells + len(state.cohorts) + 1))
    first_weights = shard.weights[block, 0] if shard.weights is not None else np.ones(len(block))
    cohorts = shard.cohort[block]
    for j, cohort in enumerate(state.cohorts):
        rows[:, n_cells + j] = first_weights * (cohorts == cohort)
    rows[:, -1] = 1.0
    rows[:, np.flatnonzero(state.kind == 2)] = np.nan

    fits = state.fits
    table = fits.table
    estimates = np.flatnonzero(state.kind == 0)
    block_codes = codes[block]
    for f in np.unique(state.cells.fit[estimates]):
        members, treated = _fit_rows(table, block_codes, f)
        if len(members) == 0:
            continue
        units = block[members]
        period = table.earlier[f]
        x = _design(shard, units, period)
        wn = _weights(shard, units, period) / fits.normalizer[f]
        d = treated.astype(float)
        unit_weights = _unit_weights(x, wn, d, treated, fits, f, state.method)
        for c in estimates[state.cells.fit[estimates] == f]:
            dy = shard.y[units, state.cells.time[c]] - shard.y[units, state.cells.base[c]]
            rows[members, c] = _influence(state, c, x, wn, d, dy, unit_weights)
    return rows


def _influence(state, c, x, wn, d, dy, unit_weights):
    """Return the influence function of one cell for its units in a block."""
    scale, eta_treat, eta_cont, mean_treat, mean_cont = state.constants[c]
    treat_direction, pscore_direction, regression_direction = state.directions[c]
    w_treat, w_cont, _, ps = unit_weights

    if state.method == "ipw":
        inf_treat = w_treat * dy / mean_treat - w_treat * eta_treat / mean_treat
        inf_cont = w_cont * dy / mean_cont - w_cont * eta_cont / mean_cont
        inf_cont_ps = (wn * (d - ps)) * (x @ pscore_direction)
        return scale * (inf_treat - (inf_cont + inf_cont_ps))

    if state.method == "reg":
        fitted = x @ state.regression[c]
        regression_terms = (wn * (1 - d) * (dy - fitted)) * (x @ regression_direction)
        inf_treat = (w_treat * dy - w_treat * eta_treat) / mean_treat
        inf_cont = (w_cont * fitted - w_cont * eta_cont + regression_terms) / mean_cont
        return scale * (inf_treat - inf_cont)

    residual = dy - x @ state.regression[c]
    comparison_residual = wn * (1 - d) * residual
    inf_treat = (w_treat * residual - w_treat * eta_treat - comparison_residual * (x @ treat_direction)) / mean_treat
    inf_cont = (
        w_cont * residual
        - w_cont * eta_cont
        + (wn * (d - ps)) * (x @ pscore_direction)
        - comparison_residual * (x @ regression_direction)
    ) / mean_cont
    return scale * (inf_treat - inf_cont)


def _n_coefficients(shard):
    """Return the number of coefficients of a design with an intercept."""
    return 1 if shard.covariates is None else shard.covariates.shape[2] + 1


def _cohort_codes(shard, cohorts):
    """Return the position of each unit's cohort among the cohorts of the panel."""
    codes = np.minimum(np.searchsorted(cohorts, shard.cohort), len(cohorts) - 1)
    if not np.array_equal(cohorts[codes], shard.cohort):
        raise RuntimeError(
            "A shard holds a unit whose cohort the panel's cohort counts leave out. Rebuild the shards from the "
            "panel that set the cohort counts."
        )
    return codes


def _fit_rows(table, codes, f):
    """Return the positions of one fit's units among the coded units and whether each is treated."""
    group = table.group[f]
    rows = np.flatnonzero((codes == group) | table.controls[f][codes])
    return rows, codes[rows] == group


def _design(shard, rows, period):
    """Return the covariates of some units in one period after a column of ones."""
    if shard.covariates is None:
        return np.ones((len(rows), 1))
    return np.column_stack([np.ones(len(rows)), shard.covariates[rows, period, :]])


def _weights(shard, rows, period):
    """Return the sampling weights of some units in one period or ones in an unweighted fit."""
    if shard.weights is None:
        return np.ones(len(rows))
    return shard.weights[rows, period]


def _factor(design, rhs):
    """Return the triangular factor of a design and the right-hand sides rotated by its orthogonal factor."""
    q, r = np.linalg.qr(design)
    return r, q.T @ rhs


def _clean(mu):
    """Clip probabilities machine precision away from zero and one as the binomial fit does."""
    eps = np.finfo(float).eps
    return np.clip(mu, eps, 1.0 - eps)


def _logit_link(mu):
    """Return the log odds of clipped probabilities."""
    p = _clean(mu)
    return np.log(p / (1.0 - p))


def _logit_derivative(mu):
    """Return the derivative of the log odds at clipped probabilities."""
    p = _clean(mu)
    return 1.0 / (p * (1 - p))


def _irls_weights(mu):
    """Return the binomial weights of an iteration as one over the squared link derivative times the variance."""
    p = _clean(mu)
    return 1.0 / (_logit_derivative(mu) ** 2 * (p * (1 - p)))


def _probability(eta):
    """Return the logistic function of a linear predictor."""
    return 1.0 / (1.0 + np.exp(-eta))


def _deviance_terms(d, mu):
    """Return each unit's binomial deviance with each ratio floored at machine precision."""
    eps = np.finfo(float).eps
    success = np.clip(d / (mu + 1e-20), eps, np.inf)
    failure = np.clip((1.0 - d) / (1.0 - mu + 1e-20), eps, np.inf)
    return 2 * (d * np.log(success) + (1 - d) * np.log(failure))
