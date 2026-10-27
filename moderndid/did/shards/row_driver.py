"""Driver that fits the group-time effects of repeated cross-sections or an unbalanced panel held in shards."""

import warnings
from functools import partial

import numpy as np

from moderndid.did.compute_att_gt import _check_cell_weights, _period_label
from moderndid.drdid.estimators.wols import _cell_name, _singular_cell_message, _treated_alternatives
from moderndid.drdid.utils import (
    _check_coefficients_validity,
    _check_extreme_weights,
    _check_wls_condition_number,
    _scaled_inverse,
    _scaled_pseudo_inverse,
    _singular_pscore_message,
)

from .container import (
    CellTable,
    FitTable,
    RowCellEstimate,
    RowCellSumsState,
    RowFitState,
    RowMomentsState,
    RowSummaryState,
)
from .driver import (
    _check_att,
    _check_logit,
    _check_random_state,
    _check_unit_count,
    _column_scales,
    _estimate_cells,
    _estimator_name,
    _fails_screens,
    _finish_shard_fit,
    _fit_subset,
    _full_length,
    _kept_block,
    _propensity_fits,
    _regression_fit,
    _treated_cohorts,
    _warn_time_varying_weights,
)
from .plan import plan_cells
from .row_kernels import row_cell_sums_kernel, row_irls_kernel, row_moments_kernel, row_summary_kernel


def fit_att_gt_on_row_shards(executor, config, cohort_counts, random_state, estimation_params, cluster_counts=None):
    """Compute the group-time average treatment effects of repeated cross-sections or an unbalanced panel in shards.

    The fit reproduces :func:`~moderndid.att_gt` on rows that never come
    together on one machine. Its passes, inference, and seed follow
    :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards`. Drawing
    multipliers from rows with neither unit nor cluster keys raises a
    RuntimeError.

    Its warnings match those of the in-memory fit in content and order,
    floating-point ones included, except in two kinds of degenerate cell. A
    cell whose comparison rows of one period are all trimmed fails without the
    division-by-zero warnings. If a cell's covariates are nearly collinear, so
    small or large that their squares leave the range of floating point, or
    collinear in an outcome regression, rounding decides which directions each
    fit keeps and whether the regression screen skips the cell. The cell's
    warnings, estimate, and standard error can then differ.

    Parameters
    ----------
    executor : object
        Object whose ``run`` works as in
        :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards`. Its
        :class:`~moderndid.did.shards.container.RowShard` shards must hold the
        rows of the data that set ``config``.
    config : DIDConfig
        Preprocessed configuration of repeated cross-sections or an unbalanced
        panel whose ``est_method`` is ``"dr"``, ``"ipw"``, or ``"reg"``. Its
        ``true_repeated_cross_sections`` tells the two apart.
    cohort_counts : DataFrame
        Number of units of each cohort in the columns ``cohort`` and
        ``cohort_size``. Each row of repeated cross-sections counts as a unit.
    random_state : int, SeedSequence, BitGenerator, Generator, optional
        Seed of the bootstrap multipliers. An integer must be non-negative.
    estimation_params : dict
        Estimation settings for the result. Its ``bootstrap`` and
        ``uniform_bands`` entries hold the caller's ``boot`` and ``cband``.
    cluster_counts : tuple, optional
        Pair of the number of clusters of all the units and a dict from each
        cohort to the number of clusters of its units, or None without a
        cluster variable besides the unit identifier.

    Returns
    -------
    MPResult
        The result of :func:`~moderndid.att_gt` with ``influence_moments`` in
        place of ``influence_func``, ``G``, and ``weights_ind``.
    """
    method = _row_method_name(config)
    _check_random_state(random_state, "a fit on shards")
    plan = plan_cells(config, cohort_counts)

    summary = executor.run(row_summary_kernel, RowSummaryState(fits=plan.fits, cells=plan.cells, method=method))
    _check_unit_count(summary, config.id_count)
    if (estimation_params["bootstrap"] or estimation_params["uniform_bands"]) and summary.maxima["unkeyed"] > 0:
        raise RuntimeError(
            "The fit draws bootstrap multipliers, but a shard holds rows without keys. Rebuild the row table with "
            "copy indices."
        )
    unbalanced = not config.true_repeated_cross_sections
    weights_vary = summary.maxima["weight_spread"] > np.sqrt(np.finfo(float).eps)
    if unbalanced and config.weightsname is not None and weights_vary:
        _warn_time_varying_weights()

    facts = _row_facts(summary)
    # att_gt fits nothing for a cell that lacks treated or comparison rows in one of its periods. Only the other cells
    # take regressions and logit iterations.
    positions = np.flatnonzero(np.all(facts["counts"] > 0, axis=1))
    regressions = _row_regressions(summary, facts, positions, method) if method != "ipw" else None
    logits = None
    if method != "reg":
        logits = _propensity_fits(
            executor, summary, facts, row_irls_kernel, partial(_row_fit_state, plan, facts), positions
        )

    sums = executor.run(
        row_cell_sums_kernel,
        RowCellSumsState(
            fits=_final_row_state(plan, facts, logits, positions),
            regression=_summed_regressions(regressions, positions, facts["n_coefficients"]),
            centers=facts["centers"][positions] if method == "ipw" else np.zeros((len(positions), 2)),
            method=method,
        ),
    )

    estimate_cell = partial(
        _estimate_row_cell,
        config=config,
        facts=facts,
        logits=logits,
        regressions=regressions,
        sums=sums,
        summed={int(c): j for j, c in enumerate(positions)},
        method=method,
    )
    kept = _estimate_cells(config, plan, estimate_cell)
    return _finish_shard_fit(
        executor,
        config,
        kept,
        random_state,
        estimation_params,
        cluster_counts,
        row_moments_kernel,
        partial(_row_moments_state, config, plan, facts, logits, kept, method),
    )


def _row_method_name(config):
    """Return the built-in estimator of a configuration of repeated cross-sections or an unbalanced panel."""
    if config.panel:
        raise ValueError(
            "The row engine fits repeated cross-sections and unbalanced panels only. Pass the configuration that "
            "preprocessing sets for panel=False or for a panel in which some unit misses a period."
        )
    return _estimator_name(config)


def _row_facts(summary):
    """Return the counts and weight sums of every cell of rows with its normalizer and column scales."""
    counts = summary.sums["counts"].astype(np.int64)
    weight_sums = summary.sums["weight_sums"]
    squares = summary.sums["squares"]
    n_cell = counts.sum(axis=1)
    # The estimators divide the cell's weights by their mean. Weights that sum to zero stay as they are.
    normalizer = np.ones(len(n_cell))
    positive = weight_sums[:, 0] > 0
    normalizer[positive] = weight_sums[positive, 0] / n_cell[positive]
    # A comparison group's mean outcome lies near its inverse probability weighted mean. That makes it the center for
    # the outcomes whose covariate products give the propensity score's estimation effect.
    centers = np.zeros((len(n_cell), 2))
    np.divide(summary.sums["outcome_sums"][:, :2], weight_sums[:, 1:3], out=centers, where=weight_sums[:, 1:3] > 0)
    return {
        "n_coefficients": squares.shape[2],
        "n_cell": n_cell,
        "counts": counts,
        "positives": summary.sums["positives"].astype(np.int64),
        "weight_sums": weight_sums,
        "normalizer": normalizer,
        "centers": centers,
        "cell_scale": _column_scales(squares[:, 0], n_cell),
        "group_scale": np.stack([_column_scales(squares[:, 1 + g], counts[:, g]) for g in range(4)], axis=1),
        "screen_gram": summary.sums["screen_gram"],
        "grams": summary.sums["grams"],
        "largest_weight": summary.maxima["largest_weight"],
        "smallest_weight": -summary.maxima["smallest_weight"],
    }


def _row_regressions(summary, facts, positions, method):
    """Solve the outcome regressions of the groups of the cells at some positions from their QR factors."""
    regressions = [None] * len(facts["n_cell"])
    for c in positions:
        fits = []
        for g in range(4 if method == "dr" else 2):
            r, rotated = summary.factors[f"regression{c}_{g}"]
            # The in-memory fit scales each regression's columns over the rows of its own group too.
            fits.append(_regression_fit(r, rotated, facts["group_scale"][c, g], int(facts["counts"][c, g])))
        regressions[c] = fits
    return regressions


def _row_fit_state(plan, facts, positions, coefficients):
    """Return the state of a pass over the cells at some positions with their propensity score coefficients."""
    return RowFitState(
        fits=_fit_subset(plan.fits, positions),
        cells=CellTable(
            fit=np.arange(len(positions), dtype=np.int64),
            time=plan.cells.time[positions],
            base=plan.cells.base[positions],
        ),
        normalizer=facts["normalizer"][positions],
        scale=facts["cell_scale"][positions],
        coefficients=coefficients,
    )


def _final_row_state(plan, facts, logits, positions):
    """Return the cells at some positions at their final propensity score coefficients for the cell-sums pass."""
    coefficients = np.zeros((len(positions), facts["n_coefficients"]))
    if logits is not None:
        for j, c in enumerate(positions):
            # A failed fit leaves zero coefficients in place. Its cell fails before the driver reads the sums.
            if np.all(np.isfinite(logits[c].coefficients)):
                coefficients[j] = logits[c].coefficients
    return _row_fit_state(plan, facts, positions, coefficients)


def _summed_regressions(regressions, positions, n_coefficients):
    """Return the regression coefficients of the groups of the cells at some positions with zero where invalid."""
    coefficients = np.zeros((len(positions), 4, n_coefficients))
    if regressions is None:
        return coefficients
    for j, c in enumerate(positions):
        for g, regression in enumerate(regressions[c]):
            # Invalid regression coefficients stay zero. The cell fails before its sums are read.
            if np.all(np.isfinite(regression.coefficients)):
                coefficients[j, g] = regression.coefficients[0]
    return coefficients


def _estimate_row_cell(spec, c, config, facts, logits, regressions, sums, summed, method):
    """Screen and estimate one cell of rows or warn and return None for a cell that fails."""
    n_cb, n_cp, n_tb, n_tp = facts["counts"][c]
    # att_gt also skips a cell without a warning when its two periods hold no treated or no comparison rows.
    if n_tb + n_tp == 0 or n_cb + n_cp == 0:
        return None

    g_label = _period_label(spec.group)
    t_label = _period_label(spec.time)
    absent = [
        (n_tp, f"No treated units in group {g_label} in time period {t_label}"),
        (n_tb, f"No treated units in group {g_label} in pre-treatment period"),
        (n_cp, f"No control units for group {g_label} in time period {t_label}"),
        (n_cb, f"No control units for group {g_label} in pre-treatment period"),
    ]
    for count, message in absent:
        if count == 0:
            warnings.warn(message, UserWarning)
    if min(n_cb, n_cp, n_tb, n_tp) == 0:
        return None

    j = summed[c]
    logit = None if logits is None else logits[c]
    if _fails_screens(logit, sums.maxima["pscore_max"][j], facts["screen_gram"][c], method, g_label, t_label):
        return None

    try:
        # Each group's total weight is all the check reads. The four totals stand in for the rows.
        base, own = config.time_periods[spec.base_index], config.time_periods[spec.time_index]
        _check_cell_weights(np.array([0.0, 0.0, 1.0, 1.0]), facts["weight_sums"][c, 1:], np.array([base, own] * 2))
        if method == "dr":
            return _doubly_robust_row_cell(c, j, facts, logit, regressions[c], sums)
        if method == "ipw":
            return _inverse_probability_row_cell(c, j, facts, logit, sums)
        return _regression_row_cell(c, j, facts, regressions[c], sums)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        warnings.warn(f"The 2x2 comparison for group {g_label} in time period {t_label} failed. {e}", UserWarning)
        return None


def _doubly_robust_row_cell(c, j, facts, logit, regressions, sums):
    """Estimate one cell of rows with the locally efficient doubly robust estimator from its sums."""
    _check_groups(facts, c, regressed=(True, True))
    _check_logit(logit, "Propensity score model coefficients have NA components. \n ")
    for g, regression in enumerate(regressions):
        _check_group_regression(facts, c, g, regression)

    n_cell = facts["n_cell"][c]
    means = sums.sums["weights"][j] / n_cell
    m_cb, m_cp, m_tb, m_tp, m_d = means
    t_cb, t_cp, t_tb, t_tp, e_post, e_d1, e_pre, e_d0 = sums.sums["terms"][j]
    # The estimator divides by each period's mean comparison weight. A period whose comparison rows are all trimmed
    # leaves a NaN effect.
    terms = np.full(8, np.nan)
    if m_cb > 0 and m_cp > 0:
        terms = np.array(
            [
                t_cb / m_cb / n_cell,
                t_cp / m_cp / n_cell,
                t_tb / m_tb / n_cell,
                t_tp / m_tp / n_cell,
                e_post / m_d / n_cell,
                e_d1 / m_tp / n_cell,
                e_pre / m_d / n_cell,
                e_d0 / m_tb / n_cell,
            ]
        )
    att_cb, att_cp, att_tb, att_tp, att_d_post, att_d1, att_d_pre, att_d0 = terms
    att = (att_tp - att_tb) - (att_cp - att_cb) + (att_d_post - att_d1) - (att_d_pre - att_d0)

    gram_inverses = [_group_gram_inverse(facts, c, g) for g in range(4)]
    hessian = _kept_block(sums.sums["hessian"][j], logit.kept)
    hessian_inverse = _scaled_inverse(hessian, _singular_pscore_message()) * n_cell
    _check_att(att)

    v_cb, v_cp, v_tb, v_tp, v_d = sums.sums["weighted_x"][j]
    u_cb, u_cp = sums.sums["terms_x"][j]
    treat_post = -(v_tp / n_cell) / m_tp
    treat_pre = -(v_tb / n_cell) / m_tb
    control_post = -(v_cp / n_cell) / m_cp
    control_pre = -(v_cb / n_cell) / m_cb
    adjustment_post = (v_d / m_d - v_tp / m_tp) / n_cell
    adjustment_pre = (v_d / m_d - v_tb / m_tb) / n_cell
    pscore_moment = ((u_cp - att_cp * v_cp) / n_cell) / m_cp - ((u_cb - att_cb * v_cb) / n_cell) / m_cb
    # Each row belongs to one group. The estimation effects of each regression fold into one direction.
    directions = np.array(
        [
            gram_inverses[0] @ (control_pre - treat_pre + adjustment_pre),
            gram_inverses[1] @ (treat_post - control_post - adjustment_post),
            gram_inverses[2] @ -adjustment_pre,
            gram_inverses[3] @ adjustment_post,
            _full_length(hessian_inverse @ _kept_block(pscore_moment, logit.kept), logit.kept, facts["n_coefficients"]),
        ]
    )
    return RowCellEstimate(
        att=att,
        terms=terms,
        means=means,
        directions=directions,
        coefficients=np.array([regression.coefficients[0] for regression in regressions]),
    )


def _inverse_probability_row_cell(c, j, facts, logit, sums):
    """Estimate one cell of rows with the standardized inverse probability weighting estimator from its sums."""
    _check_groups(facts, c, regressed=(False, False))
    _check_logit(logit, "Propensity score model coefficients have NA components. \n", "GLM algorithm did not converge.")

    n_cell = facts["n_cell"][c]
    n_coefficients = facts["n_coefficients"]
    means = sums.sums["weights"][j] / n_cell
    m_cb, m_cp = means[:2]
    terms = np.zeros(8)
    terms[:4] = np.nan
    # A period whose comparison rows are all trimmed has no mean comparison weight to divide by. Its effect stays NaN.
    if m_cb > 0 and m_cp > 0:
        terms[:4] = sums.sums["terms"][j, :4] / means[:4] / n_cell
    att_cb, att_cp, att_tb, att_tp = terms[:4]
    att = (att_tp - att_tb) - (att_cp - att_cb)

    hessian = _kept_block(sums.sums["hessian"][j], logit.kept)
    try:
        hessian_inverse = _scaled_inverse(hessian, _singular_pscore_message()) * n_cell
    except ValueError:
        warnings.warn("Failed to invert Hessian matrix. Using pseudo-inverse.", UserWarning)
        hessian_inverse = _scaled_pseudo_inverse(hessian) * n_cell
    _check_att(att)

    v_cb, v_cp = sums.sums["weighted_x"][j, :2]
    u_cb, u_cp = sums.sums["terms_x"][j]
    center_cb, center_cp = facts["centers"][c]
    moment_pre = ((u_cb - (att_cb - center_cb) * v_cb) / n_cell) / m_cb
    moment_post = ((u_cp - (att_cp - center_cp) * v_cp) / n_cell) / m_cp
    directions = np.zeros((5, n_coefficients))
    directions[4] = _full_length(
        hessian_inverse @ _kept_block(moment_post - moment_pre, logit.kept), logit.kept, n_coefficients
    )
    return RowCellEstimate(
        att=att, terms=terms, means=means, directions=directions, coefficients=np.zeros((4, n_coefficients))
    )


def _regression_row_cell(c, j, facts, regressions, sums):
    """Estimate one cell of rows with the outcome regression estimator from its sums."""
    _check_groups(facts, c, regressed=(True, False))
    n_coefficients = facts["n_coefficients"]
    reasons = ("Multicollinearity of covariates", "Multicollinearity (or lack of variation) of covariates")
    for g, (period, reason) in enumerate(zip(("pre", "post"), reasons, strict=True)):
        regression = regressions[g]
        if regression.collinear:
            raise ValueError(_singular_cell_message(_cell_name(g == 0, False), False))
        if regression.error is not None:
            raise ValueError(f"Failed to fit {period}-period regression: {regression.error}")
        if np.any(np.isnan(regression.coefficients)):
            raise ValueError(
                f"Outcome regression model coefficients have NA components. {reason} is probably the reason for it."
            )

    n_cell = facts["n_cell"][c]
    means = sums.sums["weights"][j] / n_cell
    m_tb, m_tp, m_d = means[2:]
    t_tb, t_tp, t_cont = sums.sums["terms"][j, 2:5]
    eta_tb = (t_tb / n_cell) / m_tb
    eta_tp = (t_tp / n_cell) / m_tp
    eta_cont = (t_cont / n_cell) / m_d
    att = (eta_tp - eta_tb) - eta_cont

    gram_inverses = [_group_gram_inverse(facts, c, g) for g in range(2)]
    _check_att(att)

    derivative = sums.sums["weighted_x"][j, 4] / n_cell
    directions = np.zeros((5, n_coefficients))
    directions[0] = gram_inverses[0] @ derivative
    directions[1] = gram_inverses[1] @ derivative
    terms = np.zeros(8)
    terms[2:5] = (eta_tb, eta_tp, eta_cont)
    coefficients = np.zeros((4, n_coefficients))
    coefficients[:2] = [regression.coefficients[0] for regression in regressions]
    return RowCellEstimate(att=att, terms=terms, means=means, directions=directions, coefficients=coefficients)


def _check_groups(facts, c, regressed):
    """Raise the ValueError of the first group of a cell that has no rows of positive weight or too few to regress."""
    n_coefficients = facts["n_coefficients"]
    for g in range(4):
        treat = g >= 2
        group = _cell_name(g % 2 == 0, treat)
        n_rows = int(facts["counts"][c, g])
        n_positive = int(facts["positives"][c, g])
        if n_positive == 0:
            raise ValueError(
                f"The {group} has no observations with positive weight. Pass positive weights for some of its "
                "observations."
            )
        if not regressed[treat] or n_positive >= n_coefficients:
            continue
        # Rows with zero weight drop out of the fit. The message counts only rows with positive weight.
        counted = "observations" if n_positive == n_rows else "observations with positive weight"
        raise ValueError(
            f"The {group} has fewer {counted} ({n_positive}) than regression coefficients ({n_coefficients}). "
            f"Pass fewer covariates.{_treated_alternatives(treat)}"
        )


def _check_group_regression(facts, c, g, regression):
    """Warn about the regression of one group of a doubly robust cell and raise a ValueError when it failed."""
    treat = g >= 2
    group = _cell_name(g % 2 == 0, treat)
    n_rows = int(facts["counts"][c, g])
    if n_rows < 3:
        warnings.warn(f"Only {n_rows} units available for regression. Results may be unreliable.", UserWarning)
    if n_rows > 1 and facts["positives"][c, g] > 0:
        # The check reads only the largest weight and the smallest positive one.
        extremes = np.array([facts["smallest_weight"][c, g], facts["largest_weight"][c, g]])
        _check_extreme_weights(extremes / facts["normalizer"][c])
    if regression.collinear:
        raise ValueError(_singular_cell_message(group, treat))
    solved = regression.error is None
    if solved:
        try:
            _check_wls_condition_number(regression.condition_number, threshold_error=1e15, threshold_warn=1e10)
            _check_coefficients_validity(regression.coefficients)
        except ValueError:
            solved = False
    if not solved:
        raise ValueError(
            f"The regression of the {group} could not be solved. Pass finite outcomes and covariates for that "
            "group and period."
        )


def _group_gram_inverse(facts, c, g):
    """Return the inverse of the weighted Gram matrix of one group of a cell over the cell's rows."""
    gram = facts["grams"][c, g] / facts["normalizer"][c] / facts["n_cell"][c]
    return _scaled_inverse(gram, _singular_cell_message(_cell_name(g % 2 == 0, g >= 2), g >= 2))


def _row_moments_state(config, plan, facts, logits, kept, method, seed):
    """Return the state of the moments pass over rows for the kept cells in the order of the result."""
    n_kept = len(kept)
    n_coefficients = facts["n_coefficients"]
    kind = np.array([cell_kind for _, _, cell_kind, _ in kept], dtype=np.int64)
    group = np.full(n_kept, -1, dtype=np.int64)
    controls = np.zeros((n_kept, len(plan.fits.cohorts)), dtype=bool)
    earlier = np.full(n_kept, -1, dtype=np.int64)
    time = np.full(n_kept, -1, dtype=np.int64)
    base = np.full(n_kept, -1, dtype=np.int64)
    normalizer = np.ones(n_kept)
    scale = np.ones((n_kept, n_coefficients))
    logit = np.zeros((n_kept, n_coefficients))
    regression = np.zeros((n_kept, 4, n_coefficients))
    ratio = np.zeros(n_kept)
    terms = np.zeros((n_kept, 8))
    means = np.ones((n_kept, 5))
    directions = np.zeros((n_kept, 5, n_coefficients))
    for k, (_, _, cell_kind, pair) in enumerate(kept):
        if cell_kind != 0:
            continue
        c, estimate = pair
        group[k], controls[k], earlier[k] = plan.fits.group[c], plan.fits.controls[c], plan.fits.earlier[c]
        time[k], base[k] = plan.cells.time[c], plan.cells.base[c]
        normalizer[k], scale[k] = facts["normalizer"][c], facts["cell_scale"][c]
        if logits is not None:
            logit[k] = logits[c].coefficients
        regression[k] = estimate.coefficients
        # In memory too, a cell's influence function counts every unit, not only the cell's rows.
        ratio[k] = config.id_count / facts["n_cell"][c]
        terms[k], means[k], directions[k] = estimate.terms, estimate.means, estimate.directions
    return RowMomentsState(
        fits=RowFitState(
            fits=FitTable(cohorts=plan.fits.cohorts, group=group, controls=controls, earlier=earlier),
            cells=CellTable(fit=np.arange(n_kept, dtype=np.int64), time=time, base=base),
            normalizer=normalizer,
            scale=scale,
            coefficients=logit,
        ),
        kind=kind,
        regression=regression,
        ratio=ratio,
        terms=terms,
        means=means,
        directions=directions,
        method=method,
        cohorts=_treated_cohorts(config),
        biters=config.biters if seed is not None else 0,
        random_state=seed,
    )
