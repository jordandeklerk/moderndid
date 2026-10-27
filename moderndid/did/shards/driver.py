"""Driver that fits the group-time effects of a balanced panel held in shards."""

import warnings
from functools import partial

import numpy as np
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from moderndid.core.format import warn_few_cluster_counts
from moderndid.core.preprocess import EstimationMethod
from moderndid.did.att_gt import _finish_att_gt
from moderndid.did.compute_att_gt import _check_cell_weights, _period_label
from moderndid.did.container import InfluenceMoments
from moderndid.did.mboot import _mboot_from_draws
from moderndid.drdid.estimators.wols import _singular_comparison_message
from moderndid.drdid.utils import (
    _check_coefficients_validity,
    _check_extreme_weights,
    _check_wls_condition_number,
    _dropped_covariates_message,
    _independent_columns,
    _numerical_rank,
    _scaled_inverse,
    _scaled_pseudo_inverse,
    _singular_pscore_message,
)

from .container import (
    CellEstimate,
    CellSumsState,
    CellTable,
    FitState,
    FitTable,
    LogitFit,
    MomentsState,
    RegressionFit,
    SummaryState,
)
from .kernels import cell_sums_kernel, irls_kernel, moments_kernel, summary_kernel
from .plan import plan_cells


def fit_att_gt_on_shards(executor, config, cohort_counts, random_state, estimation_params, cluster_counts=None):
    """Compute the group-time average treatment effects of a balanced panel held in shards.

    The fit reproduces :func:`~moderndid.att_gt` on a balanced panel whose units
    never come together on one machine. Its warnings match those of the
    in-memory fit in content and order. Where a cell's propensity score
    covariates are nearly collinear, or so small or large that their squares
    leave the range of floating point, rounding decides which columns each fit
    keeps. The cell's warnings and estimate can then differ.

    The influence functions never leave the shards. A last pass sums their
    outer products into :class:`~moderndid.InfluenceMoments` instead. When the
    estimation settings ask for the bootstrap or a uniform band, it also sums
    the draws of :func:`~moderndid.core.bootstrap.keyed_rademacher`. An integer
    ``random_state`` below ``2**64`` is their seed. Any other seed, or None,
    makes the fit draw a 64-bit seed that the result records. See the
    :ref:`guide to fits on shards <dev-shards>` for how the passes of a fit
    work together.

    Parameters
    ----------
    executor : object
        Object whose ``run(kernel, state)`` combines a kernel's
        :class:`~moderndid.did.shards.container.Partial` over every shard and
        finalizes the keyed sums. Its shards must hold the units of the panel
        that set ``config``.
    config : DIDConfig
        Preprocessed configuration of a balanced panel whose ``est_method`` is
        ``"dr"``, ``"ipw"``, or ``"reg"``.
    cohort_counts : DataFrame
        Number of units of each cohort in the columns ``cohort`` and
        ``cohort_size``.
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
    method = _method_name(config)
    _check_random_state(random_state, "a fit on shards")
    plan = plan_cells(config, cohort_counts)

    summary = executor.run(
        summary_kernel,
        SummaryState(fits=plan.fits, cells=plan.cells, logit=method != "reg", regression=method != "ipw"),
    )
    _check_unit_count(summary, config.id_count)
    if config.weightsname is not None and summary.maxima["weight_spread"] > np.sqrt(np.finfo(float).eps):
        _warn_time_varying_weights()

    n_fits = len(plan.fits.group)
    facts = _fit_facts(summary, n_fits)
    regressions = _outcome_regressions(summary, plan, facts) if method != "ipw" else None
    logits = None
    if method != "reg":
        logits = _propensity_fits(
            executor, summary, facts, irls_kernel, partial(_fit_state, plan.fits, facts), range(n_fits)
        )

    final = _final_state(plan.fits, facts, logits)
    regression = _cell_regressions(plan, regressions, facts["n_coefficients"])
    sums = executor.run(
        cell_sums_kernel, CellSumsState(fits=final, cells=plan.cells, regression=regression, method=method)
    )

    estimate_cell = partial(
        _estimate_cell, plan=plan, facts=facts, logits=logits, regressions=regressions, sums=sums, method=method
    )
    kept = _estimate_cells(config, plan, estimate_cell)
    return _finish_shard_fit(
        executor,
        config,
        kept,
        random_state,
        estimation_params,
        cluster_counts,
        moments_kernel,
        partial(_moments_state, config, plan, facts, final, kept, method),
    )


def _finish_shard_fit(executor, config, kept, random_state, estimation_params, cluster_counts, kernel, moments_state):
    """Assemble the att_gt result of the kept cells from a moments pass after the few-clusters warning."""
    n_units = config.id_count
    # Here and in att_gt, the few-clusters warning follows the cell warnings and names only the cohorts that keep cells.
    if cluster_counts is not None:
        total_clusters, cohort_clusters = cluster_counts
        kept_cohorts = np.unique([group for group, _, _, _ in kept])
        warn_few_cluster_counts(total_clusters, n_units, {cohort: cohort_clusters[cohort] for cohort in kept_cohorts})

    # Preprocessing turns off config.cband when two periods remain. The caller's settings decide instead, as they do in
    # att_gt.
    boot = estimation_params["bootstrap"]
    cband = estimation_params["uniform_bands"]
    draws = boot or cband
    seed = _bootstrap_seed(random_state) if draws else None
    moments = executor.run(kernel, moments_state(seed))

    n_cells = len(kept)
    cluster_gram = moments.sums.get("cluster_gram")
    n_clusters = int(moments.sums["cluster_count"]) if cluster_gram is not None else None
    # Clusters of one unit each are the units themselves and leave the covariance at the unit level.
    clustered = n_clusters is not None and n_clusters < n_units
    gram = moments.sums["gram"]
    variance_matrix = (cluster_gram if clustered else gram)[:n_cells, :n_cells] / n_units

    bootstrap_results = None
    if boot:
        # Each cluster of a cluster variable gets one multiplier here and in the in-memory bootstrap.
        n_draw_rows = n_units if n_clusters is None else n_clusters
        draw_scale = np.sqrt(n_units) if n_clusters is None else n_units / np.sqrt(n_clusters)
        bres = moments.sums["bootstrap"][:, :n_cells] / np.sqrt(n_draw_rows)
        # A cell with a negligible bootstrap scale would swamp the band. The group-time critical value leaves such cells
        # out but keeps every draw.
        bootstrap_results = _mboot_from_draws(bres, draw_scale, config.alpha, skip_small_scales=True)

    if seed is not None and random_state is not None:
        # The result records the 64-bit seed behind the stored draws for aggte to check against.
        estimation_params = {**estimation_params, "random_state": seed}

    influence_moments = InfluenceMoments(
        cohorts=_treated_cohorts(config),
        gram=gram,
        cluster_gram=cluster_gram,
        n_clusters=n_clusters,
        bootstrap_sums=moments.sums.get("bootstrap"),
        random_state=seed,
    )
    return _finish_att_gt(
        groups=np.array([group for group, _, _, _ in kept]),
        times=np.array([time for _, time, _, _ in kept]),
        att_values=np.array([_kept_att(kind, estimate) for _, _, kind, estimate in kept]),
        variance_matrix=variance_matrix,
        clustered=clustered,
        bootstrap_results=bootstrap_results,
        n_units=n_units,
        alpha=config.alpha,
        cband=cband,
        anticipation=config.anticipation,
        estimation_params=estimation_params,
        influence_moments=influence_moments,
    )


def _method_name(config):
    """Return the name of the built-in estimator of a balanced-panel configuration."""
    if not config.panel:
        raise ValueError(
            "fit_att_gt_on_shards fits balanced panels only. Fit repeated cross-sections and unbalanced panels "
            "with fit_att_gt_on_row_shards."
        )
    return _estimator_name(config)


def _estimator_name(config):
    """Return the name of the built-in estimator of a configuration or raise a ValueError for any other."""
    if callable(config.est_method) or config.est_method not in tuple(EstimationMethod):
        raise ValueError(
            f"est_method={config.est_method!r} is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'."
        )
    return EstimationMethod(config.est_method).value


def _check_unit_count(summary, n_units):
    """Raise a RuntimeError unless the shards of a summary pass hold as many units as the configuration counts."""
    if int(summary.sums["units"]) != n_units:
        raise RuntimeError(
            f"The shards hold {int(summary.sums['units'])} units but the configuration counts {n_units}. "
            "Build the shards from the panel that set the configuration."
        )


def _warn_time_varying_weights():
    """Warn that some unit carries different sampling weights in different periods."""
    warnings.warn(
        "Time-varying weights detected. For balanced panel data, each 2x2 comparison uses the weights "
        "from the earlier of its two periods, the base period for post-treatment cells.",
        UserWarning,
    )


def _fit_facts(summary, n_fits):
    """Return the counts and weight sums of every fit with its normalizer and column scales."""
    counts = summary.sums["counts"].astype(np.int64)
    weight_sums = summary.sums["weight_sums"]
    squares = summary.sums["squares"]
    n_cell = counts[:, 0]
    # The estimators divide the cell's weights by their mean. Weights that sum to zero stay as they are.
    normalizer = np.ones(n_fits)
    positive = weight_sums[:, 0] > 0
    normalizer[positive] = weight_sums[positive, 0] / n_cell[positive]
    return {
        "n_coefficients": squares.shape[2],
        "n_cell": n_cell,
        "n_comparison": counts[:, 2],
        "n_positive_comparison": counts[:, 3],
        "weight_treated": weight_sums[:, 1],
        "weight_comparison": weight_sums[:, 2],
        "normalizer": normalizer,
        "cell_scale": _column_scales(squares[:, 0], n_cell),
        "comparison_scale": _column_scales(squares[:, 1], counts[:, 2]),
        "largest_weight": summary.maxima["largest_weight"],
        "smallest_weight": -summary.maxima["smallest_weight"],
        "screen_gram": summary.sums["comparison_gram"],
    }


def _column_scales(squares, counts):
    """Return the powers of two that bring each column near unit root mean square from its sum of squares."""
    mean_squares = np.full_like(squares, np.nan)
    np.divide(squares, counts[:, None], out=mean_squares, where=counts[:, None] > 0)
    rms = np.sqrt(mean_squares)
    usable = np.isfinite(rms) & (rms > 0)
    return 2.0 ** np.where(usable, np.round(np.log2(np.where(usable, rms, 1.0))), 0.0)


def _outcome_regressions(summary, plan, facts):
    """Solve the outcome regressions of every fit from the QR factors of their comparison designs."""
    n_coefficients = facts["n_coefficients"]
    regressions = []
    for f in range(len(plan.fits.group)):
        n_rhs = int(np.sum(plan.cells.fit == f))
        r, rotated = summary.factors.get(f"regression{f}", (np.zeros((0, n_coefficients)), np.zeros((0, n_rhs))))
        # The in-memory regression scales the columns over the comparison units.
        regressions.append(_regression_fit(r, rotated, facts["comparison_scale"][f], int(facts["n_comparison"][f])))
    return regressions


def _regression_fit(factor, rotated, scale, n_rows):
    """Solve weighted least squares from the triangular factor of a design whose columns the scales divide."""
    n_coefficients = factor.shape[1]
    collinear = _has_collinear_columns(factor, n_rows)
    try:
        u, s, vt = np.linalg.svd(factor / scale, full_matrices=False)
    except np.linalg.LinAlgError as e:
        return RegressionFit(np.full((rotated.shape[1], n_coefficients), np.nan), collinear, np.inf, str(e))
    eigenvalues = np.sort(s**2)[::-1]
    condition_number = np.inf
    if len(eigenvalues) > 0 and eigenvalues[-1] > 0:
        condition_number = float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
    keep = s > 1e-15 * s.max(initial=0.0)
    params = vt[keep].T @ ((u[:, keep].T @ rotated) / s[keep][:, None])
    return RegressionFit((params / scale[:, None]).T, collinear, condition_number)


def _has_collinear_columns(factor, n_rows):
    """Tell from the triangular factor of a weighted design whether its columns lack full rank at any scale."""
    # Covariates that aren't finite have no rank to check. The solve that follows reports them instead.
    if not np.all(np.isfinite(factor)):
        return False
    if not np.all(np.linalg.norm(factor, axis=0) > 0):
        return True
    return _numerical_rank(factor, n_rows) < factor.shape[1]


def _least_squares(factor, rhs, n_rows, rcond):
    """Return the minimum-norm least squares solution of a factor and the right singular vectors it keeps."""
    u, s, vt = np.linalg.svd(factor, full_matrices=False)
    keep = s > rcond * s.max(initial=0.0)
    # The noise left in the factor of exactly collinear covariates can clear the binomial fit's cutoff. A design without
    # full rank therefore also drops its smallest singular values down to its rank.
    keep[_numerical_rank(factor, n_rows) :] = False
    return vt[keep].T @ ((u[:, keep].T @ rhs) / s[keep]), vt[keep]


def _propensity_fits(executor, summary, facts, kernel, iteration_state, positions):
    """Iterate the logit fits at some positions in shared passes until each deviance settles or 100 iterations pass."""
    n_fits = len(facts["n_cell"])
    n_coefficients = facts["n_coefficients"]
    coefficients = np.zeros((n_fits, n_coefficients))
    previous = summary.sums["deviance"] / facts["normalizer"]
    anchors = [None] * n_fits
    steps = [None] * n_fits
    kept = [None] * n_fits
    iterations = np.zeros(n_fits, dtype=np.int64)
    separated = [[] for _ in range(n_fits)]
    overflowed = [[] for _ in range(n_fits)]
    results = [None] * n_fits

    # The iterations cut singular values at machine precision and the last solve at 1e-15, the cutoffs of the binomial
    # fit.
    eps = np.finfo(float).eps
    n_cell = facts["n_cell"]
    active = []
    for f in positions:
        r, rotated = summary.factors[f"logit{f}"]
        r = r / facts["cell_scale"][f]
        # Every row carries the same binomial weight in the first iteration. Its factor has the rank of the
        # sampling-weighted design and fixes the covariates that every iteration keeps.
        columns = _independent_columns(r, n_cell[f])
        kept[f] = None if columns is None else tuple(columns)
        steps[f] = (_kept_columns(r, kept[f]), rotated[:, 0])
        try:
            coefficients[f] = _full_length(_least_squares(*steps[f], n_cell[f], eps)[0], kept[f], n_coefficients)
        except np.linalg.LinAlgError:
            results[f] = _failed_logit(n_coefficients, 0, (), (), kept[f])
            continue
        active.append(f)

    # Each fit runs to the end even when all of its cells will fail, because att_gt warns for each separated iteration
    # of every cell's logit.
    while active:
        # The caller builds each pass's state because the panel and row engines lay out their fits differently.
        partial_sums = executor.run(kernel, iteration_state(active, coefficients[active]))
        # Only the row kernels report an overflow for replay. The panel kernels let the exponential warn where it
        # overflows.
        overflow = partial_sums.maxima.get("overflow", np.zeros(len(active)))
        remaining = []
        for j, f in enumerate(active):
            iterations[f] += 1
            deviance = partial_sums.sums["deviance"][j]
            separated[f].append(bool(partial_sums.maxima["separation"][j] <= 1e-8))
            overflowed[f].append(bool(overflow[j] > 0))
            converged = bool(np.isclose(previous[f], deviance, rtol=0.0, atol=1e-8))
            if converged or iterations[f] == 100:
                try:
                    coefficients[f] = _final_coefficients(anchors[f], steps[f], n_cell[f], kept[f], n_coefficients)
                except np.linalg.LinAlgError:
                    results[f] = _failed_logit(n_coefficients, iterations[f], separated[f], overflowed[f], kept[f])
                    continue
                results[f] = LogitFit(
                    coefficients[f].copy(),
                    converged,
                    int(iterations[f]),
                    tuple(separated[f]),
                    tuple(overflowed[f]),
                    kept=kept[f],
                    dropped=_dropped_count(kept[f], n_coefficients),
                )
                continue
            r, rotated = partial_sums.factors[f"logit{j}"]
            r = _kept_columns(r, kept[f])
            try:
                step = _least_squares(r, rotated[:, 0], n_cell[f], eps)[0]
            except np.linalg.LinAlgError:
                results[f] = _failed_logit(n_coefficients, iterations[f], separated[f], overflowed[f], kept[f])
                continue
            anchors[f] = coefficients[f].copy()
            steps[f] = (r, rotated[:, 0])
            coefficients[f] = coefficients[f] + _full_length(step, kept[f], n_coefficients)
            previous[f] = deviance
            remaining.append(f)
        active = remaining
    return results


def _kept_columns(factor, kept):
    """Return the columns of a triangular factor at the kept covariates or the whole factor when all stay."""
    return factor if kept is None else factor[:, list(kept)]


def _kept_block(values, kept):
    """Return the entries of a vector or the block of a square matrix at the kept covariates or all of them."""
    if kept is None:
        return values
    positions = np.asarray(kept)
    return values[np.ix_(positions, positions)] if values.ndim == 2 else values[positions]


def _full_length(values, kept, n_coefficients):
    """Return values at the kept covariates at full length with zeros at the dropped ones."""
    if kept is None:
        return values
    full = np.zeros(n_coefficients)
    full[np.asarray(kept)] = values
    return full


def _dropped_count(kept, n_coefficients):
    """Return the number of covariates that a propensity score fit drops."""
    return 0 if kept is None else n_coefficients - len(kept)


def _fit_state(fits, facts, positions, coefficients):
    """Return the state of an iteration pass over the propensity score fits at some positions."""
    return FitState(
        table=_fit_subset(fits, positions),
        normalizer=facts["normalizer"][positions],
        scale=facts["cell_scale"][positions],
        coefficients=coefficients,
    )


def _final_coefficients(anchor, step, n_rows, kept, n_coefficients):
    """Return the coefficients of the last iteration from its pseudo-inverse as the binomial fit reports them."""
    # Regressing the current fit plus the step, the last solve projects the current coefficients onto the kept
    # directions. A full-rank fit therefore keeps its last Newton iterate.
    solution, directions = _least_squares(*step, n_rows, 1e-15)
    if anchor is not None:
        solution = directions.T @ (directions @ _kept_block(anchor, kept)) + solution
    return _full_length(solution, kept, n_coefficients)


def _failed_logit(n_coefficients, iterations, separated, overflowed, kept):
    """Return a propensity score fit whose least squares solve failed."""
    return LogitFit(
        coefficients=np.full(n_coefficients, np.nan),
        converged=False,
        iterations=int(iterations),
        separated=tuple(separated),
        overflowed=tuple(overflowed),
        error="Failed to estimate propensity scores due to singular matrix.",
        kept=kept,
        dropped=_dropped_count(kept, n_coefficients),
    )


def _fit_subset(fits, positions):
    """Return the rows of a fit table at some positions."""
    return FitTable(
        cohorts=fits.cohorts,
        group=fits.group[positions],
        controls=fits.controls[positions],
        earlier=fits.earlier[positions],
    )


def _final_state(fits, facts, logits):
    """Return the fits at their final propensity score coefficients for the passes that follow."""
    coefficients = np.zeros((len(fits.group), facts["n_coefficients"]))
    if logits is not None:
        for f, logit in enumerate(logits):
            # A failed fit fails each of its cells first. Its kernels evaluate a harmless zero in place of its
            # coefficients.
            if np.all(np.isfinite(logit.coefficients)):
                coefficients[f] = logit.coefficients
    return FitState(table=fits, normalizer=facts["normalizer"], scale=facts["cell_scale"], coefficients=coefficients)


def _cell_regressions(plan, regressions, n_coefficients):
    """Return the outcome regression coefficients of each estimable cell with zero where they are missing or invalid."""
    coefficients = np.zeros((len(plan.cells.fit), n_coefficients))
    if regressions is None:
        return coefficients
    for f, regression in enumerate(regressions):
        cells = np.flatnonzero(plan.cells.fit == f)
        # Invalid coefficients fail their cell before anything reads its sums. The kernels evaluate zero in their place.
        finite = np.all(np.isfinite(regression.coefficients), axis=1)
        coefficients[cells[finite]] = regression.coefficients[finite]
    return coefficients


def _estimate_cells(config, plan, estimate_cell):
    """Decide every cell in the order of the result with ``estimate_cell(spec, c)`` and return the kept ones."""
    universal = config.base_period == "universal"
    n_per_group = len(config.time_periods) if universal else len(config.time_periods) - 1
    time_factor = 0 if universal else 1
    estimable = {int(position): c for c, position in enumerate(plan.estimable)}

    kept = []
    for position, spec in enumerate(plan.specs):
        group_idx, time_idx = divmod(position, n_per_group)
        # Skipped cells have no spec. Their group and period come from the configuration, as in the in-memory fit.
        group = config.treated_groups[group_idx]
        time = config.time_periods[time_idx + time_factor]
        if spec is not None and spec.reference:
            kept.append((group, time, 1, None))
            continue
        if spec is not None and spec.skip_warning is not None:
            warnings.warn(spec.skip_warning, UserWarning)
        estimate = None
        if position in estimable:
            c = estimable[position]
            estimate = estimate_cell(spec, c)
        if estimate is not None:
            kept.append((group, time, 0, (c, estimate)))
        elif universal:
            kept.append((group, time, 2, None))
    return kept


def _kept_att(kind, pair):
    """Return the effect of a kept cell with zero for a reference cell and NaN for a failed one."""
    if kind == 0:
        return float(pair[1].att)
    return 0.0 if kind == 1 else np.nan


def _estimate_cell(spec, c, plan, facts, logits, regressions, sums, method):
    """Screen and estimate one cell or warn and return None for a cell that fails."""
    g_label = _period_label(spec.group)
    t_label = _period_label(spec.time)
    f = int(plan.cells.fit[c])

    logit = None if logits is None else logits[f]
    if _fails_screens(logit, sums.maxima["pscore_max"][f], facts["screen_gram"][f], method, g_label, t_label):
        return None

    try:
        # Two totals stand in for the units because the check reads only each side's total weight.
        _check_cell_weights(np.array([1.0, 0.0]), np.array([facts["weight_treated"][f], facts["weight_comparison"][f]]))
        if method == "dr":
            return _doubly_robust_cell(c, f, plan, facts, logits[f], regressions[f], sums)
        if method == "ipw":
            return _inverse_probability_cell(c, f, facts, logits[f], sums)
        return _regression_cell(c, f, plan, facts, regressions[f], sums)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        warnings.warn(f"The 2x2 comparison for group {g_label} in time period {t_label} failed. {e}", UserWarning)
        return None


def _fails_screens(logit, pscore_max, screen_gram, method, g_label, t_label):
    """Replay the overlap and regression screens of att_gt on one cell and tell whether it fails either."""
    fails = False
    if method != "reg":
        # att_gt fits each cell's logit once in the overlap screen and again in the estimator. Every cell replays the
        # screen's warnings here and the estimator's later in _check_logit.
        _warn_iterations(logit)
        if logit.error is None:
            _warn_scores(logit)
            if pscore_max >= 0.999:
                fails = True
                warnings.warn(f"Overlap condition violated for {g_label} in time period {t_label}", UserWarning)
    if method != "ipw":
        try:
            if np.linalg.cond(screen_gram) > 1.0 / np.finfo(float).eps:
                fails = True
                warnings.warn(
                    f"Not enough control units for group {g_label} in time period {t_label} "
                    "to run specified regression",
                    UserWarning,
                )
        except np.linalg.LinAlgError:
            fails = True
            warnings.warn(f"Singular covariate matrix for group {g_label} in time period {t_label}", UserWarning)
    return fails


def _doubly_robust_cell(c, f, plan, facts, logit, regression, sums):
    """Estimate one cell with the doubly robust estimator from its sums."""
    _check_comparison_size(facts, f)
    _check_logit(logit, "Propensity score model coefficients have NA components. ")

    n_comparison = int(facts["n_comparison"][f])
    if n_comparison < 5:
        warnings.warn(f"Only {n_comparison} control units available. Results may be unreliable.", UserWarning)
    if n_comparison > 1 and facts["n_positive_comparison"][f] > 0:
        # The check reads only the largest weight and the smallest positive one. Those two stand in for the units.
        _check_extreme_weights(np.array([facts["smallest_weight"][f], facts["largest_weight"][f]]))
    if regression.collinear:
        raise ValueError(_singular_comparison_message())
    if regression.error is not None:
        raise ValueError("Failed to solve linear system. The covariate matrix may be singular or ill-conditioned.")
    _check_wls_condition_number(regression.condition_number)
    coefficients = regression.coefficients[_position_in_fit(plan, c, f)]
    _check_coefficients_validity(coefficients)

    n_cell = facts["n_cell"][f]
    mean_treat, mean_cont = sums.sums["weights"][f] / n_cell
    if mean_treat == 0:
        raise ValueError("No effectively treated units after trimming. Cannot compute ATT.")
    if mean_cont == 0:
        raise ValueError("No effectively control units after trimming. Cannot compute ATT.")
    eta_treat = (sums.sums["terms"][c, 0] / n_cell) / mean_treat
    eta_cont = (sums.sums["terms"][c, 1] / n_cell) / mean_cont
    att = eta_treat - eta_cont

    gram_inverse = _scaled_inverse(sums.sums["comparison_gram"][f] / n_cell, _singular_comparison_message())
    hessian = _kept_block(sums.sums["hessian"][f], logit.kept)
    hessian_inverse = _scaled_inverse(hessian, _singular_pscore_message()) * n_cell
    treat_x, cont_x = sums.sums["weighted_x"][f] / n_cell
    pscore_derivative = (sums.sums["terms_x"][c] - eta_cont * sums.sums["weighted_x"][f, 1]) / n_cell
    pscore_direction = _full_length(
        hessian_inverse @ _kept_block(pscore_derivative, logit.kept), logit.kept, facts["n_coefficients"]
    )
    _check_att(att)
    return CellEstimate(
        att=att,
        terms=np.array([eta_treat, eta_cont, mean_treat, mean_cont]),
        directions=np.array([gram_inverse @ treat_x, pscore_direction, gram_inverse @ cont_x]),
        coefficients=coefficients,
    )


def _inverse_probability_cell(c, f, facts, logit, sums):
    """Estimate one cell with the standardized inverse probability weighting estimator from its sums."""
    _check_logit(logit, "Propensity score model coefficients have NA components. \n")

    n_cell = facts["n_cell"][f]
    mean_treat, mean_cont = sums.sums["weights"][f] / n_cell
    if mean_treat == 0:
        warnings.warn("No effectively treated units after trimming.", UserWarning)
        raise ValueError("The estimator returned NaN for the ATT.")
    if mean_cont == 0:
        warnings.warn("No effectively control units after trimming.", UserWarning)
        raise ValueError("The estimator returned NaN for the ATT.")
    att_treat = (sums.sums["terms"][c, 0] / mean_treat) / n_cell
    att_cont = (sums.sums["terms"][c, 1] / mean_cont) / n_cell
    att = att_treat - att_cont

    hessian = _kept_block(sums.sums["hessian"][f], logit.kept)
    try:
        hessian_inverse = _scaled_inverse(hessian, _singular_pscore_message()) * n_cell
    except ValueError:
        hessian_inverse = _scaled_pseudo_inverse(hessian) * n_cell
    moment = ((sums.sums["terms_x"][c] - att_cont * sums.sums["weighted_x"][f, 1]) / n_cell) / mean_cont
    _check_att(att)
    n_coefficients = facts["n_coefficients"]
    pscore_direction = _full_length(hessian_inverse @ _kept_block(moment, logit.kept), logit.kept, n_coefficients)
    return CellEstimate(
        att=att,
        terms=np.array([att_treat, att_cont, mean_treat, mean_cont]),
        directions=np.array([np.zeros(n_coefficients), pscore_direction, np.zeros(n_coefficients)]),
        coefficients=np.zeros(n_coefficients),
    )


def _regression_cell(c, f, plan, facts, regression, sums):
    """Estimate one cell with the outcome regression estimator from its sums."""
    _check_comparison_size(facts, f)
    if regression.collinear:
        raise ValueError(_singular_comparison_message())
    if regression.error is not None:
        raise ValueError(f"Failed to fit outcome regression model: {regression.error}")
    coefficients = regression.coefficients[_position_in_fit(plan, c, f)]
    if np.any(np.isnan(coefficients)):
        raise ValueError(
            "Outcome regression model coefficients have NA components. \n"
            "Multicollinearity (or lack of variation) of covariates is probably the reason for it."
        )

    n_cell = facts["n_cell"][f]
    mean_treat, mean_cont = sums.sums["weights"][f] / n_cell
    eta_treat = (sums.sums["terms"][c, 0] / n_cell) / mean_treat
    eta_cont = (sums.sums["terms"][c, 1] / n_cell) / mean_cont
    att = eta_treat - eta_cont
    _check_att(att)

    gram_inverse = _scaled_inverse(sums.sums["comparison_gram"][f] / n_cell, _singular_comparison_message())
    n_coefficients = facts["n_coefficients"]
    regression_derivative = sums.sums["weighted_x"][f, 1] / n_cell
    return CellEstimate(
        att=att,
        terms=np.array([eta_treat, eta_cont, mean_treat, mean_cont]),
        directions=np.array([np.zeros(n_coefficients), np.zeros(n_coefficients), gram_inverse @ regression_derivative]),
        coefficients=coefficients,
    )


def _check_comparison_size(facts, f):
    """Raise a ValueError when the comparison group has fewer units with positive weight than coefficients."""
    n_positive = int(facts["n_positive_comparison"][f])
    n_coefficients = facts["n_coefficients"]
    if n_positive >= n_coefficients:
        return
    # Rows with zero weight drop out of the fit. The count names only rows with positive weight.
    counted = "units" if n_positive == int(facts["n_comparison"][f]) else "units with positive weight"
    raise ValueError(
        f"The comparison group has fewer {counted} ({n_positive}) than regression coefficients ({n_coefficients}). "
        "Pass fewer covariates."
    )


def _check_logit(logit, nan_message, convergence_message="Propensity score estimation did not converge."):
    """Warn about the propensity score fit of an estimator and raise a ValueError when it failed."""
    # The warning about dropped covariates precedes the fit's warnings in memory.
    if logit.dropped > 0:
        warnings.warn(_dropped_covariates_message(logit.dropped), UserWarning)
    # att_gt refits the logit in the estimator of every cell that passes the screens. Those cells replay their iteration
    # warnings a second time here.
    _warn_iterations(logit)
    if logit.error is not None:
        raise ValueError(logit.error)
    if not logit.converged:
        warnings.warn(convergence_message, UserWarning)
    if np.any(np.isnan(logit.coefficients)):
        raise ValueError(nan_message + "Multicollinearity (or lack of variation) of covariates is a likely reason.")
    _warn_scores(logit)


def _check_att(att):
    """Raise a ValueError when an estimate is NaN."""
    if np.isnan(att):
        raise ValueError("The estimator returned NaN for the ATT.")


def _warn_iterations(logit):
    """Warn as each iteration of a binomial fit does with an overflow before a separation."""
    for overflowed, separated in zip(logit.overflowed, logit.separated, strict=True):
        if overflowed:
            _repeat_overflow()
        if separated:
            warnings.warn(
                "Perfect separation or prediction detected, parameter may not be identified",
                PerfectSeparationWarning,
            )


def _warn_scores(logit):
    """Warn as the scores of a finished binomial fit at its final coefficients do."""
    # The final coefficients are the last iteration's. Their scores overflow exactly when that iteration's scores did.
    if logit.overflowed[-1]:
        _repeat_overflow()


def _repeat_overflow():
    """Repeat the overflowing exponential behind a warning of the in-memory fit."""
    # The replay repeats the overflowing operation so the caller's floating-point error state governs the warning as it
    # does in memory.
    np.exp(np.array([1000.0]))


def _position_in_fit(plan, c, f):
    """Return the position of an estimable cell among the cells of its fit."""
    return int(np.sum(plan.cells.fit[:c] == f))


def _moments_state(config, plan, facts, final, kept, method, seed):
    """Return the state of the moments pass for the kept cells in the order of the result."""
    n_kept = len(kept)
    n_coefficients = facts["n_coefficients"]
    kind = np.array([cell_kind for _, _, cell_kind, _ in kept], dtype=np.int64)
    fit = np.full(n_kept, -1, dtype=np.int64)
    time = np.full(n_kept, -1, dtype=np.int64)
    base = np.full(n_kept, -1, dtype=np.int64)
    coefficients = np.zeros((n_kept, n_coefficients))
    constants = np.zeros((n_kept, 5))
    directions = np.zeros((n_kept, 3, n_coefficients))
    for k, (_, _, cell_kind, pair) in enumerate(kept):
        if cell_kind != 0:
            continue
        c, estimate = pair
        f = int(plan.cells.fit[c])
        fit[k], time[k], base[k] = f, plan.cells.time[c], plan.cells.base[c]
        coefficients[k] = estimate.coefficients
        # A cell's influence function counts every unit, not only the cell's. The in-memory fit scales it the same way.
        constants[k] = (config.id_count / facts["n_cell"][f], *estimate.terms)
        directions[k] = estimate.directions
    return MomentsState(
        fits=final,
        cells=CellTable(fit=fit, time=time, base=base),
        kind=kind,
        regression=coefficients,
        constants=constants,
        directions=directions,
        method=method,
        cohorts=_treated_cohorts(config),
        biters=config.biters if seed is not None else 0,
        random_state=seed,
    )


def _treated_cohorts(config):
    """Return the treated cohorts that get a weight column in increasing order."""
    return np.sort(np.asarray(config.treated_groups, dtype=float))


def _check_random_state(random_state, data_name):
    """Raise a ValueError unless the seed is None or a non-negative integer or a NumPy SeedSequence or generator."""
    if random_state is None or isinstance(
        random_state, np.random.Generator | np.random.BitGenerator | np.random.SeedSequence
    ):
        return
    if isinstance(random_state, int | np.integer) and not isinstance(random_state, bool) and int(random_state) >= 0:
        return
    raise ValueError(
        f"random_state={random_state!r} is not valid for {data_name}. Pass None, a non-negative integer, a "
        "SeedSequence, a BitGenerator, or a Generator."
    )


def _bootstrap_seed(random_state):
    """Return the 64-bit seed of the keyed bootstrap multipliers."""
    if isinstance(random_state, int | np.integer) and int(random_state) < 2**64:
        return int(random_state)
    # A larger integer, a SeedSequence, a BitGenerator, a Generator, or None each make a Generator. The seed is that
    # Generator's first 64-bit draw.
    return int(np.random.default_rng(random_state).integers(2**64, dtype=np.uint64))
