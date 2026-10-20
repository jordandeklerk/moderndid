"""Driver that fits the group-time effects of a balanced panel held in shards."""

import warnings

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
    never come together on one machine. It reads the units only through passes
    of a kernel over the shards. The propensity score fits repeat the iterations
    of the in-memory binomial fit with one pass of
    :func:`~moderndid.did.shards.kernels.irls_kernel` for each iteration. The
    warnings match those of the in-memory fit in content and order.

    The influence functions never leave the shards. A last pass sums their
    outer products into :class:`~moderndid.InfluenceMoments`. When the
    estimation settings ask for the bootstrap or a uniform band, it also sums
    the bootstrap draws of :func:`~moderndid.core.bootstrap.keyed_rademacher`.
    The standard errors, the uniform critical value, and the Wald pre-test
    follow from these sums. An integer ``random_state`` below ``2**64`` is the
    seed of the draws. From any other seed or from None, the fit draws a 64-bit
    seed that the result records.

    See the :ref:`architecture guide <architecture-shards>` for how the passes
    of a fit work together.

    Parameters
    ----------
    executor : object
        Object whose ``run(kernel, state)`` applies ``kernel(shard, state)`` to
        every shard and returns the combined
        :class:`~moderndid.did.shards.container.Partial` with its keyed sums
        finalized. Its shards must hold the units of the panel that set
        ``config``.
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
    n_units = config.id_count

    summary = executor.run(
        summary_kernel,
        SummaryState(fits=plan.fits, cells=plan.cells, logit=method != "reg", regression=method != "ipw"),
    )
    if int(summary.sums["units"]) != n_units:
        raise RuntimeError(
            f"The shards hold {int(summary.sums['units'])} units but the configuration counts {n_units}. "
            "Build the shards from the panel that set the configuration."
        )
    if config.weightsname is not None and summary.maxima["weight_spread"] > np.sqrt(np.finfo(float).eps):
        warnings.warn(
            "Time-varying weights detected. For balanced panel data, each 2x2 comparison uses the weights "
            "from the earlier of its two periods, the base period for post-treatment cells.",
            UserWarning,
        )

    facts = _fit_facts(summary, len(plan.fits.group))
    regressions = _outcome_regressions(summary, plan, facts, method) if method != "ipw" else None
    logits = _propensity_fits(executor, summary, plan.fits, facts) if method != "reg" else None

    final = _final_state(plan.fits, facts, logits)
    regression = _cell_regressions(plan, regressions, facts["n_coefficients"])
    sums = executor.run(
        cell_sums_kernel, CellSumsState(fits=final, cells=plan.cells, regression=regression, method=method)
    )

    kept = _estimate_cells(config, plan, facts, logits, regressions, sums, method)
    # As in att_gt, the few-clusters warning follows the cell warnings and names only the cohorts that keep cells.
    if cluster_counts is not None:
        total_clusters, cohort_clusters = cluster_counts
        kept_cohorts = np.unique([group for group, _, _, _ in kept])
        warn_few_cluster_counts(total_clusters, n_units, {cohort: cohort_clusters[cohort] for cohort in kept_cohorts})

    # Since preprocessing turns off config.cband when two periods remain, the caller's settings decide, as in att_gt.
    boot = estimation_params["bootstrap"]
    cband = estimation_params["uniform_bands"]
    draws = boot or cband
    seed = _bootstrap_seed(random_state) if draws else None
    moments = executor.run(moments_kernel, _moments_state(config, plan, facts, final, kept, method, seed))

    n_cells = len(kept)
    cluster_gram = moments.sums.get("cluster_gram")
    n_clusters = int(moments.sums["cluster_count"]) if cluster_gram is not None else None
    # Since clusters of one unit each are the units themselves, they leave the covariance at the unit level.
    clustered = n_clusters is not None and n_clusters < n_units
    gram = moments.sums["gram"]
    variance_matrix = (cluster_gram if clustered else gram)[:n_cells, :n_cells] / n_units

    bootstrap_results = None
    if boot:
        # As in the in-memory bootstrap, a cluster variable gives each cluster one multiplier.
        n_draw_rows = n_units if n_clusters is None else n_clusters
        draw_scale = np.sqrt(n_units) if n_clusters is None else n_units / np.sqrt(n_clusters)
        bres = moments.sums["bootstrap"][:, :n_cells] / np.sqrt(n_draw_rows)
        # Since a cell with a negligible bootstrap scale would swamp the band, the group-time critical value leaves
        # such cells out and keeps every draw.
        bootstrap_results = _mboot_from_draws(bres, draw_scale, config.alpha, skip_small_scales=True)

    if seed is not None and random_state is not None:
        # Since the stored draws come from the 64-bit seed, the result records that seed for aggte to check against.
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
    if not config.panel or config.allow_unbalanced_panel:
        raise ValueError(
            "The shard engine fits balanced panels only. Pass panel=True and allow_unbalanced_panel=False."
        )
    if callable(config.est_method) or config.est_method not in tuple(EstimationMethod):
        raise ValueError(
            f"est_method={config.est_method!r} is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'."
        )
    return EstimationMethod(config.est_method).value


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


def _outcome_regressions(summary, plan, facts, method):
    """Solve the outcome regressions of every fit from the QR factors of their comparison designs."""
    n_coefficients = facts["n_coefficients"]
    regressions = []
    for f in range(len(plan.fits.group)):
        n_rhs = int(np.sum(plan.cells.fit == f))
        r, rotated = summary.factors.get(f"regression{f}", (np.zeros((0, n_coefficients)), np.zeros((0, n_rhs))))
        collinear = _has_collinear_columns(r, int(facts["n_comparison"][f]))
        # The doubly robust regression scales the columns over the comparison units and the outcome regression
        # estimator over all the cell's units.
        scale = facts["comparison_scale"][f] if method == "dr" else facts["cell_scale"][f]
        try:
            u, s, vt = np.linalg.svd(r / scale, full_matrices=False)
        except np.linalg.LinAlgError as e:
            regressions.append(RegressionFit(np.full((n_rhs, n_coefficients), np.nan), collinear, np.inf, str(e)))
            continue
        eigenvalues = np.sort(s**2)[::-1]
        condition_number = np.inf
        if len(eigenvalues) > 0 and eigenvalues[-1] > 0:
            condition_number = float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
        keep = s > 1e-15 * s.max(initial=0.0)
        params = vt[keep].T @ ((u[:, keep].T @ rotated) / s[keep][:, None])
        regressions.append(RegressionFit((params / scale[:, None]).T, collinear, condition_number))
    return regressions


def _has_collinear_columns(factor, n_rows):
    """Tell from the triangular factor of a weighted design whether its columns lack full rank at any scale."""
    # Since a covariate that is not finite has no rank to check, the solve that follows reports it.
    if not np.all(np.isfinite(factor)):
        return False
    if not np.all(np.linalg.norm(factor, axis=0) > 0):
        return True
    return _numerical_rank(factor, n_rows) < factor.shape[1]


def _numerical_rank(factor, n_rows):
    """Return the rank of a weighted design from its triangular factor as a rank check on unit-norm columns sees it."""
    norms = np.linalg.norm(factor, axis=0)
    nonzero = norms > 0
    if not nonzero.any():
        return 0
    singular_values = np.linalg.svd(factor[:, nonzero] / norms[nonzero], compute_uv=False)
    # Since the factor stands in for the rows of the full design, the tolerance grows with that design's larger
    # dimension as a rank check's does.
    tolerance = singular_values.max() * max(n_rows, factor.shape[1]) * np.finfo(float).eps
    return int(np.count_nonzero(singular_values > tolerance))


def _least_squares(factor, rhs, n_rows, rcond):
    """Return the minimum-norm least squares solution of a factor and the right singular vectors it keeps."""
    u, s, vt = np.linalg.svd(factor, full_matrices=False)
    keep = s > rcond * s.max(initial=0.0)
    # Since the noise left in the factor of exactly collinear covariates can clear the binomial fit's cutoff, a
    # design without full rank also drops its smallest singular values down to its rank.
    keep[_numerical_rank(factor, n_rows) :] = False
    return vt[keep].T @ ((u[:, keep].T @ rhs) / s[keep]), vt[keep]


def _propensity_fits(executor, summary, fits, facts):
    """Iterate all propensity score fits in shared passes until each deviance settles or 100 iterations pass."""
    n_fits = len(fits.group)
    n_coefficients = facts["n_coefficients"]
    coefficients = np.zeros((n_fits, n_coefficients))
    previous = summary.sums["deviance"] / facts["normalizer"]
    anchors = [None] * n_fits
    steps = [None] * n_fits
    iterations = np.zeros(n_fits, dtype=np.int64)
    separated = np.zeros(n_fits, dtype=np.int64)
    results = [None] * n_fits

    # As in the binomial fit, the iterations cut singular values at machine precision and the last solve at 1e-15.
    eps = np.finfo(float).eps
    n_cell = facts["n_cell"]
    active = []
    for f in range(n_fits):
        r, rotated = summary.factors[f"logit{f}"]
        steps[f] = (r / facts["cell_scale"][f], rotated[:, 0])
        try:
            coefficients[f] = _least_squares(*steps[f], n_cell[f], eps)[0]
        except np.linalg.LinAlgError:
            results[f] = _failed_logit(n_coefficients, 0, 0)
            continue
        active.append(f)

    # Since att_gt warns for each separated iteration of every cell's logit, each fit runs to the end even when all of
    # its cells will fail.
    while active:
        state = FitState(
            table=_fit_subset(fits, active),
            normalizer=facts["normalizer"][active],
            scale=facts["cell_scale"][active],
            coefficients=coefficients[active],
        )
        partial = executor.run(irls_kernel, state)
        remaining = []
        for j, f in enumerate(active):
            iterations[f] += 1
            deviance = partial.sums["deviance"][j]
            if partial.maxima["separation"][j] <= 1e-8:
                separated[f] += 1
            converged = bool(np.isclose(previous[f], deviance, rtol=0.0, atol=1e-8))
            if converged or iterations[f] == 100:
                try:
                    coefficients[f] = _final_coefficients(anchors[f], steps[f], n_cell[f])
                except np.linalg.LinAlgError:
                    results[f] = _failed_logit(n_coefficients, iterations[f], separated[f])
                    continue
                results[f] = LogitFit(coefficients[f].copy(), converged, int(iterations[f]), int(separated[f]))
                continue
            r, rotated = partial.factors[f"logit{j}"]
            try:
                step = _least_squares(r, rotated[:, 0], n_cell[f], eps)[0]
            except np.linalg.LinAlgError:
                results[f] = _failed_logit(n_coefficients, iterations[f], separated[f])
                continue
            anchors[f] = coefficients[f].copy()
            steps[f] = (r, rotated[:, 0])
            coefficients[f] = coefficients[f] + step
            previous[f] = deviance
            remaining.append(f)
        active = remaining
    return results


def _final_coefficients(anchor, step, n_rows):
    """Return the coefficients of the last iteration from its pseudo-inverse as the binomial fit reports them."""
    # Since the last solve regresses the current fit plus the step, it projects the current coefficients onto the
    # kept directions. A full-rank fit therefore keeps its last Newton iterate.
    solution, kept = _least_squares(*step, n_rows, 1e-15)
    if anchor is None:
        return solution
    return kept.T @ (kept @ anchor) + solution


def _failed_logit(n_coefficients, iterations, separated):
    """Return a propensity score fit whose least squares solve failed."""
    return LogitFit(
        coefficients=np.full(n_coefficients, np.nan),
        converged=False,
        iterations=int(iterations),
        separated=int(separated),
        error="Failed to estimate propensity scores due to singular matrix.",
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
            # Since a fit that failed fails each of its cells first, its kernels evaluate a harmless zero instead.
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
        # Since a cell with invalid coefficients fails before its sums are read, its kernels evaluate zero instead.
        finite = np.all(np.isfinite(regression.coefficients), axis=1)
        coefficients[cells[finite]] = regression.coefficients[finite]
    return coefficients


def _estimate_cells(config, plan, facts, logits, regressions, sums, method):
    """Decide every cell in the order of the result and return the kept ones with their kinds and estimates."""
    universal = config.base_period == "universal"
    n_per_group = len(config.time_periods) if universal else len(config.time_periods) - 1
    time_factor = 0 if universal else 1
    estimable = {int(position): c for c, position in enumerate(plan.estimable)}

    kept = []
    for position, spec in enumerate(plan.specs):
        group_idx, time_idx = divmod(position, n_per_group)
        # Since a skipped cell has no spec, the group and period come from the configuration as in the in-memory fit.
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
            estimate = _estimate_cell(spec, c, plan, facts, logits, regressions, sums, method)
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

    fails = False
    if method != "reg":
        # Since att_gt fits each cell's logit in the overlap screen and again in the estimator, every cell replays the
        # separation warnings of its shared fit here before the estimator's replay in _check_logit.
        _warn_separation(logits[f].separated)
        if logits[f].error is None and sums.maxima["pscore_max"][f] >= 0.999:
            fails = True
            warnings.warn(f"Overlap condition violated for {g_label} in time period {t_label}", UserWarning)
    if method != "ipw":
        try:
            if np.linalg.cond(facts["screen_gram"][f]) > 1.0 / np.finfo(float).eps:
                fails = True
                warnings.warn(
                    f"Not enough control units for group {g_label} in time period {t_label} "
                    "to run specified regression",
                    UserWarning,
                )
        except np.linalg.LinAlgError:
            fails = True
            warnings.warn(f"Singular covariate matrix for group {g_label} in time period {t_label}", UserWarning)
    if fails:
        return None

    try:
        # Since the check reads only each side's total weight, the two totals stand in for the units.
        _check_cell_weights(np.array([1.0, 0.0]), np.array([facts["weight_treated"][f], facts["weight_comparison"][f]]))
        if method == "dr":
            return _doubly_robust_cell(c, f, plan, facts, logits[f], regressions[f], sums)
        if method == "ipw":
            return _inverse_probability_cell(c, f, facts, logits[f], sums)
        return _regression_cell(c, f, plan, facts, regressions[f], sums)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        warnings.warn(f"The 2x2 comparison for group {g_label} in time period {t_label} failed. {e}", UserWarning)
        return None


def _doubly_robust_cell(c, f, plan, facts, logit, regression, sums):
    """Estimate one cell with the doubly robust estimator from its sums."""
    _check_comparison_size(facts, f)
    _check_logit(logit, "Propensity score model coefficients have NA components. ")

    n_comparison = int(facts["n_comparison"][f])
    if n_comparison < 5:
        warnings.warn(f"Only {n_comparison} control units available. Results may be unreliable.", UserWarning)
    if n_comparison > 1 and facts["n_positive_comparison"][f] > 0:
        # Since the check reads only the largest weight and the smallest positive one, the two stand in for the units.
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
    hessian_inverse = _scaled_inverse(sums.sums["hessian"][f], _singular_pscore_message()) * n_cell
    treat_x, cont_x = sums.sums["weighted_x"][f] / n_cell
    pscore_derivative = (sums.sums["terms_x"][c] - eta_cont * sums.sums["weighted_x"][f, 1]) / n_cell
    _check_att(att)
    return CellEstimate(
        att=att,
        terms=np.array([eta_treat, eta_cont, mean_treat, mean_cont]),
        directions=np.array([gram_inverse @ treat_x, hessian_inverse @ pscore_derivative, gram_inverse @ cont_x]),
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

    try:
        hessian_inverse = _scaled_inverse(sums.sums["hessian"][f], _singular_pscore_message()) * n_cell
    except ValueError:
        hessian_inverse = _scaled_pseudo_inverse(sums.sums["hessian"][f]) * n_cell
    moment = ((sums.sums["terms_x"][c] - att_cont * sums.sums["weighted_x"][f, 1]) / n_cell) / mean_cont
    _check_att(att)
    n_coefficients = facts["n_coefficients"]
    return CellEstimate(
        att=att,
        terms=np.array([att_treat, att_cont, mean_treat, mean_cont]),
        directions=np.array([np.zeros(n_coefficients), hessian_inverse @ moment, np.zeros(n_coefficients)]),
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
    # Since a row with zero weight drops out of the fit, the count names only rows with positive weight.
    counted = "units" if n_positive == int(facts["n_comparison"][f]) else "units with positive weight"
    raise ValueError(
        f"The comparison group has fewer {counted} ({n_positive}) than regression coefficients ({n_coefficients}). "
        "Pass fewer covariates."
    )


def _check_logit(logit, nan_message):
    """Warn about the propensity score fit of an estimator and raise a ValueError when it failed."""
    # Since att_gt refits the logit in the estimator of every cell that passes the screens, such a cell replays the
    # separation warnings a second time here.
    _warn_separation(logit.separated)
    if logit.error is not None:
        raise ValueError(logit.error)
    if not logit.converged:
        warnings.warn("Propensity score estimation did not converge.", UserWarning)
    if np.any(np.isnan(logit.coefficients)):
        raise ValueError(nan_message + "Multicollinearity (or lack of variation) of covariates is a likely reason.")


def _check_att(att):
    """Raise a ValueError when an estimate is NaN."""
    if np.isnan(att):
        raise ValueError("The estimator returned NaN for the ATT.")


def _warn_separation(count):
    """Warn once for each iteration of a binomial fit whose scores separated the units."""
    for _ in range(count):
        warnings.warn(
            "Perfect separation or prediction detected, parameter may not be identified",
            PerfectSeparationWarning,
        )


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
        # As in the in-memory fit, a cell's influence function counts every unit, not only the cell's.
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
    # Since a larger integer, a SeedSequence, a BitGenerator, a Generator, or None each make a Generator, the seed is
    # that Generator's first 64-bit draw.
    return int(np.random.default_rng(random_state).integers(2**64, dtype=np.uint64))
