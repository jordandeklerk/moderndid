"""Group-time computation for the doubly robust ML DiD estimator."""

from __future__ import annotations

import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import scipy.sparse as sp
import scipy.stats

from moderndid.core.parallel import parallel_map
from moderndid.core.preprocess import BasePeriod
from moderndid.did.compute_att_gt import get_did_cohort_index
from moderndid.drdid.estimators.drdid_panel import drdid_panel

from .container import ComputeDIDMLResult, DIDMLBootstrapResult, DIDMLCellResult
from .lnw import lnw_did
from .weights import amle_weights


def compute_didml(data, *, n_jobs=1):
    r"""Run the doubly robust ML DiD score over every :math:`(g, t)` cell.

    Iterates the cohort by time-period grid, builds the stacked pre/post
    panel for each cell, fits AMLE weights and the LNW score, and stores
    the resulting CATT, score, and gamma vectors as sparse columns.

    Parameters
    ----------
    data : DIDData
        Preprocessed panel data with a ``DIDMLConfig`` attached.
    n_jobs : int, default=1
        Number of parallel workers for the cell loop. ``-1`` uses all
        cores; ``>1`` uses that many workers.

    Returns
    -------
    ComputeDIDMLResult
        Object containing the cell-loop output:

        - **cell_results**: List of per-cell :class:`DIDMLCellResult` records
        - **cates**: Sparse :math:`(n_{units}, n_{cells})` matrix of per-unit CATT predictions
        - **scores**: Sparse :math:`(n_{units}, n_{cells})` matrix of per-unit DR scores
        - **gammas**: Sparse :math:`(n_{units}, n_{cells})` matrix of per-unit AMLE weights
        - **drdid_inf_funcs**: Sparse :math:`(n_{units}, n_{cells})` matrix of benchmark influence functions
        - **influence_funcs**: Dense :math:`(n_{units}, n_{cells})` matrix of the cells' influence functions
    """
    n_units = data.config.id_count
    n_periods = len(data.config.time_periods)
    n_treated = data.config.treated_groups_count
    n_time_periods = n_periods - 1

    # Under a universal base the first calendar period can be a current period. Since the current period is
    # time_idx + 1, the grid starts one step earlier.
    t_start = -1 if data.config.base_period == BasePeriod.UNIVERSAL else 0
    pairs = [(g, t) for g in range(n_treated) for t in range(t_start, n_time_periods)]
    args_list = [(g_idx, t_idx, data) for g_idx, t_idx in pairs]

    raw_results = parallel_map(_process_didml_cell, args_list, n_jobs=n_jobs)
    cell_results = [r for r in raw_results if r is not None]

    cates = _stack_per_unit(cell_results, n_units, "tau_hat")
    scores = _stack_per_unit(cell_results, n_units, "score", average_periods=True)
    gammas = _stack_per_unit(cell_results, n_units, "gamma")
    drdid_infs = _stack_drdid_inf_funcs(cell_results, n_units)

    if cell_results:
        influence_funcs = np.column_stack(
            [np.full(n_units, np.nan) if c.inf_func is None else c.inf_func for c in cell_results]
        )
    else:
        influence_funcs = np.zeros((n_units, 0))

    return ComputeDIDMLResult(
        cell_results=cell_results,
        cates=cates,
        scores=scores,
        gammas=gammas,
        drdid_inf_funcs=drdid_infs,
        influence_funcs=influence_funcs,
    )


def bootstrap_didml(data, cell_results, *, biters, random_state=None, n_jobs=1):
    r"""Refit every group-time cell on unit bootstrap samples.

    Each draw resamples the units of the panel with replacement, keeps
    every unit's outcome and covariate history, and repeats the whole cell
    estimation on the resample. The AMLE weights, the cross-fitted
    nuisance models, and the CATT fit are refit with a fold split drawn
    for that sample. The spread of the draws therefore carries the
    sampling variability of every estimation step.

    With ``n_jobs`` above 1 the draws run in that many spawned worker
    processes. Each worker receives a copy of the data and takes a few
    seconds to start. Since every draw carries its own resample and seed,
    the draws do not depend on ``n_jobs``.

    Parameters
    ----------
    data : DIDData
        Preprocessed panel data with a ``DIDMLConfig`` attached.
    cell_results : list of DIDMLCellResult
        Cells estimated on the original sample by :func:`compute_didml`.
    biters : int
        Number of bootstrap draws.
    random_state : int, optional
        Seed for the resamples and the fold splits of the refits.
    n_jobs : int, default=1
        Number of worker processes for the draws. ``-1`` uses all cores.

    Returns
    -------
    DIDMLBootstrapResult
        Object containing the bootstrap draws:

        - **att_draws**: Array of shape ``(biters, n_cells)`` with each draw's group-time ATTs, NaN where a
          refit failed
        - **share_draws**: Array of shape ``(biters, n_cohorts)`` with each draw's weighted cohort shares,
          one column per cohort in ``np.unique`` of the cells' cohorts
    """
    n_units = data.config.id_count
    unit_groups = unit_cohorts(data)
    cohorts = np.unique([c.group for c in cell_results])
    cells = [(c.pre_idx, c.post_idx, c.cohort_idx) for c in cell_results]

    rng = np.random.default_rng(random_state)
    draws = [(rng.integers(0, n_units, n_units), int(rng.integers(0, 2**31 - 1))) for _ in range(biters)]

    n_workers = min(biters, os.cpu_count() if n_jobs == -1 else n_jobs)
    if n_workers == 1:
        out = _bootstrap_draws(draws, data, cells, unit_groups, cohorts)
    else:
        # Since the refits spend much of their time in Python code that holds the interpreter lock, threads run them
        # no faster than a loop. Spawned processes avoid both the lock and the deadlocks a fork risks in a process
        # that already runs threads.
        chunks = np.array_split(np.arange(biters), n_workers)
        try:
            with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = [
                    executor.submit(_bootstrap_draws, [draws[k] for k in chunk], data, cells, unit_groups, cohorts)
                    for chunk in chunks
                ]
                out = [row for future in futures for row in future.result()]
        except BrokenProcessPool as err:
            raise RuntimeError(
                f"n_jobs={n_jobs} started worker processes for the bootstrap draws that stopped before finishing. A "
                "script must call didml inside an if __name__ == '__main__': block to start them. Each worker also "
                "holds a copy of the data in memory. Pass n_jobs=1 to run the draws in this process."
            ) from err

    att_draws = np.array([o[0] for o in out], dtype=float).reshape(biters, len(cell_results))
    share_draws = np.array([o[1] for o in out], dtype=float).reshape(biters, len(cohorts))
    return DIDMLBootstrapResult(att_draws=att_draws, share_draws=share_draws)


def bootstrap_se(draws):
    """Return the bootstrap standard error of each column of a draw matrix over its finite draws."""
    draws = np.asarray(draws, dtype=float)
    se = np.full(draws.shape[1], np.nan)
    for j in range(draws.shape[1]):
        finite = draws[np.isfinite(draws[:, j]), j]
        if finite.size >= 2:
            se[j] = float(np.std(finite, ddof=1))
    return se


def simultaneous_critical_value(draws, estimates, standard_errors, alpha):
    r"""Return the bootstrap critical value of a uniform band across the columns of a draw matrix.

    The value is the :math:`1 - \alpha` quantile over the draws of the
    largest standardized deviation :math:`|\theta^*_j - \hat\theta_j| / \hat\sigma_j`
    among the columns with a positive standard error.
    """
    keep = np.isfinite(standard_errors) & (standard_errors > 0)
    if not keep.any():
        return float(scipy.stats.norm.ppf(1 - alpha / 2))
    deviations = np.abs(draws[:, keep] - estimates[keep]) / standard_errors[keep]
    sup_t = np.max(np.where(np.isfinite(deviations), deviations, -np.inf), axis=1)
    return float(np.quantile(sup_t[np.isfinite(sup_t)], 1 - alpha))


def unit_cohorts(data):
    """Return the cohort of each unit in the unit order of the outcome tensors."""
    return np.asarray(data.time_invariant_data[data.config.gname].to_numpy(), dtype=float)


def _bootstrap_draws(draws, data, cells, unit_groups, cohorts):
    """Estimate every cell on each resample of a list of draws."""
    return [_bootstrap_draw(units, seed, data, cells, unit_groups, cohorts) for units, seed in draws]


def _bootstrap_draw(units, seed, data, cells, unit_groups, cohorts):
    """Estimate every cell on one resample of the units."""
    weights = np.asarray(data.weights, dtype=float)[units]
    groups = unit_groups[units]
    shares = np.array([np.sum(weights * (groups == g)) for g in cohorts]) / np.sum(weights)

    att = np.full(len(cells), np.nan)
    for j, (pre_idx, post_idx, cell_cohort_idx) in enumerate(cells):
        if pre_idx == post_idx:
            att[j] = 0.0
            continue
        cohort_idx = cell_cohort_idx[units]
        valid = ~np.isnan(cohort_idx)
        if not (np.any(cohort_idx[valid] == 1) and np.any(cohort_idx[valid] == 0)):
            continue
        X, Y, T, G, unit_weights = _stack_cell(data, pre_idx, post_idx, cohort_idx, valid, units)
        try:
            gamma = _cell_amle_weights(X, T, G, unit_weights, data.config)
            att[j] = _fit_cell_score(X, Y, T, G, gamma, unit_weights, data.config, seed)[0]
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    return att, shares


def _process_didml_cell(group_idx, time_idx, data):
    """Estimate one :math:`(g, t)` cell."""
    cfg = data.config
    time_factor = 1

    treated_groups = cfg.treated_groups
    time_periods = cfg.time_periods

    universal = cfg.base_period == BasePeriod.UNIVERSAL

    pre_periods = np.where(time_periods < (treated_groups[group_idx] - cfg.anticipation))[0]
    is_post_treatment = treated_groups[group_idx] <= time_periods[time_idx + time_factor]

    if universal or is_post_treatment:
        if len(pre_periods) == 0:
            warnings.warn(
                f"No pre-treatment periods for group first treated at {treated_groups[group_idx]}; skipping.",
                UserWarning,
            )
            return None
        pre_treatment_idx = pre_periods[-1]
    else:
        pre_treatment_idx = time_idx

    if time_periods[pre_treatment_idx] == time_periods[time_idx + time_factor]:
        if not universal:
            return None
        # Since the universal base period anchors every comparison for the cohort, the cell at the base itself
        # stays as an explicit zero reference instead of being dropped.
        return DIDMLCellResult(
            group=float(treated_groups[group_idx]),
            year=float(time_periods[time_idx + time_factor]),
            post=0,
            pre_idx=int(pre_treatment_idx),
            post_idx=int(time_idx + time_factor),
            att=0.0,
            se=float("nan"),
            tau_hat=None,
            score=None,
            gamma=None,
            cohort_idx=np.full(cfg.id_count, np.nan),
            drdid_att=None,
            drdid_se=None,
            drdid_inf_func=None,
            inf_func=np.zeros(cfg.id_count),
        )

    cohort_idx = get_did_cohort_index(group_idx, time_idx, time_factor, pre_treatment_idx, data)
    valid = ~np.isnan(cohort_idx)

    if not (np.any(cohort_idx[valid] == 1) and np.any(cohort_idx[valid] == 0)):
        return None

    X, Y, T, G, unit_weights = _stack_cell(data, pre_treatment_idx, time_idx + time_factor, cohort_idx, valid)

    if X.shape[1] == 0:
        warnings.warn(
            f"No covariates after intercept removal for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}; skipping cell.",
            UserWarning,
        )
        return None

    try:
        gamma = _cell_amle_weights(X, T, G, unit_weights, cfg)
    except (ValueError, RuntimeError) as exc:
        warnings.warn(
            f"AMLE weights failed for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}: {exc}",
            UserWarning,
        )
        return None

    try:
        att, lnw_result = _fit_cell_score(X, Y, T, G, gamma, unit_weights, cfg, cfg.random_state)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"LNW estimation failed for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}: {exc}",
            UserWarning,
        )
        return None

    inf_func = _cell_influence_function(lnw_result, gamma, X, Y, T, G, unit_weights, valid, cfg.id_count)
    drdid_att, drdid_se, drdid_inf = _maybe_run_drdid_benchmark(data, cohort_idx, valid, pre_treatment_idx, time_idx)

    return DIDMLCellResult(
        group=float(treated_groups[group_idx]),
        year=float(time_periods[time_idx + time_factor]),
        post=int(is_post_treatment),
        pre_idx=int(pre_treatment_idx),
        post_idx=int(time_idx + time_factor),
        att=att,
        se=float(lnw_result["std_err"]),
        tau_hat=lnw_result["tau_hat"],
        score=lnw_result["score"],
        gamma=gamma,
        cohort_idx=cohort_idx,
        drdid_att=drdid_att,
        drdid_se=drdid_se,
        drdid_inf_func=drdid_inf,
        inf_func=inf_func,
    )


def _stack_cell(data, pre_idx, post_idx, cohort_idx, valid, units=None):
    """Stack the pre-period and post-period rows of a cell on all units or on a resample of them."""
    y_pre = data.outcomes_tensor[pre_idx]
    y_post = data.outcomes_tensor[post_idx]
    x_pre = data.covariates_tensor[pre_idx]
    x_post = data.covariates_tensor[post_idx]
    weights = np.asarray(data.weights, dtype=float)
    if units is not None:
        y_pre, y_post, x_pre, x_post, weights = y_pre[units], y_post[units], x_pre[units], x_post[units], weights[units]

    n_valid = int(valid.sum())
    cohort_valid = cohort_idx[valid]
    Y = np.concatenate([y_pre[valid], y_post[valid]])
    G = np.concatenate([cohort_valid, cohort_valid])
    T = np.concatenate([np.zeros(n_valid), np.ones(n_valid)])
    X = np.vstack([x_pre[valid], x_post[valid]])

    if X.shape[1] > 0 and np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]

    unit_weights = None if data.config.weightsname is None else weights[valid]
    return X, Y, T, G, unit_weights


def _cell_amle_weights(X, T, G, unit_weights, cfg):
    """Return the AMLE weights of one stacked cell and None when the cell uses none."""
    if not cfg.use_gamma:
        return None
    row_weights = None if unit_weights is None else np.concatenate([unit_weights, unit_weights])
    return amle_weights(X, post_indicator=T, cohort_indicator=G, zeta=cfg.zeta, sample_weights=row_weights)


def _fit_cell_score(X, Y, T, G, gamma, unit_weights, cfg, random_state):
    """Fit the LNW score of one stacked cell and return its ATT with the score output."""
    lnw_result = lnw_did(
        X,
        Y,
        T,
        G,
        constant_eff="non_constant",
        gamma=gamma,
        nu_model=cfg.nu_model,
        sigma_model=cfg.sigma_model,
        delta_model=cfg.delta_model,
        t_func=cfg.t_func,
        k_folds=cfg.k_folds,
        lambda_choice=cfg.lambda_choice,
        random_state=random_state,
    )

    if unit_weights is None:
        return float(lnw_result["TAU_hat"]), lnw_result

    # Sampling weights enter the plug-in average of the CATTs over the treated cell. Since the AMLE weights already
    # target that cell's weighted covariate mean, the residual correction keeps its plain average.
    treated_cell = T * G == 1
    row_weights = np.concatenate([unit_weights, unit_weights])
    att = float(np.average(lnw_result["tau_hat"][treated_cell], weights=row_weights[treated_cell]))
    if gamma is not None:
        att += float(np.mean(gamma * (Y - lnw_result["y_hat"])))
    return att, lnw_result


def _cell_influence_function(lnw_result, gamma, X, Y, T, G, unit_weights, valid, n_units):
    r"""Return the per-unit influence function of a cell's ATT over all units.

    Each unit contributes one pre-period and one post-period row. Its
    influence is the average of its two rows' centered contributions times
    :math:`n / n_v`. Units outside the cell have zero influence. The
    plug-in part counts the treated cell's rows only. The AMLE weights are
    first projected to balance the CATT design exactly. The estimation
    error of the CATT coefficients then enters the influence function.
    """
    if gamma is None:
        return None

    n_valid = int(valid.sum())
    n_rows = 2 * n_valid
    tau_hat = lnw_result["tau_hat"]
    resid = Y - lnw_result["y_hat"]
    row_weights = np.ones(n_rows) if unit_weights is None else np.concatenate([unit_weights, unit_weights])
    treated_weights = row_weights * (T * G == 1)
    design = np.column_stack([np.ones(n_rows), X])

    # The plug-in average moves with the CATT coefficients through the treated cell's (1, mean X) and the residual
    # correction through -mean(gamma C (1, X)). After the projection closes that gap, the coefficients' error enters.
    tau_design = lnw_result["C_hat"][:, None] * design
    imbalance = gamma @ tau_design / n_rows - np.average(design, axis=0, weights=treated_weights)
    gram = tau_design.T @ tau_design / n_rows
    gamma_tilde = gamma - tau_design @ np.linalg.lstsq(gram, imbalance, rcond=None)[0]

    plug_in = float(np.average(tau_hat, weights=treated_weights))
    row_terms = (n_rows / treated_weights.sum()) * treated_weights * (tau_hat - plug_in) + gamma_tilde * resid
    row_terms -= np.mean(gamma_tilde * resid)
    per_unit = (row_terms[:n_valid] + row_terms[n_valid:]) / 2

    inf_func = np.zeros(n_units)
    inf_func[valid] = (n_units / n_valid) * per_unit
    return inf_func


def _maybe_run_drdid_benchmark(data, cohort_idx, valid, pre_treatment_idx, time_idx):
    """Compute the doubly robust benchmark ATT for the cell when requested."""
    if not getattr(data.config, "compute_drdid_benchmark", False):
        return None, None, None

    y1 = data.outcomes_tensor[time_idx + 1][valid]
    y0 = data.outcomes_tensor[pre_treatment_idx][valid]
    d = cohort_idx[valid]
    weights = data.weights[valid]
    # Since time_idx + 1 is the current period, the minimum picks the earlier of the base and current periods
    # and stays valid when time_idx is -1.
    covariates = data.covariates_tensor[min(pre_treatment_idx, time_idx + 1)][valid]

    try:
        result = drdid_panel(
            y1=y1, y0=y0, d=d, covariates=covariates, i_weights=weights, boot=False, influence_func=True
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None, None, None

    n_units = data.config.id_count
    n_valid = int(valid.sum())
    inf_full = np.zeros(n_units)

    if n_valid > 0:
        inf_full[valid] = (n_units / n_valid) * result.att_inf_func

    return float(result.att), float(result.se), inf_full


def _stack_per_unit(cell_results, n_units, key, average_periods=False):
    """Stack a per-cell stacked-row array as a sparse :math:`(n, n_{cells})` matrix of per-unit values.

    Each unit takes its post-period row, or the average of its two rows with ``average_periods``.
    """
    if not cell_results:
        return sp.csr_matrix((n_units, 0))

    columns = []

    for cell in cell_results:
        col = np.zeros(n_units)
        values = getattr(cell, key)

        if values is None:
            columns.append(col)
            continue

        valid = ~np.isnan(cell.cohort_idx)
        n_valid = int(valid.sum())
        if average_periods:
            col[valid] = (values[:n_valid] + values[n_valid : 2 * n_valid]) / 2
        else:
            col[valid] = values[n_valid : 2 * n_valid]
        columns.append(col)

    return sp.csr_matrix(np.column_stack(columns))


def _stack_drdid_inf_funcs(cell_results, n_units):
    """Stack the optional doubly robust benchmark influence functions."""
    if not cell_results:
        return sp.csr_matrix((n_units, 0))

    columns = []
    for cell in cell_results:
        if cell.drdid_inf_func is None:
            columns.append(np.zeros(n_units))
        else:
            columns.append(cell.drdid_inf_func)

    return sp.csr_matrix(np.column_stack(columns))
