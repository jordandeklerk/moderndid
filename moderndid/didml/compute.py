"""Group-time computation for the doubly-robust ML DiD estimator."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from moderndid.core.parallel import parallel_map
from moderndid.did.compute_att_gt import get_did_cohort_index
from moderndid.drdid.estimators.drdid_panel import drdid_panel

from .container import ComputeDIDMLResult, DIDMLCellResult
from .lnw import lnw_did
from .weights import amle_weights


def compute_didml(data, *, n_jobs=1):
    r"""Run the doubly-robust ML DiD score over every :math:`(g, t)` cell.

    Iterates the cohort by time-period grid, builds the stacked pre/post
    panel for each cell, fits AMLE weights and the LNW score, and stores
    the resulting CATT / score / gamma vectors as sparse columns.

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
        - **scores**: Sparse :math:`(n_{units}, n_{cells})` matrix of per-unit DR score contributions
        - **gammas**: Sparse :math:`(n_{units}, n_{cells})` matrix of per-unit AMLE weights
        - **drdid_inf_funcs**: Sparse :math:`(n_{units}, n_{cells})` matrix of benchmark influence functions
    """
    n_units = data.config.id_count
    n_periods = len(data.config.time_periods)
    n_treated = data.config.treated_groups_count
    n_time_periods = n_periods - 1

    pairs = [(g, t) for g in range(n_treated) for t in range(n_time_periods)]
    args_list = [(g_idx, t_idx, data) for g_idx, t_idx in pairs]

    raw_results = parallel_map(_process_didml_cell, args_list, n_jobs=n_jobs)
    cell_results = [r for r in raw_results if r is not None]

    cates = _stack_per_unit(cell_results, n_units, "tau_hat")
    scores = _stack_per_unit(cell_results, n_units, "score")
    gammas = _stack_per_unit(cell_results, n_units, "gamma")
    drdid_infs = _stack_drdid_inf_funcs(cell_results, n_units)

    return ComputeDIDMLResult(
        cell_results=cell_results,
        cates=cates,
        scores=scores,
        gammas=gammas,
        drdid_inf_funcs=drdid_infs,
    )


def _process_didml_cell(group_idx, time_idx, data):
    """Estimate one :math:`(g, t)` cell."""
    cfg = data.config
    time_factor = 1

    treated_groups = cfg.treated_groups
    time_periods = cfg.time_periods

    pre_periods = np.where(time_periods < (treated_groups[group_idx] - cfg.anticipation))[0]
    is_post_treatment = treated_groups[group_idx] <= time_periods[time_idx + time_factor]

    if is_post_treatment:
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
        return None

    cohort_idx = get_did_cohort_index(group_idx, time_idx, time_factor, pre_treatment_idx, data)
    valid = ~np.isnan(cohort_idx)

    if not (np.any(cohort_idx[valid] == 1) and np.any(cohort_idx[valid] == 0)):
        return None

    Y_pre = data.outcomes_tensor[pre_treatment_idx][valid]
    Y_post = data.outcomes_tensor[time_idx + time_factor][valid]
    Y = np.concatenate([Y_pre, Y_post])

    cohort_valid = cohort_idx[valid]
    n_valid = int(valid.sum())
    G = np.concatenate([cohort_valid, cohort_valid])
    T = np.concatenate([np.zeros(n_valid), np.ones(n_valid)])

    X_pre = data.covariates_tensor[pre_treatment_idx][valid]
    X_post = data.covariates_tensor[time_idx + time_factor][valid]
    X = np.vstack([X_pre, X_post])

    if X.shape[1] > 0 and np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]

    if X.shape[1] == 0:
        warnings.warn(
            f"No covariates after intercept removal for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}; skipping cell.",
            UserWarning,
        )
        return None

    try:
        gamma = amle_weights(X, post_indicator=T, cohort_indicator=G, zeta=cfg.zeta) if cfg.use_gamma else None
    except (ValueError, RuntimeError) as exc:
        warnings.warn(
            f"AMLE weights failed for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}: {exc}",
            UserWarning,
        )
        return None

    try:
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
            tune_penalty=cfg.tune_penalty,
            lambda_choice=cfg.lambda_choice,
            random_state=cfg.random_state,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"LNW estimation failed for group {treated_groups[group_idx]}, "
            f"period {time_periods[time_idx + time_factor]}: {exc}",
            UserWarning,
        )
        return None

    drdid_att, drdid_se, drdid_inf = _maybe_run_drdid_benchmark(data, cohort_idx, valid, pre_treatment_idx, time_idx)

    return DIDMLCellResult(
        group=float(treated_groups[group_idx]),
        year=float(time_periods[time_idx + time_factor]),
        post=int(is_post_treatment),
        pre_idx=int(pre_treatment_idx),
        post_idx=int(time_idx + time_factor),
        att=float(lnw_result["TAU_hat"]),
        se=float(lnw_result["std_err"]),
        tau_hat=lnw_result["tau_hat"],
        score=lnw_result["score"],
        gamma=gamma,
        cohort_idx=cohort_idx,
        drdid_att=drdid_att,
        drdid_se=drdid_se,
        drdid_inf_func=drdid_inf,
    )


def _maybe_run_drdid_benchmark(data, cohort_idx, valid, pre_treatment_idx, time_idx):
    """Compute the doubly-robust benchmark ATT for the cell, when configured."""
    if not getattr(data.config, "compute_drdid_benchmark", False):
        return None, None, None

    y1 = data.outcomes_tensor[time_idx + 1][valid]
    y0 = data.outcomes_tensor[pre_treatment_idx][valid]
    d = cohort_idx[valid]
    weights = data.weights[valid]
    covariates = data.covariates_tensor[min(pre_treatment_idx, time_idx)][valid]

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


def _stack_per_unit(cell_results, n_units, key):
    """Stack a per-cell length-:math:`n_{cell}` array as a sparse :math:`(n, n_{cells})` column."""
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
        post_block = values[n_valid : 2 * n_valid] if values.shape[0] >= 2 * n_valid else values[:n_valid]

        col[valid] = post_block
        columns.append(col)

    return sp.csr_matrix(np.column_stack(columns))


def _stack_drdid_inf_funcs(cell_results, n_units):
    """Stack the optional doubly-robust benchmark influence functions."""
    if not cell_results:
        return sp.csr_matrix((n_units, 0))

    columns = []
    for cell in cell_results:
        if cell.drdid_inf_func is None:
            columns.append(np.zeros(n_units))
        else:
            columns.append(cell.drdid_inf_func)

    return sp.csr_matrix(np.column_stack(columns))
