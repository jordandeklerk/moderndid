"""Distributed influence function computation and variance estimation."""

from __future__ import annotations

import warnings

import numpy as np

from ._gram import _sum_gram_pair, tree_reduce


def compute_did_distributed(
    client,
    subgroup,
    covariates,
    weights,
    pscores,
    or_results,
    est_method,
    n_total,
    n_partitions,
):
    """Compute all DiD components and combine into DDD estimate.

    Distributed version of ``moderndid.didtriple.nuisance.compute_all_did``.
    Influence functions are computed locally per partition and variance is
    computed via distributed gram of the influence function matrix.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    subgroup : ndarray
        Subgroup indicators (1, 2, 3, or 4).
    covariates : ndarray
        Covariates including intercept, shape (n, k).
    weights : ndarray
        Observation weights.
    pscores : list of DistPScoreResult
        Propensity score results for comparisons [3, 2, 1].
    or_results : list of DistOutcomeRegResult
        Outcome regression results for comparisons [3, 2, 1].
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_total : int
        Total number of units.
    n_partitions : int
        Number of partitions.

    Returns
    -------
    tuple[list, float, ndarray]
        DiD results list, DDD ATT estimate, combined influence function.
    """
    did_results = []
    for i, comp_subgroup in enumerate([3, 2, 1]):
        did_result = _compute_did(
            subgroup=subgroup,
            covariates=covariates,
            weights=weights,
            comparison_subgroup=comp_subgroup,
            pscore_result=pscores[i],
            or_result=or_results[i],
            est_method=est_method,
            n_total=n_total,
        )
        did_results.append(did_result)

    dr_att_3, inf_func_3 = did_results[0]
    dr_att_2, inf_func_2 = did_results[1]
    dr_att_1, inf_func_1 = did_results[2]

    ddd_att = dr_att_3 + dr_att_2 - dr_att_1

    n = n_total
    n3 = int(np.sum((subgroup == 4) | (subgroup == 3)))
    n2 = int(np.sum((subgroup == 4) | (subgroup == 2)))
    n1 = int(np.sum((subgroup == 4) | (subgroup == 1)))

    w3 = n / n3 if n3 > 0 else 0
    w2 = n / n2 if n2 > 0 else 0
    w1 = n / n1 if n1 > 0 else 0

    inf_func = w3 * inf_func_3 + w2 * inf_func_2 - w1 * inf_func_1

    return did_results, float(ddd_att), inf_func


def compute_variance_distributed(client, inf_func_partitions, n_total, k):
    """Compute variance from distributed influence function partitions.

    Computes ``V = (1/n) * Psi' @ Psi`` without materializing the full
    matrix on one node. Each worker computes its local ``Psi_i' @ Psi_i``
    and results are tree-reduced.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    inf_func_partitions : list of ndarray
        Per-partition influence function arrays.
    n_total : int
        Total number of observations.
    k : int
        Number of influence function columns.

    Returns
    -------
    se : ndarray of shape (k,)
        Standard errors.
    """

    def _local_gram(chunk):
        return chunk.T @ chunk, np.zeros(chunk.shape[1]), chunk.shape[0]

    futures = [client.submit(_local_gram, part) for part in inf_func_partitions]
    PtP, _, _ = tree_reduce(client, futures, _sum_gram_pair)
    V = PtP / n_total
    return np.sqrt(np.diag(V) / n_total)


def _compute_did(
    subgroup,
    covariates,
    weights,
    comparison_subgroup,
    pscore_result,
    or_result,
    est_method,
    n_total,
):
    """Compute doubly robust DiD for one subgroup comparison.

    This mirrors ``moderndid.didtriple.nuisance._compute_did`` exactly,
    operating on in-memory arrays.

    Returns
    -------
    tuple[float, ndarray]
        ATT estimate and full-length influence function.
    """
    mask = (subgroup == 4) | (subgroup == comparison_subgroup)
    sub_subgroup = subgroup[mask]
    sub_covariates = covariates[mask]
    sub_weights = weights[mask]

    pscore = pscore_result.propensity_scores
    hessian = pscore_result.hessian_matrix
    keep_ps = pscore_result.keep_ps.astype(float)
    delta_y = or_result.delta_y
    or_delta = or_result.or_delta

    pa4 = (sub_subgroup == 4).astype(float)
    pa_comp = (sub_subgroup == comparison_subgroup).astype(float)

    w_treat = keep_ps * sub_weights * pa4
    if est_method == "reg":
        w_control = keep_ps * sub_weights * pa_comp
    else:
        w_control = keep_ps * sub_weights * pscore * pa_comp / (1 - pscore)

    riesz_treat = w_treat * (delta_y - or_delta)
    riesz_control = w_control * (delta_y - or_delta)

    mean_w_treat = np.mean(w_treat)
    mean_w_control = np.mean(w_control)

    if mean_w_treat == 0:
        raise ValueError(
            f"No effectively treated units (subgroup 4) in comparison with subgroup {comparison_subgroup}."
        )
    if mean_w_control == 0:
        raise ValueError(f"No effectively control units (subgroup {comparison_subgroup}) after weighting.")

    att_treat = np.mean(riesz_treat) / mean_w_treat
    att_control = np.mean(riesz_control) / mean_w_control
    dr_att = att_treat - att_control

    inf_func_sub = _compute_inf_func(
        sub_covariates=sub_covariates,
        sub_weights=sub_weights,
        pa4=pa4,
        pa_comp=pa_comp,
        pscore=pscore,
        hessian=hessian,
        delta_y=delta_y,
        or_delta=or_delta,
        w_treat=w_treat,
        w_control=w_control,
        riesz_treat=riesz_treat,
        riesz_control=riesz_control,
        att_treat=att_treat,
        att_control=att_control,
        mean_w_treat=mean_w_treat,
        mean_w_control=mean_w_control,
        est_method=est_method,
    )

    inf_func = np.zeros(n_total)
    mask_indices = np.where(mask)[0]
    inf_func[mask_indices] = inf_func_sub

    return float(dr_att), inf_func


def _compute_inf_func(
    sub_covariates,
    sub_weights,
    pa4,
    pa_comp,
    pscore,
    hessian,
    delta_y,
    or_delta,
    w_treat,
    w_control,
    riesz_treat,
    riesz_control,
    att_treat,
    att_control,
    mean_w_treat,
    mean_w_control,
    est_method,
):
    """Compute influence function for one DiD comparison.

    Mirrors ``moderndid.didtriple.nuisance._compute_inf_func``.
    """
    n_sub = len(sub_weights)

    if est_method == "reg":
        inf_control_pscore = np.zeros(n_sub)
    else:
        m2 = np.mean(
            (w_control * (delta_y - or_delta - att_control))[:, None] * sub_covariates,
            axis=0,
        )
        score_ps = (sub_weights * (pa4 - pscore))[:, None] * sub_covariates
        asy_lin_rep_ps = score_ps @ hessian
        inf_control_pscore = asy_lin_rep_ps @ m2

    if est_method == "ipw":
        inf_treat_or = np.zeros(n_sub)
        inf_cont_or = np.zeros(n_sub)
    else:
        m1 = np.mean(w_treat[:, None] * sub_covariates, axis=0)
        m3 = np.mean(w_control[:, None] * sub_covariates, axis=0)

        or_x = (sub_weights * pa_comp)[:, None] * sub_covariates
        or_ex = (sub_weights * pa_comp * (delta_y - or_delta))[:, None] * sub_covariates
        xpx = or_x.T @ sub_covariates / n_sub

        s = np.linalg.svd(xpx, compute_uv=False)
        cond_num = s[0] / s[-1] if s[-1] > 0 else float("inf")
        if cond_num > 1 / np.finfo(float).eps:
            warnings.warn(
                "Outcome regression design matrix is nearly singular.",
                UserWarning,
            )
            xpx_inv = np.linalg.pinv(xpx)
        else:
            xpx_inv = np.linalg.solve(xpx, np.eye(xpx.shape[0]))

        asy_linear_or = or_ex @ xpx_inv
        inf_treat_or = -asy_linear_or @ m1
        inf_cont_or = -asy_linear_or @ m3

    inf_control_did = riesz_control - w_control * att_control
    inf_treat_did = riesz_treat - w_treat * att_treat

    inf_control = (inf_control_did + inf_control_pscore + inf_cont_or) / mean_w_control
    inf_treat = (inf_treat_did + inf_treat_or) / mean_w_treat

    return inf_treat - inf_control
