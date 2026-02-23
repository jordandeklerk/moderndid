"""Streaming cell computation for distributed DiD."""

from __future__ import annotations

import numpy as np
from distributed import as_completed, wait

from moderndid.distributed._did_partition import (
    _attach_cell_outcomes,  # noqa: F401
    _build_did_base_partition,  # noqa: F401
    _build_did_partition_arrays,
    _build_did_partition_arrays_wide,  # noqa: F401
    _build_did_rc_partition_arrays,
    _finalize_global_stats,
    _partition_compute_did_if,
    _partition_compute_did_rc_if,
    _partition_compute_did_rc_reg_if,
    _partition_did_global_stats,
    _partition_did_or_gram,
    _partition_did_pscore_gram,
    _partition_did_rc_global_stats,
    _partition_did_rc_or_gram,
    _partition_did_rc_pscore_gram,
    _precompute_did_rc_corrections,
    _precompute_did_rc_reg_corrections,
)

from ._gpu import _maybe_to_gpu
from ._gram import tree_reduce
from ._regression import distributed_logistic_irls_from_futures, distributed_wls_from_futures
from ._utils import sum_global_stats


def streaming_did_cell_single_control(
    client,
    dask_data,
    g,
    t,
    pret,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    n_partitions,
    n_units,
    unique_ids,
    inf_func_mat,
    counter,
    trim_level=0.995,
    filtered_data=None,
    part_futures=None,
    n_cell_override=None,
    control_group="nevertreated",
    weightsname=None,
    use_gpu=False,
    ps_beta=None,
):
    r"""Streaming DiD computation for one :math:`(g, t)` cell with a single control group.

    Computes the doubly robust ATT for a single cell. Each cell requires:

    - 1 propensity score: :math:`P(D=1 \mid X)` via distributed logistic IRLS
    - 1 outcome regression: :math:`E[\Delta Y \mid X, D=0]` via distributed WLS
    - 1 influence function per unit

    All computation stays on workers — only :math:`k`-vectors and
    :math:`k \times k` Gram matrices return to the driver. The influence
    function matrix ``inf_func_mat`` is updated in-place by gathering one
    partition at a time.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    time_col : str
        Column name for time period.
    group_col : str
        Column name for treatment group.
    id_col : str
        Column name for unit identifier.
    y_col : str
        Column name for outcome variable.
    covariate_cols : list of str or None
        Covariate column names (without intercept).
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions for the merged cell data.
    n_units : int
        Total number of unique units in the full panel.
    unique_ids : ndarray
        Sorted array of all unique unit identifiers.
    inf_func_mat : ndarray of shape (n_units, n_cells)
        Influence function matrix, updated in-place at column ``counter``.
    counter : int
        Column index into ``inf_func_mat`` for this cell.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data.
    part_futures : list of Future or None, default None
        Pre-built partition futures from the wide-pivot path.
    n_cell_override : int or None, default None
        Number of units in the cell. Required when ``part_futures`` is
        provided.

    Returns
    -------
    float or None
        The DiD ATT for this cell, or ``None`` if the cell has
        insufficient data.
    """
    if part_futures is not None:
        n_cell = n_cell_override if n_cell_override is not None else 0
        part_futures = _maybe_to_gpu(client, part_futures, use_gpu)
    else:
        part_futures, n_cell = prepare_did_cell_partitions(
            client,
            dask_data,
            g,
            t,
            pret,
            control_group,
            time_col,
            group_col,
            id_col,
            y_col,
            covariate_cols,
            n_partitions,
            filtered_data=filtered_data,
            weightsname=weightsname,
            use_gpu=use_gpu,
        )

    if part_futures is None or n_cell == 0:
        return None

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    if ps_beta is not None:
        if est_method == "ipw":
            or_beta = np.zeros(k, dtype=np.float64)
        else:
            or_beta = distributed_wls_from_futures(
                client,
                part_futures,
                _partition_did_or_gram,
            )
    else:
        ps_beta, or_beta = streaming_did_nuisance_coefficients(client, part_futures, est_method, k)

    global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_did_global_stats(
        client,
        part_futures,
        ps_beta,
        or_beta,
        est_method,
        trim_level,
    )

    if global_agg is None:
        return None

    dr_att = global_agg["dr_att"]

    if_futures = [
        client.submit(
            _partition_compute_did_if,
            pf,
            ps_beta,
            or_beta,
            global_agg,
            est_method,
            trim_level,
            precomp_hess_m2,
            precomp_xpx_inv_m1,
            precomp_xpx_inv_m3,
        )
        for pf in part_futures
    ]

    scale = n_units / n_cell
    for fut in as_completed(if_futures):
        ids_part, if_part = fut.result()
        if len(ids_part) == 0:
            continue
        if_scaled = scale * if_part
        indices = np.searchsorted(unique_ids, ids_part)
        valid = (indices < len(unique_ids)) & (unique_ids[np.minimum(indices, len(unique_ids) - 1)] == ids_part)
        inf_func_mat[indices[valid], counter] = if_scaled[valid]
        del ids_part, if_part, if_scaled

    return dr_att


def streaming_did_rc_cell_single_control(
    client,
    dask_data,
    g,
    t,
    pret,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    est_method,
    n_partitions,
    n_obs,
    inf_func_mat,
    counter,
    trim_level=0.995,
    control_group="nevertreated",
    weightsname=None,
    collect_if=False,
    use_gpu=False,
):
    r"""Streaming DiD RC computation for one :math:`(g, t)` cell.

    Unlike the panel path which merges post/pre on ``id_col``, this
    concatenates post and pre observations and adds a ``post`` indicator.
    Uses 4 outcome regressions (one per ``(D, post)`` cell) and the
    RC influence function formula from ``drdid_rc.py``.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long format.
    g, t, pret : int or float
        Treatment cohort, current period, pre-treatment period.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Dask partitions.
    n_obs : int
        Total number of observations in the full long-format data.
    inf_func_mat : ndarray of shape (n_obs, n_cells) or None
        Influence function matrix, updated in-place. Ignored when
        ``collect_if=True``.
    counter : int
        Column index into ``inf_func_mat``.
    trim_level : float, default 0.995
        Propensity score trimming threshold.
    control_group : {"nevertreated", "notyettreated"}
        Control group type.
    weightsname : str or None
        Weight column name.
    collect_if : bool, default False
        When True, return ``(att, if_array)`` instead of writing IF
        directly to ``inf_func_mat``.

    Returns
    -------
    float or (float, ndarray) or None
        The DiD ATT for this cell.  When ``collect_if=True``, returns
        ``(att, if_array)`` where ``if_array`` has length ``n_obs``.
    """
    part_futures, n_cell = prepare_did_rc_cell_partitions(
        client,
        dask_data,
        g,
        t,
        pret,
        control_group,
        time_col,
        group_col,
        id_col,
        y_col,
        covariate_cols,
        n_partitions,
        weightsname=weightsname,
        use_gpu=use_gpu,
    )

    if part_futures is None or n_cell == 0:
        return None

    first = part_futures[0].result()
    if first is None:
        return None
    k = first["X"].shape[1]

    # Propensity score: P(D=1 | X) on all obs
    if est_method == "reg":
        ps_beta = np.zeros(k, dtype=np.float64)
    else:
        ps_beta = distributed_logistic_irls_from_futures(
            client,
            part_futures,
            _partition_did_rc_pscore_gram,
            k,
        )

    or_betas = {}
    if est_method == "ipw":
        for key in ["cont_pre", "cont_post", "treat_pre", "treat_post"]:
            or_betas[key] = np.zeros(k, dtype=np.float64)
    elif est_method == "reg":
        for d_val, post_val, key in [
            (0, 0, "cont_pre"),
            (0, 1, "cont_post"),
        ]:
            or_betas[key] = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )
    else:
        for d_val, post_val, key in [
            (0, 0, "cont_pre"),
            (0, 1, "cont_post"),
            (1, 0, "treat_pre"),
            (1, 1, "treat_post"),
        ]:
            or_betas[key] = distributed_wls_from_futures(
                client,
                part_futures,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )

    # Global stats via tree-reduce
    futures = [
        client.submit(
            _partition_did_rc_global_stats,
            pf,
            ps_beta,
            or_betas,
            est_method,
            trim_level,
        )
        for pf in part_futures
    ]
    agg = tree_reduce(client, futures, sum_global_stats)

    if agg is None or agg["n_sub"] == 0:
        return None

    n_sub = agg["n_sub"]

    if est_method == "reg":
        mean_w_treat_pre = agg["sum_w_treat_pre"] / n_sub
        mean_w_treat_post = agg["sum_w_treat_post"] / n_sub
        mean_w_d = agg["sum_w_d"] / n_sub

        eta_treat_pre = (agg["sum_reg_att_treat_pre"] / n_sub) / mean_w_treat_pre if mean_w_treat_pre > 0 else 0.0
        eta_treat_post = (agg["sum_reg_att_treat_post"] / n_sub) / mean_w_treat_post if mean_w_treat_post > 0 else 0.0
        eta_cont = (agg["sum_reg_att_cont"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0

        dr_att = (eta_treat_post - eta_treat_pre) - eta_cont

        global_agg = {
            "mean_w_treat_pre": mean_w_treat_pre,
            "mean_w_treat_post": mean_w_treat_post,
            "mean_w_d": mean_w_d,
            "eta_treat_pre": eta_treat_pre,
            "eta_treat_post": eta_treat_post,
            "eta_cont": eta_cont,
            "n_sub": n_sub,
            "dr_att": dr_att,
        }

        precomp = _precompute_did_rc_reg_corrections(agg, global_agg, n_sub, k)
    else:
        mean_w_treat_pre = agg["sum_w_treat_pre"] / n_sub
        mean_w_treat_post = agg["sum_w_treat_post"] / n_sub
        mean_w_cont_pre = agg["sum_w_cont_pre"] / n_sub
        mean_w_cont_post = agg["sum_w_cont_post"] / n_sub
        mean_w_d = agg["sum_w_d"] / n_sub
        mean_w_dt1 = agg["sum_w_dt1"] / n_sub
        mean_w_dt0 = agg["sum_w_dt0"] / n_sub

        att_treat_pre = (agg["sum_eta_treat_pre"] / n_sub) / mean_w_treat_pre if mean_w_treat_pre > 0 else 0.0
        att_treat_post = (agg["sum_eta_treat_post"] / n_sub) / mean_w_treat_post if mean_w_treat_post > 0 else 0.0
        att_cont_pre = (agg["sum_eta_cont_pre"] / n_sub) / mean_w_cont_pre if mean_w_cont_pre > 0 else 0.0
        att_cont_post = (agg["sum_eta_cont_post"] / n_sub) / mean_w_cont_post if mean_w_cont_post > 0 else 0.0
        att_d_post = (agg["sum_eta_d_post"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0
        att_dt1_post = (agg["sum_eta_dt1_post"] / n_sub) / mean_w_dt1 if mean_w_dt1 > 0 else 0.0
        att_d_pre = (agg["sum_eta_d_pre"] / n_sub) / mean_w_d if mean_w_d > 0 else 0.0
        att_dt0_pre = (agg["sum_eta_dt0_pre"] / n_sub) / mean_w_dt0 if mean_w_dt0 > 0 else 0.0

        dr_att = (
            (att_treat_post - att_treat_pre)
            - (att_cont_post - att_cont_pre)
            + (att_d_post - att_dt1_post)
            - (att_d_pre - att_dt0_pre)
        )

        global_agg = {
            "mean_w_treat_pre": mean_w_treat_pre,
            "mean_w_treat_post": mean_w_treat_post,
            "mean_w_cont_pre": mean_w_cont_pre,
            "mean_w_cont_post": mean_w_cont_post,
            "mean_w_d": mean_w_d,
            "mean_w_dt1": mean_w_dt1,
            "mean_w_dt0": mean_w_dt0,
            "att_treat_pre": att_treat_pre,
            "att_treat_post": att_treat_post,
            "att_cont_pre": att_cont_pre,
            "att_cont_post": att_cont_post,
            "att_d_post": att_d_post,
            "att_dt1_post": att_dt1_post,
            "att_d_pre": att_d_pre,
            "att_dt0_pre": att_dt0_pre,
            "n_sub": n_sub,
            "dr_att": dr_att,
        }

        precomp = _precompute_did_rc_corrections(agg, global_agg, est_method, n_sub, k)

    if est_method == "reg":
        if_futures = [
            client.submit(
                _partition_compute_did_rc_reg_if,
                pf,
                or_betas,
                global_agg,
                precomp,
            )
            for pf in part_futures
        ]
    else:
        if_futures = [
            client.submit(
                _partition_compute_did_rc_if,
                pf,
                ps_beta,
                or_betas,
                global_agg,
                precomp,
                est_method,
                trim_level,
            )
            for pf in part_futures
        ]

    scale = n_obs / n_cell
    if collect_if:
        if_full = np.zeros(n_obs, dtype=np.float64)
        for fut in as_completed(if_futures):
            ids_part, if_part = fut.result()
            if len(ids_part) == 0:
                continue
            if_full[ids_part] = scale * if_part
            del ids_part, if_part
        return dr_att, if_full
    else:
        for fut in as_completed(if_futures):
            ids_part, if_part = fut.result()
            if len(ids_part) == 0:
                continue
            inf_func_mat[ids_part, counter] = scale * if_part
            del ids_part, if_part
        return dr_att


def prepare_did_rc_cell_partitions(
    client,
    dask_data,
    g,
    t,
    pret,
    control_group,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    n_partitions,
    weightsname=None,
    use_gpu=False,
):
    """Filter and concatenate post/pre periods for RC, returning partition futures.

    Instead of merging on ``id_col``, concatenates post and pre rows and
    adds a ``_post`` indicator column.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    control_group : {"nevertreated", "notyettreated"}
        Which units to include as controls.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    n_partitions : int
        Number of Dask partitions.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    part_futures : list of Future or None
        Futures to partition dicts, or None if no data.
    n_cell : int
        Total number of observations (pre + post).
    """
    max_period = max(t, pret)

    if control_group == "nevertreated":
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
    else:
        group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)

    filtered = dask_data.loc[group_filter]

    time_filter = (filtered[time_col] == t) | (filtered[time_col] == pret)
    concat_dask = filtered.loc[time_filter]
    concat_dask = concat_dask.assign(_post=(concat_dask[time_col] == t).astype(int))

    keep_cols = [id_col, group_col, y_col, "_post"]
    if weightsname is not None:
        keep_cols.append(weightsname)
    if covariate_cols:
        keep_cols = keep_cols + [c for c in covariate_cols if c not in keep_cols]

    concat_dask = concat_dask[keep_cols]
    concat_dask = concat_dask.reset_index(drop=True)
    concat_dask = concat_dask.repartition(npartitions=n_partitions).persist()
    wait(concat_dask)

    n_cell = len(concat_dask)
    if n_cell == 0:
        return None, 0

    partition_lengths = concat_dask.map_partitions(len).compute()
    offsets = np.cumsum([0, *list(partition_lengths.values)[:-1]])

    delayed_parts = concat_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_did_rc_partition_arrays,
            pdf_f,
            offset,
            y_col,
            group_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f, offset in zip(pdf_futures, offsets, strict=False)
    ]
    part_futures = _maybe_to_gpu(client, part_futures, use_gpu)

    return part_futures, n_cell


def prepare_did_cell_partitions(
    client,
    dask_data,
    g,
    t,
    pret,
    control_group,
    time_col,
    group_col,
    id_col,
    y_col,
    covariate_cols,
    n_partitions,
    filtered_data=None,
    weightsname=None,
    use_gpu=False,
):
    """Filter and merge post/pre periods on workers, returning partition futures.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    dask_data : dask.dataframe.DataFrame
        Persisted Dask DataFrame in long panel format.
    g : int or float
        Treatment cohort identifier.
    t : int or float
        Current time period.
    pret : int or float
        Pre-treatment (base) period.
    control_group : {"nevertreated", "notyettreated"}
        Which units to include as controls.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    n_partitions : int
        Number of Dask partitions for the merged data.
    filtered_data : dask.dataframe.DataFrame or None, default None
        Pre-filtered cohort data.

    Returns
    -------
    part_futures : list of Future or None
        Futures resolving to numpy dicts, or ``None`` if the cell is empty.
    n_cell : int
        Number of units in the cell.
    """
    if filtered_data is not None:
        filtered = filtered_data
    else:
        max_period = max(t, pret)

        if control_group == "nevertreated":
            group_filter = (dask_data[group_col] == 0) | (dask_data[group_col] == g)
        else:
            group_filter = (
                (dask_data[group_col] == 0) | (dask_data[group_col] > max_period) | (dask_data[group_col] == g)
            )

        filtered = dask_data.loc[group_filter]

    post_cols = [id_col, group_col, y_col]
    if weightsname is not None:
        post_cols.append(weightsname)
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_dask = filtered.loc[filtered[time_col] == t][post_cols]
    pre_dask = filtered.loc[filtered[time_col] == pret][[id_col, y_col]]
    pre_dask = pre_dask.rename(columns={y_col: "_y_pre"})

    merged_dask = post_dask.merge(pre_dask, on=id_col, how="inner")
    merged_dask = merged_dask.repartition(npartitions=n_partitions).persist()
    wait(merged_dask)

    n_cell = len(merged_dask)
    if n_cell == 0:
        return None, 0

    delayed_parts = merged_dask.to_delayed()
    pdf_futures = client.compute(delayed_parts)
    part_futures = [
        client.submit(
            _build_did_partition_arrays,
            pdf_f,
            id_col,
            y_col,
            group_col,
            g,
            covariate_cols,
            weightsname,
        )
        for pdf_f in pdf_futures
    ]
    part_futures = _maybe_to_gpu(client, part_futures, use_gpu)

    return part_futures, n_cell


def streaming_did_nuisance_coefficients(client, part_futures, est_method, k):
    r"""Compute nuisance model coefficients for DiD.

    Estimates:

    - **Propensity score** (unless ``est_method="reg"``): logistic regression
      via distributed IRLS for :math:`P(D=1 \mid X)`.
    - **Outcome regression** (unless ``est_method="ipw"``): WLS regression
      of :math:`\Delta Y` on :math:`X` among control units :math:`D = 0`.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    k : int
        Number of columns in the design matrix :math:`X`.

    Returns
    -------
    ps_beta : ndarray of shape (k,)
        Propensity score coefficients.
    or_beta : ndarray of shape (k,)
        Outcome regression coefficients.
    """
    if est_method == "reg":
        ps_beta = np.zeros(k, dtype=np.float64)
    else:
        ps_beta = distributed_logistic_irls_from_futures(
            client,
            part_futures,
            _partition_did_pscore_gram,
            k,
        )

    if est_method == "ipw":
        or_beta = np.zeros(k, dtype=np.float64)
    else:
        or_beta = distributed_wls_from_futures(
            client,
            part_futures,
            _partition_did_or_gram,
        )

    return ps_beta, or_beta


def streaming_did_global_stats(client, part_futures, ps_beta, or_beta, est_method, trim_level=0.995):
    r"""Compute global aggregate statistics and precomputed correction vectors.

    Performs a single tree-reduce round to aggregate per-partition sufficient
    statistics into global means, ATT components, and the inverse Hessian and
    design matrix products needed by the influence function.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    part_futures : list of Future
        Futures to partition dicts.
    ps_beta : ndarray
        Propensity score coefficients.
    or_beta : ndarray
        Outcome regression coefficients.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    trim_level : float, default 0.995
        Propensity score trimming threshold.

    Returns
    -------
    global_agg : dict or None
        Aggregate statistics including ``mean_w_treat``, ``mean_w_control``,
        ``att_treat``, ``att_control``, ``dr_att``, and ``n_sub``.
    precomp_hess_m2 : ndarray of shape (k,) or None
        Precomputed :math:`H^{-1} m_2` vector for PS correction.
    precomp_xpx_inv_m1 : ndarray of shape (k,) or None
        Precomputed :math:`(X^T X)^{-1} m_1` vector for OR correction (treated).
    precomp_xpx_inv_m3 : ndarray of shape (k,) or None
        Precomputed :math:`(X^T X)^{-1} m_3` vector for OR correction (control).
    """
    futures = [
        client.submit(
            _partition_did_global_stats,
            pf,
            ps_beta,
            or_beta,
            est_method,
            trim_level,
        )
        for pf in part_futures
    ]

    agg = tree_reduce(client, futures, sum_global_stats)

    return _finalize_global_stats(agg, est_method)
