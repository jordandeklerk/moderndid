"""Streaming cell computation for distributed DiD via PySpark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

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
from moderndid.distributed._utils import sum_global_stats

from ._gpu import _maybe_to_gpu_dict
from ._regression import distributed_logistic_irls_from_partitions, distributed_wls_from_partitions


def streaming_did_cell_single_control(
    spark,
    sdf,
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
    part_data_list=None,
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

    All computation collects partition-level numpy dicts to the driver,
    where nuisance models and influence functions are computed. The
    influence function matrix ``inf_func_mat`` is updated in-place.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    sdf : pyspark.sql.DataFrame
        Cached Spark DataFrame in long panel format.
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
        Number of Spark partitions for the merged cell data.
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
    filtered_data : pyspark.sql.DataFrame or None, default None
        Pre-filtered cohort data.
    part_data_list : list of dict or None, default None
        Pre-built partition data dicts from the wide-pivot path.
    n_cell_override : int or None, default None
        Number of units in the cell. Required when ``part_data_list`` is
        provided.
    control_group : {"nevertreated", "notyettreated"}
        Control group type.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    float or None
        The DiD ATT for this cell, or ``None`` if the cell has
        insufficient data.
    """
    if part_data_list is not None:
        n_cell = n_cell_override if n_cell_override is not None else 0
        part_data_list = [_maybe_to_gpu_dict(pd, use_gpu) for pd in part_data_list]
    else:
        part_data_list, n_cell = prepare_did_cell_partitions(
            spark,
            sdf,
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

    if part_data_list is None or n_cell == 0:
        return None

    first = part_data_list[0]
    if first is None:
        return None
    k = first["X"].shape[1]

    if ps_beta is not None:
        if est_method == "ipw":
            or_beta = np.zeros(k, dtype=np.float64)
        else:
            or_beta = distributed_wls_from_partitions(
                spark,
                part_data_list,
                _partition_did_or_gram,
            )
    else:
        ps_beta, or_beta = streaming_did_nuisance_coefficients(spark, part_data_list, est_method, k)

    global_agg, precomp_hess_m2, precomp_xpx_inv_m1, precomp_xpx_inv_m3 = streaming_did_global_stats(
        spark,
        part_data_list,
        ps_beta,
        or_beta,
        est_method,
        trim_level,
    )

    if global_agg is None:
        return None

    dr_att = global_agg["dr_att"]

    scale = n_units / n_cell
    for pf in part_data_list:
        ids_part, if_part = _partition_compute_did_if(
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
        if len(ids_part) == 0:
            continue
        if_scaled = scale * if_part
        indices = np.searchsorted(unique_ids, ids_part)
        valid = (indices < len(unique_ids)) & (unique_ids[np.minimum(indices, len(unique_ids) - 1)] == ids_part)
        inf_func_mat[indices[valid], counter] = if_scaled[valid]
        del ids_part, if_part, if_scaled

    return dr_att


def streaming_did_rc_cell_single_control(
    spark,
    sdf,
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
    spark : pyspark.sql.SparkSession
        Active Spark session.
    sdf : pyspark.sql.DataFrame
        Cached Spark DataFrame in long format.
    g, t, pret : int or float
        Treatment cohort, current period, pre-treatment period.
    time_col, group_col, id_col, y_col : str
        Column names.
    covariate_cols : list of str or None
        Covariate column names.
    est_method : {"dr", "reg", "ipw"}
        Estimation method.
    n_partitions : int
        Number of Spark partitions.
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
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    float or (float, ndarray) or None
        The DiD ATT for this cell.  When ``collect_if=True``, returns
        ``(att, if_array)`` where ``if_array`` has length ``n_obs``.
    """
    part_data_list, n_cell = prepare_did_rc_cell_partitions(
        spark,
        sdf,
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

    if part_data_list is None or n_cell == 0:
        return None

    first = part_data_list[0]
    if first is None:
        return None
    k = first["X"].shape[1]

    if est_method == "reg":
        ps_beta = np.zeros(k, dtype=np.float64)
    else:
        ps_beta = distributed_logistic_irls_from_partitions(
            spark,
            part_data_list,
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
            or_betas[key] = distributed_wls_from_partitions(
                spark,
                part_data_list,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )
    else:
        for d_val, post_val, key in [
            (0, 0, "cont_pre"),
            (0, 1, "cont_post"),
            (1, 0, "treat_pre"),
            (1, 1, "treat_post"),
        ]:
            or_betas[key] = distributed_wls_from_partitions(
                spark,
                part_data_list,
                lambda pd, _d=d_val, _p=post_val: _partition_did_rc_or_gram(pd, _d, _p),
            )

    # Global stats via driver-side reduce
    stats_list = [
        _partition_did_rc_global_stats(
            pf,
            ps_beta,
            or_betas,
            est_method,
            trim_level,
        )
        for pf in part_data_list
    ]
    agg = None
    for s in stats_list:
        agg = sum_global_stats(agg, s)

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

    scale = n_obs / n_cell
    if collect_if:
        if_full = np.zeros(n_obs, dtype=np.float64)
        for pf in part_data_list:
            if est_method == "reg":
                ids_part, if_part = _partition_compute_did_rc_reg_if(
                    pf,
                    or_betas,
                    global_agg,
                    precomp,
                )
            else:
                ids_part, if_part = _partition_compute_did_rc_if(
                    pf,
                    ps_beta,
                    or_betas,
                    global_agg,
                    precomp,
                    est_method,
                    trim_level,
                )
            if len(ids_part) == 0:
                continue
            if_full[ids_part] = scale * if_part
            del ids_part, if_part
        return dr_att, if_full
    else:
        for pf in part_data_list:
            if est_method == "reg":
                ids_part, if_part = _partition_compute_did_rc_reg_if(
                    pf,
                    or_betas,
                    global_agg,
                    precomp,
                )
            else:
                ids_part, if_part = _partition_compute_did_rc_if(
                    pf,
                    ps_beta,
                    or_betas,
                    global_agg,
                    precomp,
                    est_method,
                    trim_level,
                )
            if len(ids_part) == 0:
                continue
            inf_func_mat[ids_part, counter] = scale * if_part
            del ids_part, if_part
        return dr_att


def prepare_did_rc_cell_partitions(
    spark,
    sdf,
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
    """Filter and concatenate post/pre periods for RC, returning partition data dicts.

    Instead of merging on ``id_col``, concatenates post and pre rows and
    adds a ``_post`` indicator column.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    sdf : pyspark.sql.DataFrame
        Cached Spark DataFrame in long format.
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
        Number of Spark partitions.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    part_data_list : list of dict or None
        Partition data dicts, or None if no data.
    n_cell : int
        Total number of observations (pre + post).
    """
    max_period = max(t, pret)

    if control_group == "nevertreated":
        group_filter = (F.col(group_col) == 0) | (F.col(group_col) == g)
    else:
        group_filter = (F.col(group_col) == 0) | (F.col(group_col) > max_period) | (F.col(group_col) == g)

    filtered = sdf.filter(group_filter)

    time_filter = (F.col(time_col) == t) | (F.col(time_col) == pret)
    concat_sdf = filtered.filter(time_filter)
    concat_sdf = concat_sdf.withColumn("_post", F.when(F.col(time_col) == t, 1).otherwise(0))

    keep_cols = [id_col, group_col, y_col, "_post"]
    if weightsname is not None:
        keep_cols.append(weightsname)
    if covariate_cols:
        keep_cols = keep_cols + [c for c in covariate_cols if c not in keep_cols]

    concat_sdf = concat_sdf.select(*keep_cols)
    concat_sdf = concat_sdf.repartition(n_partitions).cache()
    n_cell = concat_sdf.count()

    if n_cell == 0:
        concat_sdf.unpersist()
        return None, 0

    # Compute partition offsets for positional IDs
    offset_schema = StructType([StructField("_part_len", IntegerType(), False)])

    def _count_partition(iterator):
        for pdf in iterator:
            yield pd.DataFrame({"_part_len": [len(pdf)]})

    len_rows = concat_sdf.mapInPandas(_count_partition, schema=offset_schema).collect()
    partition_lengths = [row["_part_len"] for row in len_rows]
    offsets = np.cumsum([0, *partition_lengths[:-1]])

    # Collect partition data
    part_data_list = []
    offset_idx = 0
    for pdf in concat_sdf.toLocalIterator(prefetchPartitions=True):
        if len(pdf) == 0:
            offset_idx += 1
            continue
        offset = int(offsets[offset_idx]) if offset_idx < len(offsets) else 0
        part = _build_did_rc_partition_arrays(pdf, offset, y_col, group_col, g, covariate_cols, weightsname)
        if part is not None:
            part = _maybe_to_gpu_dict(part, use_gpu)
            part_data_list.append(part)
        offset_idx += 1

    concat_sdf.unpersist()

    if not part_data_list:
        return None, 0

    return part_data_list, n_cell


def prepare_did_cell_partitions(
    spark,
    sdf,
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
    """Filter and merge post/pre periods, returning partition data dicts.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    sdf : pyspark.sql.DataFrame
        Cached Spark DataFrame in long panel format.
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
        Number of Spark partitions for the merged data.
    filtered_data : pyspark.sql.DataFrame or None, default None
        Pre-filtered cohort data.
    weightsname : str or None
        Weight column name.
    use_gpu : bool, default False
        Convert partition arrays to CuPy for GPU-accelerated computation.

    Returns
    -------
    part_data_list : list of dict or None
        Partition data dicts, or ``None`` if the cell is empty.
    n_cell : int
        Number of units in the cell.
    """
    if filtered_data is not None:
        filtered = filtered_data
    else:
        max_period = max(t, pret)

        if control_group == "nevertreated":
            group_filter = (F.col(group_col) == 0) | (F.col(group_col) == g)
        else:
            group_filter = (F.col(group_col) == 0) | (F.col(group_col) > max_period) | (F.col(group_col) == g)

        filtered = sdf.filter(group_filter)

    post_cols = [id_col, group_col, y_col]
    if weightsname is not None:
        post_cols.append(weightsname)
    if covariate_cols:
        post_cols = post_cols + [c for c in covariate_cols if c not in post_cols]

    post_sdf = filtered.filter(F.col(time_col) == t).select(*post_cols)
    pre_sdf = filtered.filter(F.col(time_col) == pret).select(F.col(id_col), F.col(y_col).alias("_y_pre"))

    merged_sdf = post_sdf.join(pre_sdf, on=id_col, how="inner")
    merged_sdf = merged_sdf.repartition(n_partitions).cache()
    n_cell = merged_sdf.count()

    if n_cell == 0:
        merged_sdf.unpersist()
        return None, 0

    part_data_list = _collect_partition_data(
        merged_sdf,
        _build_did_partition_arrays,
        (id_col, y_col, group_col, g, covariate_cols, weightsname),
    )
    part_data_list = [_maybe_to_gpu_dict(pd, use_gpu) for pd in part_data_list]

    merged_sdf.unpersist()

    if not part_data_list:
        return None, 0

    return part_data_list, n_cell


def streaming_did_nuisance_coefficients(spark, part_data_list, est_method, k):
    r"""Compute nuisance model coefficients for DiD.

    Estimates:

    - **Propensity score** (unless ``est_method="reg"``): logistic regression
      via distributed IRLS for :math:`P(D=1 \mid X)`.
    - **Outcome regression** (unless ``est_method="ipw"``): WLS regression
      of :math:`\Delta Y` on :math:`X` among control units :math:`D = 0`.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    part_data_list : list of dict
        Partition data dicts.
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
        ps_beta = distributed_logistic_irls_from_partitions(
            spark,
            part_data_list,
            _partition_did_pscore_gram,
            k,
        )

    if est_method == "ipw":
        or_beta = np.zeros(k, dtype=np.float64)
    else:
        or_beta = distributed_wls_from_partitions(
            spark,
            part_data_list,
            _partition_did_or_gram,
        )

    return ps_beta, or_beta


def streaming_did_global_stats(spark, part_data_list, ps_beta, or_beta, est_method, trim_level=0.995):
    r"""Compute global aggregate statistics and precomputed correction vectors.

    Performs a driver-side reduce to aggregate per-partition sufficient
    statistics into global means, ATT components, and the inverse Hessian and
    design matrix products needed by the influence function.

    Parameters
    ----------
    spark : pyspark.sql.SparkSession
        Active Spark session.
    part_data_list : list of dict
        Partition data dicts.
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
    stats_list = [
        _partition_did_global_stats(
            pf,
            ps_beta,
            or_beta,
            est_method,
            trim_level,
        )
        for pf in part_data_list
    ]

    agg = None
    for s in stats_list:
        agg = sum_global_stats(agg, s)

    return _finalize_global_stats(agg, est_method)


def _collect_partition_data(cached_df, build_fn, build_args):
    """Collect partition-level numpy dicts from a cached Spark DataFrame.

    Iterates over Spark partitions using ``toLocalIterator``, applying
    ``build_fn`` to each pandas partition to produce numpy dicts.

    Parameters
    ----------
    cached_df : pyspark.sql.DataFrame
        Cached Spark DataFrame.
    build_fn : callable
        Function ``(pandas_df, *build_args) -> dict or None``.
    build_args : tuple
        Additional arguments for ``build_fn``.

    Returns
    -------
    list of dict
        Non-None partition dicts.
    """
    part_data_list = []
    for pdf in cached_df.toLocalIterator(prefetchPartitions=True):
        if len(pdf) == 0:
            continue
        part = build_fn(pdf, *build_args)
        if part is not None:
            part_data_list.append(part)
    return part_data_list
