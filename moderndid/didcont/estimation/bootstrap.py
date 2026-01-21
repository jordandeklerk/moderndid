"""Empirical bootstrap for panel data."""

import warnings

import numpy as np
import polars as pl

from ...core.dataframe import to_pandas, to_polars
from ..utils import _quantile_basis
from .container import PteEmpBootResult


def panel_empirical_bootstrap(
    attgt_list, pte_params, setup_pte_fun, subset_fun, attgt_fun, extra_gt_returns, compute_pte_fun, **kwargs
):
    """Compute empirical bootstrap standard errors for panel treatment effects.

    Parameters
    ----------
    attgt_list : list
        List of ATT(g,t) results from compute_pte.
    pte_params : PTEParams
        Parameters object with estimation settings.
    setup_pte_fun : callable
        Function to setup PTE parameters.
    subset_fun : callable
        Function to create data subsets for each (g,t).
    attgt_fun : callable
        Function to compute ATT for single group-time.
    extra_gt_returns : list
        Extra returns from group-time calculations.
    compute_pte_fun : callable
        Function to compute PTE for bootstrap samples.
    **kwargs
        Additional arguments passed through.

    Returns
    -------
    PteEmpBootResult
        Bootstrap results with standard errors for all aggregations.
    """
    data = to_pandas(pte_params.data)
    idname = pte_params.idname
    boot_type = pte_params.boot_type
    n_boot = pte_params.biters
    gt_type = pte_params.gt_type

    if gt_type == "qtt":
        aggte_results = qtt_pte_aggregations(attgt_list, pte_params, extra_gt_returns)
    elif gt_type == "qott":
        aggte_results = qott_pte_aggregations(attgt_list, pte_params, extra_gt_returns)
    else:
        aggte_results = attgt_pte_aggregations(attgt_list, pte_params)

    original_periods = np.sort(data[pte_params.tname].unique())
    if extra_gt_returns:
        extra_gt_returns = _convert_to_original_time(extra_gt_returns, original_periods)

    bootstrap_results = []

    for _ in range(n_boot):
        boot_data = block_boot_sample(data, idname)

        boot_params = setup_pte_fun(
            yname=pte_params.yname,
            gname=pte_params.gname,
            tname=pte_params.tname,
            idname=pte_params.idname,
            data=boot_data,
            alp=pte_params.alp,
            boot_type=boot_type,
            gt_type=gt_type,
            biters=pte_params.biters,
            **kwargs,
        )

        boot_gt_results = compute_pte_fun(ptep=boot_params, subset_fun=subset_fun, attgt_fun=attgt_fun, **kwargs)

        if gt_type == "qtt":
            boot_aggte = qtt_pte_aggregations(
                boot_gt_results["attgt_list"], boot_params, boot_gt_results["extra_gt_returns"]
            )
        elif gt_type == "qott":
            boot_aggte = qott_pte_aggregations(
                boot_gt_results["attgt_list"], boot_params, boot_gt_results["extra_gt_returns"]
            )
        else:
            boot_aggte = attgt_pte_aggregations(boot_gt_results["attgt_list"], boot_params)

        bootstrap_results.append(boot_aggte)

    attgt_boots = pl.concat([res["attgt_results"] for res in bootstrap_results if "attgt_results" in res])
    attgt_boots = attgt_boots.with_columns(
        [
            pl.col("group").cast(pl.Float64),
            pl.col("time_period").cast(pl.Float64),
        ]
    )
    attgt_se = attgt_boots.group_by(["group", "time_period"]).agg(pl.col("att").std().alias("se"))
    attgt_results = aggte_results["attgt_results"].clone()
    attgt_results = attgt_results.with_columns(
        [
            pl.col("group").cast(pl.Float64),
            pl.col("time_period").cast(pl.Float64),
        ]
    )
    attgt_results = attgt_results.join(attgt_se, on=["group", "time_period"], how="left")

    if aggte_results.get("dyn_results") is not None:
        dyn_boots = pl.concat([res["dyn_results"] for res in bootstrap_results if res.get("dyn_results") is not None])

        counts = dyn_boots.group_by("e").len()
        original_e_count = len(counts)
        complete_groups = counts.filter(pl.col("len") == n_boot)["e"].to_list()
        new_e_count = len(complete_groups)

        if new_e_count != original_e_count:
            warnings.warn("dropping some event times due to small groups")

        if complete_groups:
            filtered_boots = dyn_boots.filter(pl.col("e").is_in(complete_groups))
            dyn_se = filtered_boots.group_by("e").agg(pl.col("att_e").std().alias("se"))

            dyn_results = aggte_results["dyn_results"].clone()
            dyn_results = dyn_results.join(dyn_se, on="e", how="inner")
        else:
            dyn_results = aggte_results["dyn_results"].clone()
            dyn_results = dyn_results.with_columns(pl.lit(np.nan).alias("se"))
            dyn_results = dyn_results.filter(pl.col("e").is_in([]))
    else:
        dyn_results = None

    if aggte_results.get("group_results") is not None:
        group_boots = pl.concat(
            [res["group_results"] for res in bootstrap_results if res.get("group_results") is not None]
        )

        counts = group_boots.group_by("group").len()
        original_g_count = len(counts)
        complete_groups = counts.filter(pl.col("len") == n_boot)["group"].to_list()
        new_g_count = len(complete_groups)

        if new_g_count != original_g_count:
            warnings.warn("dropping some groups due to small groups")

        if complete_groups:
            filtered_boots = group_boots.filter(pl.col("group").is_in(complete_groups))
            group_se = filtered_boots.group_by("group").agg(pl.col("att_g").std().alias("se"))

            group_results = aggte_results["group_results"].clone()
            group_results = group_results.join(group_se, on="group", how="inner")
        else:
            group_results = aggte_results["group_results"].clone()
            group_results = group_results.with_columns(pl.lit(np.nan).alias("se"))
            group_results = group_results.filter(pl.col("group").is_in([]))
    else:
        group_results = None

    overall_boots = [res["overall_results"] for res in bootstrap_results if "overall_results" in res]
    overall_se = np.std(overall_boots) if len(overall_boots) > 1 else np.nan

    overall_results = {"att": aggte_results["overall_results"], "se": overall_se}

    return PteEmpBootResult(
        attgt_results=attgt_results,
        overall_results=overall_results,
        group_results=group_results,
        dyn_results=dyn_results,
        extra_gt_returns=extra_gt_returns,
    )


def attgt_pte_aggregations(attgt_list, pte_params):
    """Aggregate average treatment effects into overall, group, and dynamic effects.

    Parameters
    ----------
    attgt_list : list
        List of average treatment effects with 'att', 'group', 'time_period'.
    pte_params : PTEParams
        Parameters object containing data and settings.

    Returns
    -------
    dict
        Dictionary containing aggregated results and weights.
    """
    time_periods = pte_params.t_list
    groups = pte_params.g_list

    data = to_polars(pte_params.data)
    original_periods = np.sort(data[pte_params.tname].unique().to_numpy())

    attgt_df = pl.DataFrame(attgt_list)

    if "group" in attgt_df.columns:
        attgt_df = attgt_df.with_columns(pl.col("group").cast(pl.Float64))
    if "time_period" in attgt_df.columns:
        attgt_df = attgt_df.with_columns(pl.col("time_period").cast(pl.Float64))

    if attgt_df.is_empty() or "att" not in attgt_df.columns:
        return {
            "attgt_results": pl.DataFrame(
                {"group": [], "time_period": [], "att": []},
                schema={"group": pl.Float64, "time_period": pl.Float64, "att": pl.Float64},
            ),
            "dyn_results": None,
            "dyn_weights": [],
            "group_results": None,
            "group_weights": [],
            "overall_results": np.nan,
            "overall_weights": np.array([]),
        }

    attgt_df = attgt_df.drop_nulls(subset=["att"])

    if not np.array_equal(time_periods, original_periods):
        time_map = {i + 1: orig for i, orig in enumerate(original_periods)}
        attgt_df = attgt_df.with_columns(
            pl.col("time_period").replace(time_map).alias("time_period"),
            pl.col("group").replace(time_map).alias("group"),
        )
        groups = np.array([time_map.get(g, g) for g in groups])
        time_periods = np.array([time_map.get(t, t) for t in time_periods])

    attgt_df = attgt_df.with_columns((pl.col("time_period") - pl.col("group")).alias("e"))

    first_period = time_periods[0]
    group_sizes = (
        data.filter(pl.col(pte_params.tname) == first_period)
        .group_by(pte_params.gname)
        .len()
        .rename({pte_params.gname: "group", "len": "n_group"})
        .with_columns(pl.col("group").cast(pl.Float64))
    )
    attgt_df = attgt_df.join(group_sizes, on="group", how="left")
    attgt_df = attgt_df.with_columns(pl.col("n_group").fill_null(0))

    if "e" in attgt_df.columns and attgt_df["e"].drop_nulls().len() > 0:
        e_sums = attgt_df.group_by("e").agg(pl.col("n_group").sum().alias("e_sum"))
        attgt_df = attgt_df.join(e_sums, on="e", how="left")
        attgt_df = attgt_df.with_columns((pl.col("n_group") / pl.col("e_sum")).alias("dyn_w"))

        dyn_df = attgt_df.group_by("e").agg((pl.col("att") * pl.col("dyn_w")).sum().alias("att_e"))

        e_values = attgt_df["e"].unique().sort().to_list()
        dyn_weights = []
        for e in e_values:
            weights = attgt_df.with_columns(pl.when(pl.col("e") == e).then(pl.col("dyn_w")).otherwise(0).alias("w"))[
                "w"
            ].to_numpy()
            dyn_weights.append({"e": e, "weights": weights})
    else:
        dyn_df = None
        dyn_weights = []

    post_treatment_df = attgt_df.filter(pl.col("time_period") >= pl.col("group"))
    if not post_treatment_df.is_empty():
        group_df = post_treatment_df.group_by("group").agg(
            pl.col("att").mean().alias("att_g"),
            pl.col("n_group").first().alias("n_group"),
            pl.col("att").len().alias("group_post_length"),
        )

        group_counts = post_treatment_df.group_by("group").len().rename({"len": "group_size"})
        post_treatment_df = post_treatment_df.join(group_counts, on="group", how="left")
        post_treatment_df = post_treatment_df.with_columns((1.0 / pl.col("group_size")).alias("group_w"))

        group_values = post_treatment_df["group"].unique().sort().to_list()
        group_weights = []
        for g in group_values:
            weights = np.zeros(len(attgt_df))
            post_mask = (attgt_df["time_period"] >= attgt_df["group"]).to_numpy()
            group_mask = (attgt_df["group"] == g).to_numpy()
            combined_mask = post_mask & group_mask
            if combined_mask.any():
                group_size = combined_mask.sum()
                weights[combined_mask] = 1.0 / group_size
            group_weights.append({"g": g, "weights": weights})
    else:
        group_df = None
        group_weights = []

    if group_df is not None and not group_df.is_empty():
        valid_groups = group_df.drop_nulls(subset=["att_g"])
        if not valid_groups.is_empty():
            n_group_sum = valid_groups["n_group"].sum()
            valid_groups = valid_groups.with_columns((pl.col("n_group") / n_group_sum).alias("overall_w"))
            overall_att = (valid_groups["att_g"] * valid_groups["overall_w"]).sum()

            attgt_df = attgt_df.join(
                valid_groups.select(["group", "overall_w", "group_post_length"]),
                on="group",
                how="left",
            )
            attgt_df = attgt_df.with_columns(pl.col("overall_w").fill_null(0))

            e_values = attgt_df["e"].to_numpy()
            overall_w_values = attgt_df["overall_w"].to_numpy()
            group_post_length_values = attgt_df["group_post_length"].to_numpy()

            e_mask = e_values >= 0
            overall_weights = np.zeros(len(attgt_df))
            with np.errstate(divide="ignore", invalid="ignore"):
                weights_calc = overall_w_values / group_post_length_values
                weights_calc = np.nan_to_num(weights_calc, nan=0.0, posinf=0.0, neginf=0.0)
            overall_weights[e_mask] = weights_calc[e_mask]
        else:
            overall_att = np.nan
            overall_weights = np.zeros(len(attgt_df))
    else:
        overall_att = np.nan
        overall_weights = np.zeros(len(attgt_df))

    attgt_results = attgt_df.select(["group", "time_period", "att"]).filter(
        pl.col("att").is_not_null() & pl.col("att").is_not_nan()
    )

    if group_df is not None:
        group_results = group_df.select(["group", "att_g"])
    else:
        group_results = None

    return {
        "attgt_results": attgt_results,
        "dyn_results": dyn_df,
        "dyn_weights": dyn_weights,
        "group_results": group_results,
        "group_weights": group_weights,
        "overall_results": overall_att,
        "overall_weights": overall_weights,
    }


def qtt_pte_aggregations(attgt_list, pte_params, extra_gt_returns):
    """Aggregate QTT (quantile of treatment effect) distributions into overall, group, and dynamic effects.

    Parameters
    ----------
    attgt_list : list
        List of average treatment effects.
    pte_params : PTEParams
        Parameters with ret_quantile specifying which quantile to compute.
    extra_gt_returns : list
        Contains F0 and F1 distributions for each (g,t).

    Returns
    -------
    dict
        Same structure as attgt_pte_aggregations but with QTT estimates.
    """
    ret_quantile = pte_params.ret_quantile
    att_results = attgt_pte_aggregations(attgt_list, pte_params)

    f0_list = [egr["extra_gt_returns"]["F0"] for egr in extra_gt_returns]
    f1_list = [egr["extra_gt_returns"]["F1"] for egr in extra_gt_returns]

    qtt_gt = []
    for f0, f1 in zip(f0_list, f1_list):
        q1 = _quantile_basis(f1, ret_quantile)
        q0 = _quantile_basis(f0, ret_quantile)
        qtt_gt.append(q1 - q0)

    groups = [item["group"] for item in attgt_list]
    time_periods = [item["time_period"] for item in attgt_list]

    yname = pte_params.yname

    data = to_polars(pte_params.data)
    y_values = data[yname].to_numpy()
    y_seq = np.quantile(y_values, np.linspace(0, 1, 1000))

    overall_weights = att_results["overall_weights"]
    f0_overall = _combine_ecdfs(y_seq, f0_list, overall_weights)
    f1_overall = _combine_ecdfs(y_seq, f1_list, overall_weights)

    q1_overall_cdf = f1_overall(y_seq)
    q0_overall_cdf = f0_overall(y_seq)
    q1_idx = np.clip(np.searchsorted(q1_overall_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
    q0_idx = np.clip(np.searchsorted(q0_overall_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
    overall_qtt = y_seq[q1_idx] - y_seq[q0_idx]

    dyn_qtt = []
    if att_results.get("dyn_weights"):
        for dyn_weight in att_results["dyn_weights"]:
            e = dyn_weight["e"]
            weights = dyn_weight["weights"]

            f0_e = _combine_ecdfs(y_seq, f0_list, weights)
            f1_e = _combine_ecdfs(y_seq, f1_list, weights)

            q1_e_cdf = f1_e(y_seq)
            q0_e_cdf = f0_e(y_seq)
            q1_e_idx = np.clip(np.searchsorted(q1_e_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            q0_e_idx = np.clip(np.searchsorted(q0_e_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            qtt_e = y_seq[q1_e_idx] - y_seq[q0_e_idx]

            dyn_qtt.append({"e": e, "att_e": qtt_e})

    group_qtt = []
    if att_results.get("group_weights"):
        for group_weight in att_results["group_weights"]:
            g = group_weight["g"]
            weights = group_weight["weights"]

            f0_g = _combine_ecdfs(y_seq, f0_list, weights)
            f1_g = _combine_ecdfs(y_seq, f1_list, weights)

            q1_g_cdf = f1_g(y_seq)
            q0_g_cdf = f0_g(y_seq)
            q1_g_idx = np.clip(np.searchsorted(q1_g_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            q0_g_idx = np.clip(np.searchsorted(q0_g_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            qtt_g = y_seq[q1_g_idx] - y_seq[q0_g_idx]

            group_qtt.append({"group": g, "att_g": qtt_g})

    return {
        "attgt_results": pl.DataFrame({"group": groups, "time_period": time_periods, "att": qtt_gt}),
        "dyn_results": pl.DataFrame(dyn_qtt) if dyn_qtt else None,
        "group_results": pl.DataFrame(group_qtt) if group_qtt else None,
        "overall_results": overall_qtt,
    }


def qott_pte_aggregations(attgt_list, pte_params, extra_gt_returns):
    """Aggregate QOTT (quantile of treatment effect) distributions.

    Parameters
    ----------
    attgt_list : list
        List of average treatment effects.
    pte_params : PTEParams
        Parameters with ret_quantile.
    extra_gt_returns : list
        Contains Fte (treatment effect distribution) for each (g,t).

    Returns
    -------
    dict
        Same structure as attgt_pte_aggregations but with QOTT estimates.
    """
    ret_quantile = pte_params.ret_quantile
    att_results = attgt_pte_aggregations(attgt_list, pte_params)

    fte_list = [egr["extra_gt_returns"]["Fte"] for egr in extra_gt_returns]
    qott_gt = [_quantile_basis(fte, ret_quantile) for fte in fte_list]

    groups = [item["group"] for item in attgt_list]
    time_periods = [item["time_period"] for item in attgt_list]

    yname = pte_params.yname
    data = to_polars(pte_params.data)
    y_max = data[yname].max()
    y_seq = np.linspace(-y_max, y_max, 1000)

    overall_weights = att_results["overall_weights"]
    fte_overall_cdf = _combine_ecdfs(y_seq, fte_list, overall_weights)(y_seq)
    overall_idx = np.clip(np.searchsorted(fte_overall_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
    overall_qott = y_seq[overall_idx]

    dyn_qott = []
    if att_results.get("dyn_weights"):
        for dyn_weight in att_results["dyn_weights"]:
            e = dyn_weight["e"]
            weights = dyn_weight["weights"]

            fte_e_cdf = _combine_ecdfs(y_seq, fte_list, weights)(y_seq)
            dyn_idx = np.clip(np.searchsorted(fte_e_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            qott_e = y_seq[dyn_idx]

            dyn_qott.append({"e": e, "att_e": qott_e})

    group_qott = []
    if att_results.get("group_weights"):
        for group_weight in att_results["group_weights"]:
            g = group_weight["g"]
            weights = group_weight["weights"]

            fte_g_cdf = _combine_ecdfs(y_seq, fte_list, weights)(y_seq)
            group_idx = np.clip(np.searchsorted(fte_g_cdf, ret_quantile, side="left"), 0, len(y_seq) - 1)
            qott_g = y_seq[group_idx]

            group_qott.append({"group": g, "att_g": qott_g})

    return {
        "attgt_results": pl.DataFrame({"group": groups, "time_period": time_periods, "att": qott_gt}),
        "dyn_results": pl.DataFrame(dyn_qott) if dyn_qott else None,
        "group_results": pl.DataFrame(group_qott) if group_qott else None,
        "overall_results": overall_qott,
    }


def block_boot_sample(data, id_column):
    """Draw a block bootstrap sample from panel data.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        Panel data with unit identifiers.
    id_column : str
        Name of column containing unit IDs.

    Returns
    -------
    pl.DataFrame
        Bootstrap sample with same structure as input data.
    """
    data_pl = to_polars(data)

    unique_ids = data_pl[id_column].unique().to_numpy()
    n_units = len(unique_ids)

    rng = np.random.default_rng()
    sampled_ids = rng.choice(unique_ids, size=n_units, replace=True)

    bootstrap_data = []
    for new_id, old_id in enumerate(sampled_ids):
        unit_data = data_pl.filter(pl.col(id_column) == old_id).clone()
        unit_data = unit_data.with_columns(pl.lit(new_id).alias(id_column))
        bootstrap_data.append(unit_data)

    return pl.concat(bootstrap_data)


def _make_ecdf(y_values, cdf_values):
    """Create an empirical CDF function from values and probabilities."""

    def ecdf(x):
        """Evaluate empirical CDF at point :math:`x`."""
        return np.interp(x, y_values, cdf_values, left=0, right=1)

    return ecdf


def _combine_ecdfs(y_seq, ecdf_list, weights=None):
    """Combine multiple empirical CDFs using weights."""
    if weights is None:
        weights = np.ones(len(ecdf_list)) / len(ecdf_list)
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()

    y_seq = np.sort(y_seq)

    cdf_matrix = np.zeros((len(y_seq), len(ecdf_list)))

    for i, ecdf in enumerate(ecdf_list):
        if callable(ecdf):
            cdf_matrix[:, i] = ecdf(y_seq)
        else:
            cdf_matrix[:, i] = np.mean(y_seq[:, np.newaxis] >= ecdf, axis=1)

    combined_cdf_values = cdf_matrix @ weights

    return _make_ecdf(y_seq, combined_cdf_values)


def _convert_to_original_time(extra_gt_returns, original_periods):
    """Convert time indices back to original scale."""
    time_map = {i + 1: orig for i, orig in enumerate(original_periods)}

    converted = []
    for egr in extra_gt_returns:
        new_egr = egr.copy()
        if "group" in new_egr:
            new_egr["group"] = time_map.get(egr["group"], egr["group"])
        if "time_period" in new_egr:
            new_egr["time_period"] = time_map.get(egr["time_period"], egr["time_period"])
        converted.append(new_egr)

    return converted
