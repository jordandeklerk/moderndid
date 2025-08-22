"""Empirical bootstrap for panel data."""

import warnings

import numpy as np
import pandas as pd

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
    data = pte_params.data
    idname = pte_params.idname
    boot_type = pte_params.boot_type
    n_boot = pte_params.biters
    n_clusters = pte_params.cl
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
            cl=n_clusters,
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

    attgt_boots = pd.concat([res["attgt_results"] for res in bootstrap_results if "attgt_results" in res])
    attgt_se = attgt_boots.groupby(["group", "time_period"])["att"].std().reset_index(name="se")
    attgt_results = aggte_results["attgt_results"].copy()
    attgt_results = attgt_results.merge(attgt_se, on=["group", "time_period"], how="left")

    if aggte_results.get("dyn_results") is not None:
        dyn_boots = pd.concat([res["dyn_results"] for res in bootstrap_results if res.get("dyn_results") is not None])

        counts = dyn_boots.groupby("e").size()
        original_e_count = len(counts)
        complete_groups = counts[counts == n_boot].index
        new_e_count = len(complete_groups)

        if new_e_count != original_e_count:
            warnings.warn("dropping some event times due to small groups")

        if not complete_groups.empty:
            filtered_boots = dyn_boots[dyn_boots["e"].isin(complete_groups)]
            dyn_se = filtered_boots.groupby("e")["att_e"].std().reset_index(name="se")

            dyn_results = aggte_results["dyn_results"].copy()
            dyn_results = dyn_results.merge(dyn_se, on="e", how="inner")
        else:
            dyn_results = aggte_results["dyn_results"].copy()
            dyn_results["se"] = np.nan
            dyn_results = dyn_results[dyn_results["e"].isin([])]
    else:
        dyn_results = None

    if aggte_results.get("group_results") is not None:
        group_boots = pd.concat(
            [res["group_results"] for res in bootstrap_results if res.get("group_results") is not None]
        )

        counts = group_boots.groupby("group").size()
        original_g_count = len(counts)
        complete_groups = counts[counts == n_boot].index
        new_g_count = len(complete_groups)

        if new_g_count != original_g_count:
            warnings.warn("dropping some groups due to small groups")

        if not complete_groups.empty:
            filtered_boots = group_boots[group_boots["group"].isin(complete_groups)]
            group_se = filtered_boots.groupby("group")["att_g"].std().reset_index(name="se")

            group_results = aggte_results["group_results"].copy()
            group_results = group_results.merge(group_se, on="group", how="inner")
        else:
            group_results = aggte_results["group_results"].copy()
            group_results["se"] = np.nan
            group_results = group_results[group_results["group"].isin([])]
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

    data = pte_params.data
    original_periods = np.sort(data[pte_params.tname].unique())

    attgt_df = pd.DataFrame(attgt_list)

    if attgt_df.empty or "att" not in attgt_df.columns:
        return {
            "attgt_results": pd.DataFrame(columns=["group", "time_period", "att"]),
            "dyn_results": None,
            "dyn_weights": [],
            "group_results": None,
            "group_weights": [],
            "overall_results": np.nan,
            "overall_weights": np.array([]),
        }

    attgt_df = attgt_df.dropna(subset=["att"]).reset_index(drop=True)

    if not np.array_equal(time_periods, original_periods):
        time_map = {i + 1: orig for i, orig in enumerate(original_periods)}
        attgt_df["time_period"] = attgt_df["time_period"].map(time_map)
        attgt_df["group"] = attgt_df["group"].map(time_map)
        groups = np.array([time_map.get(g, g) for g in groups])
        time_periods = np.array([time_map.get(t, t) for t in time_periods])

    attgt_df["e"] = attgt_df["time_period"] - attgt_df["group"]

    first_period = time_periods[0]
    group_sizes = data[data[pte_params.tname] == first_period].groupby(pte_params.gname).size().rename("n_group")
    attgt_df = attgt_df.merge(group_sizes, left_on="group", right_index=True, how="left")
    attgt_df["n_group"] = attgt_df["n_group"].fillna(0)

    if "e" in attgt_df.columns and attgt_df["e"].notna().any():
        attgt_df["dyn_w"] = attgt_df["n_group"] / attgt_df.groupby("e")["n_group"].transform("sum")
        dyn_df = (
            attgt_df.groupby("e")
            .apply(lambda x: (x["att"] * x["dyn_w"]).sum(), include_groups=False)
            .reset_index(name="att_e")
        )

        dyn_weights_pivot = attgt_df.pivot_table(index=attgt_df.index, columns="e", values="dyn_w", fill_value=0)
        dyn_weights = [{"e": e, "weights": dyn_weights_pivot[e].values} for e in dyn_weights_pivot.columns]
    else:
        dyn_df = None
        dyn_weights = []

    post_treatment_df = attgt_df[attgt_df["time_period"] >= attgt_df["group"]].copy()
    if not post_treatment_df.empty:
        group_df = (
            post_treatment_df.groupby("group")
            .agg(
                att_g=("att", "mean"),
                n_group=("n_group", "first"),
                group_post_length=("att", "size"),
            )
            .reset_index()
        )

        post_treatment_df["group_w"] = 1.0 / post_treatment_df.groupby("group")["group"].transform("size")
        group_weights_pivot = post_treatment_df.pivot_table(
            index=post_treatment_df.index, columns="group", values="group_w", fill_value=0
        )
        group_weights_pivot = group_weights_pivot.reindex(attgt_df.index, fill_value=0)
        group_weights = [{"g": g, "weights": group_weights_pivot[g].values} for g in group_weights_pivot.columns]
    else:
        group_df = None
        group_weights = []

    if group_df is not None and not group_df.empty:
        valid_groups = group_df.dropna(subset=["att_g"])
        if not valid_groups.empty:
            valid_groups["overall_w"] = valid_groups["n_group"] / valid_groups["n_group"].sum()
            overall_att = (valid_groups["att_g"] * valid_groups["overall_w"]).sum()

            attgt_df = attgt_df.merge(
                valid_groups[["group", "overall_w", "group_post_length"]],
                on="group",
                how="left",
            )
            attgt_df["overall_w"] = attgt_df["overall_w"].fillna(0)

            e_mask = attgt_df["e"] >= 0
            overall_weights = np.zeros(len(attgt_df))
            overall_weights[e_mask] = (
                attgt_df.loc[e_mask, "overall_w"] / attgt_df.loc[e_mask, "group_post_length"]
            ).fillna(0)
        else:
            overall_att = np.nan
            overall_weights = np.zeros(len(attgt_df))
    else:
        overall_att = np.nan
        overall_weights = np.zeros(len(attgt_df))

    attgt_results = attgt_df[["group", "time_period", "att"]]

    if group_df is not None:
        group_results = group_df[["group", "att_g"]]
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
    y_values = pte_params.data[yname].values
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
        "attgt_results": pd.DataFrame({"group": groups, "time_period": time_periods, "att": qtt_gt}),
        "dyn_results": pd.DataFrame(dyn_qtt) if dyn_qtt else None,
        "group_results": pd.DataFrame(group_qtt) if group_qtt else None,
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
    y_max = pte_params.data[yname].max()
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
        "attgt_results": pd.DataFrame({"group": groups, "time_period": time_periods, "att": qott_gt}),
        "dyn_results": pd.DataFrame(dyn_qott) if dyn_qott else None,
        "group_results": pd.DataFrame(group_qott) if group_qott else None,
        "overall_results": overall_qott,
    }


def block_boot_sample(data, id_column):
    """Draw a block bootstrap sample from panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit identifiers.
    id_column : str
        Name of column containing unit IDs.

    Returns
    -------
    pd.DataFrame
        Bootstrap sample with same structure as input data.
    """
    unique_ids = data[id_column].unique()
    n_units = len(unique_ids)

    rng = np.random.default_rng()
    sampled_ids = rng.choice(unique_ids, size=n_units, replace=True)

    bootstrap_data = []
    for new_id, old_id in enumerate(sampled_ids):
        unit_data = data[data[id_column] == old_id].copy()
        unit_data[id_column] = new_id
        bootstrap_data.append(unit_data)

    return pd.concat(bootstrap_data, ignore_index=True)


def _make_ecdf(y_values, cdf_values):
    """Create an empirical CDF function from values and probabilities."""

    def ecdf(x):
        """Evaluate empirical CDF at point x."""
        if np.isscalar(x):
            return np.interp(x, y_values, cdf_values, left=0, right=1)
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
