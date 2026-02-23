from __future__ import annotations

import io

import numpy as np
import polars as pl

from moderndid.didcont.cont_did import cont_did_acrt
from moderndid.didcont.estimation.estimators import pte_attgt


def build_cell_subset(data, g, tp, control_group, anticipation, base_period):
    """Build a serialized subset of the panel for a single (group, time) cell.

    Filters the full panel to the relevant units (treated group *g* plus the
    chosen control group) and the two time periods needed for estimation
    (the evaluation period *tp* and the appropriate base period).  The
    resulting subset is serialized to Arrow IPC bytes so it can be shipped
    to a distributed worker without requiring Polars on the driver.

    Parameters
    ----------
    data : pl.DataFrame
        Full panel dataset with columns ``id``, ``period``, ``G``
        (group / first-treatment period), and ``D`` (treatment indicator).
    g : int or float
        Group identifier (first period of treatment).
    tp : int or float
        Evaluation time period.
    control_group : {"notyettreated", "nevertreated"}
        Which units serve as the control group.  ``"notyettreated"`` uses
        units not yet treated by period *tp*; ``"nevertreated"`` uses units
        with ``G == inf``.
    anticipation : int
        Number of anticipation periods to allow when computing the base
        period.
    base_period : {"varying", "universal"}
        Strategy for choosing the pre-treatment reference period.
        ``"varying"`` picks ``tp - 1`` when *tp* precedes the treatment
        onset minus anticipation; ``"universal"`` always uses
        ``g - anticipation - 1``.

    Returns
    -------
    arrow_bytes : bytes
        The subset DataFrame serialized as Arrow IPC bytes.
    n1 : int
        Number of unique units in the subset.
    disidx : np.ndarray of bool
        Boolean mask over all sorted unit IDs indicating which units
        belong to this subset.
    """
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        base_period_val = tp - 1 if tp < g - anticipation else main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        unit_mask = (pl.col("G") == g) | (pl.col("G") > tp)
    else:
        unit_mask = (pl.col("G") == g) | pl.col("G").is_infinite()

    subset_data = data.filter(unit_mask)
    time_mask = (pl.col("period") == tp) | (pl.col("period") == base_period_val)
    subset_data = subset_data.filter(time_mask)
    subset_data = subset_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name")
    )
    subset_data = subset_data.with_columns((pl.col("D") * (pl.col("G") == g).cast(pl.Float64)).alias("D"))

    n1 = subset_data["id"].n_unique()
    all_ids = data["id"].unique().sort().to_numpy()
    subset_ids = subset_data["id"].unique().to_numpy()
    disidx = np.isin(all_ids, subset_ids)

    buf = io.BytesIO()
    subset_data.write_ipc(buf)
    arrow_bytes = buf.getvalue()

    return arrow_bytes, n1, disidx


def process_pte_cell_from_subset(
    subset_arrow_bytes,
    tp,
    g,
    gt_type,
    n_units,
    n1,
    cell_kwargs,
):
    """Estimate the ATT for a single (group, time) cell from serialized data.

    Deserializes the Arrow IPC bytes produced by :func:`build_cell_subset`,
    selects the appropriate estimator based on *gt_type*, computes the
    group-time ATT, and returns the point estimate together with the
    re-scaled influence function needed for downstream inference.

    Parameters
    ----------
    subset_arrow_bytes : bytes
        Arrow IPC-serialized DataFrame for this (group, time) cell,
        as returned by :func:`build_cell_subset`.
    tp : int or float
        Evaluation time period.
    g : int or float
        Group identifier (first period of treatment).
    gt_type : {"dose", "attgt"}
        Type of group-time estimator to use.  ``"dose"`` invokes the
        continuous-treatment dose-response estimator; any other value
        uses the standard PTE ATT(g,t) estimator.
    n_units : int
        Total number of unique units in the full panel (used to re-scale
        the influence function).
    n1 : int
        Number of unique units in this cell's subset (used to re-scale
        the influence function).
    cell_kwargs : dict
        Extra keyword arguments forwarded to the estimator.  For
        ``gt_type="dose"`` the recognized keys are ``dvals``, ``knots``,
        ``degree``, and ``num_knots``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"att_entry"`` : dict with ``att``, ``group``, and
          ``time_period``.
        - ``"extra_entry"`` : dict with ``extra_gt_returns``, ``group``,
          and ``time_period``.
        - ``"inf_func_data"`` : tuple ``("values", np.ndarray)`` holding
          the re-scaled influence function, or ``None`` if the estimator
          did not produce one.
    """
    gt_data = pl.read_ipc(subset_arrow_bytes)

    attgt_fun = cont_did_acrt if gt_type == "dose" else pte_attgt

    attgt_kwargs = {}
    if gt_type == "dose":
        for key in ("dvals", "knots", "degree", "num_knots"):
            if key in cell_kwargs:
                attgt_kwargs[key] = cell_kwargs[key]

    attgt_result = attgt_fun(gt_data=gt_data, **attgt_kwargs)

    inf_func_data = None
    if attgt_result.inf_func is not None:
        adjusted_inf_func = (n_units / max(n1, 1)) * attgt_result.inf_func
        inf_func_data = ("values", adjusted_inf_func)

    return {
        "att_entry": {"att": attgt_result.attgt, "group": g, "time_period": tp},
        "extra_entry": {
            "extra_gt_returns": attgt_result.extra_gt_returns,
            "group": g,
            "time_period": tp,
        },
        "inf_func_data": inf_func_data,
    }
