"""Control variable adjustment for DIDInter."""

import numpy as np
import polars as pl


def compute_control_coefficients(df, config, horizon):
    """Compute regression coefficients for control adjustment.

    Parameters
    ----------
    df : pl.DataFrame
        Data with outcome and control columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon.

    Returns
    -------
    dict
        Mapping from baseline treatment level to coefficient vector.
    """
    controls = config.controls

    if not controls:
        return {}

    diff_y_col = f"diff_y_{horizon}"
    coefficients = {}

    baseline_levels = df.filter(pl.col("F_g") == float("inf"))["d_sq"].unique().to_list()

    for d_level in baseline_levels:
        subset = df.filter(
            (pl.col("d_sq") == d_level) & (pl.col("F_g") == float("inf")) & pl.col(diff_y_col).is_not_null()
        )

        if len(subset) < len(controls) + 1:
            coefficients[d_level] = np.zeros(len(controls))
            continue

        y = subset.select(diff_y_col).to_numpy().flatten()
        weights = subset.select("weight_gt").to_numpy().flatten()

        X_cols = []
        for ctrl in controls:
            lag_col = f"lag_{ctrl}_{horizon}"
            if lag_col in subset.columns:
                diff_ctrl = subset.select(pl.col(ctrl) - pl.col(lag_col)).to_numpy().flatten()
            else:
                diff_ctrl = np.zeros(len(subset))
            X_cols.append(diff_ctrl)

        X = np.column_stack(X_cols)

        valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if np.sum(valid_mask) < len(controls) + 1:
            coefficients[d_level] = np.zeros(len(controls))
            continue

        y_valid = y[valid_mask]
        X_valid = X[valid_mask]
        w_valid = weights[valid_mask]

        try:
            W = np.diag(w_valid)
            XtWX = X_valid.T @ W @ X_valid
            XtWy = X_valid.T @ W @ y_valid
            theta = np.linalg.solve(XtWX, XtWy)
            coefficients[d_level] = theta
        except np.linalg.LinAlgError:
            coefficients[d_level] = np.zeros(len(controls))

    return coefficients


def apply_control_adjustment(df, config, horizon, coefficients):
    """Apply control adjustment.

    Parameters
    ----------
    df : pl.DataFrame
        Data with outcome and control columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon.
    coefficients : dict
        Mapping from baseline treatment level to coefficient vector.

    Returns
    -------
    pl.DataFrame
        DataFrame with adjusted outcome differences.
    """
    gname = config.gname
    controls = config.controls

    if not controls or not coefficients:
        return df

    diff_y_col = f"diff_y_{horizon}"

    for ctrl in controls:
        lag_col = f"lag_{ctrl}_{horizon}"

        if lag_col not in df.columns:
            df = df.sort([gname, config.tname])
            df = df.with_columns(pl.col(ctrl).shift(horizon).over(gname).alias(lag_col))

        diff_ctrl_col = f"diff_{ctrl}_{horizon}"
        df = df.with_columns((pl.col(ctrl) - pl.col(lag_col)).alias(diff_ctrl_col))

    for d_level, theta in coefficients.items():
        adjustment = pl.lit(0.0)
        for ctrl_idx, ctrl in enumerate(controls):
            diff_ctrl_col = f"diff_{ctrl}_{horizon}"
            adjustment = adjustment + pl.lit(theta[ctrl_idx]) * pl.col(diff_ctrl_col).fill_null(0.0)

        df = df.with_columns(
            pl.when(pl.col("d_sq") == d_level)
            .then(pl.col(diff_y_col) - adjustment)
            .otherwise(pl.col(diff_y_col))
            .alias(diff_y_col)
        )

    return df
