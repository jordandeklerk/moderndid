"""Control variable adjustments."""

import warnings

import numpy as np
import polars as pl
from scipy.linalg import lapack

from moderndid.core.preprocess.utils import get_covariate_names_from_formula


def compute_control_coefficients(df, config, n_groups):
    r"""Estimate the control coefficients for each baseline treatment.

    The coefficients come from a regression of the outcome's first differences on the
    controls' first differences with period fixed effects. For each baseline treatment
    :math:`d` the regression uses the observations of groups with :math:`D_{g,1} = d`
    whose treatment has not changed yet. Levels whose groups all share one switch date
    contribute no comparison and keep their outcomes unadjusted.

    The returned data carry one influence column per control with each group's
    contribution to the coefficients of its baseline treatment.
    :func:`compute_variance_adjustment` turns these columns into the variance term of
    every horizon.

    The controls are the covariates of ``xformla``. With a continuous treatment they also
    include the interactions of period indicators with powers of the baseline treatment that
    preprocessing adds to the data.

    Parameters
    ----------
    df : pl.DataFrame
        Preprocessed panel with the outcome, the controls, ``F_g``, ``d_sq``, its rank ``d_sq_int``,
        and ``weight_gt``.
    config : DIDInterConfig
        Configuration object.
    n_groups : int
        Number of groups :math:`G`.

    Returns
    -------
    df : pl.DataFrame
        Data with the influence columns ``.ctrl_influence_{j}``.
    coefficients : dict
        Mapping from the rank ``d_sq_int`` of each baseline treatment to its coefficients:

        - **theta**: Coefficients :math:`\hat{\boldsymbol{\theta}}_d` of the control differences
        - **inv_denom**: Inverse of the residualized control cross product scaled by the sample weight and :math:`G`
        - **useful**: Whether the outcomes of groups with this baseline treatment are adjusted

    Notes
    -----
    Let :math:`\Delta Y_{g,t} = Y_{g,t} - Y_{g,t-1}` and let :math:`\Delta \mathbf{X}_{g,t}`
    be the first differences of the controls. For baseline treatment :math:`d` the sample
    :math:`\mathcal{S}_d` holds the pairs :math:`(g, t)` with :math:`D_{g,1} = d`,
    :math:`t < F_g`, and observed differences. Let :math:`\widetilde{\Delta \mathbf{X}}_{g,t}`
    subtract the weighted mean of :math:`\Delta \mathbf{X}` over the observations of
    :math:`\mathcal{S}_d` in the same period and ``trends_nonparam`` cell. With

    .. math::

        \mathbf{A}_d = \sum_{(g,t) \in \mathcal{S}_d} N_{g,t}
        \widetilde{\Delta \mathbf{X}}_{g,t} \widetilde{\Delta \mathbf{X}}_{g,t}'

    the coefficients are

    .. math::

        \hat{\boldsymbol{\theta}}_d = \mathbf{A}_d^{-1} \sum_{(g,t) \in \mathcal{S}_d}
        N_{g,t} \widetilde{\Delta \mathbf{X}}_{g,t} \Delta Y_{g,t}

    where the inverse comes from a pivoted Cholesky factor that drops collinear controls.

    The influence of group :math:`g` on :math:`\hat{\boldsymbol{\theta}}_d` is

    .. math::

        \boldsymbol{\psi}_g = G \frac{W_d}{N^c_d} \mathbf{A}_d^{-1}
        \sum_{t : (g,t) \in \mathcal{S}_d} N_{g,t} \widetilde{\Delta \mathbf{X}}_{g,t}
        \kappa_{d,t} \left(\Delta Y_{g,t} - \hat{E}_{g,t}\right)

    where :math:`W_d` sums the weights in :math:`\mathcal{S}_d` and :math:`N^c_d` sums
    the weights of the not-yet-switched observations with baseline treatment :math:`d`
    and an observed outcome difference. The fitted value :math:`\hat{E}_{g,t}` comes from
    a weighted regression of :math:`\Delta Y` on :math:`\Delta \mathbf{X}` with period
    fixed effects in :math:`\mathcal{S}_d`. When :math:`n_{d,t} \geq 2` not-yet-switched
    observations of the baseline treatment are observed in period :math:`t`, the factor
    :math:`\kappa_{d,t} = \sqrt{n_{d,t} / (n_{d,t} - 1)}` corrects for degrees of freedom.
    Otherwise :math:`\kappa_{d,t} = 1` and :math:`\hat{E}_{g,t} = 0`.
    """
    controls = _control_names(df, config)
    if not controls:
        return df, {}

    gname = config.gname
    tname = config.tname
    first_diff_y = ".ctrl_first_diff_y"
    first_diffs = [f".ctrl_first_diff_{k}" for k in range(len(controls))]
    centered = [f".ctrl_centered_{k}" for k in range(len(controls))]
    influence_cols = [f".ctrl_influence_{k}" for k in range(len(controls))]
    cells = [tname, "d_sq_int", *(config.trends_nonparam or [])]
    raw_weight = pl.col(config.weightsname).fill_null(0.0) if config.weightsname else pl.lit(1.0)

    df = df.drop(influence_cols, strict=False).sort([gname, tname])
    df = df.with_columns(
        (pl.col(config.yname) - pl.col(config.yname).shift(1).over(gname)).alias(first_diff_y),
        *[
            (pl.col(ctrl) - pl.col(ctrl).shift(1).over(gname)).alias(name)
            for ctrl, name in zip(controls, first_diffs, strict=True)
        ],
        raw_weight.cast(pl.Float64).alias(".ctrl_raw_weight"),
    )
    not_yet_switched = (pl.col(tname) < pl.col("F_g")) & pl.col(first_diff_y).is_not_null()
    observed = pl.all_horizontal(pl.col(name).is_not_null() for name in first_diffs)
    df = df.with_columns(
        (not_yet_switched & observed).alias(".ctrl_sample"),
        not_yet_switched.sum().over([tname, "d_sq_int"]).cast(pl.Float64).alias(".ctrl_period_count"),
    )
    sample_weight = pl.when(pl.col(".ctrl_sample")).then(pl.col("weight_gt")).otherwise(0.0)
    df = df.with_columns(
        pl.when(pl.col(".ctrl_sample"))
        .then(pl.col(name) - (sample_weight * pl.col(name)).sum().over(cells) / sample_weight.sum().over(cells))
        .alias(centered_name)
        for name, centered_name in zip(first_diffs, centered, strict=True)
    )

    sample = df.filter(pl.col(".ctrl_sample"))
    baselines = dict(df.group_by("d_sq_int").agg(pl.col("d_sq").first()).drop_nulls().iter_rows())
    coefficients = {}
    influence_frames = []
    dropped = []
    collinear = []
    for level in sorted(baselines):
        coefficients[level] = {"theta": np.zeros(len(controls)), "inv_denom": None, "useful": False}
        is_level = pl.col("d_sq_int") == level
        if df.filter(is_level)["F_g"].n_unique() < 2:
            continue

        rows = sample.filter(is_level)
        root_weight = np.sqrt(rows["weight_gt"].to_numpy())
        x = rows.select(centered).to_numpy() * root_weight[:, None]
        y = rows[first_diff_y].to_numpy() * root_weight
        if rows.height == 0 or np.isnan(x).any():
            dropped.append(level)
            continue

        inverse, rank = _invert_symmetric(x.T @ x)
        if rank < len(controls):
            collinear.append(level)
        inv_denom = inverse * rows["weight_gt"].sum() * n_groups
        coefficients[level] = {"theta": inverse @ (x.T @ y), "inv_denom": inv_denom, "useful": True}

        n_control = df.filter(is_level & not_yet_switched)["weight_gt"].sum()
        scores = _control_scores(rows, config, first_diffs, centered) / n_control
        score_cols = [f".ctrl_score_{k}" for k in range(len(controls))]
        in_sum = (
            rows.select(gname, *[pl.Series(name, scores[:, k]) for k, name in enumerate(score_cols)])
            .group_by(gname)
            .agg(pl.col(score_cols).sum())
        )
        influence = in_sum.select(score_cols).to_numpy() @ inv_denom.T
        influence_frames.append(
            in_sum.select(gname).with_columns(pl.Series(name, influence[:, k]) for k, name in enumerate(influence_cols))
        )

    if dropped:
        warnings.warn(
            f"Groups with baseline treatment {_format_levels([baselines[level] for level in dropped])} were dropped "
            "because none of their not-yet-switched observations has every control difference.",
            UserWarning,
            stacklevel=4,
        )
        df = df.filter(~pl.col("d_sq_int").is_in(dropped))
    if collinear:
        warnings.warn(
            f"Some controls are not taken into account for groups with baseline treatment "
            f"{_format_levels([baselines[level] for level in collinear])} because their differences are collinear "
            "among the not-yet-switched observations.",
            UserWarning,
            stacklevel=4,
        )

    df = df.drop(first_diff_y, *first_diffs, *centered, ".ctrl_raw_weight", ".ctrl_sample", ".ctrl_period_count")
    if influence_frames:
        df = df.join(pl.concat(influence_frames), on=gname, how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name) for name in influence_cols)
    df = df.with_columns(pl.col(influence_cols).fill_null(0.0)).sort([gname, tname])

    return df, coefficients


def apply_control_adjustment(df, config, horizon, coefficients, horizon_type):
    r"""Remove the control component from the outcome differences.

    At an effect horizon :math:`\ell` the outcome difference of group :math:`g` becomes

    .. math::

        Y_{g,t} - Y_{g,t-\ell} - \hat{\boldsymbol{\theta}}_{D_{g,1}}'
        \left(\mathbf{X}_{g,t} - \mathbf{X}_{g,t-\ell}\right)

    At a placebo horizon the difference :math:`Y_{g,t-2\ell} - Y_{g,t-\ell}` loses the same
    coefficients times :math:`\mathbf{X}_{g,t-2\ell} - \mathbf{X}_{g,t-\ell}`. The coefficients
    come from :func:`compute_control_coefficients`.

    Parameters
    ----------
    df : pl.DataFrame
        Data sorted by group and period with the outcome difference ``.diff_y_{horizon}``.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.
    coefficients : dict
        Mapping from baseline treatment rank to coefficients from :func:`compute_control_coefficients`.
    horizon_type : {"effect", "placebo"}
        Whether the horizon is an effect or a placebo.

    Returns
    -------
    pl.DataFrame
        Data with the adjusted outcome difference and the control differences ``.ctrl_diff_{j}_{horizon}``.
    """
    controls = _control_names(df, config)
    if not controls or not coefficients:
        return df

    gname = config.gname
    diff_col = f".diff_y_{horizon}"
    diff_cols = [f".ctrl_diff_{k}_{horizon}" for k in range(len(controls))]
    if horizon_type == "effect":
        differences = [pl.col(ctrl) - pl.col(ctrl).shift(horizon).over(gname) for ctrl in controls]
    else:
        differences = [
            pl.col(ctrl).shift(2 * horizon).over(gname) - pl.col(ctrl).shift(horizon).over(gname) for ctrl in controls
        ]
    df = df.with_columns(difference.alias(name) for difference, name in zip(differences, diff_cols, strict=True))

    adjusted = pl.col(diff_col)
    for level, coef in coefficients.items():
        if not coef["useful"]:
            continue
        explained = sum(float(theta) * pl.col(name) for theta, name in zip(coef["theta"], diff_cols, strict=True))
        adjusted = pl.when(pl.col("d_sq_int") == level).then(pl.col(diff_col) - explained).otherwise(adjusted)

    return df.with_columns(adjusted.alias(diff_col))


def compute_variance_adjustment(df, config, horizon, coefficients, n_switchers, dist_col):
    r"""Compute the variance term for the estimated control coefficients.

    The adjusted estimator depends on the coefficients through the control differences of
    the switchers and their controls. This term carries the estimation error of the
    coefficients into each group's influence function at the horizon. The influence of
    each group on the coefficients comes from :func:`compute_control_coefficients`.

    Parameters
    ----------
    df : pl.DataFrame
        Data with the influence columns, the control differences from
        :func:`apply_control_adjustment`, and the horizon's switcher and control columns.
    config : DIDInterConfig
        Configuration object.
    horizon : int
        Current horizon :math:`\ell`.
    coefficients : dict
        Mapping from baseline treatment rank to coefficients from :func:`compute_control_coefficients`.
    n_switchers : float
        Weighted number of switchers :math:`N_\ell` at this horizon.
    dist_col : str
        Name of the column flagging the switchers at this horizon.

    Returns
    -------
    pl.DataFrame
        Data with the variance term ``.part2_{horizon}`` of each group.

    Notes
    -----
    For baseline treatment :math:`d` and control :math:`j` let

    .. math::

        M_{d,j,\ell} = \frac{1}{N_\ell} \sum_{g,t} \mathbb{1}\{D_{g,1} = d\} N_{g,t}
        \left(S_{g,t} - \frac{N^S_{t}}{N^C_{t}} C_{g,t}\right) \Delta_\ell X_{j,g,t}

    where :math:`S_{g,t}` flags the switchers at the horizon and :math:`C_{g,t}` flags their
    controls. Their weighted counts in the period and baseline treatment are :math:`N^S_t`
    and :math:`N^C_t`. Control :math:`j` enters through the same difference
    :math:`\Delta_\ell X_{j,g,t}` that adjusts the outcome at the horizon. The term of
    group :math:`g` is

    .. math::

        \sum_{d} \sum_{j} M_{d,j,\ell} \left(\psi_{g,d,j} - \hat{\theta}_{d,j}\right)

    where :math:`\psi_{g,d,j}` is zero unless :math:`D_{g,1} = d`.
    """
    controls = _control_names(df, config)
    part2_col = f".part2_{horizon}"
    useful = {level: coef for level, coef in coefficients.items() if coef["useful"]}
    if not controls or not useful:
        return df.with_columns(pl.lit(0.0).alias(part2_col))

    tname = config.tname
    n_control = pl.col(f".n_control_{horizon}")
    safe_n_control = pl.when(n_control.is_null() | (n_control == 0)).then(1.0).otherwise(n_control)
    comparison = (
        ((pl.col("T_g") - 2) >= horizon).cast(pl.Float64)
        * ((pl.col(tname) >= horizon + 1) & (pl.col(tname) <= pl.col("T_g"))).cast(pl.Float64)
        * pl.col("weight_gt")
        * (
            pl.col(dist_col)
            - (pl.col(f".n_treated_{horizon}") / safe_n_control) * pl.col(f".never_change_{horizon}").fill_null(0.0)
        )
        / n_switchers
    )
    totals = df.select(
        pl.when(pl.col("d_sq_int") == level)
        .then(comparison * pl.col(f".ctrl_diff_{k}_{horizon}"))
        .sum()
        .alias(f".ctrl_m_{index}_{k}")
        for index, level in enumerate(useful)
        for k in range(len(controls))
    ).row(0)
    weights = np.array(totals, dtype=float).reshape(len(useful), len(controls))

    part2 = pl.lit(0.0)
    for level_weights, (level, coef) in zip(weights, useful.items(), strict=True):
        influence = sum(float(m) * pl.col(f".ctrl_influence_{k}") for k, m in enumerate(level_weights))
        part2 = (
            part2
            + pl.when(pl.col("d_sq_int") == level).then(influence).otherwise(0.0)
            - float(level_weights @ coef["theta"])
        )

    return df.with_columns(part2.alias(part2_col))


def _control_names(df, config):
    """Get the control columns of the formula and the continuous baseline trends."""
    controls = get_covariate_names_from_formula(config.xformla) or []
    if config.continuous > 0:
        controls = [*controls, *(name for name in df.columns if name.startswith(".baseline_trend_"))]
    return controls


def _control_scores(rows, config, first_diffs, centered):
    """Score each sample row against a period fixed-effects fit of the outcome differences."""
    tname = config.tname
    times = rows[tname].to_numpy()
    raw_weight = rows[".ctrl_raw_weight"].to_numpy()
    dx = rows.select(first_diffs).to_numpy()
    dy = rows[".ctrl_first_diff_y"].to_numpy()

    # Since a zero-weight row cannot set a period's mean, it only receives a fitted value.
    fit = raw_weight > 0
    fitted = np.full(len(dy), np.nan)
    if fit.any():
        periods, index = np.unique(times[fit], return_inverse=True)
        period_weight = np.bincount(index, weights=raw_weight[fit])
        dx_mean = np.column_stack([np.bincount(index, weights=raw_weight[fit] * col) for col in dx[fit].T])
        dx_mean = dx_mean / period_weight[:, None]
        dy_mean = np.bincount(index, weights=raw_weight[fit] * dy[fit]) / period_weight
        root_weight = np.sqrt(raw_weight[fit])[:, None]
        beta = np.linalg.lstsq(
            (dx[fit] - dx_mean[index]) * root_weight,
            (dy[fit] - dy_mean[index]) * root_weight[:, 0],
            rcond=None,
        )[0]
        position = np.minimum(np.searchsorted(periods, times), len(periods) - 1)
        covered = periods[position] == times
        fitted[covered] = (dy_mean - dx_mean @ beta)[position[covered]] + dx[covered] @ beta

    count = rows[".ctrl_period_count"].to_numpy()
    enough = count >= 2
    kappa = np.ones(len(count))
    kappa[enough] = np.sqrt(count[enough] / (count[enough] - 1))
    residual = kappa * (dy - fitted * enough)
    scores = rows["weight_gt"].to_numpy()[:, None] * rows.select(centered).to_numpy() * residual[:, None]
    # A row without a fitted value drops out of the sums like a missing value.
    return np.where(np.isnan(scores), 0.0, scores)


def _invert_symmetric(matrix):
    """Invert a symmetric matrix through a pivoted Cholesky factor and zero out dependent columns."""
    size = matrix.shape[0]
    factor, pivot, rank, _ = lapack.dpstrf(matrix)
    inverse = np.zeros((size, size))
    if rank > 0:
        block, _ = lapack.dpotri(np.triu(factor[:rank, :rank]))
        kept = pivot[:rank] - 1
        inverse[np.ix_(kept, kept)] = np.triu(block) + np.triu(block, 1).T
    return inverse, rank


def _format_levels(levels):
    """Format baseline treatment levels for a warning."""
    return ", ".join(f"{level:g}" for level in levels)
