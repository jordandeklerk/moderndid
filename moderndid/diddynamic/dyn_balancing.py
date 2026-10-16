"""Dynamic covariate balancing treatment effect estimation for panel data with time-varying treatments."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from moderndid.core.dataframe import to_polars
from moderndid.core.parallel import parallel_map
from moderndid.core.preprocess import DynBalancingConfig, PreprocessDataBuilder
from moderndid.core.preprocess.validators import check_columns

from .container import DynBalancingHetResult, DynBalancingHistoryResult, DynBalancingResult
from .estimation.inference import (
    compute_ate_variance,
    compute_quantiles,
    compute_variance,
    compute_variance_clustered,
)


def dyn_balancing(
    data,
    yname,
    tname,
    idname,
    treatment_name,
    ds1,
    ds2,
    xformla=None,
    fixed_effects=None,
    pooled=False,
    clustervars=None,
    balancing="dcb",
    method="lasso_plain",
    alpha=0.05,
    final_period=None,
    initial_period=None,
    adaptive_balancing=True,
    debias=False,
    continuous_treatment=False,
    lb=0.0005,
    ub=2.0,
    regularization=True,
    fast_adaptive=False,
    grid_length=1000,
    n_beta_nonsparse=1e-4,
    ratio_coefficients=1 / 3,
    nfolds=10,
    lags=None,
    robust_quantile=False,
    demeaned_fe=False,
    histories_length=None,
    final_periods=None,
    impulse_response=False,
    n_jobs=1,
    random_state=None,
):
    r"""Estimate treatment effects under dynamic treatment regimes.

    Implements the dynamic covariate balancing estimator of [1]_ for comparing potential
    outcomes under the treatment histories :math:`d_{1:T}` in ``ds1`` and
    :math:`d'_{1:T}` in ``ds2``. The average treatment effect is

    .. math::

        \text{ATE}(d_{1:T}, d'_{1:T}) = \mu_T(d_{1:T}) - \mu_T(d'_{1:T}),

    where :math:`\mu_T(d_{1:T}) = \mathbb{E}[Y_T(d_{1:T})]` is the potential outcome
    under treatment history :math:`d_{1:T}`.

    Under sequential ignorability and overlap, the estimator projects the outcome
    backward from the final period. For each period :math:`t`, it solves a quadratic
    program for the balancing weights :math:`\hat{\gamma}_t` of smallest :math:`\ell_2`
    norm that satisfy the dynamic covariate balance constraints. These weights correct
    the projections. The standard errors are analytic and treat the first-period
    covariates as fixed.

    See the :ref:`dynamic covariate balancing example <example_dyn_balancing>` for a
    full analysis of the democracy and economic growth data.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column. The periods must be consecutive integers.
    idname : str
        The name of the unit identifier column.
    treatment_name : str
        The name of the binary treatment column.
    ds1 : list[int]
        Treatment history of the first potential outcome over the ``len(ds1)`` periods
        that end at ``final_period``.
    ds2 : list[int]
        Treatment history of the second potential outcome, as long as ``ds1``.
    xformla : str, optional
        Formula for the covariates, such as ``"~ X1 + X2"``. Each term must name a
        column. The estimator needs at least one covariate from ``xformla``,
        ``fixed_effects``, or both.
    fixed_effects : list[str], optional
        Columns to include as fixed-effect dummies.
    pooled : bool, default=False
        Whether to add every complete earlier window that ends at or after
        ``initial_period`` as a separate unit history. Standard errors then cluster on
        ``idname`` unless ``clustervars`` is set.
    clustervars : str or list[str], optional
        The column to cluster standard errors on. A list must hold one name.
    balancing : {"dcb", "aipw", "ipw"}, default="dcb"
        Weights of the estimator. ``"dcb"`` balances the covariates. ``"ipw"`` uses
        inverse probability weights and ``"aipw"`` corrects the projections with them.
    method : {"lasso_plain", "lasso_subsample"}, default="lasso_plain"
        LASSO strategy of the coefficient stage. The projections of
        ``balancing="aipw"`` always use ``"lasso_subsample"`` with 10 folds.
    alpha : float, default=0.05
        Significance level of the confidence interval and p-value.
    final_period : int, optional
        Last period of the treatment histories. Defaults to the latest period.
    initial_period : int, optional
        Earliest final period of the windows that ``pooled=True`` stacks. Defaults to
        the first period at which a full history ends.
    adaptive_balancing : bool, default=True
        Whether to tighten balance on covariates with large estimated coefficients.
    debias : bool, default=False
        Whether to subtract a bootstrap estimate from 20 resamples of the bias that the
        projection coefficients leave in each potential outcome.
    continuous_treatment : bool, default=False
        Whether the treatment is continuous. Only False is supported.
    lb : float, default=0.0005
        Lower bound of the grid of tuning constants.
    ub : float, default=2.0
        Upper bound of the grid of tuning constants.
    regularization : bool, default=True
        Whether to fit the coefficient stage by cross-validated LASSO rather than ridge
        with a negligible penalty.
    fast_adaptive : bool, default=False
        Whether to use a flat grid search instead of the three-segment nested search.
    grid_length : int, default=1000
        Number of points in the grid of tuning constants.
    n_beta_nonsparse : float, default=1e-4
        Threshold below which a rescaled coefficient counts as zero.
    ratio_coefficients : float, default=1/3
        Fraction of the largest coefficients to prioritize when sparsity is low.
    nfolds : int, default=10
        Number of cross-validation folds for LASSO.
    lags : int, optional
        Number of most recent treatment indicators that ``"lasso_plain"`` leaves
        unpenalized. Defaults to ``len(ds1)``.
    robust_quantile : bool, default=False
        Whether the interval and p-value use the square root of a chi-squared critical
        value with :math:`2T` degrees of freedom instead of the Gaussian one.
    demeaned_fe : bool, default=False
        Option for continuous treatments. Only False is supported.
    histories_length : list[int], optional
        History lengths, each from 1 to ``len(ds1)``, at which to estimate the ATE on
        the last entries of ``ds1`` and ``ds2``. Can't be combined with
        ``final_periods``.
    final_periods : list[int], optional
        Final periods at which to estimate the ATE. Can't be combined with
        ``histories_length``.
    impulse_response : bool, default=False
        Whether each history length compares ``[1, 0, ..., 0]`` with ``[0, ..., 0]`` in
        place of ``ds1`` and ``ds2``. Requires ``histories_length``.
    n_jobs : int, default=1
        Number of parallel workers for ``histories_length`` or ``final_periods``.
        ``-1`` uses all cores.
    random_state : int, Generator, optional
        Seeds the resamples of ``debias=True``. Pass an int for reproducible results.

    Returns
    -------
    DynBalancingResult or DynBalancingHistoryResult or DynBalancingHetResult
        With ``histories_length`` or ``final_periods``, a container whose ``summary``
        table and ``results`` list hold one entry per history length or final period.
        Otherwise a DynBalancingResult containing:

        - **att**: The ATE point estimate (:math:`\mu_1 - \mu_2`)
        - **var_att**: Variance of the ATE
        - **mu1**: Potential outcome estimate under ``ds1``
        - **mu2**: Potential outcome estimate under ``ds2``
        - **var_mu1**: Variance of ``mu1``
        - **var_mu2**: Variance of ``mu2``
        - **robust_quantile**: Chi-squared critical value for the ATE, or the Gaussian one
          when ``robust_quantile=False``
        - **gaussian_quantile**: Gaussian critical value for inference
        - **gammas**: Weights per treatment history
        - **coefficients**: LASSO coefficients per treatment history (DCB only)
        - **imbalances**: Standardized imbalance of each covariate and fixed-effect dummy per period
        - **estimation_params**: Metadata such as the observation count and the variable names

    Notes
    -----
    The variance of the ATE is the sum of the variances of ``mu1`` and ``mu2`` minus
    twice their covariance. Since both estimates weight the same units until the two
    histories first differ, the covariance comes from those shared periods. With
    clustered standard errors, it also comes from units of one cluster that follow
    different histories. Without clusters, two histories that differ in their first
    treatment have zero covariance. The variance then matches the ATE variance of [1]_.
    The :ref:`dynamic covariate balancing background <background-diddynamic>` derives
    the weights and these variances.

    Rows with a null, NaN, or infinite unit, period, or cluster are dropped with a
    warning. An infinite outcome, treatment, covariate, or fixed effect counts as
    missing. So does a null or NaN fixed effect.

    References
    ----------
    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016

    """
    # Since the plots module imports the diddynamic containers, this package must load without scikit-learn.
    from .estimation.weights_dcb import compute_dcb_estimator, compute_imbalances
    from .estimation.weights_ipw import compute_ipw_estimator

    if histories_length is not None and final_periods is not None:
        raise ValueError("histories_length and final_periods are mutually exclusive.")
    if impulse_response and histories_length is None:
        raise ValueError("impulse_response=True requires histories_length.")
    if not ds1:
        raise ValueError("ds1 must be a non-empty list of treatment values.")
    if not ds2:
        raise ValueError("ds2 must be a non-empty list of treatment values.")
    if len(ds1) != len(ds2):
        raise ValueError(f"ds1 and ds2 must have the same length, got {len(ds1)} and {len(ds2)}.")
    if balancing == "ipw_msm":
        raise ValueError(
            "balancing='ipw_msm' is not available. Its stabilized weights would only rescale the 'ipw' weights "
            "of the target treatment history by a constant. Since the normalized estimate cancels that "
            "constant, the result would match balancing='ipw'. Use balancing='ipw'."
        )
    if balancing not in ("dcb", "aipw", "ipw"):
        raise ValueError(f"balancing must be one of 'dcb', 'aipw', 'ipw', got {balancing!r}.")
    if method not in ("lasso_plain", "lasso_subsample"):
        raise ValueError(f"method must be one of 'lasso_plain', 'lasso_subsample', got {method!r}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}.")
    if lb > ub:
        raise ValueError(f"lb ({lb}) must be less than or equal to ub ({ub}).")
    if lags is not None and lags < 0:
        raise ValueError(f"lags must be a nonnegative integer or None, got {lags}.")
    if isinstance(clustervars, str):
        clustervars = [clustervars]
    if clustervars is not None and len(clustervars) > 1:
        raise ValueError(
            f"clustervars must name a single column because only one-way clustering is supported, got {clustervars}."
        )
    if continuous_treatment:
        raise NotImplementedError("Continuous treatment estimation is not implemented yet.")
    if demeaned_fe:
        raise NotImplementedError(
            "demeaned_fe=True applies only to continuous treatments. Since those are not implemented yet, "
            "leave demeaned_fe=False."
        )
    if alpha > 0.1:
        warnings.warn("Significance level larger than 0.1 selected.", stacklevel=2)
    if histories_length is None and final_periods is None and len(ds1) == 1:
        warnings.warn("ds1 contains one element. No dynamics will be considered.", stacklevel=2)
    if pooled and clustervars is None:
        clustervars = [idname]
    data = to_polars(data)
    check_columns(
        data,
        yname=yname,
        tname=tname,
        idname=idname,
        treatment_name=treatment_name,
        xformla=xformla,
        fixed_effects=fixed_effects,
        clustervars=clustervars,
    )

    if histories_length is not None:
        return _run_history(
            data=data,
            yname=yname,
            tname=tname,
            idname=idname,
            treatment_name=treatment_name,
            ds1=ds1,
            ds2=ds2,
            histories_length=histories_length,
            xformla=xformla,
            fixed_effects=fixed_effects,
            pooled=pooled,
            clustervars=clustervars,
            balancing=balancing,
            method=method,
            alpha=alpha,
            final_period=final_period,
            initial_period=initial_period,
            adaptive_balancing=adaptive_balancing,
            debias=debias,
            continuous_treatment=continuous_treatment,
            lb=lb,
            ub=ub,
            regularization=regularization,
            fast_adaptive=fast_adaptive,
            grid_length=grid_length,
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            nfolds=nfolds,
            lags=lags,
            robust_quantile=robust_quantile,
            demeaned_fe=demeaned_fe,
            impulse_response=impulse_response,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if final_periods is not None:
        return _run_het(
            data=data,
            yname=yname,
            tname=tname,
            idname=idname,
            treatment_name=treatment_name,
            ds1=ds1,
            ds2=ds2,
            final_periods=final_periods,
            xformla=xformla,
            fixed_effects=fixed_effects,
            pooled=pooled,
            clustervars=clustervars,
            balancing=balancing,
            method=method,
            alpha=alpha,
            initial_period=initial_period,
            adaptive_balancing=adaptive_balancing,
            debias=debias,
            continuous_treatment=continuous_treatment,
            lb=lb,
            ub=ub,
            regularization=regularization,
            fast_adaptive=fast_adaptive,
            grid_length=grid_length,
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            nfolds=nfolds,
            lags=lags,
            robust_quantile=robust_quantile,
            demeaned_fe=demeaned_fe,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    config = DynBalancingConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        treatment_name=treatment_name,
        ds1=list(ds1),
        ds2=list(ds2),
        xformla=xformla,
        fixed_effects=fixed_effects,
        pooled=pooled,
        clustervars=clustervars,
        balancing=balancing,
        method=method,
        alpha=alpha,
        final_period=final_period,
        initial_period=initial_period,
        adaptive_balancing=adaptive_balancing,
        debias=debias,
        continuous_treatment=continuous_treatment,
        lb=lb,
        ub=ub,
        regularization=regularization,
        fast_adaptive=fast_adaptive,
        grid_length=grid_length,
        n_beta_nonsparse=n_beta_nonsparse,
        ratio_coefficients=ratio_coefficients,
        nfolds=nfolds,
        lags=lags,
        robust_quantile=robust_quantile,
        demeaned_fe=demeaned_fe,
    )

    dp = PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()

    n_periods = config.n_periods
    outcome = dp.outcome_vector
    treatment_matrix = dp.treatment_matrix
    cluster = dp.cluster
    dim_fe = dp.dim_fe

    covariates_t = _reindex_covariates(dp.covariate_dict, config.time_periods)
    if not covariates_t:
        raise ValueError(
            "dyn_balancing balances covariates and needs at least one. Pass xformla, fixed_effects, or both."
        )

    ds1_arr = np.array(ds1, dtype=float)
    ds2_arr = np.array(ds2, dtype=float)

    if balancing == "dcb":
        res1 = compute_dcb_estimator(
            n_periods,
            outcome,
            treatment_matrix,
            covariates_t,
            ds1_arr,
            method=method,
            adaptive_balancing=adaptive_balancing,
            debias=debias,
            regularization=regularization,
            nfolds=nfolds,
            lb=lb,
            ub=ub,
            grid_length=grid_length,
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            lags=lags,
            dim_fe=dim_fe,
            fast_adaptive=fast_adaptive,
            random_state=random_state,
        )
        res2 = compute_dcb_estimator(
            n_periods,
            outcome,
            treatment_matrix,
            covariates_t,
            ds2_arr,
            method=method,
            adaptive_balancing=adaptive_balancing,
            debias=debias,
            regularization=regularization,
            nfolds=nfolds,
            lb=lb,
            ub=ub,
            grid_length=grid_length,
            n_beta_nonsparse=n_beta_nonsparse,
            ratio_coefficients=ratio_coefficients,
            lags=lags,
            dim_fe=dim_fe,
            fast_adaptive=fast_adaptive,
            random_state=random_state,
        )

        mu1 = res1.mu_hat
        mu2 = res2.mu_hat

        if debias:
            mu1 -= res1.bias
            mu2 -= res2.bias

        coefficients_out = {"ds1": res1.coef_t, "ds2": res2.coef_t}
    else:
        res1 = compute_ipw_estimator(
            n_periods,
            outcome,
            treatment_matrix,
            covariates_t,
            ds1_arr,
            method=balancing,
            regularization=regularization,
            lags=lags,
            dim_fe=dim_fe,
        )
        res2 = compute_ipw_estimator(
            n_periods,
            outcome,
            treatment_matrix,
            covariates_t,
            ds2_arr,
            method=balancing,
            regularization=regularization,
            lags=lags,
            dim_fe=dim_fe,
        )

        mu1 = res1.mu_hat
        mu2 = res2.mu_hat
        coefficients_out = {}

    if cluster is not None:
        var1 = compute_variance_clustered(res1.gammas, res1.predictions, res1.not_nas, outcome, cluster)
        var2 = compute_variance_clustered(res2.gammas, res2.predictions, res2.not_nas, outcome, cluster)
    else:
        var1 = compute_variance(res1.gammas, res1.predictions, res1.not_nas, outcome)
        var2 = compute_variance(res2.gammas, res2.predictions, res2.not_nas, outcome)

    labels = _covariate_labels(config, dp.panel)
    imbalances_out = {
        "ds1": _imbalance_table(compute_imbalances(res1.gammas, res1.not_nas, covariates_t), labels),
        "ds2": _imbalance_table(compute_imbalances(res2.gammas, res2.not_nas, covariates_t), labels),
    }

    ate = mu1 - mu2
    var_ate = compute_ate_variance(res1, res2, outcome, cluster)

    quantiles = compute_quantiles(alpha, n_periods, robust_quantile)

    estimation_params = {
        "yname": yname,
        "tname": tname,
        "idname": idname,
        "treatment_name": treatment_name,
        "balancing": balancing,
        "method": method,
        "n_units": dp.panel[idname].n_unique(),
        "n_obs": len(dp.panel),
        "n_periods": n_periods,
        "ds1": list(ds1),
        "ds2": list(ds2),
        "alpha": alpha,
        "robust_quantile": robust_quantile,
        "adaptive_balancing": adaptive_balancing,
        "debias": debias,
        "clustervars": clustervars,
    }
    if pooled:
        # Each stacked window counts as its own unit history in the estimation.
        estimation_params["n_stacked_units"] = config.n_units

    return DynBalancingResult(
        att=ate,
        var_att=var_ate,
        mu1=mu1,
        mu2=mu2,
        var_mu1=var1,
        var_mu2=var2,
        robust_quantile=quantiles.robust_quantile_ate,
        gaussian_quantile=quantiles.gaussian_quantile_ate,
        gammas={"ds1": res1.gammas, "ds2": res2.gammas},
        coefficients=coefficients_out,
        imbalances=imbalances_out,
        estimation_params=estimation_params,
    )


def _run_history(
    *, ds1, ds2, histories_length, impulse_response=False, n_jobs=1, **kwargs
) -> DynBalancingHistoryResult:
    """Dispatch for histories_length mode."""
    if not histories_length:
        raise ValueError("histories_length must be a non-empty list.")
    t_all = len(ds1)
    for h in histories_length:
        if h < 1 or h > t_all:
            raise ValueError(f"All entries in histories_length must be between 1 and {t_all} (len(ds1)), got {h}.")

    sorted_lengths = sorted(histories_length)
    if impulse_response:
        args_list = [([1] + [0] * (h - 1), [0] * h, kwargs) for h in sorted_lengths]
    else:
        args_list = [(ds1[-h:], ds2[-h:], kwargs) for h in sorted_lengths]
    results = parallel_map(_call_dyn_balancing, args_list, n_jobs=n_jobs)

    summary = pl.DataFrame(
        {
            "period_length": sorted_lengths,
            "att": [r.att for r in results],
            "var_att": [r.var_att for r in results],
            "mu1": [r.mu1 for r in results],
            "var_mu1": [r.var_mu1 for r in results],
            "mu2": [r.mu2 for r in results],
            "var_mu2": [r.var_mu2 for r in results],
            "robust_quantile": [r.robust_quantile for r in results],
            "gaussian_quantile": [r.gaussian_quantile for r in results],
        }
    )
    return DynBalancingHistoryResult(summary=summary, results=results)


def _run_het(*, ds1, ds2, final_periods, n_jobs=1, **kwargs) -> DynBalancingHetResult:
    """Dispatch for final_periods mode."""
    if not final_periods:
        raise ValueError("final_periods must be a non-empty list.")

    sorted_periods = sorted(final_periods)
    args_list = [(ds1, ds2, {**kwargs, "final_period": p}) for p in sorted_periods]
    results = parallel_map(_call_dyn_balancing, args_list, n_jobs=n_jobs)

    summary = pl.DataFrame(
        {
            "final_period": sorted_periods,
            "att": [r.att for r in results],
            "var_att": [r.var_att for r in results],
            "mu1": [r.mu1 for r in results],
            "var_mu1": [r.var_mu1 for r in results],
            "mu2": [r.mu2 for r in results],
            "var_mu2": [r.var_mu2 for r in results],
            "robust_quantile": [r.robust_quantile for r in results],
            "gaussian_quantile": [r.gaussian_quantile for r in results],
        }
    )
    return DynBalancingHetResult(summary=summary, results=results)


def _call_dyn_balancing(dd1, dd2, kwargs):
    """Call dyn_balancing with unpacked arguments for parallel_map."""
    return dyn_balancing(ds1=dd1, ds2=dd2, **kwargs)


def _reindex_covariates(covariate_dict: dict[int, np.ndarray], time_periods: np.ndarray) -> dict[int, np.ndarray]:
    """Re-key covariate dict from actual period values to 0-based indices."""
    sorted_periods = sorted(time_periods)
    return {i: covariate_dict[p] for i, p in enumerate(sorted_periods) if p in covariate_dict}


def _covariate_labels(config, panel):
    """Return the covariate names in the column order of the covariate matrices."""
    labels = list(config.covariate_names)
    # The preprocessing names each fixed-effect dummy after its column and appends them in this order.
    for fe_col in config.fixed_effects or []:
        # Since pooling keeps the calendar period in new_Time, its dummies get the time column's name back.
        name = config.tname if config.pooled and fe_col == "new_Time" else fe_col
        labels.extend(name + c[len(fe_col) :] for c in panel.columns if c.startswith(f"{fe_col}_"))
    return labels


def _imbalance_table(imbalances, labels):
    """Return one row per period and covariate with its standardized imbalance."""
    n_periods = imbalances.shape[0]
    return pl.DataFrame(
        {
            "period": np.repeat(np.arange(1, n_periods + 1), len(labels)),
            "covariate": labels * n_periods,
            "imbalance": imbalances.ravel(),
        }
    )
