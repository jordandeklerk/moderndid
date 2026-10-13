"""ML-based doubly robust DiD estimator."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as la
import scipy.stats

from moderndid.core.preprocess import (
    BasePeriod,
    ControlGroup,
    DIDMLConfig,
    PreprocessDataBuilder,
)
from moderndid.core.preprocess.validators import check_columns

from .compute import bootstrap_didml, bootstrap_se, compute_didml, simultaneous_critical_value, unit_cohorts
from .container import didml_result


def didml(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    weightsname=None,
    alpha=0.05,
    cband=False,
    boot=True,
    biters=100,
    clustervars=None,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    nu_model="rlearner",
    sigma_model="rlearner",
    delta_model="glm",
    k_folds=10,
    lambda_choice="lambda.min",
    t_func=False,
    use_gamma=True,
    zeta=0.5,
    compute_drdid_benchmark=True,
    random_state=None,
    n_jobs=1,
):
    r"""Compute group-time ATTs and CATTs with cross-fitted ML nuisances.

    Implements the staggered-adoption ML doubly robust DiD estimator of [1]_. For cohort
    :math:`g` in period :math:`t`, it estimates the group-time average treatment effect
    on the treated and its conditional version, the CATT,

    .. math::

        ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g], \qquad
        \tau_{g,t}(x) = \mathbb{E}[Y_t(g) - Y_t(0) \mid X = x, G = g],

    where :math:`Y_t(g)` is the potential outcome at time :math:`t` for a unit first
    treated at :math:`g` and :math:`Y_t(0)` is the never-treated potential outcome.

    Each cell stacks one pre-period and one post-period row per unit of the treated
    cohort and of the controls that ``control_group`` selects. :func:`lnw_did` fits the
    Nie-Lu-Wager score [2]_ with cross-fitted nuisances and predicts the CATT of each
    unit. The cell ATT averages the CATTs of the treated cohort and adds a residual
    correction with the augmented minimax-linear weights of [3]_.

    Standard errors come from a unit bootstrap by default. Each of the ``biters`` draws
    resamples the units and refits every cell under a new fold split. The bootstrap
    therefore takes about ``biters`` times as long as the estimates.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column.
    idname : str
        The name of the unit identifier column. Since every cell compares two periods of
        each unit, units missing from any period are dropped with a warning.
    gname : str
        The name of the column holding the first period in which each unit is treated,
        or 0 for never-treated units.
    xformla : str, optional
        Formula for the covariates, such as ``"~ x1 + x2"``. If None, the fit uses no
        covariates.
    weightsname : str, optional
        The name of the per-unit sampling weights column. The weights enter the AMLE
        balance target, the average of the CATTs, the cohort shares of
        :func:`aggte_didml`, and the bootstrap. The nuisance models ignore them.
    alpha : float, default=0.05
        The significance level.
    cband : bool, default=False
        Whether to compute a uniform confidence band over all group-time ATTs. Only
        applicable if ``boot=True``.
    boot : bool, default=True
        Whether to use the unit bootstrap. With False, the standard errors come from the
        influence functions. They treat the fold split as fixed, can understate the
        sampling variability, and need ``use_gamma=True``.
    biters : int, default=100
        Number of bootstrap draws, at least 2 when ``boot=True``. The bootstrap pre-test
        needs more draws than pre-treatment cells and otherwise warns and reports none.
    clustervars : list[str], optional
        Variables to cluster on. The standard errors don't support clusters yet and
        ignore them with a warning.
    control_group : {"nevertreated", "notyettreated"}, default="notyettreated"
        Comparison group for the doubly robust score.
    anticipation : int, default=0
        Number of periods before treatment in which treated units may anticipate it.
    base_period : {"varying", "universal"}, default="varying"
        Base period of each comparison. ``"varying"`` compares each pre-treatment period
        with the one before it. ``"universal"`` anchors every comparison of cohort ``g``
        at the last period before ``g`` less the anticipation and reports that reference
        cell as zero.
    nu_model : {"rlearner", "cf"}, default="rlearner"
        Backend for the conditional time trend :math:`\nu(x)` and the propensity
        :math:`t(x)` of :func:`lnw_did`.
    sigma_model : {"rlearner", "cf"}, default="rlearner"
        Backend for the conditional cohort contrast :math:`\zeta(x)` and the propensity
        :math:`g(x)` of :func:`lnw_did`.
    delta_model : {"glm", "stack"}, default="glm"
        Backend for the conditional covariance term :math:`\Delta(x)`. ``"glm"`` fits a
        cross-validated lasso and ``"stack"`` a stacking ensemble.
    k_folds : int, default=10
        Number of folds for cross-fitting the nuisance models.
    lambda_choice : {"lambda.min", "lambda.1se"}, default="lambda.min"
        Cross-validation rule for the lasso penalty of the R-learner residual fits and
        the linear CATT fit.
    t_func : bool, default=False
        Whether to use the estimated post-period propensity instead of the constant 0.5
        that :func:`lnw_did` uses by default.
    use_gamma : bool, default=True
        Whether to apply AMLE weights. Without them, the cell ATT is the mean CATT of
        the treated cohort in the post period.
    zeta : float, default=0.5
        AMLE objective mixing weight in :math:`(0, 1)`.
    compute_drdid_benchmark : bool, default=True
        Whether to compute the doubly robust group-time ATT of [4]_ in each cell with
        the estimator of [5]_ as a benchmark.
    random_state : int, optional
        Seed for the fold splits, the inner cross-validation, and the bootstrap
        resamples. The estimates depend on it through the fold splits.
    n_jobs : int, default=1
        Number of workers for the cells and the bootstrap draws. ``-1`` uses all cores.
        The draws run in worker processes that each copy the data and take a few seconds
        to start. A script that sets ``n_jobs`` must call ``didml`` inside an
        ``if __name__ == "__main__":`` block.

    Returns
    -------
    DIDMLResult
        Object containing group-time ML ATT and CATT results:

        - **groups**: Array indicating which cohort (period first treated) each ATT is for
        - **times**: Array indicating which time period each ATT is for
        - **att_gt**: Array of group-time ML ATT estimates
        - **se_gt**: Standard errors from the unit bootstrap, or from the influence functions with ``boot=False``
        - **critical_value**: Critical value of the intervals, simultaneous with ``cband=True`` and ``boot=True``
        - **influence_func**: Per-unit influence-function matrix of shape ``(n_units, n_cells)``
        - **cates**: Sparse matrix of per-unit CATT predictions
        - **scores**: Sparse matrix of per-unit DR scores, each the average of the unit's two periods
        - **gammas**: Sparse matrix of per-unit AMLE weights
        - **unit_ids**: Unit identifiers indexing the sparse-matrix rows
        - **unit_periods**: Placeholder period index of zeros for the sparse-matrix rows
        - **drdid_benchmark**: DRDID benchmark ATTs per cell when ``compute_drdid_benchmark=True``
        - **drdid_benchmark_se**: Standard errors of the DRDID benchmark
        - **n_units**: Number of unique cross-sectional units
        - **wald_stat**: Wald statistic for pre-testing parallel trends
        - **wald_pvalue**: P-value of the parallel-trends pre-test
        - **aggregate_effects**: Always None. ``aggte_didml`` returns the aggregate effects separately
        - **alpha**: Significance level used
        - **estimation_params**: Dictionary of the estimation settings
        - **G**: Cohort of each unit, in the row order of ``influence_func``
        - **weights_ind**: Sampling weight of each unit, in the same order
        - **att_gt_draws**: Bootstrap draws of the group-time ATTs of shape ``(biters, n_cells)``, or None
        - **cohort_share_draws**: Bootstrap draws of the weighted cohort shares, or None

    Examples
    --------
    Estimate the group-time ATTs and CATTs on a small simulated panel. The influence
    function standard errors of ``boot=False`` keep the example fast.

    .. ipython::
        :okwarning:

        In [1]: from moderndid import didml, gen_did_scalable
           ...:
           ...: df = gen_did_scalable(
           ...:     n=300, n_periods=4, n_cohorts=2, n_covariates=4, random_state=0
           ...: )["data"]
           ...: result = didml(
           ...:     df, yname="y", tname="time", idname="id", gname="group",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4", k_folds=5, boot=False,
           ...:     random_state=0,
           ...: )
           ...: print(result)

    Notes
    -----
    A cell with :math:`n_v` of the :math:`n` units stacks a pre-period row
    :math:`s = 0` and a post-period row :math:`s = 1` per unit. The influence function
    of its ATT averages each unit's two rows. For a unit :math:`i` in the cell, it is

    .. math::

        \psi_i = \frac{n\, w_i}{W_1}\, G_i
            \bigl(\hat\tau(X_{i1}) - \hat\tau_1\bigr)
            + \frac{n}{2 n_v} \sum_{s \in \{0, 1\}}
            \bigl(\tilde\gamma_{is}\,\hat\varepsilon_{is}
            - \overline{\tilde\gamma\,\hat\varepsilon}\bigr),

    and zero for the units outside it. Here :math:`G_i` marks cohort :math:`g` and
    :math:`w_i` is the sampling weight. The treated cell has weight sum :math:`W_1` and
    weighted mean CATT :math:`\hat\tau_1`. The residual
    :math:`\hat\varepsilon = Y - \hat y` comes from the orthogonal decomposition in
    :func:`lnw_did`. The weights :math:`\tilde\gamma` are the AMLE weights projected to
    balance the CATT design :math:`W_{is} = C_{is} (1, X_{is})` exactly,

    .. math::

        \tilde\gamma = \hat\gamma - W M^{-1} d, \qquad
        M = \frac{W^\top W}{2 n_v}, \qquad
        d = \frac{W^\top \hat\gamma}{2 n_v} - (1, \bar X_1),

    where :math:`\bar X_1` is the weighted covariate mean of the treated cell. The
    projection carries the estimation error of the CATT coefficients into
    :math:`\psi_i`. With ``boot=False``, the variance of the cell ATT is
    :math:`\sum_i \psi_i^2 / n^2`. The covariance of two cells follows from their
    influence functions in the same way.

    The parallel-trends pre-test stacks the :math:`q` pre-treatment cells in
    :math:`\hat\theta_{pre}` and forms

    .. math::

        W = \hat\theta_{pre}^\top \hat V_{pre}^{-1} \hat\theta_{pre}.

    With ``boot=False``, the covariance :math:`\hat V_{pre}` comes from the influence
    functions and the p-value from the chi-square with :math:`q` degrees of freedom.
    With the bootstrap, :math:`\hat V_{pre}` is the covariance of the :math:`B` draws in
    which every cell estimated. Since that covariance is itself noisy, the p-value uses
    Hotelling's distribution

    .. math::

        \frac{B - q}{q\,(B - 1)}\, W \sim F(q,\, B - q).

    No pre-test is reported when :math:`B \le q`.

    See Also
    --------
    aggte_didml : Aggregate ML group-time ATTs to a dynamic event-study summary.
    lnw_did : Per-cell doubly robust ML score.
    amle_weights : Augmented minimax-linear weighting solver.

    References
    ----------

    .. [1] Hatamyar, J., Kreif, N., Rocha, R., & Huber, M. (2023).
           "Machine learning for staggered difference-in-differences and
           dynamic treatment effect heterogeneity." arXiv:2310.11962.
           https://arxiv.org/abs/2310.11962

    .. [2] Nie, X., Lu, C., & Wager, S. (2024). "Nonparametric heterogeneous
           treatment effect estimation in repeated cross sectional designs."
           In E. Laber, B. Chakraborty, E. E. M. Moodie, T. Cai, & M. van
           der Laan (Eds.), Handbook of Statistical Methods for Precision
           Medicine (Ch. 9). Chapman and Hall/CRC.
           https://doi.org/10.1201/9781003216223-9

    .. [3] Hirshberg, D. A., & Wager, S. (2021). "Augmented minimax linear
           estimation." The Annals of Statistics, 49(6), 3206-3227.
           https://doi.org/10.1214/21-AOS2080

    .. [4] Callaway, B., & Sant'Anna, P. H. C. (2021).
           "Difference-in-differences with multiple time periods."
           Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001

    .. [5] Sant'Anna, P. H. C., & Zhao, J. (2020). "Doubly robust
           difference-in-differences estimators." Journal of Econometrics,
           219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if idname is None:
        raise ValueError("idname is required for the panel ML estimator.")
    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError(f"control_group={control_group!r} is not valid. Must be 'nevertreated' or 'notyettreated'.")
    if base_period not in ("varying", "universal"):
        raise ValueError(f"base_period={base_period!r} is not valid. Must be 'varying' or 'universal'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} is not valid. Must be a non-negative number.")
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs={n_jobs} is not valid. Must be a positive integer or -1 for all cores.")
    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")
    if boot and (isinstance(biters, bool) or not isinstance(biters, int | np.integer) or biters < 2):
        raise ValueError(f"biters={biters!r} is not valid. Must be an integer of at least 2 when boot=True.")
    check_columns(
        data,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla,
        weightsname=weightsname,
        clustervars=clustervars,
    )

    config = DIDMLConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        # Since preprocessing builds the outcome tensor only for a balanced panel and every cell reads it,
        # units missing from any period are dropped as att_gt does by default.
        allow_unbalanced_panel=False,
        weightsname=weightsname,
        alpha=alpha,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        anticipation=anticipation,
        control_group=ControlGroup(control_group),
        base_period=BasePeriod(base_period),
        nu_model=nu_model,
        sigma_model=sigma_model,
        delta_model=delta_model,
        k_folds=k_folds,
        lambda_choice=lambda_choice,
        t_func=t_func,
        use_gamma=use_gamma,
        zeta=zeta,
        compute_drdid_benchmark=compute_drdid_benchmark,
        random_state=random_state,
    )

    builder = PreprocessDataBuilder()
    dp = builder.with_data(data).with_config(config).validate().transform().build()

    compute_out = compute_didml(dp, n_jobs=n_jobs)
    cell_results = compute_out.cell_results

    if not cell_results:
        raise RuntimeError(
            "compute_didml produced no usable cells. Check that the data has at least one "
            "treated cohort and a valid control group."
        )

    groups = np.array([c.group for c in cell_results])
    times = np.array([c.year for c in cell_results])
    att_values = np.array([c.att for c in cell_results])

    n_units = dp.config.id_count
    influence_funcs = compute_out.influence_funcs

    att_draws = None
    share_draws = None
    critical_value = scipy.stats.norm.ppf(1 - alpha / 2)

    if boot:
        boot_out = bootstrap_didml(dp, cell_results, biters=biters, random_state=random_state, n_jobs=n_jobs)
        att_draws = boot_out.att_draws
        share_draws = boot_out.share_draws
        _warn_failed_draws(att_draws, groups, times)
        standard_errors = bootstrap_se(att_draws)
        vcov, n_complete_draws = _draws_covariance(att_draws)
    else:
        vcov = influence_funcs.T @ influence_funcs / n_units**2
        standard_errors = np.sqrt(np.diag(vcov))
        n_complete_draws = None

    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    if boot and cband:
        critical_value = simultaneous_critical_value(att_draws, att_values, standard_errors, alpha)

    if clustervars is not None and idname in clustervars:
        clustervars = [v for v in clustervars if v != idname]

    if clustervars:
        warnings.warn(
            "Cluster-robust standard errors are not yet implemented for didml. The reported standard errors "
            "treat units as independent and ignore the clusters.",
            UserWarning,
            stacklevel=2,
        )

    if not boot and not use_gamma:
        warnings.warn(
            "With use_gamma=False the influence-function standard errors are undefined and se_gt holds NaN. "
            "Pass boot=True for bootstrap standard errors.",
            UserWarning,
            stacklevel=2,
        )

    wald_statistic, wald_pvalue = _pretest_wald(att_values, vcov, groups, times, anticipation, n_complete_draws)

    drdid_benchmark = None
    drdid_benchmark_se = None

    if compute_drdid_benchmark and any(c.drdid_att is not None for c in cell_results):
        drdid_benchmark = np.array([np.nan if c.drdid_att is None else c.drdid_att for c in cell_results])
        drdid_benchmark_se = np.array([np.nan if c.drdid_se is None else c.drdid_se for c in cell_results])

    cohort_counts = _derive_cohort_counts(dp, gname)

    estimation_params = {
        "yname": yname,
        "control_group": control_group,
        "anticipation_periods": anticipation,
        "estimation_method": "ml_dr",
        "bootstrap": boot,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": True,
        "clustervars": clustervars,
        "cluster": getattr(dp, "cluster", None),
        "biters": biters,
        "random_state": random_state,
        "n_units": n_units,
        "n_obs": len(dp.data),
        "alpha": alpha,
        "nu_model": nu_model,
        "sigma_model": sigma_model,
        "delta_model": delta_model,
        "k_folds": k_folds,
        "t_func": t_func,
        "use_gamma": use_gamma,
        "zeta": zeta,
        "cohort_counts": cohort_counts,
    }

    unit_ids, unit_periods = _build_unit_period_index(dp)

    return didml_result(
        groups=groups,
        times=times,
        att_gt=att_values,
        se_gt=standard_errors,
        critical_value=critical_value,
        influence_func=influence_funcs,
        cates=compute_out.cates,
        scores=compute_out.scores,
        gammas=compute_out.gammas,
        unit_ids=unit_ids,
        unit_periods=unit_periods,
        drdid_benchmark=drdid_benchmark,
        drdid_benchmark_se=drdid_benchmark_se,
        n_units=n_units,
        wald_stat=wald_statistic,
        wald_pvalue=wald_pvalue,
        alpha=alpha,
        estimation_params=estimation_params,
        G=unit_cohorts(dp),
        weights_ind=np.asarray(dp.weights, dtype=float),
        att_gt_draws=att_draws,
        cohort_share_draws=share_draws,
    )


def _draws_covariance(draws):
    """Return the covariance over the draws where every cell estimated and the number of those draws."""
    complete = np.all(np.isfinite(draws), axis=1)
    n_complete = int(complete.sum())
    if n_complete < 2:
        return np.full((draws.shape[1], draws.shape[1]), np.nan), n_complete
    return np.atleast_2d(np.cov(draws[complete], rowvar=False)), n_complete


def _warn_failed_draws(att_draws, groups, times):
    """Warn about cells whose refit failed in some bootstrap draws."""
    failed = np.isnan(att_draws).sum(axis=0)
    for j in np.flatnonzero(failed):
        warnings.warn(
            f"The bootstrap refit of the cell for group {groups[j]:g}, period {times[j]:g} failed in "
            f"{failed[j]} of {att_draws.shape[0]} draws. Its standard error uses the other draws.",
            UserWarning,
            stacklevel=3,
        )


def _derive_cohort_counts(dp, gname):
    """Count unique treated units per cohort for event-study weighting."""
    df = dp.data
    treated = df.filter(df[gname] > 0)
    counts = treated.unique(subset=[dp.config.idname, gname]).group_by(gname).len()
    return {float(row[gname]): int(row["len"]) for row in counts.iter_rows(named=True)}


def _pretest_wald(att_values, vcov, groups, times, anticipation, n_draws=None):
    """Compute the parallel-trends Wald pre-test against Hotelling's distribution or the chi-square."""
    pre_indices = np.where(groups > times)[0]
    # Zero-variance cells, including universal-base reference cells, carry no
    # information and would make the variance singular.
    pre_indices = pre_indices[np.diag(vcov)[pre_indices] != 0]

    if len(pre_indices) == 0:
        msg = "No pre-treatment periods available for the parallel-trends Wald pre-test."
        if anticipation > 0:
            msg += f" Note: anticipation={anticipation} reduces available pre-treatment cells."
        warnings.warn(msg, UserWarning)
        return None, None

    n_pre = len(pre_indices)
    if n_draws is not None and n_draws <= n_pre:
        warnings.warn(
            f"Pre-test Wald not returned because {n_draws} complete bootstrap draws cannot estimate the covariance "
            f"of {n_pre} pre-treatment cells. Pass biters above {n_pre} or boot=False.",
            UserWarning,
            stacklevel=3,
        )
        return None, None

    pre_att = att_values[pre_indices]
    pre_var = vcov[np.ix_(pre_indices, pre_indices)]

    if np.any(np.isnan(pre_var)):
        warnings.warn("Pre-test Wald not returned due to NaN entries in pre-treatment variance.", UserWarning)
        return None, None

    if la.norm(pre_var) == 0 or np.linalg.matrix_rank(pre_var) < pre_var.shape[0]:
        warnings.warn("Pre-test Wald not returned due to a singular pre-treatment variance.", UserWarning)
        return None, None

    try:
        wald_stat = float(pre_att.T @ np.linalg.solve(pre_var, pre_att))
    except np.linalg.LinAlgError:
        warnings.warn("Pre-test Wald not returned due to a numerical solver failure.", UserWarning)
        return None, None

    if n_draws is None:
        return wald_stat, round(1 - scipy.stats.chi2.cdf(wald_stat, n_pre), 5)

    # Since a covariance from n_draws draws carries their sampling noise, the statistic is Hotelling's T-squared,
    # a scaled F, rather than a chi-square.
    scaled = wald_stat * (n_draws - n_pre) / (n_pre * (n_draws - 1))
    return wald_stat, round(float(scipy.stats.f.sf(scaled, n_pre, n_draws - n_pre)), 5)


def _build_unit_period_index(dp):
    """Return the unit id and period index of each sparse-matrix row."""
    n_units = dp.config.id_count
    idname = dp.config.idname
    tname = dp.config.tname
    gname = dp.config.gname

    # Since the per-cell tensors slice the data in (tname, gname, idname) order, the first
    # time-period block of that sort gives the unit order of every sparse-matrix row.
    first_period_block = dp.data.sort([tname, gname, idname]).slice(0, n_units)
    unit_ids = first_period_block[idname].to_numpy()

    if unit_ids.shape[0] != n_units:
        unit_ids = np.arange(n_units)

    unit_periods = np.zeros(n_units, dtype=int)
    return unit_ids, unit_periods
