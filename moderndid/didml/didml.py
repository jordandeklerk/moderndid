"""ML-based doubly-robust DiD estimator."""

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

from .compute import compute_didml
from .container import didml_result


def didml(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    weightsname=None,
    alp=0.05,
    cband=False,
    biters=1000,
    clustervars=None,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    nu_model="rlearner",
    sigma_model="rlearner",
    delta_model="glm",
    k_folds=10,
    tune_penalty=False,
    lambda_choice="lambda.min",
    t_func=False,
    use_gamma=True,
    zeta=0.5,
    compute_drdid_benchmark=True,
    random_state=None,
    n_jobs=1,
):
    r"""Compute group-time ATTs and CATTs with cross-fitted ML nuisances.

    Implements the staggered-adoption ML doubly-robust DiD estimator of
    [1]_. The target parameter for cohort :math:`g` and time period
    :math:`t` is the group-time average treatment effect on the treated,

    .. math::

        ATT(g, t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g],

    where :math:`Y_t(g)` is the potential outcome at time :math:`t` for a
    unit first treated at :math:`g` and :math:`Y_t(0)` is the
    never-treated potential outcome. The conditional version is the CATT

    .. math::

        \tau_{g,t}(x) = \mathbb{E}[Y_t(g) - Y_t(0) \mid X = x, G = g],

    which the function recovers as a per-unit prediction.

    For each cell, the function stacks the relevant pre and post periods
    over the treated cohort and the not-yet-treated controls, runs
    :func:`lnw_did` to compute the cell-level Nie-Lu-Wager [2]_ score
    with cross-fitted nuisances, and combines the per-unit conditional
    treatment effects with augmented minimax-linear weights from [3]_.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars,
        pandas, pyarrow Table, and cudf DataFrames.
    yname : str
        Name of the outcome variable.
    tname : str
        Name of the column containing the time periods.
    idname : str, optional
        The individual unit identifier. Required for panel data.
    gname : str
        Name of the variable holding the first period a unit is treated.
        Should be a positive integer for treated units and 0 for the
        never-treated group.
    xformla : str, optional
        A formula for the covariates, e.g. ``"~ x1 + x2"``. Defaults to
        ``"~1"`` (no covariates) when ``None``.
    weightsname : str, optional
        Name of the column containing per-unit sampling weights.
    alp : float, default=0.05
        Significance level for confidence intervals.
    cband : bool, default=False
        Whether to compute uniform confidence bands. Currently uses the
        pointwise normal critical value; bootstrap-based simultaneous
        bands are forthcoming.
    biters : int, default=1000
        Number of bootstrap iterations (reserved for forthcoming bootstrap
        support; not currently used).
    clustervars : list[str], optional
        Variables to cluster on. Currently passed through to the
        configuration but ignored at the variance-estimation step;
        clustering support is forthcoming.
    control_group : {"nevertreated", "notyettreated"}, default="notyettreated"
        Comparison group for the doubly-robust score.
    anticipation : int, default=0
        Number of pre-treatment periods where treated units may anticipate
        treatment.
    base_period : {"varying", "universal"}, default="varying"
        Choice of base period for each comparison. With ``"varying"``,
        each pre-treatment period is compared with the immediately
        preceding period. With ``"universal"``, every comparison for
        cohort ``g`` is anchored at the last period before ``g`` minus
        the anticipation periods, the reference cell at that base period
        is reported as zero, and earlier calendar periods are included
        as placebo comparisons.
    nu_model : {"rlearner", "cf"}, default="rlearner"
        Backend for the conditional time trend :math:`\nu(x)` (the effect
        of the post-period indicator on the outcome marginal over cohort)
        and the propensity :math:`t(x)`.
    sigma_model : {"rlearner", "cf"}, default="rlearner"
        Backend for the conditional cohort contrast :math:`\zeta(x)` (the
        effect of cohort membership on the outcome marginal over time) and
        the propensity :math:`g(x)`.
    delta_model : {"glm", "stack"}, default="glm"
        Backend for the conditional covariance term :math:`\Delta(x)`.
        ``"glm"`` fits a cross-validated lasso with optional penalty-factor
        tuning; ``"stack"`` uses a stacking ensemble.
    k_folds : int, default=10
        Number of folds for cross-fitting the nuisance models.
    tune_penalty : bool, default=False
        Whether to grid-search penalty factors in the lasso-based
        backends.
    lambda_choice : {"lambda.min", "lambda.1se"}, default="lambda.min"
        Cross-validation rule for selecting the lasso penalty in the
        non-constant tau fit.
    t_func : bool, default=False
        Whether to use the estimated post-period propensity. The default
        replaces the estimate with the constant 0.5, which is exact
        because each stacked cell is balanced by construction. See
        :func:`lnw_did`.
    use_gamma : bool, default=True
        Whether to compute and apply AMLE weights. When ``False``, the
        cell ATT reduces to the simple mean of cross-fitted CATTs.
    zeta : float, default=0.5
        AMLE objective mixing weight in :math:`(0, 1)`.
    compute_drdid_benchmark : bool, default=True
        Whether to compute the standard DR-DiD benchmark (Sant'Anna-Zhao
        2020) for each cell as a comparison reference.
    random_state : int, optional
        Seed for cross-fitting fold splits and the inner CV.
    n_jobs : int, default=1
        Parallel workers for the group-time cell loop. ``-1`` uses all
        cores; ``>1`` uses that many workers.

    Returns
    -------
    DIDMLResult
        Object containing group-time ML ATT and CATT results:

        - **groups**: Array indicating which cohort (period first treated) each ATT is for
        - **times**: Array indicating which time period each ATT is for
        - **att_gt**: Array of group-time ML ATT estimates
        - **se_gt**: Per-cell analytical doubly-robust standard errors of each group-time ATT
        - **critical_value**: Critical value for confidence intervals
        - **influence_func**: Per-unit DR-score matrix of shape ``(n_units, n_cells)``
        - **cates**: Sparse matrix of per-unit CATT predictions
        - **scores**: Sparse matrix of per-unit DR score contributions
        - **gammas**: Sparse matrix of per-unit AMLE weights
        - **unit_ids**: Unit identifiers indexing the sparse-matrix rows
        - **unit_periods**: Time periods indexing the sparse-matrix rows
        - **drdid_benchmark**: DR-DiD benchmark ATTs per cell (when ``compute_drdid_benchmark=True``)
        - **drdid_benchmark_se**: Standard errors for the DR-DiD benchmark
        - **n_units**: Number of unique cross-sectional units
        - **wald_stat**: Wald statistic for pre-testing parallel trends
        - **wald_pvalue**: P-value for the parallel-trends pre-test
        - **aggregate_effects**: Aggregate treatment effects object populated by ``aggte_didml``
        - **alpha**: Significance level used
        - **estimation_params**: Dictionary with estimation details (nu_model, sigma_model, delta_model, k_folds, etc.)

    Examples
    --------
    Generate a small staggered-adoption panel with four covariates, two
    treated cohorts, and four periods, then estimate group-time ATTs and
    CATTs with R-learner nuisance backends and the lasso delta model:

    .. ipython::
        :okwarning:

        In [1]: from moderndid import didml, gen_did_scalable
           ...:
           ...: df = gen_did_scalable(
           ...:     n=300, n_periods=4, n_cohorts=2,
           ...:     n_covariates=4, random_state=0,
           ...: )["data"]
           ...: print(df.head())

    The output is a ``DIDMLResult`` carrying group-time ATT estimates,
    per-unit CATT predictions, doubly-robust scores, AMLE weights, and the
    optional Callaway-Sant'Anna DRDID benchmark for direct comparison:

    .. ipython::
        :okwarning:

        In [2]: result = didml(
           ...:     df,
           ...:     yname="y",
           ...:     tname="time",
           ...:     gname="group",
           ...:     idname="id",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4",
           ...:     k_folds=5,
           ...:     tune_penalty=False,
           ...:     random_state=0,
           ...: )
           ...: print(result)

    See Also
    --------
    aggte_didml : Aggregate ML group-time ATTs to a dynamic event-study summary.
    lnw_did : Per-cell doubly-robust ML score.
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
    """
    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if idname is None:
        raise ValueError("idname is required for the panel ML estimator.")
    if not 0 < alp < 1:
        raise ValueError(f"alp={alp} must lie in (0, 1).")
    if isinstance(control_group, str) and control_group not in ("nevertreated", "notyettreated"):
        raise ValueError(f"control_group='{control_group}' is not valid. Must be 'nevertreated' or 'notyettreated'.")
    if isinstance(base_period, str) and base_period not in ("varying", "universal"):
        raise ValueError(f"base_period='{base_period}' is not valid. Must be 'varying' or 'universal'.")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} must be a non-negative number.")
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs={n_jobs} must be a positive integer or -1.")
    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")

    config = DIDMLConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        weightsname=weightsname,
        alp=alp,
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
        tune_penalty=tune_penalty,
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
    se_values = np.array([c.se for c in cell_results], dtype=float)

    n_units = dp.config.id_count
    influence_funcs = compute_out.scores.toarray().astype(float)

    variance_matrix = influence_funcs.T @ influence_funcs / n_units

    if clustervars is not None and idname in clustervars:
        clustervars = [v for v in clustervars if v != idname]

    if clustervars:
        warnings.warn(
            "Cluster-robust standard errors are not yet implemented for didml; reporting "
            "analytical influence-function-based SEs that ignore clustering.",
            UserWarning,
        )

    wald_statistic, wald_pvalue = _pretest_wald(att_values, variance_matrix, groups, times, n_units, anticipation)
    critical_value = scipy.stats.norm.ppf(1 - alp / 2)

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
        "bootstrap": False,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": True,
        "clustervars": clustervars,
        "cluster": getattr(dp, "cluster", None),
        "biters": biters,
        "random_state": random_state,
        "n_units": n_units,
        "n_obs": len(dp.data),
        "alpha": alp,
        "nu_model": nu_model,
        "sigma_model": sigma_model,
        "delta_model": delta_model,
        "k_folds": k_folds,
        "tune_penalty": tune_penalty,
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
        se_gt=se_values,
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
        alpha=alp,
        estimation_params=estimation_params,
    )


def _derive_cohort_counts(dp, gname):
    """Count unique treated units per cohort for event-study weighting."""
    df = dp.data
    treated = df.filter(df[gname] > 0)
    counts = treated.unique(subset=[dp.config.idname, gname]).group_by(gname).len()
    return {float(row[gname]): int(row["len"]) for row in counts.iter_rows(named=True)}


def _pretest_wald(att_values, variance_matrix, groups, times, n_units, anticipation):
    """Compute the parallel-trends Wald statistic from pre-treatment cells."""
    pre_indices = np.where(groups > times)[0]
    # Zero-variance cells, including universal-base reference cells, carry no
    # information and would make the variance singular.
    pre_indices = pre_indices[np.diag(variance_matrix)[pre_indices] != 0]

    if len(pre_indices) == 0:
        msg = "No pre-treatment periods available for the parallel-trends Wald pre-test."
        if anticipation > 0:
            msg += f" Note: anticipation={anticipation} reduces available pre-treatment cells."
        warnings.warn(msg, UserWarning)
        return None, None

    pre_att = att_values[pre_indices]
    pre_var = variance_matrix[np.ix_(pre_indices, pre_indices)]

    if np.any(np.isnan(pre_var)):
        warnings.warn("Pre-test Wald not returned due to NaN entries in pre-treatment variance.", UserWarning)
        return None, None

    if la.norm(pre_var) == 0 or np.linalg.matrix_rank(pre_var) < pre_var.shape[0]:
        warnings.warn("Pre-test Wald not returned due to a singular pre-treatment variance.", UserWarning)
        return None, None

    try:
        wald_stat = float(n_units * pre_att.T @ np.linalg.solve(pre_var, pre_att))
        wald_pvalue = round(1 - scipy.stats.chi2.cdf(wald_stat, len(pre_indices)), 5)
    except np.linalg.LinAlgError:
        warnings.warn("Pre-test Wald not returned due to a numerical solver failure.", UserWarning)
        return None, None

    return wald_stat, wald_pvalue


def _build_unit_period_index(dp):
    """Return arrays mapping each sparse-matrix row to (unit_id, period_index)."""
    n_units = dp.config.id_count
    idname = dp.config.idname
    tname = dp.config.tname
    gname = dp.config.gname

    # The per-cell tensors slice the data in (tname, gname, idname) order, so
    # the first time-period block fixes the unit order of every sparse-matrix
    # row. Reproduce that block order here so each row carries the right id.
    first_period_block = dp.data.sort([tname, gname, idname]).slice(0, n_units)
    unit_ids = first_period_block[idname].to_numpy()

    if unit_ids.shape[0] != n_units:
        unit_ids = np.arange(n_units)

    unit_periods = np.zeros(n_units, dtype=int)
    return unit_ids, unit_periods
