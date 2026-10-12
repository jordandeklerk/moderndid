"""Multi-period difference-in-differences group-time average treatment effects estimation."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import scipy.linalg as la
import scipy.stats

from moderndid.core.format import warn_few_clusters
from moderndid.core.preprocess import (
    WEIGHTS_COLUMN,
    BasePeriod,
    ControlGroup,
    DIDConfig,
    EstimationMethod,
    PreprocessDataBuilder,
)
from moderndid.core.preprocess.validators import check_columns
from moderndid.cupy.backend import to_numpy, use_backend

from .compute_att_gt import _is_unbalanced_panel, compute_att_gt
from .container import mp
from .mboot import _cluster_sums, _mboot


def att_gt(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    weightsname=None,
    alpha=0.05,
    cband=True,
    boot=False,
    biters=1000,
    clustervars=None,
    est_method="dr",
    panel=True,
    allow_unbalanced_panel=False,
    control_group="nevertreated",
    anticipation=0,
    base_period="varying",
    random_state=None,
    n_jobs=1,
    backend=None,
):
    r"""Compute group-time average treatment effects.

    Implements difference-in-differences estimation for staggered adoption designs
    where treatment timing varies across units, following [1]_. This approach
    addresses the challenges of standard two-way fixed-effects regressions by
    providing flexible estimators that allow for treatment effect heterogeneity
    across groups and over time.

    Let :math:`G_i` denote the time period when unit :math:`i` is first treated,
    with :math:`G_i = \infty` for never-treated units, and let :math:`C_i` be an
    indicator for never-treated status. The fundamental parameter
    of interest is the group-time average treatment effect, :math:`ATT(g,t)`,
    which measures the average effect for units first treated in period :math:`g`
    as of time :math:`t`

    .. math::

        ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g].

    Identification relies on a conditional parallel trends assumption. When using
    never-treated units as the comparison group, the assumption requires that
    trends in untreated potential outcomes are the same for the treatment group
    and never-treated units conditional on covariates :math:`X`

    .. math::

        \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, G = g]
        = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, C = 1],

    where :math:`C = 1` indicates never-treated units. The doubly robust estimand
    combines inverse probability weighting and outcome regression, providing
    consistency if either the propensity score or outcome model is correctly
    specified

    .. math::

        ATT_{dr}(g,t) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]}
        - \frac{\frac{p_g(X) C}{1 - p_g(X)}}
        {\mathbb{E}\left[\frac{p_g(X) C}{1 - p_g(X)}\right]}\right)
        \left(\Delta Y_t - m_{g,t}(X)\right)\right],

    where :math:`p_g(X)` is the propensity score, :math:`\Delta Y_t` is the
    change in outcomes, and :math:`m_{g,t}(X)` is the expected outcome change
    for the comparison group.

    Sampling weights may change over time within a unit of a balanced panel.
    Each 2x2 comparison then uses the weights from the earlier of its two
    periods. A warning names this rule. :func:`~moderndid.aggte` weights each
    unit by its weight in the first period.

    A group-time comparison can fail to estimate, such as when its fitted
    propensity score reaches 0.999 or its estimator returns no estimate. A
    warning then names the cell. Under the varying base the cell drops out of
    the result. Under the universal base it stays in the result as NaN.

    A cluster variable besides ``idname`` in ``clustervars`` makes the
    inference cluster-robust. The analytic covariance of the estimates then
    sums the influence function within each cluster before the outer
    product. The standard errors without the bootstrap use that covariance.
    The multiplier bootstrap draws one multiplier for each cluster.

    Since a Wald test on a cluster-robust covariance rejects too often at
    realistic cluster counts, a fit with such clusters reports no pre-test
    of parallel trends. The bootstrap's uniform bands over the
    pre-treatment cells take its place.

    Since cluster-robust inference from very few clusters can give confidence
    bands that are too narrow, a warning names each treated cohort whose
    units fall in fewer than 5 clusters. It also appears when all units fall
    in fewer than 30 clusters.

    See the :ref:`staggered DiD example <example_staggered_did>` for a full analysis of
    the minimum wage data with ``att_gt``.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        The name of the outcome variable.
    tname : str
        The name of the column containing the time periods.
    idname : str, optional
        The individual (cross-sectional unit) id name. Required for panel data.
    gname : str
        The name of the variable that contains the first period when a particular
        observation is treated. This should be a positive number for all observations
        in treated groups. It defines which "group" a unit belongs to. It should be 0
        for units in the untreated group.
    xformla : str, optional
        A formula for the covariates to include in the model. It should be of the
        form "~ X1 + X2". Default is None which is equivalent to xformla="~1".
    weightsname : str, optional
        The name of the column containing the sampling weights. If not set, all
        observations have same weight.
    alpha : float, default=0.05
        The significance level.
    cband : bool, default=True
        Whether or not to compute a uniform confidence band that covers all of the
        group-time average treatment effects with fixed probability 1 - alpha.
    boot : bool, default=False
        Whether to compute standard errors with the multiplier bootstrap
        instead of analytically.
    biters : int, default=1000
        The number of bootstrap iterations to use. Only applicable if boot=True.
    clustervars : list[str], optional
        Variables to cluster the standard errors on. Besides idname, the list
        may hold one variable that stays fixed within each unit of a panel.
    est_method : {"dr", "ipw", "reg"} or callable, default="dr"
        The method to compute group-time average treatment effects. The default is
        "dr" which uses the doubly robust approach. Other built-in methods include
        "ipw" for inverse probability weighting and "reg" for first step regression
        estimators. The user can also pass their own function for estimating group
        time average treatment effects.
    panel : bool, default=True
        Whether or not the data is a panel dataset. The panel dataset should be
        provided in long format. With panel=False every row is its own observation,
        even when idname is given.
    allow_unbalanced_panel : bool, default=False
        Whether to keep units observed in only some periods. If False, att_gt drops
        every unit missing from any period. It applies only when panel=True.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use the control group. The default is "nevertreated" which
        sets the control group to be the group of units that never participate in
        the treatment.
    anticipation : int, default=0
        The number of time periods before participating in the treatment where units
        can anticipate participating in the treatment and therefore it can affect
        their untreated potential outcomes.
    base_period : {"varying", "universal"}, default="varying"
        Whether to use a "varying" base period or a "universal" base period.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.
    n_jobs : int, default=1
        Number of parallel jobs for group-time estimation. 1 = sequential
        (default), -1 = all cores, >1 = that many workers.
    backend : {"numpy", "cupy"} or None, default=None
        Array backend to use for this call only. When set, the backend is
        activated for the duration of this call and reverted automatically
        when the call returns. ``None`` (the default) uses whatever backend
        is currently active (see :func:`~moderndid.set_backend`).

    Returns
    -------
    MPResult
        Object containing group-time average treatment effect results:

        - **groups**: Array indicating which group (period first treated) each ATT is for
        - **times**: Array indicating which time period each ATT is for
        - **att_gt**: Array of group-time average treatment effects
        - **se_gt**: Standard errors for each ATT(g,t)
        - **vcov_analytical**: Analytical variance-covariance matrix, cluster-robust when the fit clusters
        - **critical_value**: Critical value for confidence intervals (simultaneous if bootstrap with cband=True)
        - **influence_func**: Influence function matrix for each ATT(g,t)
        - **n_units**: Number of unique cross-sectional units
        - **wald_stat**: Wald statistic for pre-testing parallel trends, or None for a fit with clusters
        - **wald_pvalue**: P-value for the parallel trends pre-test, or None for a fit with clusters
        - **alpha**: Significance level used
        - **estimation_params**: Dictionary of the estimation settings, such as control_group and
          anticipation_periods
        - **G**: Unit-level group assignments
        - **weights_ind**: Unit-level sampling weights (if provided)

    See Also
    --------
    aggte : Aggregate group-time average treatment effects.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences
           with multiple time periods." Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    if backend is not None:
        with use_backend(backend):
            return att_gt(
                data=data,
                yname=yname,
                tname=tname,
                idname=idname,
                gname=gname,
                xformla=xformla,
                weightsname=weightsname,
                alpha=alpha,
                cband=cband,
                boot=boot,
                biters=biters,
                clustervars=clustervars,
                est_method=est_method,
                panel=panel,
                allow_unbalanced_panel=allow_unbalanced_panel,
                control_group=control_group,
                anticipation=anticipation,
                base_period=base_period,
                random_state=random_state,
                n_jobs=n_jobs,
                backend=None,
            )

    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True.")

    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError(f"control_group={control_group!r} is not valid. Must be 'nevertreated' or 'notyettreated'.")
    if not callable(est_method) and est_method not in ("dr", "ipw", "reg"):
        raise ValueError(f"est_method={est_method!r} is not valid. Must be 'dr', 'ipw', or 'reg'.")
    if base_period not in ("varying", "universal"):
        raise ValueError(f"base_period={base_period!r} is not valid. Must be 'varying' or 'universal'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} is not valid. Must be a non-negative number.")
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs={n_jobs} is not valid. Must be a positive integer or -1 for all cores.")
    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")
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

    if anticipation > 0:
        warnings.warn(
            f"anticipation = {anticipation}. Never-treated units (with group status 0 or inf) "
            "are assumed to never anticipate treatment. Anticipation only applies to eventually-treated units.",
            UserWarning,
        )

    control_group_enum = ControlGroup(control_group)
    est_method_enum = EstimationMethod(est_method) if isinstance(est_method, str) else est_method
    base_period_enum = BasePeriod(base_period)

    config = DIDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group_enum,
        anticipation=anticipation,
        weightsname=weightsname,
        alpha=alpha,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=est_method_enum,
        base_period=base_period_enum,
    )

    builder = PreprocessDataBuilder()
    dp = builder.with_data(data).with_config(config).validate().transform().build()
    results = compute_att_gt(dp, n_jobs=n_jobs)

    att_gt_list = results.attgt_list
    influence_functions = results.influence_functions

    groups = np.array([att.group for att in att_gt_list])
    times = np.array([att.year for att in att_gt_list])
    att_values = np.array([float(att.att) for att in att_gt_list])

    if hasattr(influence_functions, "toarray"):
        influence_functions_dense = to_numpy(influence_functions.toarray())
    else:
        influence_functions_dense = to_numpy(np.array(influence_functions))

    n_units = dp.config.id_count

    if clustervars is not None and idname in clustervars:
        clustervars = [v for v in clustervars if v != idname]

    # Since clustering on idname alone is unit-level inference, the fit records a cluster vector only for another
    # variable. The standard errors, the Wald pre-test, aggte, and honest_did cluster on that vector.
    cluster = dp.time_invariant_data[clustervars[0]].to_numpy() if clustervars else None

    if cluster is not None:
        unit_cohorts = dp.time_invariant_data[gname].to_numpy()
        warn_few_clusters(cluster, {g: cluster[unit_cohorts == g] for g in np.unique(groups)})

    # Since clusters of one unit each are the units themselves, they leave the covariance at the unit level.
    clustered = cluster is not None and len(np.unique(cluster)) < len(cluster)
    if clustered:
        # Since units in one cluster share shocks, their influence functions are summed before the outer product.
        cluster_sums = _cluster_sums(influence_functions_dense, cluster)
        variance_matrix = cluster_sums.T @ cluster_sums / n_units
    else:
        variance_matrix = influence_functions_dense.T @ influence_functions_dense / n_units

    standard_errors = np.sqrt(np.diag(variance_matrix) / n_units)
    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    zero_na_sd_indices = np.unique(np.where(np.isnan(standard_errors))[0])

    if boot:
        # Since a cell with a negligible bootstrap scale would swamp the band, the group-time critical value leaves
        # such cells out and keeps every draw.
        bootstrap_results = _mboot(
            inf_func=influence_functions_dense,
            n_units=n_units,
            biters=biters,
            alpha=alpha,
            cluster=cluster,
            random_state=random_state,
            skip_small_scales=True,
        )

        if len(zero_na_sd_indices) > 0:
            standard_errors[~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)] = bootstrap_results["se"][
                ~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)
            ]
        else:
            standard_errors = bootstrap_results["se"]

    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # Since a Wald test on a cluster-robust covariance rejects a true null too often at realistic cluster counts,
    # a clustered fit reports no pre-test. Its report points to the bootstrap's uniform bands instead.
    if clustered:
        wald_statistic = None
        wald_pvalue = None
    else:
        pre_treatment_indices = np.where(groups > times)[0]

        if len(zero_na_sd_indices) > 0:
            pre_treatment_indices = pre_treatment_indices[~np.isin(pre_treatment_indices, zero_na_sd_indices)]

        pre_treatment_att = att_values[pre_treatment_indices]
        pre_treatment_variance = variance_matrix[np.ix_(pre_treatment_indices, pre_treatment_indices)]

        if len(pre_treatment_indices) == 0:
            msg = (
                "No pre-treatment periods available for the Wald pre-test of parallel trends. "
                "This can happen when all groups are first treated early in the panel, "
                "such as in the second time period, so that no pre-treatment ATT(g,t) estimates exist."
            )
            if anticipation > 0:
                msg += f" With anticipation={anticipation}, fewer pre-treatment periods remain."
            warnings.warn(msg, UserWarning)
            wald_statistic = None
            wald_pvalue = None
        elif np.any(np.isnan(pre_treatment_variance)):
            warnings.warn(
                "Not returning pre-test Wald statistic due to NaN pre-treatment values",
                UserWarning,
            )
            wald_statistic = None
            wald_pvalue = None
        elif (
            la.norm(pre_treatment_variance) == 0
            or np.linalg.matrix_rank(pre_treatment_variance) < pre_treatment_variance.shape[0]
        ):
            warnings.warn(
                "Not returning pre-test Wald statistic due to singular covariance matrix",
                UserWarning,
            )
            wald_statistic = None
            wald_pvalue = None
        else:
            try:
                wald_statistic = (
                    n_units * pre_treatment_att.T @ np.linalg.solve(pre_treatment_variance, pre_treatment_att)
                )
                q = len(pre_treatment_indices)
                wald_pvalue = round(1 - scipy.stats.chi2.cdf(wald_statistic, q), 5)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Not returning pre-test Wald statistic due to numerical issues",
                    UserWarning,
                )
                wald_statistic = None
                wald_pvalue = None

    critical_value = scipy.stats.norm.ppf(1 - alpha / 2)

    if boot and cband:
        critical_value = bootstrap_results["crit_val"]
        if not np.isnan(critical_value) and critical_value >= 7:
            warnings.warn(
                "Simultaneous critical value is arguably 'too large' to be reliable. "
                "This usually happens when number of observations per group is small "
                "and/or there is not much variation in outcomes.",
                UserWarning,
            )

    estimation_params = {
        "yname": yname,
        "idname": idname,
        "control_group": control_group,
        "anticipation_periods": anticipation,
        "estimation_method": est_method if isinstance(est_method, str) else "custom",
        "bootstrap": boot,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": panel,
        "clustervars": clustervars,
        "cluster": cluster,
        "biters": biters,
        "random_state": random_state,
        "n_units": n_units,
        "n_obs": len(dp.data),
    }

    group_assignments = None
    sampling_weights = None

    if hasattr(dp, "time_invariant_data"):
        if gname in dp.time_invariant_data.columns:
            group_assignments = dp.time_invariant_data[gname]
        # Preprocessing keeps the normalized weights in one internal column whatever the user's column is called.
        if weightsname is not None:
            sampling_weights = dp.time_invariant_data[WEIGHTS_COLUMN]
            if _is_unbalanced_panel(dp):
                # Since a unit of an unbalanced panel can carry a different weight in each of its rows, aggte
                # weighs the unit by the mean of those weights.
                unit_ids = dp.time_invariant_data[dp.config.idname]
                unit_means = dp.data.group_by(dp.config.idname).agg(pl.col(WEIGHTS_COLUMN).mean())
                sampling_weights = unit_ids.replace_strict(
                    unit_means[dp.config.idname], unit_means[WEIGHTS_COLUMN]
                ).alias(WEIGHTS_COLUMN)

    return mp(
        groups=groups,
        times=times,
        att_gt=att_values,
        vcov_analytical=variance_matrix,
        se_gt=standard_errors,
        critical_value=critical_value,
        influence_func=influence_functions_dense,
        n_units=n_units,
        wald_stat=wald_statistic,
        wald_pvalue=wald_pvalue,
        alpha=alpha,
        estimation_params=estimation_params,
        G=group_assignments,
        weights_ind=sampling_weights,
    )
