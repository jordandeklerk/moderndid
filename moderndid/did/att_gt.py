"""Multi-period difference-in-differences group-time average treatment effects estimation."""

from __future__ import annotations

import sys
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

    Estimates the average treatment effect for the cohort of units first treated in
    period :math:`g` at each period :math:`t` of a staggered adoption design [1]_,

    .. math::

        ATT(g,t) = \mathbb{E}[Y_t(g) - Y_t(0) \mid G = g].

    Each effect compares the cohort with never-treated or not-yet-treated units under
    conditional parallel trends. Using only untreated units as controls avoids the bias
    that a two-way fixed effects regression suffers when effects vary across cohorts or
    over time. The standard errors come from the influence functions, analytically or
    through the multiplier bootstrap. :func:`~moderndid.aggte` averages the effects into
    an event study or an overall effect.

    See the :ref:`staggered DiD example <example_staggered_did>` for a full analysis of
    the minimum wage data with ``att_gt``. The :ref:`Spark guide <spark>` shows how to
    fit a balanced panel held in a Spark DataFrame.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow PyCapsule
        Interface (``__arrow_c_stream__``), including polars, pandas, pyarrow Table, and
        cudf DataFrames. Also accepts a Spark DataFrame from a classic or Spark Connect
        session.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column.
    idname : str, optional
        The name of the unit identifier column. Required for panel data.
    gname : str
        The name of the column holding the period in which each unit is first treated.
        It is 0 for never-treated units.
    xformla : str, optional
        A formula for the covariates, such as ``"~ X1 + X2"``.
    weightsname : str, optional
        The name of the sampling weights column.
    alpha : float, default=0.05
        The significance level.
    cband : bool, default=True
        Whether to report a uniform confidence band that covers all the group-time
        effects with probability 1 - alpha. Applies only with ``boot=True``.
    boot : bool, default=False
        Whether to compute the standard errors with the multiplier bootstrap.
    biters : int, default=1000
        The number of bootstrap draws. Only applicable if ``boot=True``.
    clustervars : list[str], optional
        Variables to cluster the standard errors on. Besides ``idname``, the list may
        hold one variable that stays fixed within each unit.
    est_method : {"dr", "ipw", "reg"} or callable, default="dr"
        The estimator of each effect, doubly robust, inverse probability weighting, or
        outcome regression. A callable supplies a custom estimator.
    panel : bool, default=True
        Whether the data is a panel. With ``panel=False`` every row is its own
        observation, even when ``idname`` is given.
    allow_unbalanced_panel : bool, default=False
        Whether to keep the units of a panel observed in only some periods. If False,
        att_gt drops every unit missing from any period.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Whether the comparison group holds only the never-treated units or also the
        units not yet treated.
    anticipation : int, default=0
        The number of periods before treatment in which units may already respond to it.
    base_period : {"varying", "universal"}, default="varying"
        ``"varying"`` compares each pre-treatment period with the one before it.
        ``"universal"`` compares every period with one base period before treatment.
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible results
        across calls.
    n_jobs : int, default=1
        The number of parallel jobs for the group-time cells. Pass -1 to use all cores.
    backend : {"numpy", "cupy"}, optional
        The array backend for this call only. None uses the backend that
        :func:`~moderndid.set_backend` made active.

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
        - **influence_moments**: Sums over units as an :class:`~moderndid.InfluenceMoments` in place of influence_func,
          G, and weights_ind for a fit on a Spark DataFrame, or None

    See Also
    --------
    aggte : Aggregate group-time average treatment effects.

    Notes
    -----
    Let :math:`G` be the period in which a unit is first treated, :math:`G_g` the
    indicator of :math:`G = g`, and :math:`C = \mathbf{1}\{G = \infty\}` the indicator
    of never-treated units. With never-treated units as the comparison group,
    identification rests on conditional parallel trends given the covariates :math:`X`,

    .. math::

        \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, G = g]
        = \mathbb{E}[Y_t(0) - Y_{t-1}(0) \mid X, C = 1].

    With the propensity score :math:`p_g(X)` and the regression :math:`m_{g,t}(X)` of
    the comparison group's outcome change :math:`\Delta Y_t`, the doubly robust
    estimand is

    .. math::

        ATT_{dr}(g,t) = \mathbb{E}\left[\left(\frac{G_g}{\mathbb{E}[G_g]}
        - \frac{\frac{p_g(X) C}{1 - p_g(X)}}
        {\mathbb{E}\left[\frac{p_g(X) C}{1 - p_g(X)}\right]}\right)
        \left(\Delta Y_t - m_{g,t}(X)\right)\right].

    It recovers the effect if either model is correctly specified. The
    :ref:`staggered DiD background <background-did>` develops the identifying
    assumptions, the estimators, and the inference.

    A cell that fails to estimate, such as one whose propensity score reaches 0.999,
    triggers a warning that names it. The cell then drops out under the varying base
    period and stays as NaN under the universal one.

    When sampling weights change over time within a unit of a balanced panel, a warning
    notes that each comparison uses the weights of its earlier period.
    :func:`~moderndid.aggte` weights each unit by its weight in the first period.

    With a cluster variable besides ``idname``, a warning appears when a treated cohort's
    units fall in fewer than 5 clusters or all units fall in fewer than 30.

    References
    ----------

    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences
           with multiple time periods." Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    # Since att_gt never imports pyspark, a Spark DataFrame can only arrive after the caller has imported it. Because a
    # Spark Connect DataFrame of pyspark 3.5 shares no class with a classic one, its class is checked as well.
    spark_frames = tuple(
        frame
        for frame in (
            getattr(sys.modules.get("pyspark.sql"), "DataFrame", None),
            getattr(sys.modules.get("pyspark.sql.connect.dataframe"), "DataFrame", None),
        )
        if frame is not None
    )
    spark_data = isinstance(data, spark_frames)
    if backend is not None:
        if spark_data:
            raise ValueError(
                f"backend={backend!r} is not supported for a Spark DataFrame. Pass backend=None to fit on the Spark "
                "executors."
            )
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

    if spark_data:
        # Since pyspark is optional, the Spark path loads only for a Spark DataFrame. It comes before check_columns
        # because converting the data for that check would collect its rows to the driver.
        from moderndid.spark.att_gt import spark_att_gt

        return spark_att_gt(
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
        )

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

    bootstrap_results = None
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

    return _finish_att_gt(
        groups=groups,
        times=times,
        att_values=att_values,
        variance_matrix=variance_matrix,
        clustered=clustered,
        bootstrap_results=bootstrap_results,
        n_units=n_units,
        alpha=alpha,
        cband=cband,
        anticipation=anticipation,
        estimation_params=estimation_params,
        influence_func=influence_functions_dense,
        G=group_assignments,
        weights_ind=sampling_weights,
    )


def _finish_att_gt(
    groups,
    times,
    att_values,
    variance_matrix,
    clustered,
    bootstrap_results,
    n_units,
    alpha,
    cband,
    anticipation,
    estimation_params,
    influence_func=None,
    G=None,
    weights_ind=None,
    influence_moments=None,
):
    r"""Assemble the att_gt result from the effects and their covariance for a fit in memory or on shards.

    Effect :math:`k` has the analytic standard error :math:`\sqrt{V_{kk} / n}`
    for :math:`V` the ``variance_matrix`` and :math:`n` the number of units. A
    standard error of at most about 1.5e-7 is NaN. Where the analytic standard
    error is not NaN, the bootstrap standard error in ``bootstrap_results``
    replaces it.
    """
    standard_errors = np.sqrt(np.diag(variance_matrix) / n_units)
    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    zero_na_sd_indices = np.unique(np.where(np.isnan(standard_errors))[0])

    if bootstrap_results is not None:
        standard_errors = np.where(np.isnan(standard_errors), standard_errors, bootstrap_results["se"])

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

    if bootstrap_results is not None and cband:
        critical_value = bootstrap_results["crit_val"]
        if not np.isnan(critical_value) and critical_value >= 7:
            warnings.warn(
                "Simultaneous critical value is arguably 'too large' to be reliable. "
                "This usually happens when number of observations per group is small "
                "and/or there is not much variation in outcomes.",
                UserWarning,
            )

    return mp(
        groups=groups,
        times=times,
        att_gt=att_values,
        vcov_analytical=variance_matrix,
        se_gt=standard_errors,
        critical_value=critical_value,
        influence_func=influence_func,
        n_units=n_units,
        wald_stat=wald_statistic,
        wald_pvalue=wald_pvalue,
        alpha=alpha,
        estimation_params=estimation_params,
        G=G,
        weights_ind=weights_ind,
        influence_moments=influence_moments,
    )
