"""Multi-period difference-in-differences group-time average treatment effects estimation."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as la
import scipy.stats

from moderndid.core.preprocess import (
    BasePeriod,
    ControlGroup,
    DIDConfig,
    EstimationMethod,
    PreprocessDataBuilder,
)
from moderndid.cupy.backend import to_numpy

from .compute_att_gt import compute_att_gt
from .mboot import mboot
from .multiperiod_obj import mp


def att_gt(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    xformla=None,
    weightsname=None,
    alp=0.05,
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
    n_partitions=None,
    max_cohorts=None,
    progress_bar=False,
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
    alp : float, default=0.05
        The significance level.
    cband : bool, default=True
        Whether or not to compute a uniform confidence band that covers all of the
        group-time average treatment effects with fixed probability 1-alp.
    boot : bool, default=False
        Whether or not to compute standard errors using the multiplier bootstrap.
        If standard errors are clustered, then one must set boot=True.
    biters : int, default=1000
        The number of bootstrap iterations to use. Only applicable if boot=True.
    clustervars : list[str], optional
        A list of variables names to cluster on. At most, there can be two variables
        (otherwise will throw an error) and one of these must be the same as idname
        which allows for clustering at the individual level.
    est_method : {"dr", "ipw", "reg"} or callable, default="dr"
        The method to compute group-time average treatment effects. The default is
        "dr" which uses the doubly robust approach. Other built-in methods include
        "ipw" for inverse probability weighting and "reg" for first step regression
        estimators. The user can also pass their own function for estimating group
        time average treatment effects.
    panel : bool, default=True
        Whether or not the data is a panel dataset. The panel dataset should be
        provided in long format.
    allow_unbalanced_panel : bool, default=False
        Whether or not function should "balance" the panel with respect to time and
        id. The default values if False which means that att_gt will drop all units
        where data is not observed in all periods.
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
    n_partitions : int or None, default=None
        Number of Dask partitions per cell. Only used when ``data`` is a Dask
        DataFrame; ignored for non-Dask inputs.
    max_cohorts : int or None, default=None
        Maximum number of treatment cohorts to process in parallel. Only used
        when ``data`` is a Dask DataFrame; ignored for non-Dask inputs.
    progress_bar : bool, default=False
        Whether to display a tqdm progress bar during distributed computation.
        Only used when ``data`` is a Dask DataFrame; ignored for non-Dask inputs.

    Returns
    -------
    MPResult
        Object containing group-time average treatment effect results:

        - **groups**: Array indicating which group (period first treated) each ATT is for
        - **times**: Array indicating which time period each ATT is for
        - **att_gt**: Array of group-time average treatment effects
        - **se_gt**: Standard errors for each ATT(g,t)
        - **vcov_analytical**: Analytical variance-covariance matrix
        - **critical_value**: Critical value for confidence intervals (simultaneous if bootstrap with cband=True)
        - **influence_func**: Influence function matrix for each ATT(g,t)
        - **n_units**: Number of unique cross-sectional units
        - **wald_stat**: Wald statistic for pre-testing parallel trends
        - **wald_pvalue**: P-value for the parallel trends pre-test
        - **alpha**: Significance level used
        - **estimation_params**: Dictionary with estimation details (control_group, anticipation_periods, etc.)
        - **G**: Unit-level group assignments
        - **weights_ind**: Unit-level sampling weights (if provided)

    Examples
    --------
    The dataset below contains 500 observations of county-level teen employment rates from 2003-2007.
    Some states are first treated in 2004, some in 2006, and some in 2007. The variable ``first.treat``
    indicates the first period in which a state is treated:

    .. ipython::

        In [1]: import numpy as np
           ...: from moderndid import att_gt, load_mpdta
           ...:
           ...: df = load_mpdta()
           ...: print(df.head())

    We can compute group-time average treatment effects for a staggered adoption design
    where different units adopt treatment at different time periods. The output is an object of type
    ``MPResult`` which is a container for the results:

    .. ipython::
        :okwarning:

        In [2]: result = att_gt(
           ...:     data=df,
           ...:     yname="lemp",
           ...:     tname="year",
           ...:     gname="first.treat",
           ...:     idname="countyreal",
           ...:     est_method="dr",
           ...:     boot=False
           ...: )
           ...: print(result)

    See Also
    --------
    aggte : Aggregate group-time average treatment effects.

    References
    ----------
    .. [1] Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences
           with multiple time periods." Journal of Econometrics, 225(2), 200-230.
           https://doi.org/10.1016/j.jeconom.2020.12.001
    """
    from moderndid.dask._utils import is_dask_collection

    if is_dask_collection(data):
        from moderndid.dask._did import dask_att_gt

        return dask_att_gt(
            data,
            yname,
            tname,
            idname,
            gname,
            xformla=xformla,
            control_group=control_group,
            base_period=base_period,
            anticipation=anticipation,
            est_method=est_method,
            panel=panel,
            weightsname=weightsname,
            boot=boot,
            biters=biters,
            cband=cband,
            alp=alp,
            clustervars=clustervars,
            allow_unbalanced_panel=allow_unbalanced_panel,
            trim_level=0.995,
            random_state=random_state,
            n_partitions=n_partitions,
            max_cohorts=max_cohorts,
            progress_bar=progress_bar,
        )

    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True.")

    if isinstance(control_group, str) and control_group not in ("nevertreated", "notyettreated"):
        raise ValueError(f"control_group='{control_group}' is not valid. Must be 'nevertreated' or 'notyettreated'.")
    if isinstance(est_method, str) and est_method not in ("dr", "ipw", "reg"):
        raise ValueError(f"est_method='{est_method}' is not valid. Must be 'dr', 'ipw', or 'reg'.")
    if isinstance(base_period, str) and base_period not in ("varying", "universal"):
        raise ValueError(f"base_period='{base_period}' is not valid. Must be 'varying' or 'universal'.")
    if not 0 < alp < 1:
        raise ValueError(f"alp={alp} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} is not valid. Must be a non-negative number.")
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs={n_jobs} is not valid. Must be a positive integer or -1 for all cores.")
    if clustervars is not None and isinstance(clustervars, str):
        raise TypeError(f"clustervars must be a list of strings, not a string. Use clustervars=['{clustervars}'].")

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
        alp=alp,
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
    variance_matrix = influence_functions_dense.T @ influence_functions_dense / n_units
    standard_errors = np.sqrt(np.diag(variance_matrix) / n_units)
    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # If clustering along another dimension we require using the bootstrap
    if (clustervars is not None and len(clustervars) > 0) and not boot:
        warnings.warn(
            "Clustering the standard errors requires using the bootstrap, "
            "resulting standard errors are NOT accounting for clustering",
            UserWarning,
        )

    zero_na_sd_indices = np.unique(np.where(np.isnan(standard_errors))[0])

    if boot:
        cluster = None
        if clustervars is not None and len(clustervars) > 0:
            if len(clustervars) > 2:
                raise ValueError("Can cluster on at most 2 variables.")

            if not hasattr(dp, "time_invariant_data"):
                raise RuntimeError(
                    "Clustering requires 'time_invariant_data' in the pre-processed data, but it was not found."
                )
            cluster_data = dp.time_invariant_data

            if len(clustervars) == 1:
                cluster = cluster_data[clustervars[0]].to_numpy()
            else:
                combined = cluster_data[clustervars[0]].cast(str) + "_" + cluster_data[clustervars[1]].cast(str)
                unique_vals = combined.unique()
                val_to_code = {v: i for i, v in enumerate(unique_vals.to_list())}
                cluster = np.array([val_to_code[v] for v in combined.to_list()])

        bootstrap_results = mboot(
            inf_func=influence_functions_dense,
            n_units=n_units,
            biters=biters,
            alp=alp,
            cluster=cluster,
            random_state=random_state,
        )

        if len(zero_na_sd_indices) > 0:
            standard_errors[~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)] = bootstrap_results["se"][
                ~np.isin(np.arange(len(standard_errors)), zero_na_sd_indices)
            ]
        else:
            standard_errors = bootstrap_results["se"]

    standard_errors[standard_errors <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # Wald pre-test
    pre_treatment_indices = np.where(groups > times)[0]

    if len(zero_na_sd_indices) > 0:
        pre_treatment_indices = pre_treatment_indices[~np.isin(pre_treatment_indices, zero_na_sd_indices)]

    # Pseudo-atts in pre-treatment periods
    pre_treatment_att = att_values[pre_treatment_indices]
    pre_treatment_variance = variance_matrix[np.ix_(pre_treatment_indices, pre_treatment_indices)]

    if len(pre_treatment_indices) == 0:
        warnings.warn("No pre-treatment periods to test", UserWarning)
        wald_statistic = None
        wald_pvalue = None
    if np.any(np.isnan(pre_treatment_variance)):
        warnings.warn(
            "Not returning pre-test Wald statistic due to NA pre-treatment values",
            UserWarning,
        )
        wald_statistic = None
        wald_pvalue = None
    if (
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
            wald_statistic = n_units * pre_treatment_att.T @ np.linalg.solve(pre_treatment_variance, pre_treatment_att)
            q = len(pre_treatment_indices)
            wald_pvalue = round(1 - scipy.stats.chi2.cdf(wald_statistic, q), 5)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Not returning pre-test Wald statistic due to numerical issues",
                UserWarning,
            )
            wald_statistic = None
            wald_pvalue = None

    critical_value = scipy.stats.norm.ppf(1 - alp / 2)

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
        "control_group": control_group,
        "anticipation_periods": anticipation,
        "estimation_method": est_method if isinstance(est_method, str) else "custom",
        "bootstrap": boot,
        "uniform_bands": cband,
        "base_period": base_period,
        "panel": panel,
        "clustervars": clustervars,
        "biters": biters,
        "random_state": random_state,
    }

    group_assignments = None
    sampling_weights = None

    if hasattr(dp, "time_invariant_data"):
        if gname in dp.time_invariant_data.columns:
            group_assignments = dp.time_invariant_data[gname]
        if weightsname is not None and weightsname in dp.time_invariant_data.columns:
            sampling_weights = dp.time_invariant_data[weightsname]

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
        alpha=alp,
        estimation_params=estimation_params,
        G=group_assignments,
        weights_ind=sampling_weights,
    )
