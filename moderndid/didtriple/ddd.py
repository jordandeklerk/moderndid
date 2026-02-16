"""Main wrapper for Triple Difference-in-Differences estimation."""

import numpy as np
import polars as pl

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocessing import preprocess_ddd_2periods

from .estimators.ddd_mp import ddd_mp
from .estimators.ddd_mp_rc import ddd_mp_rc
from .estimators.ddd_panel import ddd_panel
from .estimators.ddd_rc import _ddd_rc_2period
from .utils import add_intercept, detect_multiple_periods, detect_rcs_mode, get_covariate_names


def ddd(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    pname=None,
    xformla=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    weightsname=None,
    boot=False,
    boot_type="multiplier",
    biters=1000,
    cluster=None,
    alpha=0.05,
    trim_level=0.995,
    panel=True,
    allow_unbalanced_panel=False,
    random_state=None,
    n_jobs=1,
):
    r"""Compute the doubly robust Triple Difference-in-Differences estimator for the ATT.

    Implements triple difference-in-differences (DDD) estimation following [1]_. DDD
    extends standard DiD by incorporating a partition variable :math:`Q` that identifies
    eligible units within treatment-enabling groups :math:`S`, allowing for violations
    of traditional DiD parallel trends as long as these violations are stable across
    groups.

    Let :math:`S_i` denote the period when treatment is enabled for unit :math:`i`'s
    group, and :math:`Q_i \in \{0,1\}` indicate eligibility within that group. The
    group-time average treatment effect measures the effect among eligible units in
    group :math:`g` at time :math:`t`

    .. math::

        ATT(g,t) = \mathbb{E}[Y_{i,t}(g) - Y_{i,t}(\infty) \mid S_i = g, Q_i = 1].

    Identification relies on a DDD conditional parallel trends assumption that allows
    for differential trends between eligible and ineligible units, provided these
    differentials are stable across treatment-enabling groups. For groups :math:`g`
    and :math:`g'` where :math:`g' > \max\{g,t\}`

    .. math::

        &\mathbb{E}[\Delta Y(\infty) \mid S=g, Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g, Q=0, X] \\
        &= \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=1, X]
        - \mathbb{E}[\Delta Y(\infty) \mid S=g', Q=0, X],

    where :math:`\Delta Y(\infty) = Y_t(\infty) - Y_{t-1}(\infty)` denotes the change
    in untreated potential outcomes. This assumption does not impose standard DiD
    parallel trends within or across groups, making DDD appealing when such assumptions
    are implausible.

    Parameters
    ----------
    data : DataFrame
        Data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    idname : str, optional
        Name of unit identifier column. Required for panel data. For repeated
        cross-section data (panel=False), this can be omitted and a row index
        will be used automatically.
    gname : str
        Name of treatment group column. For 2-period data, this should be
        0 for never-treated and a positive value for treated units. For
        multi-period data, this is the first period when treatment is enabled
        for the unit's group (use 0 or np.inf for never-treated units).
    pname : str
        Name of partition/eligibility column (1=eligible, 0=ineligible).
        This identifies which units within a treatment group are actually
        eligible to receive the treatment effect.
    xformla : str, optional
        Formula for covariates in the form "~ x1 + x2 + x3". If None, only an
        intercept is used.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use as controls in multi-period settings.
        This parameter is ignored for 2-period data.
    base_period : {"universal", "varying"}, default="universal"
        Base period selection for multi-period settings.
        This parameter is ignored for 2-period data.
    est_method : {"dr", "reg", "ipw"}, default="dr"
        Estimation method: doubly robust, regression, or IPW.
    weightsname : str, optional
        Name of the column containing observation weights.
    boot : bool, default=False
        Whether to use bootstrap for inference.
    boot_type : {"multiplier", "weighted"}, default="multiplier"
        Type of bootstrap for 2-period data (only used if boot=True).
        Multi-period data always uses multiplier bootstrap.
    biters : int, default=1000
        Number of bootstrap repetitions (only used if boot=True).
    cluster : str, optional
        Name of the clustering variable for clustered standard errors.
        Currently only supported for 2-period data with bootstrap.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    trim_level : float, default=0.995
        Trimming level for propensity scores. Only used for repeated cross-section
        data (panel=False).
    panel : bool, default=True
        Whether the data is panel data (True) or repeated cross-section data (False).
        Panel data has the same units observed across time periods. Repeated
        cross-section data has different samples in each period.
    allow_unbalanced_panel : bool, default=False
        If True and panel=True, allows unbalanced panel data by treating it as
        repeated cross-section data. If the panel is unbalanced and this is False,
        an error will be raised.
    random_state : int, Generator, optional
        Random seed for reproducibility of bootstrap.
    n_jobs : int, default=1
        Number of parallel jobs for group-time estimation in multi-period
        settings. 1 = sequential (default), -1 = all cores, >1 = that many
        workers. Ignored for 2-period data.

    Returns
    -------
    DDDPanelResult, DDDRCResult, DDDMultiPeriodResult, or DDDMultiPeriodRCResult
        For 2-period panel data (panel=True), returns DDDPanelResult containing:

        - **att**: The DDD point estimate
        - **se**: Standard error
        - **uci**, **lci**: Confidence interval bounds
        - **boots**: Bootstrap draws (if requested)
        - **att_inf_func**: Influence function
        - **did_atts**: Individual DiD ATT estimates
        - **subgroup_counts**: Number of units per subgroup
        - **args**: Estimation arguments

        For 2-period repeated cross-section data (panel=False), returns DDDRCResult
        with the same structure.

        For multi-period panel data, returns DDDMultiPeriodResult containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**, **lci**: Confidence interval bounds
        - **groups**, **times**: Treatment cohort and time for each estimate
        - **glist**, **tlist**: Unique cohorts and periods
        - **inf_func_mat**: Influence function matrix
        - **n**: Number of units
        - **args**: Estimation arguments

        For multi-period repeated cross-section data, returns DDDMultiPeriodRCResult
        with the same structure.

    Examples
    --------
    We can generate synthetic data for a 2-period DDD setup using the ``gen_dgp_2periods``
    function. The data contains treatment status (``state``), eligibility within treatment
    groups  (``partition``), and covariates.

    .. ipython::

        In [1]: import numpy as np
           ...: from moderndid import ddd, gen_dgp_2periods
           ...:
           ...: dgp = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
           ...: df = dgp["data"]
           ...: df.head()

    Now we can compute the DDD estimate using the doubly robust estimator. The ``pname``
    parameter identifies which units within a treatment group are eligible to receive
    treatment, which is the key distinction from standard DiD.

    .. ipython::
        :okwarning:

        In [2]: result = ddd(
           ...:     data=df,
           ...:     yname="y",
           ...:     tname="time",
           ...:     idname="id",
           ...:     gname="state",
           ...:     pname="partition",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4",
           ...:     est_method="dr",
           ...: )
           ...: result

    The function automatically detects multi-period data with staggered treatment adoption.
    When there are more than two time periods or treatment cohorts, it returns group-time
    ATT estimates that can be aggregated using ``agg_ddd``.

    .. ipython::
        :okwarning:

        In [3]: from moderndid import gen_dgp_mult_periods
           ...:
           ...: dgp_mp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
           ...: result_mp = ddd(
           ...:     data=dgp_mp["data"],
           ...:     yname="y",
           ...:     tname="time",
           ...:     idname="id",
           ...:     gname="group",
           ...:     pname="partition",
           ...:     control_group="nevertreated",
           ...:     base_period="varying",
           ...:     est_method="dr",
           ...: )
           ...: result_mp

    The function also supports repeated cross-section data where different units are
    sampled in each time period. Set ``panel=False`` to use this mode.

    .. ipython::
        :okwarning:

        In [4]: dgp_rcs = gen_dgp_2periods(n=2000, dgp_type=1, panel=False, random_state=42)
           ...: result_rcs = ddd(
           ...:     data=dgp_rcs["data"],
           ...:     yname="y",
           ...:     tname="time",
           ...:     gname="state",
           ...:     pname="partition",
           ...:     xformla="~ cov1 + cov2 + cov3 + cov4",
           ...:     est_method="dr",
           ...:     panel=False,
           ...: )
           ...: result_rcs

    For multi-period repeated cross-section data with staggered treatment adoption,
    set ``panel=False`` with multiple time periods.

    .. ipython::
        :okwarning:

        In [5]: dgp_mp_rcs = gen_dgp_mult_periods(n=500, dgp_type=1, panel=False, random_state=42)
           ...: result_mp_rcs = ddd(
           ...:     data=dgp_mp_rcs["data"],
           ...:     yname="y",
           ...:     tname="time",
           ...:     gname="group",
           ...:     pname="partition",
           ...:     control_group="notyettreated",
           ...:     base_period="universal",
           ...:     est_method="dr",
           ...:     panel=False,
           ...: )
           ...: result_mp_rcs

    Notes
    -----
    The DDD estimator identifies treatment effects in settings where units must satisfy
    two criteria to be treated: belonging to a group that enables treatment (e.g., a state
    that passes a policy) and being in an eligible partition (e.g., women eligible for
    maternity benefits). This allows for violations of standard DiD parallel trends
    assumptions, as long as these violations are stable across groups.

    When ``est_method="dr"`` (the default), the function implements doubly robust
    DDD estimators that combine outcome regression and inverse probability weighting.
    These estimators are consistent if either the outcome model or the propensity
    score model is correctly specified.

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.
    ddd_rc : Two-period DDD estimator for repeated cross-section data.
    ddd_mp : Multi-period DDD estimator for staggered adoption with panel data.
    ddd_mp_rc : Multi-period DDD estimator for staggered adoption with RCS data.
    agg_ddd : Aggregate group-time DDD effects.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
    """
    from moderndid.dask._utils import is_dask_collection

    if is_dask_collection(data):
        from moderndid.dask._ddd import dask_ddd

        return dask_ddd(
            data,
            yname,
            tname,
            idname,
            gname,
            pname,
            xformla,
            control_group=control_group,
            base_period=base_period,
            est_method=est_method,
            boot=boot,
            biters=biters,
            cluster=cluster,
            alpha=alpha,
            random_state=random_state,
        )

    if gname is None:
        raise ValueError("gname is required. Please specify the treatment group column.")
    if pname is None:
        raise ValueError("pname is required. Please specify the partition/eligibility column.")
    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True.")
    if est_method not in ("dr", "reg", "ipw"):
        raise ValueError(f"est_method='{est_method}' is not valid. Must be 'dr', 'reg', or 'ipw'.")
    if control_group not in ("nevertreated", "notyettreated"):
        raise ValueError(f"control_group='{control_group}' is not valid. Must be 'nevertreated' or 'notyettreated'.")
    if base_period not in ("universal", "varying"):
        raise ValueError(f"base_period='{base_period}' is not valid. Must be 'universal' or 'varying'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if boot_type not in ("weighted", "multiplier"):
        raise ValueError(f"boot_type='{boot_type}' is not valid. Must be 'weighted' or 'multiplier'.")
    if not 0 < trim_level < 1:
        raise ValueError(f"trim_level={trim_level} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(n_jobs, int) or (n_jobs < 1 and n_jobs != -1):
        raise ValueError(f"n_jobs={n_jobs} is not valid. Must be a positive integer or -1 for all cores.")

    is_rcs = detect_rcs_mode(data, tname, idname, panel, allow_unbalanced_panel)

    data = to_polars(data)
    if is_rcs and idname is None:
        data = data.with_columns(pl.Series("_row_id", np.arange(len(data))))
        idname = "_row_id"

    multiple_periods = detect_multiple_periods(data, tname, gname)

    if multiple_periods:
        covariate_cols = get_covariate_names(xformla)

        if covariate_cols is not None:
            missing_covs = [c for c in covariate_cols if c not in data.columns]
            if missing_covs:
                raise ValueError(f"Covariates not found in data: {missing_covs}")

        if is_rcs:
            return ddd_mp_rc(
                data=data,
                y_col=yname,
                time_col=tname,
                id_col=idname,
                group_col=gname,
                partition_col=pname,
                covariate_cols=covariate_cols,
                control_group=control_group,
                base_period=base_period,
                est_method=est_method,
                boot=boot,
                biters=biters,
                cband=False,
                cluster=cluster,
                alpha=alpha,
                trim_level=trim_level,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        return ddd_mp(
            data=data,
            y_col=yname,
            time_col=tname,
            id_col=idname,
            group_col=gname,
            partition_col=pname,
            covariate_cols=covariate_cols,
            control_group=control_group,
            base_period=base_period,
            est_method=est_method,
            boot=boot,
            biters=biters,
            cband=False,
            cluster=cluster,
            alpha=alpha,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if is_rcs:
        return _ddd_rc_2period(
            data=data,
            yname=yname,
            tname=tname,
            gname=gname,
            pname=pname,
            xformla=xformla,
            weightsname=weightsname,
            est_method=est_method,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            alpha=alpha,
            trim_level=trim_level,
            random_state=random_state,
        )

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        pname=pname,
        xformla=xformla,
        est_method=est_method,
        weightsname=weightsname,
        boot=boot,
        boot_type=boot_type,
        n_boot=biters,
        cluster=cluster,
        alp=alpha,
        inf_func=True,
    )

    covariates_with_intercept = add_intercept(ddd_data.covariates)

    return ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates_with_intercept,
        i_weights=ddd_data.weights,
        est_method=est_method,
        boot=ddd_data.config.boot,
        boot_type=ddd_data.config.boot_type.value,
        biters=ddd_data.config.n_boot,
        influence_func=True,
        alpha=ddd_data.config.alp,
        random_state=random_state,
    )
