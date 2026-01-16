"""Main wrapper for Triple Difference-in-Differences estimation."""

import numpy as np

from moderndid.core.preprocess.utils import parse_formula
from moderndid.core.preprocessing import preprocess_ddd_2periods

from .estimators.ddd_mp import ddd_mp
from .estimators.ddd_panel import ddd_panel


def ddd(
    data,
    yname,
    tname,
    idname,
    gname,
    pname,
    xformla=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    weightsname=None,
    boot=False,
    boot_type="multiplier",
    nboot=999,
    cluster=None,
    alpha=0.05,
    random_state=None,
):
    r"""Compute the doubly robust Triple Difference-in-Differences estimator for the ATT.

    Wrapper for triple difference-in-differences (DDD) estimators that automatically
    detects whether the data has two periods or multiple periods with staggered treatment
    adoption, calling the appropriate estimator. DDD extends standard DiD by incorporating
    a partition variable that identifies eligible units within treatment groups, allowing
    for both group-specific and partition-specific violations of parallel trends.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    idname : str
        Name of unit identifier column.
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
    nboot : int, default=999
        Number of bootstrap repetitions (only used if boot=True).
    cluster : str, optional
        Name of the clustering variable for clustered standard errors.
        Currently only supported for 2-period data with bootstrap.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    random_state : int, Generator, optional
        Random seed for reproducibility of bootstrap.

    Returns
    -------
    DDDPanelResult or DDDMultiPeriodResult
        For 2-period data, returns DDDPanelResult containing:

        - *att*: The DDD point estimate
        - *se*: Standard error
        - *uci*, *lci*: Confidence interval bounds
        - *boots*: Bootstrap draws (if requested)
        - *att_inf_func*: Influence function
        - *did_atts*: Individual DiD ATT estimates
        - *subgroup_counts*: Number of units per subgroup
        - *args*: Estimation arguments

        For multi-period data, returns DDDMultiPeriodResult containing:

        - *att*: Array of ATT(g,t) point estimates
        - *se*: Standard errors for each ATT(g,t)
        - *uci*, *lci*: Confidence interval bounds
        - *groups*, *times*: Treatment cohort and time for each estimate
        - *glist*, *tlist*: Unique cohorts and periods
        - *inf_func_mat*: Influence function matrix
        - *n*: Number of units
        - *args*: Estimation arguments

    Examples
    --------
    First, we generate synthetic data for a 2-period DDD setup. The data contains
    treatment status (``state``), eligibility within treatment groups (``partition``),
    and covariates.

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

    For two-period data, the DDD estimate can be expressed as a combination of three
    DiD terms comparing treated-eligible units against: (1) treated-ineligible units,
    (2) control-eligible units, and (3) control-ineligible units.

    For multi-period data with staggered adoption, the function computes group-time
    average treatment effects ATT(g,t) for each treatment cohort g and time period t.
    These can be aggregated into summary measures using ``agg_ddd``.

    See Also
    --------
    ddd_panel : Two-period DDD estimator for panel data.
    ddd_mp : Multi-period DDD estimator for staggered adoption.
    agg_ddd : Aggregate group-time DDD effects.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
           "Better Understanding Triple Differences Estimators."
           arXiv preprint arXiv:2505.09942.
    """
    multiple_periods = _multiple_periods(data, tname, gname)

    if multiple_periods:
        covariates = _extract_covariates(data, xformla)

        return ddd_mp(
            data=data,
            y_col=yname,
            time_col=tname,
            id_col=idname,
            group_col=gname,
            partition_col=pname,
            covariates=covariates,
            control_group=control_group,
            base_period=base_period,
            est_method=est_method,
            boot=boot,
            nboot=nboot,
            cband=False,
            cluster=cluster,
            alpha=alpha,
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
        n_boot=nboot,
        cluster=cluster,
        alp=alpha,
        inf_func=True,
    )

    covariates_with_intercept = _intercept(ddd_data.covariates)

    return ddd_panel(
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates_with_intercept,
        i_weights=ddd_data.weights,
        est_method=est_method,
        boot=ddd_data.config.boot,
        boot_type=ddd_data.config.boot_type.value,
        nboot=ddd_data.config.n_boot,
        influence_func=True,
        alpha=ddd_data.config.alp,
        random_state=random_state,
    )


def _multiple_periods(data, tname, gname):
    """Detect whether data has multiple periods (staggered adoption)."""
    n_time_periods = data[tname].nunique()

    gvals = data[gname].unique()
    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time_periods, n_groups) > 2


def _extract_covariates(data, xformla):
    """Extract covariate matrix for multi-period estimation."""
    if xformla is None or xformla == "~1":
        return None

    formula_str = xformla.strip()
    if formula_str.startswith("~"):
        formula_str = "y " + formula_str

    parsed = parse_formula(formula_str)
    covariate_names = parsed["predictors"]

    if not covariate_names or covariate_names == ["1"]:
        return None

    covariate_names = [c for c in covariate_names if c != "1"]

    missing_covs = [c for c in covariate_names if c not in data.columns]
    if missing_covs:
        raise ValueError(f"Covariates not found in data: {missing_covs}")

    X = data[covariate_names].values
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def _intercept(covariates):
    """Add intercept column to covariate matrix if needed."""
    if covariates is None or covariates.shape[1] == 0:
        return None

    intercept = np.ones((covariates.shape[0], 1))
    return np.hstack([intercept, covariates])
