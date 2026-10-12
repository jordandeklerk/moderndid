"""Wrapper for doubly robust DiD estimators."""

from moderndid.core.preprocess import preprocess_drdid
from moderndid.core.preprocess.validators import check_columns

from .container import DRDIDResult
from .estimators.drdid_imp_local_rc import drdid_imp_local_rc
from .estimators.drdid_imp_panel import drdid_imp_panel
from .estimators.drdid_panel import drdid_panel
from .estimators.drdid_rc import drdid_rc


def drdid(
    data,
    yname,
    tname,
    idname=None,
    treatname=None,
    xformla=None,
    panel=True,
    est_method="imp",
    weightsname=None,
    boot=False,
    boot_type="weighted",
    biters=999,
    inf_func=False,
    trim_level=0.995,
):
    r"""Compute the doubly robust DiD estimator for the ATT.

    Implements the doubly robust difference-in-differences estimators of [2]_ for the
    average treatment effect on the treated (ATT) with two periods. They combine inverse
    probability weighting with outcome regression. The estimate is consistent if either
    the propensity score model or the outcome model of the comparison group is correctly
    specified.

    The function accepts panel data and stationary repeated cross-sections and calls the
    estimator that matches ``panel`` and ``est_method``. With either design, both
    methods give a locally efficient estimator.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        The input data containing outcome, time, unit ID, treatment,
        and optionally covariates and weights. Accepts both pandas and polars DataFrames.
    yname : str
        Name of the column containing the outcome variable.
    tname : str
        Name of the column containing the time periods (must have exactly 2 periods).
    idname : str | None, default None
        Name of the column containing the unit ID. Required if panel=True.
    treatname : str
        Name of the column containing the treatment group indicator. With panel data it
        is constant within a unit, 1 if ever treated and 0 if never treated. With
        repeated cross-sections it marks the group treated in the post-period and equals
        1 for that group's rows in both periods.
    xformla : str | None, default None
        Formula that joins covariate column names with ``+``, as in
        ``"~ x1 + x2"``. The model always includes an intercept. Transformed
        terms such as ``I(x**2)`` or ``C(x)`` and columns of strings or
        categories require formulaic. If None, only the intercept enters.
    panel : bool, default True
        Whether the data is panel (True) or repeated cross-sections (False).
        Panel data should be in long format with each row representing
        a unit-time observation.
    est_method : {"imp", "trad"}, default "imp"
        The estimator to compute. The default "imp" fits the propensity score by inverse
        probability tilting [1]_ and the outcome regressions of the comparison group by
        weighted least squares. It is the estimator in equation (3.7) of [2]_ for panel
        data and in equation (3.10) for repeated cross-sections. With repeated
        cross-sections, it also fits outcome regressions for the treated group in each
        period by ordinary least squares. "trad" fits the propensity score by maximum
        likelihood and the outcome regressions by ordinary least squares. It is the
        estimator in equation (3.1) for panel data and in equation (3.4) for repeated
        cross-sections.
    weightsname : str | None, default None
        Name of the column containing sampling weights.
        If None, all observations have equal weight.
        Weights are normalized to have mean 1.
    boot : bool, default False
        Whether to compute bootstrap standard errors. If False,
        analytical standard errors are reported.
    boot_type : {"weighted", "multiplier"}, default "weighted"
        Type of bootstrap to perform (only relevant if boot=True).
    biters : int, default 999
        Number of bootstrap repetitions (only relevant if boot=True).
    inf_func : bool, default False
        Whether to return the influence function values.
    trim_level : float, default 0.995
        The level of trimming for the propensity score.

    Returns
    -------
    DRDIDResult
        NamedTuple containing:

        - **att**: The DR DiD point estimate.
        - **se**: The DR DiD standard error.
        - **uci**: The upper bound of a 95 percent confidence interval.
        - **lci**: The lower bound of a 95 percent confidence interval.
        - **boots**: Bootstrap draws of the ATT if boot=True.
        - **att_inf_func**: Influence function values if inf_func=True.
        - **call_params**: Original function call parameters.
        - **args**: Arguments used in the estimation.

    Examples
    --------
    Estimate the average treatment effect on the treated (ATT) using panel data from a job training
    program. The data tracks the same individuals over time, before and after
    some received training.

    .. ipython::
        :okwarning:

        In [1]: import moderndid
           ...: from moderndid import load_nsw
           ...:
           ...: nsw_data = load_nsw()
           ...:
           ...: att_result = moderndid.drdid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...:     est_method="imp",
           ...: )

        In [2]: print(att_result)

    Notes
    -----
    The parameter of interest is the ATT in a two-period setting

    .. math::

        \tau = \mathbb{E}[Y_1(1) - Y_1(0) \mid D = 1].

    Identification relies on a conditional parallel trends assumption requiring
    that treated and comparison groups would have evolved similarly absent
    treatment, conditional on covariates :math:`X`

    .. math::

        \mathbb{E}[Y_1(0) - Y_0(0) \mid D=1, X] = \mathbb{E}[Y_1(0) - Y_0(0) \mid D=0, X].

    For panel data, the doubly robust estimand combines outcome regression
    :math:`\mu_{0,\Delta}(X) = \mathbb{E}[\Delta Y \mid D=0, X]` with propensity
    score weighting :math:`\pi(X) = \mathbb{P}(D=1 \mid X)`

    .. math::

        \tau^{dr} = \mathbb{E}\left[\left(\frac{D}{\mathbb{E}[D]}
        - \frac{\pi(X)(1-D)/(1-\pi(X))}
        {\mathbb{E}[\pi(X)(1-D)/(1-\pi(X))]}\right)
        (\Delta Y - \mu_{0,\Delta}(X))\right].

    An estimator is locally efficient when its asymptotic variance attains the
    semiparametric efficiency bound once every working model is correctly specified
    [2]_. With panel data, both methods model the propensity score and the outcome
    evolution of the comparison group. With stationary repeated cross-sections, they
    also model the outcome of the treated group in each period. Those regressions cannot
    be fit in a period where the treated group has fewer observations than coefficients
    or a covariate that is constant or collinear within it. The function then raises a
    ValueError that names the group and period. The estimators in equations (3.3) and
    (3.9) of [2]_ leave those regressions out and are not locally efficient. Their
    variance can be many times larger even when every working model is correct.
    :func:`drdid_trad_rc` and :func:`drdid_imp_rc` compute them.

    With the inverse probability tilting and weighted least squares fits of ``"imp"``,
    the asymptotic variance does not depend on which working models are correct. That
    estimator is therefore doubly robust for inference as well as for consistency [2]_.

    See Also
    --------
    ipwdid : Inverse propensity weighted DiD estimator.
    ordid : Outcome regression DiD estimator.

    References
    ----------

    .. [1] Graham, B., Pinto, C., and Egel, D. (2012),
           "Inverse Probability Tilting for Moment Condition Models with Missing Data."
           Review of Economic Studies, vol. 79 (3), pp. 1053-1079.
           https://doi.org/10.1093/restud/rdr047

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020),
           "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122.
           https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    if treatname is None:
        raise ValueError("treatname is required. Please specify the treatment column.")

    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True")

    if est_method not in ("imp", "trad"):
        raise ValueError(f"est_method='{est_method}' is not valid. Must be 'imp' or 'trad'.")

    if not 0 < trim_level < 1:
        raise ValueError(f"trim_level={trim_level} is not valid. Must be between 0 and 1 (exclusive).")
    if boot_type not in ("weighted", "multiplier"):
        raise ValueError(f"boot_type='{boot_type}' is not valid. Must be 'weighted' or 'multiplier'.")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    check_columns(
        data,
        yname=yname,
        tname=tname,
        idname=idname if panel else None,
        treatname=treatname,
        xformla=xformla,
        weightsname=weightsname,
    )

    call_params = {
        "yname": yname,
        "tname": tname,
        "idname": idname,
        "treatname": treatname,
        "xformla": xformla,
        "data_shape": data.shape,
        "panel": panel,
        "est_method": est_method,
        "weightsname": weightsname,
        "boot": boot,
        "boot_type": boot_type,
        "biters": biters,
        "inf_func": inf_func,
        "trim_level": trim_level,
    }

    dp = preprocess_drdid(
        data=data,
        yname=yname,
        tname=tname,
        treat_col=treatname,
        idname=idname if panel else None,
        xformla=xformla,
        panel=panel,
        weightsname=weightsname,
        boot=boot,
        boot_type=boot_type,
        biters=biters,
        inf_func=inf_func,
        est_method=est_method,
        trim_level=trim_level,
    )

    if panel:
        if est_method == "imp":
            result = drdid_imp_panel(
                y1=dp.y1,
                y0=dp.y0,
                d=dp.D,
                covariates=dp.covariates,
                i_weights=dp.weights,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                influence_func=inf_func,
                trim_level=trim_level,
            )
        else:
            result = drdid_panel(
                y1=dp.y1,
                y0=dp.y0,
                d=dp.D,
                covariates=dp.covariates,
                i_weights=dp.weights,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                influence_func=inf_func,
                trim_level=trim_level,
            )
    else:
        # Since drdid_imp_rc's improved estimator is not locally efficient, "imp" runs the one that is.
        if est_method == "imp":
            result = drdid_imp_local_rc(
                y=dp.y,
                post=dp.post,
                d=dp.D,
                covariates=dp.covariates,
                i_weights=dp.weights,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                influence_func=inf_func,
                trim_level=trim_level,
            )
        else:
            result = drdid_rc(
                y=dp.y,
                post=dp.post,
                d=dp.D,
                covariates=dp.covariates,
                i_weights=dp.weights,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                influence_func=inf_func,
                trim_level=trim_level,
            )

    args = result.args.copy()
    args.update(
        {
            "panel": panel,
            "estMethod": est_method,
            "normalized": True,
            "boot": boot,
            "boot_type": boot_type,
            "biters": biters,
            "type": "dr",
            "trim_level": trim_level,
            "n_obs": 2 * dp.n_units if panel else dp.n_obs,
            "n_covariates": sum(name != "Intercept" for name in dp.covariate_names),
        }
    )

    return DRDIDResult(
        att=result.att,
        se=result.se,
        uci=result.uci,
        lci=result.lci,
        boots=result.boots,
        att_inf_func=result.att_inf_func,
        call_params=call_params,
        args=args,
    )
