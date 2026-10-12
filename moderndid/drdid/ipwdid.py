"""Wrapper for inverse propensity weighted DiD estimators."""

from moderndid.core.preprocess import preprocess_drdid
from moderndid.core.preprocess.validators import check_columns

from .container import IPWDIDResult
from .estimators.ipw_did_panel import ipw_did_panel
from .estimators.ipw_did_rc import ipw_did_rc
from .estimators.std_ipw_did_panel import std_ipw_did_panel
from .estimators.std_ipw_did_rc import std_ipw_did_rc


def ipwdid(
    data,
    yname,
    tname,
    idname=None,
    treatname=None,
    xformla=None,
    panel=True,
    est_method="ipw",
    weightsname=None,
    boot=False,
    boot_type="weighted",
    biters=999,
    inf_func=False,
    trim_level=0.995,
):
    r"""Wrap the inverse propensity weighted DiD estimators for the ATT.

    This function is a wrapper for inverse propensity weighted (IPW) DiD estimators [1]_, [2]_.
    It can be used with panel or stationary repeated cross-section data and calls the
    appropriate estimator based on the panel argument and estimation method.

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
    est_method : {"ipw", "std_ipw"}, default "ipw"
        The IPW estimator to compute. The default "ipw" is the Horvitz-Thompson type
        estimator of [1]_. Its weights are not normalized to sum to one. "std_ipw" is
        the standardized estimator of the Hajek type. With panel data its weights sum to
        one within each group. With repeated cross-sections they sum to one within each
        group and period.
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
    IPWDIDResult
        NamedTuple containing:

        - **att**: The IPW DiD point estimate.
        - **se**: The IPW DiD standard error.
        - **uci**: The upper bound of a 95 percent confidence interval.
        - **lci**: The lower bound of a 95 percent confidence interval.
        - **boots**: Bootstrap draws of the ATT if boot=True.
        - **att_inf_func**: Influence function values if inf_func=True.
        - **call_params**: Original function call parameters.
        - **args**: Arguments used in the estimation.

    Examples
    --------
    Estimate the average treatment effect on the treated (ATT) using inverse propensity
    weighting with panel data from a job training program. IPW reweights observations
    to create balance between treated and control groups.

    .. ipython::
        :okwarning:

        In [1]: import moderndid
           ...: from moderndid import load_nsw
           ...:
           ...: nsw_data = load_nsw()
           ...:
           ...: att_result = moderndid.ipwdid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...:     est_method="ipw",
           ...: )

        In [2]: print(att_result)

    The standardized (Hajek-type) estimator normalizes the weights to sum to one and
    can be more stable with extreme propensity scores.

    .. ipython::
        :okwarning:

        In [3]: att_result_std = moderndid.ipwdid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...:     est_method="std_ipw",
           ...: )

        In [4]: print(att_result_std)

    Notes
    -----
    The IPW estimators reweight the comparison units by the odds of the propensity
    score, the probability of being in the treated group, so that they resemble the
    treated units. Normalized weights make the estimate much more stable when some
    propensity scores are close to one [2]_.

    Unlike doubly robust methods, IPW estimators are not robust to misspecification
    of the propensity score model. They also ignore the outcome model. In the
    simulations of [2]_ with a correctly specified propensity score model, both IPW
    estimators have a larger variance than the doubly robust estimators.

    See Also
    --------
    drdid : Doubly robust DiD estimator.
    ordid : Outcome regression DiD estimator.

    References
    ----------
    .. [1] Abadie, A. (2005), "Semiparametric Difference-in-Differences Estimators",
           Review of Economic Studies, vol. 72(1), pp. 1-19.
           https://doi.org/10.1111/0034-6527.00321

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020),
           "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122.
           https://doi.org/10.1016/j.jeconom.2020.06.003
    """
    if treatname is None:
        raise ValueError("treatname is required. Please specify the treatment column.")

    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True")

    if est_method not in ("ipw", "std_ipw"):
        raise ValueError(f"est_method='{est_method}' is not valid. Must be 'ipw' or 'std_ipw'.")
    if boot_type not in ("weighted", "multiplier"):
        raise ValueError(f"boot_type='{boot_type}' is not valid. Must be 'weighted' or 'multiplier'.")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if not 0 < trim_level < 1:
        raise ValueError(f"trim_level={trim_level} is not valid. Must be between 0 and 1 (exclusive).")
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
        trim_level=trim_level,
    )

    if panel:
        if est_method == "ipw":
            result = ipw_did_panel(
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
            result = std_ipw_did_panel(
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
        if est_method == "ipw":
            result = ipw_did_rc(
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
            result = std_ipw_did_rc(
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
            "normalized": est_method == "std_ipw",
            "boot": boot,
            "boot_type": boot_type,
            "biters": biters,
            "type": "ipw",
            "trim_level": trim_level,
            "n_obs": 2 * dp.n_units if panel else dp.n_obs,
            "n_covariates": sum(name != "Intercept" for name in dp.covariate_names),
        }
    )

    return IPWDIDResult(
        att=result.att,
        se=result.se,
        uci=result.uci,
        lci=result.lci,
        boots=result.boots,
        att_inf_func=result.att_inf_func,
        call_params=call_params,
        args=args,
    )
