"""Wrapper for outcome regression DiD estimators."""

from moderndid.core.preprocess import preprocess_drdid
from moderndid.core.preprocess.validators import check_columns

from .container import ORDIDResult
from .estimators.reg_did_panel import reg_did_panel
from .estimators.reg_did_rc import reg_did_rc


def ordid(
    data,
    yname,
    tname,
    idname=None,
    treatname=None,
    xformla=None,
    panel=True,
    weightsname=None,
    boot=False,
    boot_type="weighted",
    biters=999,
    inf_func=False,
):
    r"""Wrap the outcome regression DiD estimators for the ATT.

    Estimates the average treatment effect on the treated (ATT) of a design with two
    periods by outcome regression [1]_, [2]_. Linear regressions fit on the comparison
    group predict the untreated outcome change of the treated units from their
    covariates. The estimate subtracts that predicted change from the treated group's
    observed change. Unlike the doubly robust :func:`~moderndid.drdid`, it is consistent
    only if the outcome regression is correctly specified.

    The function runs :func:`~moderndid.reg_did_panel` on a panel and
    :func:`~moderndid.reg_did_rc` on stationary repeated cross-sections.

    Parameters
    ----------
    data : DataFrame
        Data in long format. Accepts pandas and polars DataFrames.
    yname : str
        The name of the outcome column.
    tname : str
        The name of the time period column. It must hold exactly two periods.
    idname : str, optional
        The name of the unit identifier column. Required if ``panel=True``.
    treatname : str
        The name of the treatment group column, 1 in both periods for the group treated in
        the post-period and 0 for the comparison group.
    xformla : str, optional
        A formula joining covariate columns with ``+``, such as ``"~ x1 + x2"``. The
        model always includes an intercept. Terms such as ``I(x**2)`` or ``C(x)`` and
        string or categorical columns require formulaic.
    panel : bool, default=True
        Whether the data is a panel (True) or repeated cross-sections (False).
    weightsname : str, optional
        The name of the sampling weights column. The weights are normalized to mean 1.
    boot : bool, default=False
        Whether to compute bootstrap standard errors instead of analytical ones.
    boot_type : {"weighted", "multiplier"}, default="weighted"
        The kind of bootstrap. Only applicable if ``boot=True``.
    biters : int, default=999
        The number of bootstrap draws. Only applicable if ``boot=True``.
    inf_func : bool, default=False
        Whether to return the influence function values.

    Returns
    -------
    ORDIDResult
        NamedTuple containing:

        - **att**: The OR DiD point estimate
        - **se**: The OR DiD standard error
        - **uci**: The upper bound of a 95 percent confidence interval
        - **lci**: The lower bound of a 95 percent confidence interval
        - **boots**: Bootstrap draws of the ATT if boot=True
        - **att_inf_func**: Influence function values if inf_func=True
        - **call_params**: Original function call parameters
        - **args**: Arguments used in the estimation

    See Also
    --------
    drdid : Doubly robust DiD estimator.
    ipwdid : Inverse propensity weighted DiD estimator.

    Notes
    -----
    The :ref:`doubly robust DiD background <background-drdid>` develops the outcome
    regression estimand and the conditional parallel trends it relies on.

    References
    ----------

    .. [1] Heckman, J., Ichimura, H., and Todd, P. (1997),
           "Matching as an Econometric Evaluation Estimator: Evidence from
           Evaluating a Job Training Programme", Review of Economic Studies,
           vol. 64(4), pp. 605-654. https://doi.org/10.2307/2971733

    .. [2] Sant'Anna, P. H. C. and Zhao, J. (2020),
           "Doubly Robust Difference-in-Differences Estimators."
           Journal of Econometrics, Vol. 219 (1), pp. 101-122.
           https://doi.org/10.1016/j.jeconom.2020.06.003

    Examples
    --------
    Estimate the ATT of a job training program on earnings by outcome regression.

    .. ipython::

        In [1]: import moderndid
           ...: from moderndid import load_nsw
           ...:
           ...: nsw_data = load_nsw()
           ...:
           ...: att_result = moderndid.ordid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...: )

        In [2]: print(att_result)
    """
    if treatname is None:
        raise ValueError("treatname is required. Please specify the treatment column.")

    if panel and idname is None:
        raise ValueError("idname must be provided when panel=True")

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
        "weightsname": weightsname,
        "boot": boot,
        "boot_type": boot_type,
        "biters": biters,
        "inf_func": inf_func,
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
    )

    if panel:
        result = reg_did_panel(
            y1=dp.y1,
            y0=dp.y0,
            d=dp.D,
            covariates=dp.covariates,
            i_weights=dp.weights,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            influence_func=inf_func,
        )
    else:
        result = reg_did_rc(
            y=dp.y,
            post=dp.post,
            d=dp.D,
            covariates=dp.covariates,
            i_weights=dp.weights,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            influence_func=inf_func,
        )

    args = result.args.copy()
    args.update(
        {
            "panel": panel,
            "normalized": True,
            "boot": boot,
            "boot_type": boot_type,
            "biters": biters,
            "type": "or",
            "n_obs": 2 * dp.n_units if panel else dp.n_obs,
            "n_covariates": sum(name != "Intercept" for name in dp.covariate_names),
        }
    )

    return ORDIDResult(
        att=result.att,
        se=result.se,
        uci=result.uci,
        lci=result.lci,
        boots=result.boots,
        att_inf_func=result.att_inf_func,
        call_params=call_params,
        args=args,
    )
