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

    Estimates the average treatment effect on the treated (ATT) of a design with two
    periods [2]_,

    .. math::

        \tau = \mathbb{E}[Y_1(1) - Y_1(0) \mid D = 1].

    The estimators combine inverse probability weighting with an outcome regression for
    the comparison group. The estimate is consistent if either the propensity score
    model or the outcome model is correctly specified. With panel data and with
    stationary repeated cross-sections, both methods give a locally efficient estimator.
    The function runs :func:`~moderndid.drdid_imp_panel` or
    :func:`~moderndid.drdid_panel` on a panel and :func:`~moderndid.drdid_imp_local_rc`
    or :func:`~moderndid.drdid_rc` on repeated cross-sections, as ``est_method``
    selects.

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
    est_method : {"imp", "trad"}, default="imp"
        ``"imp"`` fits the propensity score by inverse probability tilting [1]_ and the
        outcome regressions by weighted least squares. ``"trad"`` uses maximum
        likelihood and ordinary least squares.
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
    trim_level : float, default=0.995
        The propensity score level at which comparison units are trimmed.

    Returns
    -------
    DRDIDResult
        NamedTuple containing:

        - **att**: The DR DiD point estimate
        - **se**: The DR DiD standard error
        - **uci**: The upper bound of a 95 percent confidence interval
        - **lci**: The lower bound of a 95 percent confidence interval
        - **boots**: Bootstrap draws of the ATT if boot=True
        - **att_inf_func**: Influence function values if inf_func=True
        - **call_params**: Original function call parameters
        - **args**: Arguments used in the estimation

    See Also
    --------
    ipwdid : Inverse propensity weighted DiD estimator.
    ordid : Outcome regression DiD estimator.

    Notes
    -----
    Identification rests on conditional parallel trends given the covariates
    :math:`X`,

    .. math::

        \mathbb{E}[Y_1(0) - Y_0(0) \mid D=1, X] = \mathbb{E}[Y_1(0) - Y_0(0) \mid D=0, X].

    With panel data, the outcome regression
    :math:`\mu_{0,\Delta}(X) = \mathbb{E}[\Delta Y \mid D=0, X]` and the propensity
    score :math:`\pi(X) = \mathbb{P}(D=1 \mid X)` give the doubly robust estimand

    .. math::

        \tau^{dr} = \mathbb{E}\left[\left(\frac{D}{\mathbb{E}[D]}
        - \frac{\pi(X)(1-D)/(1-\pi(X))}
        {\mathbb{E}[\pi(X)(1-D)/(1-\pi(X))]}\right)
        (\Delta Y - \mu_{0,\Delta}(X))\right].

    The :ref:`doubly robust DiD background <background-drdid>` develops both designs and
    the efficiency of each estimator.

    With repeated cross-sections, both methods also fit an outcome regression for the
    treated group in each period. When a period's treated group has fewer observations
    than coefficients, or a covariate constant or collinear within it, the function
    raises a ValueError that names the group and period. :func:`~moderndid.drdid_imp_rc`
    and :func:`~moderndid.drdid_trad_rc` leave those regressions out but are not locally
    efficient.

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

    Examples
    --------
    Estimate the ATT of a job training program on earnings with panel data.

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
        # drdid_imp_rc's improved estimator is not locally efficient. "imp" runs the one that is.
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
