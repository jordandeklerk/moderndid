"""Wrapper for inverse propensity weighted DiD estimators."""

from typing import Any, NamedTuple

import numpy as np

from moderndid.core.preprocess import preprocess_drdid

from .estimators.ipw_did_panel import ipw_did_panel
from .estimators.ipw_did_rc import ipw_did_rc
from .estimators.std_ipw_did_panel import std_ipw_did_panel
from .estimators.std_ipw_did_rc import std_ipw_did_rc
from .print import print_did_result


class IPWDIDResult(NamedTuple):
    """Result from the inverse propensity weighted DiD estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    call_params: dict[str, Any]
    args: dict[str, Any]


IPWDIDResult = print_did_result(IPWDIDResult)


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
    n_boot=999,
    inf_func=False,
    trim_level=0.995,
):
    r"""Wrap the inverse propensity weighted DiD estimators for the ATT.

    This function is a wrapper for inverse propensity weighted (IPW) DiD estimators.
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
        Name of the column containing the treatment group indicator.
        For panel data: time-invariant indicator (1 if ever treated, 0 if never treated).
        For repeated cross-sections: treatment status in the post-period.
    xformla : str | None, default None
        A formula for the covariates to include in the model.
        Should be of the form "~ X1 + X2" (intercept is always included).
        If None, equivalent to "~ 1" (intercept only).
    panel : bool, default True
        Whether the data is panel (True) or repeated cross-sections (False).
        Panel data should be in long format with each row representing
        a unit-time observation.
    est_method : {"ipw", "std_ipw"}, default "ipw"
        The IPW estimation method to use.

        - "ipw": Standard inverse propensity weighted estimator (Horvitz-Thompson type).
          Weights are not normalized to sum to one. This is based on Abadie (2005).
        - "std_ipw": Standardized (Hajek-type) inverse propensity weighted estimator.
          Weights are normalized to sum to one, which can improve finite sample
          performance when propensity scores are close to 0 or 1.
    weightsname : str | None, default None
        Name of the column containing sampling weights.
        If None, all observations have equal weight.
        Weights are normalized to have mean 1.
    boot : bool, default False
        Whether to compute bootstrap standard errors. If False,
        analytical standard errors are reported.
    boot_type : {"weighted", "multiplier"}, default "weighted"
        Type of bootstrap to perform (only relevant if boot=True).
    n_boot : int, default 999
        Number of bootstrap repetitions (only relevant if boot=True).
    inf_func : bool, default False
        Whether to return the influence function values.
    trim_level : float, default 0.995
        The level of trimming for the propensity score.

    Returns
    -------
    IPWDIDResult
        NamedTuple containing:

        - *att*: The IPW DiD point estimate.
        - *se*: The IPW DiD standard error.
        - *uci*: The upper bound of a 95% confidence interval.
        - *lci*: The lower bound of a 95% confidence interval.
        - *boots*: Bootstrap draws of the ATT if boot=True.
        - *att_inf_func*: Influence function values if inf_func=True.
        - *call_params*: Original function call parameters.
        - *args*: Arguments used in the estimation.

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

    We can also use the standardized (Hajek-type) IPW estimator, which normalizes
    weights to sum to one and can be more stable with extreme propensity scores.

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

    For more robust inference, we can use weighted-bootstrap standard errors with
    propensity score trimming to handle extreme weights.

    .. ipython::
        :okwarning:

        In [5]: att_result_boot = moderndid.ipwdid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...:     est_method="ipw",
           ...:     boot=True,
           ...: )

        In [6]: print(att_result_boot)

    Notes
    -----
    The IPW estimator uses the propensity score (probability of being in the treated
    group) to reweight observations and create a balanced comparison between treated
    and control units. The standard IPW estimator ("ipw") uses unnormalized weights,
    while the standardized version ("std_ipw") normalizes weights to sum to one within
    each group, which can improve performance when propensity scores are extreme.

    Unlike doubly robust methods, IPW estimators are not robust to misspecification
    of the propensity score model. However, they can be more efficient when the
    propensity score model is correctly specified and there is substantial overlap
    between treated and control groups.

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
        "n_boot": n_boot,
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
        bstrap=boot,
        boot_type=boot_type,
        biters=n_boot,
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
                nboot=n_boot,
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
                nboot=n_boot,
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
                nboot=n_boot,
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
                nboot=n_boot,
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
            "nboot": n_boot,
            "type": "ipw",
            "trim_level": trim_level,
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
