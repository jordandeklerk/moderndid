"""Wrapper for doubly robust DiD estimators."""

from typing import Any, NamedTuple

import numpy as np

from moderndid.core.preprocess import preprocess_drdid

from .estimators.drdid_imp_local_rc import drdid_imp_local_rc
from .estimators.drdid_imp_panel import drdid_imp_panel
from .estimators.drdid_imp_rc import drdid_imp_rc
from .estimators.drdid_panel import drdid_panel
from .estimators.drdid_rc import drdid_rc
from .estimators.drdid_trad_rc import drdid_trad_rc
from .print import print_did_result


class DRDIDResult(NamedTuple):
    """Result from the doubly robust DiD estimator."""

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    call_params: dict[str, Any]
    args: dict[str, Any]


DRDIDResult = print_did_result(DRDIDResult)


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
    n_boot=999,
    inf_func=False,
    trim_level=0.995,
):
    r"""Wrap the locally efficient doubly robust DiD estimators for the ATT.

    This function is a wrapper for doubly robust difference-in-differences (DiD) estimators.
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
    est_method : {"imp", "trad", "imp_local", "trad_local"}, default "imp"
        The method to estimate the nuisance parameters.

        - "imp": Uses weighted least squares to estimate outcome regressions and
          inverse probability tilting to estimate the propensity score, leading to
          the improved locally efficient DR DiD estimator. For panel data, this
          corresponds to equation (3.1) in Sant'Anna and Zhao (2020). For repeated
          cross-sections, this uses a single propensity score model.
        - "trad": Uses OLS to estimate outcome regressions and maximum likelihood
          to estimate propensity score, leading to the "traditional" locally
          efficient DR DiD estimator.
        - "imp_local": For repeated cross-sections only. Implements the locally
          efficient estimator from equation (3.4) in Sant'Anna and Zhao (2020)
          with separate outcome regressions for each group and time period.
        - "trad_local": For repeated cross-sections only. Traditional DR DiD
          estimator from equation (3.3) in Sant'Anna and Zhao (2020) that is
          not locally efficient.
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
    DRDIDResult
        NamedTuple containing:

        - *att*: The DR DiD point estimate.
        - *se*: The DR DiD standard error.
        - *uci*: The upper bound of a 95% confidence interval.
        - *lci*: The lower bound of a 95% confidence interval.
        - *boots*: Bootstrap draws of the ATT if boot=True.
        - *att_inf_func*: Influence function values if inf_func=True.
        - *call_params*: Original function call parameters.
        - *args*: Arguments used in the estimation.

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

    For more robust inference, we can use bootstrapped standard errors with
    propensity score trimming to handle extreme weights.

    .. ipython::
        :okwarning:

        In [3]: att_result_boot = moderndid.drdid(
           ...:     data=nsw_data,
           ...:     yname="re",
           ...:     tname="year",
           ...:     idname="id",
           ...:     treatname="experimental",
           ...:     xformla="~ age + educ + black + married + nodegree + hisp + re74",
           ...:     panel=True,
           ...:     est_method="imp",
           ...:     boot=True,
           ...: )

        In [4]: print(att_result_boot)

    Notes
    -----
    When panel data are available (`panel=True`), the function implements the
    locally efficient doubly robust DiD estimator for the ATT defined in
    equation (3.1) in [2]_. This estimator makes use of a logistic propensity score
    model for the probability of being in the treated group, and of a linear regression
    model for the outcome evolution among the comparison units.

    When only stationary repeated cross-section data are available (`panel=False`),
    the function implements the locally efficient doubly robust DiD estimator
    for the ATT defined in equation (3.4) in [2]_. This estimator makes use of a
    logistic propensity score model for the probability of being in the treated group,
    and of (separate) linear regression models for the outcome of both treated and
    comparison units, in both pre and post-treatment periods.

    When `est_method="imp"` (the default), the nuisance parameters are estimated
    using the methods described in Sections 3.1 and 3.2 of [2]_.
    The propensity score parameters are estimated using the inverse probability
    tilting estimator proposed by [1]_, and the outcome regression coefficients are
    estimated using weighted least squares.

    When `est_method="trad"`, the propensity score parameters are estimated using
    maximum likelihood, and the outcome regression coefficients are estimated
    using ordinary least squares.

    The main advantage of using `est_method="imp"` is that the resulting estimator
    is not only locally efficient and doubly robust for the ATT, but it is also
    doubly robust for inference; see [2]_ for details.

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
        est_method=est_method,
        trim_level=trim_level,
    )

    if panel:
        if est_method in ["imp_local", "trad_local"]:
            raise ValueError(f"est_method '{est_method}' is only available for repeated cross-sections (panel=False)")

        if est_method == "imp":
            result = drdid_imp_panel(
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
            result = drdid_panel(
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
        if est_method == "imp":
            result = drdid_imp_rc(
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
        elif est_method == "trad":
            result = drdid_rc(
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
        elif est_method == "imp_local":
            result = drdid_imp_local_rc(
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
            result = drdid_trad_rc(
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
            "type": "dr",
            "trim_level": trim_level,
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
