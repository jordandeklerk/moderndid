# pylint: disable=unused-argument
"""Continuous treatment difference-in-differences estimation."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .npiv import gsl_bs, npiv
from .panel import (
    AttgtResult,
    pte,
    pte_default,
    setup_pte_cont,
)
from .panel.estimators import pte_attgt
from .panel.process_dose import DoseResult
from .panel.process_panel import _get_first_difference, _get_group, _make_balanced_panel
from .spline import BSpline


def cont_did(
    yname,
    dname,
    tname,
    idname,
    data,
    gname=None,
    xformula="~1",
    target_parameter="level",
    aggregation="dose",
    treatment_type="continuous",
    dose_est_method="parametric",
    dvals=None,
    degree=3,
    num_knots=0,
    allow_unbalanced_panel=False,
    control_group="notyettreated",
    anticipation=0,
    weightsname=None,
    alp=0.05,
    cband=False,
    boot_type="multiplier",
    biters=1000,
    clustervars=None,
    est_method=None,
    base_period="varying",
    **kwargs,
):
    r"""Compute difference-in-differences with a continuous treatment.

    Implements difference-in-differences estimation for settings with continuous treatment
    variables in a staggered treatment adoption framework. It extends traditional DiD methods
    to handle treatment intensity or dose, supporting both parametric estimation using B-splines
    and non-parametric estimation following the methods in [2]_.

    Parameters
    ----------
    yname : str
        Name of the column containing the outcome variable.
    dname : str
        Name of the column containing the continuous treatment variable.
        This should represent the "dose" or amount of treatment received,
        and should be constant across time periods for each unit.
        Use 0 for never-treated units.
    tname : str
        Name of the column containing the time period variable.
    idname : str
        Name of the column containing the unit ID variable.
    data : pd.DataFrame
        The input panel data containing outcome, treatment, time, unit ID,
        and optionally covariates and weights. Should be in long format with
        each row representing a unit-time observation.
    gname : str, optional
        Name of the column containing the timing-group variable indicating
        when treatment starts for each unit. If None, it will be computed
        from the treatment variable. Should be 0 for never-treated units.
    xformula : str, default="~1"
        A formula for the covariates to include in the model.
        Should be of the form "~ X1 + X2" (intercept is always included).
        Currently only "~1" (no covariates) is supported.
    target_parameter : {"level", "slope"}, default="level"
        Type of treatment effect to estimate:

        - "level": Average treatment effect (ATT) at different dose levels
        - "slope": Average causal response (ACRT), the derivative of the dose-response curve
    aggregation : {"dose", "eventstudy"}, default="dose"
        How to aggregate the treatment effects:

        - "dose": Average across timing-groups and time periods, report by dose
        - "eventstudy": Average across timing-groups and doses, report by event time
    treatment_type : {"continuous", "discrete"}, default="continuous"
        Nature of the treatment variable. Only "continuous" is currently supported.
    dose_est_method : {"parametric", "cck"}, default="parametric"
        Method for estimating dose-specific effects:

        - "parametric": Use B-splines with specified degree and knots
        - "cck": Use non-parametric method based on [2]_.
    dvals : array-like, optional
        Values of the treatment dose at which to compute effects.
        If None, uses quantiles of the dose distribution among treated units.
    degree : int, default=3
        Degree of the B-spline basis functions. Combined with num_knots=0 (default),
        this fits a global polynomial of the specified degree.
    num_knots : int, default=0
        Number of interior knots for the B-spline. More knots allow more
        flexibility but may increase variance.
    allow_unbalanced_panel : bool, default=False
        Whether to allow unbalanced panel data. Currently not supported.
    control_group : {"notyettreated", "nevertreated"}, default="notyettreated"
        Which units to use as controls:

        - "notyettreated": Units not yet treated by time t
        - "nevertreated": Only never-treated units
    anticipation : int, default=0
        Number of time periods before treatment where effects may appear.
    weightsname : str, optional
        Name of the column containing sampling weights.
        If None, all observations have equal weight.
    alp : float, default=0.05
        Significance level for confidence intervals (e.g., 0.05 for 95% CI).
    cband : bool, default=False
        Whether to compute uniform confidence bands over all dose values.
    boot_type : str, default="multiplier"
        Type of bootstrap to perform ("multiplier" or "empirical").
    biters : int, default=1000
        Number of bootstrap iterations for inference.
    clustervars : str, optional
        Variable(s) for clustering standard errors. Not currently supported.
    est_method : str, optional
        Estimation method for the outcome model. Must be None as covariates
        are not currently supported.
    base_period : {"varying", "universal"}, default="varying"
        How to choose the base period for comparisons:

        - "varying": Use different base periods for different timing groups
        - "universal": Use the same base period for all comparisons
    **kwargs
        Additional keyword arguments passed to internal functions.

    Returns
    -------
    DoseResult or PTEResult
        Results object containing:

        - **dose** : Array of dose values at which effects are evaluated
        - **att_d** : Dose-specific ATT estimates
        - **att_d_se** : Standard errors for dose-specific ATT
        - **acrt_d** : Dose-specific ACRT estimates (if target_parameter="slope")
        - **acrt_d_se** : Standard errors for dose-specific ACRT
        - **overall_att** : Overall average treatment effect
        - **overall_att_se** : Standard error for overall ATT
        - **overall_acrt** : Overall average causal response (if applicable)
        - **overall_acrt_se** : Standard error for overall ACRT

    Examples
    --------
    We can estimate the dose-response function for a continuous treatment using simulated
    data where units receive different treatment intensities at different times.

    First, we need to generate simulated data with a continuous treatment:

    .. ipython::
        :okwarning:

        In [1]: import numpy as np
           ...: import pandas as pd
           ...: import moderndid
           ...:
           ...: np.random.seed(42)
           ...: n = 1000
           ...: n_periods = 4
           ...: rng = np.random.default_rng(42)
           ...:
           ...: time_periods = np.arange(1, n_periods + 1)
           ...: groups = np.concatenate(([0], time_periods[1:]))
           ...: p = np.repeat(1/len(groups), len(groups))
           ...: group = rng.choice(groups, n, replace=True, p=p)
           ...: dose = rng.uniform(0, 1, n)
           ...:
           ...: eta = rng.normal(loc=group, scale=1, size=n)
           ...: time_effects = np.arange(1, n_periods + 1)
           ...: y0_t = time_effects + eta[:, np.newaxis] + rng.normal(size=(n, n_periods))
           ...:
           ...: dose_effect = 0.5
           ...: y1_t = (dose_effect * dose[:, np.newaxis] + time_effects +
           ...:         eta[:, np.newaxis] + rng.normal(size=(n, n_periods)))
           ...:
           ...: post_matrix = (group[:, np.newaxis] <= time_periods) & (group[:, np.newaxis] != 0)
           ...: y = post_matrix * y1_t + (1 - post_matrix) * y0_t
           ...:
           ...: df = pd.DataFrame(y, columns=[f"Y_{t}" for t in time_periods])
           ...: df["id"] = np.arange(1, n + 1)
           ...: df["G"] = group
           ...: df["D"] = dose
           ...:
           ...: df_long = pd.melt(df, id_vars=["id", "G", "D"],
           ...:                   value_vars=[f"Y_{t}" for t in time_periods],
           ...:                   var_name="time_period", value_name="Y")
           ...: df_long["time_period"] = df_long["time_period"].str.replace("Y_", "").astype(int)
           ...:
           ...: df_long.loc[df_long["G"] == 0, "D"] = 0
           ...: df_long.loc[df_long["time_period"] < df_long["G"], "D"] = 0
           ...: data = df_long.sort_values(["id", "time_period"]).reset_index(drop=True)

    Now we can estimate the dose-response function using the parametric estimator which
    uses B-splines to approximate the dose-response function:

    .. ipython::
        :okwarning:

        In [2]: dose_result = moderndid.cont_did(
           ...:     yname="Y",
           ...:     dname="D",
           ...:     tname="time_period",
           ...:     idname="id",
           ...:     gname="G",
           ...:     data=data,
           ...:     target_parameter="level",
           ...:     aggregation="dose",
           ...:     degree=3,
           ...:     biters=100
           ...: )
           ...: dose_result

    For the non-parametric CCK estimator, we need exactly 2 groups and 2 time periods,
    which we can simulate as follows:

    .. ipython::
        :okwarning:

        In [3]: np.random.seed(42)
           ...: n = 1000
           ...: rng = np.random.default_rng(42)
           ...:
           ...: group = rng.choice([0, 2], n, replace=True, p=[0.5, 0.5])
           ...: dose = rng.uniform(0, 1, n)
           ...: time_periods = [1, 2]
           ...:
           ...: eta = rng.normal(size=n)
           ...: y0_t = np.array(time_periods) + eta[:, np.newaxis] + rng.normal(size=(n, 2))
           ...:
           ...: dose_effect = 0.5
           ...: y1_t = (dose_effect * dose[:, np.newaxis] + np.array(time_periods) +
           ...:         eta[:, np.newaxis] + rng.normal(size=(n, 2)))
           ...:
           ...: post_matrix = (group[:, np.newaxis] == 2) & (np.array(time_periods) >= 2)
           ...: y = post_matrix * y1_t + (1 - post_matrix) * y0_t
           ...:
           ...: df_cck = pd.DataFrame(y, columns=["Y_1", "Y_2"])
           ...: df_cck["id"] = np.arange(1, n + 1)
           ...: df_cck["G"] = group
           ...: df_cck["D"] = dose
           ...:
           ...: df_cck_long = pd.melt(df_cck, id_vars=["id", "G", "D"],
           ...:                       value_vars=["Y_1", "Y_2"],
           ...:                       var_name="time_period", value_name="Y")
           ...: df_cck_long["time_period"] = df_cck_long["time_period"].str.replace("Y_", "").astype(int)
           ...:
           ...: df_cck_long.loc[df_cck_long["G"] == 0, "D"] = 0
           ...: df_cck_long.loc[df_cck_long["time_period"] < df_cck_long["G"], "D"] = 0
           ...: data_cck = df_cck_long.sort_values(["id", "time_period"]).reset_index(drop=True)

    Now we can estimate the dose-response function using the non-parametric CCK estimator,
    which is a non-parametric estimator that does not require any assumptions about the
    dose-response function:

    .. ipython::
        :okwarning:

        In [4]: cck_result = moderndid.cont_did(
           ...:     yname="Y",
           ...:     dname="D",
           ...:     tname="time_period",
           ...:     idname="id",
           ...:     gname="G",
           ...:     data=data_cck,
           ...:     dose_est_method="cck",
           ...:     target_parameter="level",
           ...:     aggregation="dose",
           ...:     biters=100
           ...: )
           ...: cck_result

    Notes
    -----
    The estimator accommodates settings where treatment intensity varies across
    treated units but remains constant over time for each unit. This is common
    in policy evaluations where different regions or units receive different
    levels of treatment (e.g., different minimum wage increases, varying subsidy
    amounts, or different policy intensities).

    The combination of `target_parameter` and `aggregation` determines the
    specific analysis:

    - `target_parameter="level"` + `aggregation="dose"`: Estimates ATT as a
      function of dose, showing how treatment effects vary with treatment intensity
    - `target_parameter="slope"` + `aggregation="dose"`: Estimates ACRT, the
      marginal effect of increasing treatment intensity
    - `target_parameter="level"` + `aggregation="eventstudy"`: Event study
      showing ATT over time, averaged across doses
    - `target_parameter="slope"` + `aggregation="eventstudy"`: Event study
      showing ACRT over time

    When using `dose_est_method="parametric"` (default), the dose-response function
    is approximated using B-splines. The flexibility of the approximation is
    controlled by the `degree` and `num_knots` parameters. With `num_knots=0`,
    a global polynomial of the specified degree is fit.

    When using `dose_est_method="cck"`, the non-parametric estimator from
    [2]_ is used. This method requires exactly 2 groups and 2 time periods and
    currently only supports `aggregation="dose"`.

    References
    ----------

    .. [1] Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. (2024).
           "Difference-in-differences with a continuous treatment."
           Journal of Econometrics, forthcoming.
           https://arxiv.org/abs/2107.02637

    .. [2] Chen, X., Christensen, T. M., & Kankanala, S. (2024).
           "Adaptive Estimation and Uniform Confidence Bands for Nonparametric
           Structural Functions and Elasticities."
           https://arxiv.org/abs/2107.11869
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    required_cols = [yname, dname, tname, idname]
    if gname is not None:
        required_cols.append(gname)

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    if xformula != "~1":
        raise NotImplementedError("Covariates not currently supported, use xformula='~1'")

    if treatment_type == "discrete":
        raise NotImplementedError("Discrete treatment not yet supported")

    if allow_unbalanced_panel:
        raise NotImplementedError("Unbalanced panel not currently supported")

    if clustervars is not None:
        warnings.warn("Two-way clustering not currently supported", UserWarning)

    if est_method is not None:
        raise ValueError("Covariates not supported yet, set est_method=None")

    if anticipation != 0:
        warnings.warn("Anticipation not fully tested yet, may not work correctly", UserWarning)

    if weightsname is not None:
        warnings.warn("Sampling weights not fully tested yet", UserWarning)

    if gname is None:
        data = data.copy()
        data[".G"] = _get_group(data, idname=idname, tname=tname, treatname=dname)
        gname = ".G"

    if dose_est_method == "cck":
        if aggregation != "dose":
            raise ValueError("Event study not supported with CCK estimator yet, use aggregation='dose'")

        return _cck_estimator(
            data=data,
            yname=yname,
            dname=dname,
            gname=gname,
            tname=tname,
            idname=idname,
            dvals=dvals,
            alp=alp,
            cband=cband,
            target_parameter=target_parameter,
            **kwargs,
        )

    if aggregation == "eventstudy":
        if target_parameter == "slope":
            attgt_fun = cont_did_acrt
            gt_type = "dose"
        else:
            attgt_fun = pte_attgt
            gt_type = "att"
    elif target_parameter in ["level", "slope"]:
        attgt_fun = cont_did_acrt
        gt_type = "dose"
    else:
        raise ValueError(f"Invalid combination of parameters: {target_parameter}, {aggregation}, {treatment_type}")

    pte_kwargs = kwargs.copy()
    if aggregation == "eventstudy":
        pte_kwargs["d_outcome"] = True

    return pte(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        setup_pte_fun=setup_pte_cont,
        subset_fun=cont_two_by_two_subset,
        attgt_fun=attgt_fun,
        xformula=xformula,
        target_parameter=target_parameter,
        aggregation=aggregation,
        treatment_type=treatment_type,
        dose_est_method=dose_est_method,
        anticipation=anticipation,
        gt_type=gt_type,
        cband=cband,
        alp=alp,
        boot_type=boot_type,
        biters=biters,
        dname=dname,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals,
        control_group=control_group,
        base_period=base_period,
        weightsname=weightsname,
        **pte_kwargs,
    )


def cont_did_acrt(gt_data, dvals=None, degree=3, knots=None, **kwargs):
    """Compute Average Causal Response on Treated (ACRT) for a timing group and period.

    Estimates dose-specific treatment effects using B-splines for a particular
    timing group and time period combination.

    Parameters
    ----------
    gt_data : pd.DataFrame
        Data subset for this group-time combination with columns:

        - id: Unit identifier
        - Y: Outcome variable
        - D: Treatment dose
        - period: Time period
        - name: "pre" or "post" indicator
    dvals : array-like, optional
        Dose values at which to evaluate effects. If None, uses quantiles
        of the treated dose distribution.
    degree : int, default=3
        Degree of the B-spline basis.
    knots : array-like, optional
        Interior knot positions for the B-spline. If None, uses quantiles
        based on the degree.
    **kwargs
        Additional arguments.

    Returns
    -------
    AttgtResult
        NamedTuple containing:

        - **attgt**: Overall ACRT estimate
        - **inf_func**: Influence function for inference
        - **extra_gt_returns**: Dictionary with detailed results including
          dose-specific ATT and ACRT estimates
    """
    gt_data["dy"] = _get_first_difference(gt_data, "id", "Y", "period")

    post_data = gt_data[gt_data["name"] == "post"].copy()
    dose = post_data["D"].values
    dy = post_data["dy"].values

    if dvals is None or len(dvals) == 0:
        return AttgtResult(attgt=0.0, inf_func=np.zeros(len(post_data)), extra_gt_returns=None)

    treated_mask = dose > 0
    if not np.any(treated_mask):
        return AttgtResult(attgt=0.0, inf_func=np.zeros(len(post_data)), extra_gt_returns=None)

    positive_doses = dose[treated_mask]
    if len(np.unique(positive_doses)) < 2:
        return AttgtResult(attgt=0.0, inf_func=np.zeros(len(post_data)), extra_gt_returns=None)

    boundary_knots = [np.min(positive_doses), np.max(positive_doses)]
    if len(np.unique(boundary_knots)) < 2:
        boundary_knots = None

    bspline_treated = BSpline(x=dose[treated_mask], degree=degree, internal_knots=knots, boundary_knots=boundary_knots)
    x_treated = bspline_treated.basis(complete_basis=False)
    y_treated = dy[treated_mask]

    x_treated = np.column_stack([np.ones(x_treated.shape[0]), x_treated])

    try:
        model = sm.OLS(y_treated, x_treated)
        results = model.fit()
        coef = results.params
    except (np.linalg.LinAlgError, ValueError):
        return AttgtResult(attgt=0.0, inf_func=np.zeros(len(post_data)), extra_gt_returns=None)

    bspline_grid = BSpline(x=dvals, degree=degree, internal_knots=knots, boundary_knots=boundary_knots)
    x_grid = bspline_grid.basis(complete_basis=False)
    x_grid = np.column_stack([np.ones(x_grid.shape[0]), x_grid])
    att_d = x_grid @ coef - np.mean(dy[dose == 0])

    x_deriv = bspline_grid.derivative(derivs=1, complete_basis=False)
    acrt_d = x_deriv @ coef[1:]

    x_overall = x_treated
    att_overall = np.mean(x_overall @ coef) - np.mean(dy[dose == 0])

    x_deriv_overall = bspline_treated.derivative(derivs=1, complete_basis=False)
    acrt_overall = np.mean(x_deriv_overall @ coef[1:])

    inf_func1 = x_deriv_overall @ coef[1:] - acrt_overall

    score = results.resid[:, np.newaxis] * x_treated
    bread = np.linalg.inv(x_treated.T @ x_treated / len(x_treated))
    avg_deriv = np.mean(x_deriv_overall, axis=0)
    inf_func2 = score @ bread @ np.concatenate([[0], avg_deriv])

    inf_func = np.zeros(len(post_data))
    inf_func[treated_mask] = inf_func1 + inf_func2

    extra_gt_returns = {
        "att_d": att_d,
        "acrt_d": acrt_d,
        "att_overall": att_overall,
        "acrt_overall": acrt_overall,
        "dvals": dvals,
        "coef": coef,
        "bread": bread,
        "score": score,
    }

    return AttgtResult(attgt=acrt_overall, inf_func=inf_func, extra_gt_returns=extra_gt_returns)


def cont_two_by_two_subset(
    data,
    g,
    tp,
    control_group="notyettreated",
    anticipation=0,
    base_period="varying",
    **kwargs,
):
    """Create a two-by-two subset for continuous treatment DiD."""
    main_base_period = g - anticipation - 1

    if base_period == "varying":
        if tp < (g - anticipation):
            base_period_val = tp - 1
        else:
            base_period_val = main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        unit_mask = (data["G"] == g) | (data["G"] > tp) | (data["G"] == 0)
    else:  # 'nevertreated'
        unit_mask = (data["G"] == g) | (data["G"] == 0)

    subset_data = data.loc[unit_mask].copy()

    time_mask = (subset_data["period"] == tp) | (subset_data["period"] == base_period_val)
    subset_data = subset_data.loc[time_mask].copy()

    subset_data["name"] = np.where(subset_data["period"] == tp, "post", "pre")
    subset_data["D"] = subset_data["D"] * (subset_data["G"] == g)

    n1 = subset_data["id"].nunique()
    all_ids = data["id"].unique()
    subset_ids = subset_data["id"].unique()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": subset_data, "n1": n1, "disidx": disidx}


def _cck_estimator(data, yname, dname, gname, tname, idname, dvals, alp, cband, target_parameter, **kwargs):
    """Compute the CCK non-parametric estimator for continuous treatment."""
    unique_groups = data[gname].unique()
    unique_times = data[tname].unique()

    if len(unique_groups) != 2 or len(unique_times) != 2:
        raise ValueError(
            f"CCK estimator requires exactly 2 groups and 2 time periods "
            f"(found {len(unique_groups)} groups and {len(unique_times)} periods)"
        )

    data = _make_balanced_panel(data, idname, tname)
    data[".dy"] = _get_first_difference(data, idname, yname, tname)

    max_t = data[tname].max()
    post_data = data[data[tname] == max_t].copy()

    dose = post_data[dname].values
    dy = post_data[".dy"].values

    m0 = np.mean(dy[dose == 0])
    dy_centered = dy - m0

    if dvals is None:
        positive_doses = dose[dose > 0]
        if len(positive_doses) > 0:
            dvals = np.linspace(positive_doses.min(), positive_doses.max(), 50)
        else:
            raise ValueError("No treated units found")

    dvals = np.asarray(dvals).reshape(-1, 1)

    cck_res = npiv(
        y=dy_centered[dose > 0],
        x=dose[dose > 0].reshape(-1, 1),
        w=dose[dose > 0].reshape(-1, 1),
        x_grid=dvals,
        knots="quantiles",
        boot_num=999,
        j_x_degree=3,
        k_w_degree=3,
    )

    att_d = cck_res.h
    att_d_se = cck_res.asy_se

    if cband:
        att_d_crit_val = (
            (cck_res.h_upper[0] - att_d[0]) / att_d_se[0] if att_d_se[0] > 0 else stats.norm.ppf(1 - alp / 2)
        )
    else:
        att_d_crit_val = stats.norm.ppf(1 - alp / 2)

    acrt_d = cck_res.deriv if hasattr(cck_res, "deriv") else np.gradient(att_d, dvals.flatten())
    acrt_d_se = cck_res.deriv_asy_se if hasattr(cck_res, "deriv_asy_se") else np.full_like(acrt_d, np.nan)

    if cband and hasattr(cck_res, "h_upper_deriv") and acrt_d_se[0] > 0:
        acrt_d_crit_val = (cck_res.h_upper_deriv[0] - acrt_d[0]) / acrt_d_se[0]
    else:
        acrt_d_crit_val = att_d_crit_val

    ptep = setup_pte_cont(
        yname=yname,
        gname=gname,
        tname=tname,
        idname=idname,
        data=data,
        dname=dname,
        target_parameter=target_parameter,
        aggregation="dose",
        treatment_type="continuous",
        dose_est_method="cck",
        cband=cband,
        alp=alp,
        boot_type="multiplier",
        gt_type="att",
        **kwargs,
    )

    overall_att_res = pte_default(
        yname=ptep.yname,
        gname=ptep.gname,
        tname=ptep.tname,
        idname=ptep.idname,
        data=ptep.data,
        d_outcome=True,
        anticipation=ptep.anticipation,
        base_period=ptep.base_period,
        control_group=ptep.control_group,
        weightsname=ptep.weightsname,
        boot_type="multiplier",
        biters=ptep.biters,
        alp=ptep.alp,
    )

    w_treated = dose[dose > 0]
    n_breaks = cck_res.j_x_segments + 1

    knots = np.quantile(w_treated, np.linspace(0, 1, n_breaks))
    spline_dosage_result = gsl_bs(
        w_treated,
        degree=cck_res.j_x_degree,
        knots=knots,
        nbreak=n_breaks,
        deriv=0,
        intercept=True,
    )
    spline_dosage = spline_dosage_result.basis

    y_treated = dy_centered[dose > 0]
    n_treated_val = len(y_treated)

    beta_array = cck_res.beta.flatten() if cck_res.beta.ndim > 1 else cck_res.beta

    h_hat_w_treated = spline_dosage @ beta_array
    infl_reg = (y_treated - h_hat_w_treated.flatten())[:, np.newaxis] * (
        spline_dosage @ np.linalg.pinv(spline_dosage.T @ spline_dosage / n_treated_val)
    )

    deriv_spline_basis_w = gsl_bs(
        w_treated,
        degree=cck_res.j_x_degree,
        knots=knots,
        nbreak=n_breaks,
        deriv=1,
        intercept=True,
    ).basis

    average_spline_deriv = deriv_spline_basis_w.mean(axis=0)
    deriv_at_w = (deriv_spline_basis_w @ beta_array).flatten()

    average_acr = np.mean(deriv_at_w)
    infl_avg_acr = (deriv_at_w - average_acr) + infl_reg @ average_spline_deriv
    se_avg_acr = np.std(infl_avg_acr) / np.sqrt(n_treated_val)

    overall_att = overall_att_res.overall_att.overall_att
    overall_att_se = overall_att_res.overall_att.overall_se
    overall_att_inf_func = overall_att_res.overall_att.influence_func

    result = DoseResult(
        dose=dvals.flatten() if dvals.ndim > 1 else dvals,
        overall_att=overall_att,
        overall_att_se=overall_att_se,
        overall_att_inf_func=overall_att_inf_func,
        overall_acrt=average_acr,
        overall_acrt_se=se_avg_acr,
        overall_acrt_inf_func=infl_avg_acr,
        att_d=att_d,
        att_d_se=att_d_se,
        att_d_crit_val=att_d_crit_val,
        att_d_inf_func=None,
        acrt_d=acrt_d,
        acrt_d_se=acrt_d_se,
        acrt_d_crit_val=acrt_d_crit_val,
        acrt_d_inf_func=None,
        pte_params=ptep,
    )

    return result
