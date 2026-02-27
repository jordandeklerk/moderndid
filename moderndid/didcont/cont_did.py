"""Continuous treatment difference-in-differences estimation."""

import warnings
from functools import partial

import numpy as np
import polars as pl
from scipy import stats

from moderndid.core.dataframe import to_polars
from moderndid.core.preprocess import (
    get_first_difference as _get_first_difference,
)
from moderndid.core.preprocess import (
    get_group,
)
from moderndid.core.preprocess import (
    make_balanced_panel as _make_balanced_panel,
)
from moderndid.core.preprocessing import preprocess_cont_did
from moderndid.cupy.backend import get_backend, to_device, to_numpy, use_backend

from .estimation import (
    AttgtResult,
    _build_pte_params,
    pte,
    pte_default,
)
from .estimation.estimators import pte_attgt
from .estimation.process_dose import DoseResult
from .npiv import gsl_bs, npiv
from .spline import BSpline


def cont_did(
    data,
    yname,
    tname,
    idname,
    gname=None,
    dname=None,
    xformla="~1",
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
    boot=False,
    boot_type="multiplier",
    biters=1000,
    clustervars=None,
    base_period="varying",
    random_state=None,
    n_partitions=None,
    backend=None,
    **kwargs,
):
    r"""Compute difference-in-differences with a continuous treatment.

    Implements difference-in-differences estimation for settings where treatment
    intensity varies across units but remains constant over time for each unit,
    following [1]_.

    With continuous treatments, two distinct causal parameters are of interest.
    The average treatment effect on the treated at dose :math:`d`, denoted
    :math:`ATT(d|d)`, measures the effect of receiving dose :math:`d` compared
    to no treatment among units that actually received dose :math:`d`

    .. math::

        ATT(d|d) = \mathbb{E}[Y_{t}(d) - Y_{t}(0) \mid D = d].

    The average causal response on the treated, :math:`ACRT(d|d)`, measures the
    marginal effect of increasing the dose, i.e., the slope of the dose-response
    function

    .. math::

        ACRT(d|d) = \left.\frac{\partial}{\partial l}
        \mathbb{E}[Y_{t}(l) \mid D = d]\right|_{l=d}.

    Under a parallel trends assumption, the :math:`ATT(d|d)` is identified by
    comparing outcome changes between dose group :math:`d` and the untreated

    .. math::

        ATT(d|d) = \mathbb{E}[\Delta Y \mid D = d] - \mathbb{E}[\Delta Y \mid D = 0].

    Aggregating over the dose distribution among treated units yields the overall
    average treatment effect and average causal response

    .. math::

        ATT^o = \mathbb{E}[ATT(D|D) \mid D > 0], \quad
        ACRT^o = \mathbb{E}[ACRT(D|D) \mid D > 0].

    Parameters
    ----------
    data : DataFrame
        Panel data in long format. Accepts any object implementing the Arrow
        PyCapsule Interface (``__arrow_c_stream__``), including polars, pandas,
        pyarrow Table, and cudf DataFrames.
    yname : str
        Name of the column containing the outcome variable.
    tname : str
        Name of the column containing the time period variable.
    idname : str
        Name of the column containing the unit ID variable.
    gname : str, optional
        Name of the column containing the timing-group variable indicating
        when treatment starts for each unit. If None, it will be computed
        from the treatment variable. Should be 0 for never-treated units.
    dname : str
        Name of the column containing the continuous treatment variable.
        This should represent the "dose" or amount of treatment received,
        and should be constant across time periods for each unit.
        Use 0 for never-treated units.
    xformla : str, default="~1"
        A formula for the covariates to include in the model.
        Should be of the form "~ X1 + X2" (intercept is always included).
        Currently only "~1" (no covariates) is supported.
    target_parameter : {"level", "slope"}, default="level"
        Type of treatment effect to focus on:

        - "level": Average treatment effect (ATT) at different dose levels
        - "slope": Average causal response (ACRT), the derivative of the dose-response curve

        For ``aggregation="dose"``, both ATT(d) and ACRT(d) are always computed
        and reported regardless of this setting. This parameter mainly affects
        ``aggregation="eventstudy"``, where it determines whether to aggregate
        ATT or ACRT over event time.
    aggregation : {"dose", "eventstudy"}, default="dose"
        How to aggregate the treatment effects:

        - "dose": Average across timing-groups and time periods, report by dose.
          Both ATT(d) and ACRT(d) curves are returned.
        - "eventstudy": Average across timing-groups and doses, report by event time.
          Returns ATT or ACRT by event time depending on ``target_parameter``.
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
    boot : bool, default=False
        Whether to use bootstrap inference. If False, uses analytical
        standard errors.
    boot_type : str, default="multiplier"
        Type of bootstrap to perform ("multiplier" or "empirical").
        Only used when ``boot=True``.
    biters : int, default=1000
        Number of bootstrap iterations for inference. Only used when
        ``boot=True``.
    clustervars : str, optional
        Variable(s) for clustering standard errors. Not currently supported.
    base_period : {"varying", "universal"}, default="varying"
        How to choose the base period for comparisons:

        - "varying": Use different base periods for different timing groups
        - "universal": Use the same base period for all comparisons
    random_state : int, Generator, optional
        Controls the randomness of the bootstrap. Pass an int for reproducible
        results across multiple function calls. Can also accept a NumPy
        ``Generator`` instance.
    n_partitions : int, optional
        Number of partitions for distributed computation when ``data`` is a
        Dask or Spark DataFrame. If ``None``, defaults to the framework's
        default parallelism.
    backend : {"numpy", "cupy"} or None, default=None
        Array backend to use for this call only. When set, the backend is
        activated before estimation and the previous backend is restored
        when the call returns. ``None`` (the default) uses whatever backend
        is currently active (see :func:`~moderndid.set_backend`). Ignored
        when ``data`` is a Dask or Spark DataFrame.
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
    Estimate the dose-response function using simulated data with continuous treatment:

    .. ipython::
        :okwarning:

        In [1]: import moderndid
           ...: data = moderndid.simulate_cont_did_data(n=500, seed=42)
           ...: data.head()

    Estimate ATT as a function of dose using the parametric (B-spline) estimator:

    .. ipython::
        :okwarning:

        In [2]: result = moderndid.cont_did(
           ...:     data=data,
           ...:     yname="Y",
           ...:     tname="time_period",
           ...:     idname="id",
           ...:     gname="G",
           ...:     dname="D",
           ...:     target_parameter="level",
           ...:     aggregation="dose",
           ...:     degree=3,
           ...:     biters=100
           ...: )
           ...: result

    For the non-parametric CCK estimator, we need exactly 2 groups and 2 time periods:

    .. ipython::
        :okwarning:

        In [3]: data_cck = moderndid.simulate_cont_did_data(
           ...:     n=500, num_time_periods=2, seed=42
           ...: )
           ...: cck_result = moderndid.cont_did(
           ...:     data=data_cck,
           ...:     yname="Y",
           ...:     tname="time_period",
           ...:     idname="id",
           ...:     gname="G",
           ...:     dname="D",
           ...:     dose_est_method="cck",
           ...:     target_parameter="level",
           ...:     aggregation="dose",
           ...:     biters=100
           ...: )
           ...: cck_result

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
    if backend is not None:
        with use_backend(backend):
            return cont_did(
                data=data,
                yname=yname,
                tname=tname,
                idname=idname,
                gname=gname,
                dname=dname,
                xformla=xformla,
                target_parameter=target_parameter,
                aggregation=aggregation,
                treatment_type=treatment_type,
                dose_est_method=dose_est_method,
                dvals=dvals,
                degree=degree,
                num_knots=num_knots,
                allow_unbalanced_panel=allow_unbalanced_panel,
                control_group=control_group,
                anticipation=anticipation,
                weightsname=weightsname,
                alp=alp,
                cband=cband,
                boot=boot,
                boot_type=boot_type,
                biters=biters,
                clustervars=clustervars,
                base_period=base_period,
                random_state=random_state,
                n_partitions=n_partitions,
                backend=None,
                **kwargs,
            )

    if dname is None:
        raise ValueError("dname is required. Please specify the dose/treatment column.")

    if aggregation not in ("dose", "eventstudy"):
        raise ValueError(f"aggregation='{aggregation}' is not valid. Must be 'dose' or 'eventstudy'.")
    if target_parameter not in ("level", "slope"):
        raise ValueError(f"target_parameter='{target_parameter}' is not valid. Must be 'level' or 'slope'.")
    if dose_est_method not in ("parametric", "cck"):
        raise ValueError(f"dose_est_method='{dose_est_method}' is not valid. Must be 'parametric' or 'cck'.")
    if control_group not in ("notyettreated", "nevertreated"):
        raise ValueError(f"control_group='{control_group}' is not valid. Must be 'notyettreated' or 'nevertreated'.")
    if isinstance(base_period, str) and base_period not in ("varying", "universal"):
        raise ValueError(f"base_period='{base_period}' is not valid. Must be 'varying' or 'universal'.")
    if not 0 < alp < 1:
        raise ValueError(f"alp={alp} is not valid. Must be between 0 and 1 (exclusive).")
    if not isinstance(biters, int) or biters < 1:
        raise ValueError(f"biters={biters} is not valid. Must be a positive integer.")
    if boot_type not in ("weighted", "multiplier", "empirical"):
        raise ValueError(f"boot_type='{boot_type}' is not valid. Must be 'weighted', 'multiplier', or 'empirical'.")
    if not isinstance(anticipation, int | float) or anticipation < 0:
        raise ValueError(f"anticipation={anticipation} is not valid. Must be a non-negative number.")
    if degree < 1:
        raise ValueError(f"degree={degree} is not valid. Must be at least 1.")
    if num_knots < 0:
        raise ValueError(f"num_knots={num_knots} is not valid. Must be non-negative.")
    if treatment_type not in ("continuous", "discrete"):
        raise ValueError(f"treatment_type='{treatment_type}' is not valid. Must be 'continuous' or 'discrete'.")

    from moderndid.dask._utils import is_dask_collection

    if is_dask_collection(data):
        from moderndid.dask._didcont import dask_cont_did

        return dask_cont_did(
            data,
            yname,
            tname,
            idname,
            gname=gname,
            dname=dname,
            xformla=xformla,
            target_parameter=target_parameter,
            aggregation=aggregation,
            treatment_type=treatment_type,
            dose_est_method=dose_est_method,
            dvals=dvals,
            degree=degree,
            num_knots=num_knots,
            allow_unbalanced_panel=allow_unbalanced_panel,
            control_group=control_group,
            anticipation=anticipation,
            weightsname=weightsname,
            alp=alp,
            cband=cband,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            clustervars=clustervars,
            base_period=base_period,
            random_state=random_state,
            n_partitions=n_partitions,
            **kwargs,
        )

    from moderndid.spark._utils import is_spark_dataframe

    if is_spark_dataframe(data):
        from moderndid.spark._didcont import spark_cont_did

        return spark_cont_did(
            data,
            yname,
            tname,
            idname,
            gname=gname,
            dname=dname,
            xformla=xformla,
            target_parameter=target_parameter,
            aggregation=aggregation,
            treatment_type=treatment_type,
            dose_est_method=dose_est_method,
            dvals=dvals,
            degree=degree,
            num_knots=num_knots,
            allow_unbalanced_panel=allow_unbalanced_panel,
            control_group=control_group,
            anticipation=anticipation,
            weightsname=weightsname,
            alp=alp,
            cband=cband,
            boot=boot,
            boot_type=boot_type,
            biters=biters,
            clustervars=clustervars,
            base_period=base_period,
            random_state=random_state,
            n_partitions=n_partitions,
            **kwargs,
        )

    data = to_polars(data)

    if xformla != "~1":
        raise NotImplementedError("Covariates not currently supported, use xformla='~1'")

    if treatment_type == "discrete":
        raise NotImplementedError("Discrete treatment not yet supported")

    if allow_unbalanced_panel:
        raise NotImplementedError("Unbalanced panel not currently supported")

    if clustervars is not None:
        warnings.warn("Two-way clustering not currently supported", UserWarning)
        clustervars = None

    if anticipation != 0:
        warnings.warn("Anticipation not fully tested yet, may not work correctly", UserWarning)

    if weightsname is not None:
        warnings.warn("Sampling weights not fully tested yet", UserWarning)

    if dose_est_method == "cck" and aggregation != "dose":
        raise ValueError("Event study not supported with CCK estimator yet, use aggregation='dose'")

    missing_cols = []
    required_cols = [yname, dname, tname, idname]
    for col in required_cols:
        if col not in data.columns:
            missing_cols.append(col)
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    if gname is None:
        data = get_group(data, idname=idname, tname=tname, treatname=dname)
        data = data.rename({"G": ".G"})
        gname = ".G"

    req_pre_periods = 0 if dose_est_method == "cck" else 1

    cont_did_data = preprocess_cont_did(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        dname=dname,
        idname=idname,
        xformla=xformla,
        panel=True,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals,
        target_parameter=target_parameter,
        aggregation=aggregation,
        base_period=base_period,
        boot_type=boot_type,
        required_pre_periods=req_pre_periods,
        dose_est_method=dose_est_method,
    )

    if dose_est_method == "cck":
        return _estimate_cck(
            cont_did_data=cont_did_data,
            original_data=data,
            random_state=random_state,
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

    setup_fn = partial(_build_pte_params, cont_did_data, gt_type=gt_type)

    return pte(
        yname=cont_did_data.config.yname,
        gname=cont_did_data.config.gname,
        tname=cont_did_data.config.tname,
        idname=cont_did_data.config.idname,
        data=cont_did_data.data,
        setup_pte_fun=setup_fn,
        subset_fun=cont_two_by_two_subset,
        attgt_fun=attgt_fun,
        xformla=xformla,
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
        random_state=random_state,
        **pte_kwargs,
    )


def cont_did_acrt(gt_data, dvals=None, degree=3, knots=None, **kwargs):
    """Compute Average Causal Response on Treated (ACRT) for a timing group and period.

    Estimates dose-specific treatment effects using B-splines for a particular
    timing group and time period combination.

    Parameters
    ----------
    gt_data : pl.DataFrame
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
    gt_data = _get_first_difference(gt_data, "id", "Y", "period")

    post_data = gt_data.filter(pl.col("name") == "post")
    dose = post_data["D"].to_numpy()
    dy = post_data["dy"].to_numpy()

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

    control_mean = np.mean(dy[dose == 0])

    bspline_treated = BSpline(x=dose[treated_mask], degree=degree, internal_knots=knots, boundary_knots=boundary_knots)
    x_treated = np.asarray(bspline_treated.basis(complete_basis=False))
    y_treated = dy[treated_mask]

    x_treated = np.column_stack([np.ones(x_treated.shape[0]), x_treated])

    try:
        coef = np.linalg.lstsq(x_treated, y_treated, rcond=None)[0]
        resid = y_treated - x_treated @ coef
    except (np.linalg.LinAlgError, ValueError):
        return AttgtResult(attgt=0.0, inf_func=np.zeros(len(post_data)), extra_gt_returns=None)

    bspline_grid = BSpline(x=dvals, degree=degree, internal_knots=knots, boundary_knots=boundary_knots)
    x_grid = np.asarray(bspline_grid.basis(complete_basis=False))
    x_grid = np.column_stack([np.ones(x_grid.shape[0]), x_grid])
    att_d = x_grid @ coef - control_mean

    x_deriv = np.asarray(bspline_grid.derivative(derivs=1, complete_basis=False))
    acrt_d = x_deriv @ coef[1:]

    x_overall = x_treated
    att_overall = float(np.mean(x_overall @ coef) - control_mean)

    x_deriv_overall = np.asarray(bspline_treated.derivative(derivs=1, complete_basis=False))
    acrt_overall = float(np.mean(x_deriv_overall @ coef[1:]))

    inf_func1 = x_deriv_overall @ coef[1:] - acrt_overall

    score = resid[:, None] * x_treated
    n_treated = len(x_treated)
    bread = np.linalg.inv(x_treated.T @ x_treated / n_treated)

    x_expanded = score
    avg_deriv = np.mean(x_deriv_overall, axis=0)
    inf_func2 = score @ bread @ np.concatenate([np.zeros(1), avg_deriv])

    inf_func = np.zeros(len(post_data))
    inf_func[treated_mask] = inf_func1 + inf_func2

    extra_gt_returns = {
        "att_d": att_d,
        "acrt_d": acrt_d,
        "att_overall": att_overall,
        "acrt_overall": acrt_overall,
        "dvals": np.asarray(dvals) if hasattr(dvals, "__array__") else dvals,
        "coef": coef,
        "bread": bread,
        "x_expanded": x_expanded,
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
        base_period_val = tp - 1 if tp < g - anticipation else main_base_period
    else:
        base_period_val = main_base_period

    if control_group == "notyettreated":
        unit_mask = (pl.col("G") == g) | (pl.col("G") > tp)
    else:
        unit_mask = (pl.col("G") == g) | pl.col("G").is_infinite()

    subset_data = data.filter(unit_mask)
    time_mask = (pl.col("period") == tp) | (pl.col("period") == base_period_val)
    subset_data = subset_data.filter(time_mask)
    subset_data = subset_data.with_columns(
        pl.when(pl.col("period") == tp).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("name")
    )

    subset_data = subset_data.with_columns((pl.col("D") * (pl.col("G") == g).cast(pl.Float64)).alias("D"))

    n1 = subset_data["id"].n_unique()
    all_ids = data["id"].unique().to_numpy()
    subset_ids = subset_data["id"].unique().to_numpy()
    disidx = np.isin(all_ids, subset_ids)

    return {"gt_data": subset_data, "n1": n1, "disidx": disidx}


def _estimate_cck(cont_did_data, original_data, random_state=None, **kwargs):
    """Compute the CCK non-parametric estimator."""
    config = cont_did_data.config
    data = cont_did_data.data.clone()

    unique_groups = config.treated_groups
    unique_times = config.time_periods

    n_groups = len(unique_groups) + 1
    if n_groups != 2 or len(unique_times) != 2:
        raise ValueError(
            f"CCK estimator requires exactly 2 groups and 2 time periods "
            f"(found {n_groups} groups and {len(unique_times)} periods)"
        )

    data = _make_balanced_panel(data, config.idname, config.tname)
    data = _get_first_difference(data, config.idname, config.yname, config.tname)
    data = data.rename({"dy": ".dy"})

    max_t = data[config.tname].max()
    post_data = data.filter(pl.col(config.tname) == max_t)

    dose = post_data[config.dname].to_numpy()
    dy = post_data[".dy"].to_numpy()

    m0 = np.mean(dy[dose == 0])
    dy_centered = dy - m0

    dvals = config.dvals
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
        seed=random_state,
    )

    att_d = cck_res.h
    att_d_se = cck_res.asy_se

    alp = config.alp
    cband = config.cband

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

    ptep = _build_pte_params(cont_did_data, gt_type="att")

    overall_att_res = pte_default(
        yname=config.yname,
        gname=config.gname,
        tname=config.tname,
        idname=config.idname,
        data=original_data,
        d_outcome=True,
        anticipation=config.anticipation,
        base_period=config.base_period.value if hasattr(config.base_period, "value") else config.base_period,
        control_group=config.control_group.value if hasattr(config.control_group, "value") else config.control_group,
        weightsname=config.weightsname,
        boot_type="multiplier",
        biters=config.biters,
        alp=config.alp,
        random_state=random_state,
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

    xp = get_backend()

    spline_dosage = to_device(spline_dosage)
    y_treated_dev = to_device(y_treated)
    beta_array = to_device(cck_res.beta.flatten() if cck_res.beta.ndim > 1 else cck_res.beta)

    h_hat_w_treated = spline_dosage @ beta_array
    infl_reg = (y_treated_dev - h_hat_w_treated.flatten())[:, None] * (
        spline_dosage @ xp.linalg.pinv(spline_dosage.T @ spline_dosage / n_treated_val)
    )

    deriv_spline_basis_w = to_device(
        gsl_bs(
            w_treated,
            degree=cck_res.j_x_degree,
            knots=knots,
            nbreak=n_breaks,
            deriv=1,
            intercept=True,
        ).basis
    )

    average_spline_deriv = xp.mean(deriv_spline_basis_w, axis=0)
    deriv_at_w = (deriv_spline_basis_w @ beta_array).flatten()

    average_acr = float(xp.mean(deriv_at_w))
    infl_avg_acr = (deriv_at_w - average_acr) + infl_reg @ average_spline_deriv
    se_avg_acr = float(xp.std(infl_avg_acr) / xp.sqrt(xp.asarray(n_treated_val, dtype=float)))

    overall_att = overall_att_res.overall_att.overall_att
    overall_att_se = overall_att_res.overall_att.overall_se
    overall_att_inf_func = overall_att_res.overall_att.influence_func

    result = DoseResult(
        dose=to_numpy(dvals.flatten() if dvals.ndim > 1 else dvals),
        overall_att=overall_att,
        overall_att_se=overall_att_se,
        overall_att_inf_func=overall_att_inf_func,
        overall_acrt=average_acr,
        overall_acrt_se=se_avg_acr,
        overall_acrt_inf_func=to_numpy(infl_avg_acr),
        att_d=to_numpy(att_d),
        att_d_se=to_numpy(att_d_se),
        att_d_crit_val=att_d_crit_val,
        att_d_inf_func=None,
        acrt_d=to_numpy(acrt_d),
        acrt_d_se=to_numpy(acrt_d_se),
        acrt_d_crit_val=acrt_d_crit_val,
        acrt_d_inf_func=None,
        pte_params=ptep,
    )

    return result
