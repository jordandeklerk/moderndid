"""DiD preprocessing."""

import warnings

import numpy as np

from .preprocess.builders import PreprocessDataBuilder
from .preprocess.config import ContDIDConfig, DDDConfig, DIDConfig, DIDInterConfig, TwoPeriodDIDConfig
from .preprocess.constants import EstimationMethod


def preprocess_drdid(
    data,
    yname,
    tname,
    treat_col,
    idname=None,
    xformla=None,
    panel=True,
    weightsname=None,
    boot=False,
    boot_type="weighted",
    biters=999,
    inf_func=False,
    est_method="imp",
    trim_level=0.995,
    normalized=True,
):
    """Process data for 2-period doubly robust DiD estimation.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel or repeated cross-section data.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column (must have exactly 2 periods).
    treat_col : str
        Name of treatment indicator column (0=control, 1=treated).
        For panel data: time-invariant indicator.
        For repeated cross-sections: treatment status in post-period.
    idname : str | None, default None
        Name of entity/unit identifier column. Required for panel data.
    xformla : str | None, default None
        Formula for the covariates, such as ``"~ x1 + x2"``. A transformed
        term such as ``I(x**2)`` requires formulaic. If None, only the
        intercept is included.
    panel : bool, default True
        Whether the data are a panel rather than repeated cross-sections.
    weightsname : str | None, default None
        Name of sampling weights column.
    boot : bool, default False
        Whether to use bootstrap for inference.
    boot_type : {"weighted", "multiplier"}, default "weighted"
        Type of bootstrap.
    biters : int, default 999
        Number of bootstrap iterations.
    inf_func : bool, default False
        Whether to compute influence functions.
    est_method : {"imp", "trad"}, default "imp"
        Estimation method for the nuisance parameters. "imp" selects the
        improved doubly robust estimator and "trad" the traditional one.
    trim_level : float, default 0.995
        Propensity score trimming level.
    normalized : bool, default True
        Whether to normalize weights.

    Returns
    -------
    TwoPeriodDIDData
        Container with the data of the two periods:

        - **y1**: Post-treatment outcome of each unit in a panel, or None
        - **y0**: Pre-treatment outcome of each unit in a panel, or None
        - **y**: Outcome of each row in repeated cross-sections, or None
        - **post**: Post-treatment indicator of each row in repeated cross-sections, or None
        - **D**: Treatment indicator of each unit or row
        - **covariates**: Covariate matrix with the intercept
        - **weights**: Normalized sampling weights
        - **covariate_names**: Names of the covariate columns
        - **n_units**: Number of units in a panel
        - **n_obs**: Number of rows in repeated cross-sections
        - **config**: Configuration with all settings
    """
    config = TwoPeriodDIDConfig(
        yname=yname,
        tname=tname,
        treat_col=treat_col,
        idname=idname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        weightsname=weightsname,
        boot=boot,
        boot_type=boot_type,
        biters=biters,
        est_method=est_method,
        trim_level=trim_level,
        inf_func=inf_func,
        normalized=normalized,
    )

    builder = PreprocessDataBuilder()
    two_period_data = builder.with_data(data).with_config(config).validate().transform().build()

    return two_period_data


def preprocess_did(
    data,
    yname,
    tname,
    gname,
    idname=None,
    xformla=None,
    panel=True,
    allow_unbalanced_panel=True,
    control_group="nevertreated",
    anticipation=0,
    weightsname=None,
    alpha=0.05,
    boot=False,
    cband=False,
    biters=1000,
    clustervars=None,
    est_method="dr",
    base_period="varying",
    faster_mode=False,
    pl_parallel=False,
    cores=1,
):
    """Process data for multi-period difference-in-differences.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel or repeated cross-section data.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    gname : str
        Name of treatment group column. Should contain the time period
        when a unit is first treated (0 for never-treated).
    idname : str | None, default None
        Name of entity/unit identifier column. Required for panel data.
    xformla : str | None, default None
        Formula for covariates in Wilkinson notation, such as ``"~ x1 + x2"``.
        If None, no covariates are included.
    panel : bool, default True
        Whether the data are a panel rather than repeated cross-sections.
    allow_unbalanced_panel : bool, default True
        Whether to allow unbalanced panels.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as control group.
    anticipation : int, default 0
        Number of time periods before treatment where effects may appear.
    weightsname : str | None, default None
        Name of sampling weights column.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    boot : bool, default False
        Whether to use bootstrap for inference.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    biters : int, default 1000
        Number of bootstrap iterations.
    clustervars : list[str] | None, default None
        Variables to cluster standard errors on.
    est_method : {"dr", "ipw", "reg"} or callable, default "dr"
        Estimation method. The built-in methods are doubly robust, IPW, and
        regression. A function that estimates each 2x2 comparison works too.
    base_period : {"universal", "varying"}, default "varying"
        How to choose base period for comparisons.
    faster_mode : bool, default False
        Whether to use computational shortcuts.
    pl_parallel : bool, default False
        Whether to use parallel processing.
    cores : int, default 1
        Number of cores for parallel processing.

    Returns
    -------
    DIDData
        Container with the preprocessed data and settings:

        - **data**: Standardized panel or cross-section data
        - **time_invariant_data**: One row for each unit with its group and weight
        - **weights**: Normalized sampling weight of each unit
        - **cohort_counts**: Number of units in each cohort
        - **period_counts**: Number of rows in each period
        - **crosstable_counts**: Number of rows in each period and cohort
        - **config**: Configuration with all settings and the categorical options stored as enums
        - **cluster**: Cluster of each unit, or None
        - **outcomes_tensor**: Outcomes of each period for a balanced panel, or None
        - **covariates_matrix**: Covariates of each row for repeated cross-sections and unbalanced panels, or None
        - **covariates_tensor**: Covariates of each period for a balanced panel, or None
        - **weights_tensor**: Weights of each period for a balanced panel, or None
    """
    config = DIDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alpha=alpha,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=est_method,
        base_period=base_period,
        faster_mode=faster_mode,
        pl=pl_parallel,
        cores=cores,
    )

    builder = PreprocessDataBuilder()
    did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return did_data


def preprocess_cont_did(
    data,
    yname,
    tname,
    gname,
    dname,
    idname=None,
    xformla=None,
    panel=True,
    allow_unbalanced_panel=False,
    control_group="notyettreated",
    anticipation=0,
    weightsname=None,
    alpha=0.05,
    cband=False,
    biters=1000,
    clustervars=None,
    degree=3,
    num_knots=0,
    dvals=None,
    target_parameter="level",
    aggregation="dose",
    base_period="varying",
    boot_type="multiplier",
    required_pre_periods=0,
    dose_est_method="parametric",
):
    """Process data for continuous treatment difference-in-differences.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel or repeated cross-section data.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    gname : str
        Name of treatment group column. Should contain the time period
        when a unit is first treated (0 for never-treated).
    dname : str
        Name of the column containing the continuous treatment dose.
        Should be constant across time periods for each unit.
        Use 0 for never-treated units.
    idname : str | None, default None
        Name of entity/unit identifier column. Required for panel data.
    xformla : str | None, default None
        Formula for the covariates, such as ``"~ x1 + x2"``. If None, no
        covariates are included. Currently only ``"~1"`` is supported.
    panel : bool, default True
        Whether the data are a panel rather than repeated cross-sections.
    allow_unbalanced_panel : bool, default False
        Whether to allow unbalanced panels. Currently not supported.
    control_group : {"nevertreated", "notyettreated"}, default "notyettreated"
        Which units to use as control group.
    anticipation : int, default 0
        Number of time periods before treatment where effects may appear.
    weightsname : str | None, default None
        Name of sampling weights column.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    biters : int, default 1000
        Number of bootstrap iterations.
    clustervars : list[str] | None, default None
        Variables to cluster standard errors on.
    degree : int, default 3
        Degree of the B-spline basis functions for dose-response estimation.
    num_knots : int, default 0
        Number of interior knots for the B-spline.
    dvals : array-like | None, default None
        Doses at which to evaluate the dose-response curves. If None, the
        estimator uses 50 evenly spaced doses between the smallest and largest
        treated dose.
    target_parameter : {"level", "slope"}, default "level"
        Type of treatment effect to estimate. "level" gives the average
        treatment effect on the treated at each dose. "slope" gives the
        average causal response, the derivative of the dose-response curve.
    aggregation : {"dose", "eventstudy"}, default "dose"
        How to aggregate treatment effects. "dose" averages across timing
        groups and periods and reports by dose. "eventstudy" averages across
        timing groups and doses and reports by event time.
    base_period : {"universal", "varying"}, default "varying"
        How to choose base period for comparisons.
    boot_type : {"multiplier", "weighted", "empirical"}, default "multiplier"
        Type of bootstrap to perform.
    required_pre_periods : int, default 0
        Minimum number of pre-treatment periods required.
    dose_est_method : {"parametric", "cck"}, default "parametric"
        Estimator of the dose-response curves.

    Returns
    -------
    ContDIDData
        Container with the preprocessed data and settings:

        - **data**: Standardized panel or cross-section data with each time period recoded to its position
        - **time_invariant_data**: One row for each unit with its group and weight
        - **weights**: Normalized sampling weight of each unit
        - **cohort_counts**: Number of units in each cohort
        - **period_counts**: Number of rows in each period
        - **crosstable_counts**: Number of rows in each period and cohort
        - **config**: Configuration with all settings and the categorical options stored as enums
        - **cluster**: Cluster of each unit, or None
        - **time_map**: Mapping from each time period in the data to its position
        - **original_time_periods**: Time periods of the preprocessed data before recoding
    """
    dvals_array = np.asarray(dvals) if dvals is not None else None

    if clustervars is None:
        clustervars_list: list[str] = []
    elif isinstance(clustervars, str):
        clustervars_list = [clustervars]
    else:
        clustervars_list = list(clustervars)

    config = ContDIDConfig(
        yname=yname,
        tname=tname,
        gname=gname,
        dname=dname,
        idname=idname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group,
        anticipation=anticipation,
        weightsname=weightsname,
        alpha=alpha,
        cband=cband,
        biters=biters,
        clustervars=clustervars_list,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals_array,
        target_parameter=target_parameter,
        aggregation=aggregation,
        base_period=base_period,
        boot_type=boot_type,
        required_pre_periods=required_pre_periods,
        dose_est_method=dose_est_method,
    )

    builder = PreprocessDataBuilder()
    cont_did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return cont_did_data


def preprocess_ddd_2periods(
    data,
    yname,
    tname,
    idname,
    gname,
    pname,
    xformla=None,
    est_method="dr",
    weightsname=None,
    boot=False,
    boot_type="multiplier",
    biters=999,
    cluster=None,
    cband=False,
    alpha=0.05,
    inf_func=False,
):
    """Preprocess data for 2-period DDD estimation.

    The inference options are stored as given. :func:`~moderndid.ddd` settles
    them for every data layout before it calls this function.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel data with exactly 2 time periods.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    idname : str
        Name of unit identifier column.
    gname : str
        Name of treatment group column. Should be 0 for never-treated and
        positive for treated units.
    pname : str
        Name of partition/eligibility column (1=eligible, 0=ineligible).
    xformla : str | None, default None
        Formula for the covariates, such as ``"~ x1 + x2"``. If None, no
        covariates are included.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method: doubly robust, regression, or IPW.
    weightsname : str | None, default None
        Name of sampling weights column.
    boot : bool, default False
        Whether to use bootstrap for inference.
    boot_type : {"multiplier", "weighted"}, default "multiplier"
        Type of bootstrap.
    biters : int, default 999
        Number of bootstrap iterations.
    cluster : str | None, default None
        Name of cluster variable for clustered standard errors.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    inf_func : bool, default False
        Whether to compute influence functions.

    Returns
    -------
    DDDData
        Container with the data of the two periods for each unit:

        - **y1**: Post-treatment outcomes
        - **y0**: Pre-treatment outcomes
        - **treat**: Treatment indicator
        - **partition**: Eligibility indicator
        - **subgroup**: Subgroup assignment from 1 to 4
        - **covariates**: Covariate matrix
        - **weights**: Normalized sampling weights
        - **cluster**: Cluster of each unit, or None
        - **n_units**: Number of units
        - **subgroup_counts**: Count of units in each subgroup
        - **covariate_names**: Names of the covariate columns
        - **config**: Configuration with all settings and the categorical options stored as enums
    """
    if xformla is None:
        xformla = "~1"

    if boot and biters is None:
        warnings.warn("biters not specified. Using 999.", stacklevel=2)
        biters = 999

    if est_method not in ["dr", "reg", "ipw"]:
        raise ValueError(f"est_method must be 'dr', 'reg', or 'ipw', got '{est_method}'.")

    if biters is None:
        biters = 999

    config = DDDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        pname=pname,
        xformla=xformla,
        est_method=EstimationMethod(est_method),
        weightsname=weightsname,
        boot=boot,
        boot_type=boot_type,
        biters=biters,
        cluster=cluster,
        cband=cband,
        alpha=alpha,
        inf_func=inf_func,
    )

    return PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()


def preprocess_didinter(
    data,
    yname,
    tname,
    gname,
    dname,
    cluster=None,
    weightsname=None,
    controls=None,
    trends_nonparam=None,
    effects=1,
    placebo=0,
    normalized=False,
    effects_equal=False,
    switchers="",
    only_never_switchers=False,
    same_switchers=False,
    same_switchers_pl=False,
    alpha=0.05,
    less_conservative_se=False,
    trends_lin=False,
    continuous=0,
    allow_unbalanced_panel=False,
):
    """Process data for heterogeneous and dynamic treatment effects.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Panel data in long format.
    yname : str
        Name of outcome variable column.
    tname : str
        Name of time period column.
    gname : str
        Name of unit/group identifier column.
    dname : str
        Name of treatment variable column.
    cluster : str, optional
        Name of clustering variable for standard errors.
    weightsname : str, optional
        Name of sampling weights column.
    controls : list of str, optional
        Names of control variable columns.
    trends_nonparam : list of str, optional
        Variables for non-parametric trends.
    effects : int, default 1
        Number of post-treatment periods to estimate.
    placebo : int, default 0
        Number of pre-treatment placebo periods.
    normalized : bool, default False
        Whether to normalize by treatment intensity.
    effects_equal : bool, default False
        Whether to test equality of effects.
    switchers : str, default ""
        Filter switchers: "" (all), "in" (switchers-in), "out" (switchers-out).
    only_never_switchers : bool, default False
        Use only never-switchers as controls.
    same_switchers : bool, default False
        Use same switchers across all effects.
    same_switchers_pl : bool, default False
        Use same switchers for placebos.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    less_conservative_se : bool, default False
        Use less conservative standard errors.
    trends_lin : bool, default False
        Include linear trends.
    continuous : int, default 0
        Polynomial degree for continuous treatment.
    allow_unbalanced_panel : bool, default False
        Allow unbalanced panel data.

    Returns
    -------
    DIDInterData
        Container with the preprocessed data and settings:

        - **data**: Standardized panel with each unit's switching information
        - **time_invariant_data**: One row for each unit with its weight and switching information
        - **weights**: Normalized sampling weight of each unit
        - **cohort_counts**: Number of units by the direction of their first switch
        - **period_counts**: Number of rows in each period
        - **crosstable_counts**: Number of rows in each period by direction of the first switch
        - **config**: Configuration with all settings
        - **cluster**: Cluster of each unit, or None
    """
    config = DIDInterConfig(
        yname=yname,
        tname=tname,
        gname=gname,
        dname=dname,
        cluster=cluster,
        weightsname=weightsname,
        controls=controls,
        trends_nonparam=trends_nonparam,
        effects=effects,
        placebo=placebo,
        normalized=normalized,
        effects_equal=effects_equal,
        switchers=switchers,
        only_never_switchers=only_never_switchers,
        same_switchers=same_switchers,
        same_switchers_pl=same_switchers_pl,
        alpha=alpha,
        less_conservative_se=less_conservative_se,
        trends_lin=trends_lin,
        continuous=continuous,
        allow_unbalanced_panel=allow_unbalanced_panel,
    )

    return PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()
