"""DiD preprocessing."""

import warnings

import numpy as np

from .dataframe import DataFrame
from .preprocess.builders import PreprocessDataBuilder
from .preprocess.config import ContDIDConfig, DDDConfig, DIDConfig, DIDInterConfig, TwoPeriodDIDConfig
from .preprocess.constants import BasePeriod, BootstrapType, ControlGroup, EstimationMethod
from .preprocess.utils import check_partition_collinearity as _check_partition_collinearity  # noqa: F401


def preprocess_drdid(
    data: DataFrame,
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
        Formula for covariates as a string (e.g., "x1 + x2").
        If None, only intercept is included.
    panel : bool, default True
        Whether data is in panel format (vs repeated cross-sections).
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
        Estimation method for nuisance parameters.
    trim_level : float, default 0.995
        Propensity score trimming level.
    normalized : bool, default True
        Whether to normalize weights.

    Returns
    -------
    TwoPeriodDIDData
        Container with processed data for 2-period DiD estimation.
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
        boot_type=BootstrapType(boot_type),
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
    data: DataFrame,
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
    alp=0.05,
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
        Formula for covariates in Wilkinson notation (e.g., "~ x1 + x2").
        If None, no covariates are included.
    panel : bool, default True
        Whether data is in panel format (vs repeated cross-sections).
    allow_unbalanced_panel : bool, default True
        Whether to allow unbalanced panels.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as control group.
    anticipation : int, default 0
        Number of time periods before treatment where effects may appear.
    weightsname : str | None, default None
        Name of sampling weights column.
    alp : float, default 0.05
        Significance level for confidence intervals.
    boot : bool, default False
        Whether to use bootstrap for inference.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    biters : int, default 1000
        Number of bootstrap iterations.
    clustervars : list[str] | None, default None
        Variables to cluster standard errors on.
    est_method : {"dr", "ipw", "reg"}, default "dr"
        Estimation method: doubly robust, IPW, or regression.
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
        Container with all preprocessed data and parameters including:

        - data: Standardized panel/cross-section data
        - weights: Normalized sampling weights
        - config: Configuration with all settings
        - Various tensors and matrices for computation
    """
    control_group_enum = ControlGroup(control_group)
    est_method_enum = EstimationMethod(est_method)
    base_period_enum = BasePeriod(base_period)

    config = DIDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=control_group_enum,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=est_method_enum,
        base_period=base_period_enum,
        faster_mode=faster_mode,
        pl=pl_parallel,
        cores=cores,
    )

    builder = PreprocessDataBuilder()
    did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return did_data


def preprocess_cont_did(
    data: DataFrame,
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
    alp=0.05,
    boot=False,
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
        Formula for covariates as a string (e.g., "x1 + x2").
        If None, no covariates are included. Currently only "~1" is supported.
    panel : bool, default True
        Whether data is in panel format (vs repeated cross-sections).
    allow_unbalanced_panel : bool, default False
        Whether to allow unbalanced panels. Currently not supported.
    control_group : {"nevertreated", "notyettreated"}, default "notyettreated"
        Which units to use as control group.
    anticipation : int, default 0
        Number of time periods before treatment where effects may appear.
    weightsname : str | None, default None
        Name of sampling weights column.
    alp : float, default 0.05
        Significance level for confidence intervals.
    boot : bool, default False
        Whether to use bootstrap for inference.
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
        Values of the treatment dose at which to compute effects.
        If None, uses quantiles of the dose distribution among treated units.
    target_parameter : {"level", "slope"}, default "level"
        Type of treatment effect to estimate:
        - "level": Average treatment effect (ATT) at different dose levels
        - "slope": Average causal response (ACRT), the derivative of dose-response
    aggregation : {"dose", "eventstudy"}, default "dose"
        How to aggregate treatment effects:
        - "dose": Average across timing-groups and time, report by dose
        - "eventstudy": Average across timing-groups and doses, report by event time
    base_period : {"universal", "varying"}, default "varying"
        How to choose base period for comparisons.
    boot_type : {"multiplier", "weighted"}, default "multiplier"
        Type of bootstrap to perform.
    required_pre_periods : int, default 0
        Minimum number of pre-treatment periods required.

    Returns
    -------
    ContDIDData
        Container with all preprocessed data and parameters including:

        - data: Standardized panel/cross-section data with recoded time periods
        - time_invariant_data: Unit-level data with group and dose info
        - weights: Normalized sampling weights
        - config: Configuration with all settings
        - time_map: Mapping from original to recoded time periods
        - cohort_counts: Count of units in each treatment cohort
        - period_counts: Count of observations in each time period
    """
    control_group_enum = ControlGroup(control_group)
    base_period_enum = BasePeriod(base_period)
    boot_type_enum = BootstrapType(boot_type)
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
        control_group=control_group_enum,
        anticipation=anticipation,
        weightsname=weightsname,
        alp=alp,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars_list,
        degree=degree,
        num_knots=num_knots,
        dvals=dvals_array,
        target_parameter=target_parameter,
        aggregation=aggregation,
        base_period=base_period_enum,
        boot_type=boot_type_enum,
        required_pre_periods=required_pre_periods,
        dose_est_method=dose_est_method,
    )

    builder = PreprocessDataBuilder()
    cont_did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return cont_did_data


def preprocess_ddd_2periods(
    data: DataFrame,
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
    n_boot=999,
    cluster=None,
    cband=False,
    alp=0.05,
    inf_func=False,
):
    """Preprocess data for 2-period DDD estimation.

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
        Formula for covariates as a string (e.g., "x1 + x2").
        If None, no covariates are included.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method: doubly robust, regression, or IPW.
    weightsname : str | None, default None
        Name of sampling weights column.
    boot : bool, default False
        Whether to use bootstrap for inference.
    boot_type : {"multiplier", "weighted"}, default "multiplier"
        Type of bootstrap.
    n_boot : int, default 999
        Number of bootstrap iterations.
    cluster : str | None, default None
        Name of cluster variable for clustered standard errors.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    alp : float, default 0.05
        Significance level for confidence intervals.
    inf_func : bool, default False
        Whether to compute influence functions.

    Returns
    -------
    DDDData
        Container with preprocessed data for DDD estimation including:

        - y1, y0: Post and pre-treatment outcomes
        - treat: Treatment indicator
        - partition: Eligibility indicator
        - subgroup: Subgroup assignment (1-4)
        - covariates: Covariate matrix
        - weights: Normalized sampling weights
        - subgroup_counts: Count of units in each subgroup
    """
    if xformla is None:
        xformla = "~1"

    if alp > 0.10:
        warnings.warn(f"alp={alp} is high. Using alp=0.05.", stacklevel=2)
        alp = 0.05

    if boot and n_boot is None:
        warnings.warn("n_boot not specified. Using 999.", stacklevel=2)
        n_boot = 999

    if boot and not cband:
        warnings.warn("Setting cband=True for bootstrap.", stacklevel=2)
        cband = True

    if cluster is not None and not boot:
        warnings.warn(
            "Clustered SEs require bootstrap. Setting boot=True, cband=True.",
            stacklevel=2,
        )
        boot = True
        cband = True
        if n_boot is None:
            n_boot = 999

    if est_method not in ["dr", "reg", "ipw"]:
        raise ValueError(f"est_method must be 'dr', 'reg', or 'ipw', got '{est_method}'.")

    if n_boot is None:
        n_boot = 999

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
        boot_type=BootstrapType(boot_type),
        n_boot=n_boot,
        cluster=cluster,
        cband=cband,
        alp=alp,
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
    ci_level=95.0,
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
    ci_level : float, default 95.0
        Confidence interval level.
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
        Container with processed data for DIDInter estimation.
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
        ci_level=ci_level,
        less_conservative_se=less_conservative_se,
        trends_lin=trends_lin,
        continuous=continuous,
        allow_unbalanced_panel=allow_unbalanced_panel,
    )

    return PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()
