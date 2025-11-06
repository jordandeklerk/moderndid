"""DiD preprocessing."""

from .builders import PreprocessDataBuilder
from .config import DIDConfig, TwoPeriodDIDConfig
from .constants import BasePeriod, BootstrapType, ControlGroup, EstimationMethod


def preprocess_drdid(
    data,
    yname,
    tname,
    treat_col,
    idname=None,
    xformla=None,
    panel=True,
    weightsname=None,
    bstrap=False,
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
    data : pd.DataFrame
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
        Formula for covariates in Wilkinson notation (e.g., "~ x1 + x2").
        If None, only intercept is included.
    panel : bool, default True
        Whether data is in panel format (vs repeated cross-sections).
    weightsname : str | None, default None
        Name of sampling weights column.
    bstrap : bool, default False
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
        bstrap=bstrap,
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
    alp=0.05,
    bstrap=False,
    cband=False,
    biters=1000,
    clustervars=None,
    est_method="dr",
    base_period="varying",
    faster_mode=False,
    pl=False,
    cores=1,
):
    """Process data for multi-period difference-in-differences.

    Parameters
    ----------
    data : pd.DataFrame
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
    bstrap : bool, default False
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
    pl : bool, default False
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
        bstrap=bstrap,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=est_method_enum,
        base_period=base_period_enum,
        faster_mode=faster_mode,
        pl=pl,
        cores=cores,
    )

    builder = PreprocessDataBuilder()
    did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return did_data
