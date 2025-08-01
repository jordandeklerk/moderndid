"""DiD preprocessing."""

from .preprocess.builders import DIDDataBuilder
from .preprocess.constants import BasePeriod, ControlGroup, EstimationMethod
from .preprocess.models import DIDConfig


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
    print_details=True,
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
    print_details : bool, default True
        Whether to print progress messages.
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
        print_details=print_details,
        faster_mode=faster_mode,
        pl=pl,
        cores=cores,
    )

    builder = DIDDataBuilder()
    did_data = builder.with_data(data).with_config(config).validate().transform().build()

    return did_data
