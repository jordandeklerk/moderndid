"""DiD preprocessing."""

import warnings

import numpy as np
import polars as pl
import scipy.linalg

from .dataframe import DataFrame, to_polars
from .preprocess.builders import PreprocessDataBuilder
from .preprocess.config import ContDIDConfig, DDDConfig, DIDConfig, TwoPeriodDIDConfig
from .preprocess.constants import BasePeriod, BootstrapType, ControlGroup, EstimationMethod
from .preprocess.models import DDDData
from .preprocess.utils import extract_vars_from_formula, make_balanced_panel


def preprocess_drdid(
    data: DataFrame,
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
    bstrap=False,
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
        bstrap=bstrap,
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
    bstrap=False,
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
    bstrap : bool, default False
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
        bstrap=bstrap,
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

    df = to_polars(data)
    _validate_ddd_inputs(df, yname, tname, idname, gname, pname, xformla, cluster, weightsname)

    config_params = _apply_ddd_defaults(
        alp=alp,
        boot=boot,
        boot_type=boot_type,
        n_boot=n_boot,
        cluster=cluster,
        cband=cband,
        est_method=est_method,
        inf_func=inf_func,
    )

    if weightsname is not None:
        weights = df[weightsname].to_numpy().astype(float)
        if np.any(np.isnan(weights)):
            raise ValueError("Missing values in weights column.")
        _check_weights_uniqueness(df, idname, weightsname)
    else:
        weights = np.ones(len(df))
    weights = weights / np.mean(weights)
    df = df.with_columns(pl.Series("_weights", weights))

    tlist = np.sort(df[tname].unique().to_numpy())
    if len(tlist) != 2:
        raise ValueError(f"Data must have exactly 2 time periods, found {len(tlist)}.")

    glist = np.sort(df[gname].unique().to_numpy())
    if len(glist) != 2:
        raise ValueError(f"Treatment variable must have exactly 2 values (0 and treated group), found {len(glist)}.")
    if glist[0] != 0:
        raise ValueError("Treatment variable must include 0 for never-treated units.")

    df = df.with_columns((pl.col(tname) == tlist[1]).cast(pl.Int64).alias("_post"))

    if xformla != "~1":
        _check_covariates_time_invariant(df, xformla, idname)

    df = df.sort([idname, tname])
    df = make_balanced_panel(df, idname, tname)

    if len(df) == 0:
        raise ValueError("No observations remain after creating balanced panel.")

    treat_val = glist[1]
    subgroup = _create_subgroups(df[gname].to_numpy(), df[pname].to_numpy(), treat_val)
    df = df.with_columns(pl.Series("_subgroup", subgroup))

    subgroup_counts = _compute_subgroup_counts(df, idname)
    _validate_subgroup_sizes(subgroup_counts)

    covariates, covariate_names = _extract_covariates(df, xformla)

    cluster_arr = None
    if config_params["cluster"] is not None:
        _check_cluster_time_invariant(df, config_params["cluster"], idname)
        cluster_arr = df.filter(pl.col("_post") == 0)[config_params["cluster"]].to_numpy()

    df_pre = df.filter(pl.col("_post") == 0)
    df_post = df.filter(pl.col("_post") == 1)

    y0 = df_pre[yname].to_numpy().astype(float)
    y1 = df_post[yname].to_numpy().astype(float)
    treat = df_pre[gname].to_numpy()
    treat = (treat == treat_val).astype(int)
    partition = df_pre[pname].to_numpy().astype(int)
    subgroup = df_pre["_subgroup"].to_numpy()
    weights_arr = df_pre["_weights"].to_numpy()

    n_units = len(y0)

    ddd_config = DDDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        pname=pname,
        xformla=xformla,
        est_method=EstimationMethod(config_params["est_method"]),
        weightsname=weightsname,
        boot=config_params["boot"],
        boot_type=BootstrapType(config_params["boot_type"]),
        n_boot=config_params["n_boot"],
        cluster=config_params["cluster"],
        cband=config_params["cband"],
        alp=config_params["alp"],
        inf_func=config_params["inf_func"],
        time_periods=tlist,
        time_periods_count=len(tlist),
        n_units=n_units,
    )

    return DDDData(
        y1=y1,
        y0=y0,
        treat=treat,
        partition=partition,
        subgroup=subgroup,
        covariates=covariates,
        weights=weights_arr,
        cluster=cluster_arr,
        n_units=n_units,
        subgroup_counts=subgroup_counts,
        covariate_names=covariate_names,
        config=ddd_config,
    )


def _validate_ddd_inputs(data: pl.DataFrame, yname, tname, idname, gname, pname, xformla, cluster, weightsname):
    """Validate DDD input arguments."""
    required_cols = [yname, tname, idname, gname, pname]
    col_names = ["yname", "tname", "idname", "gname", "pname"]

    for col, name in zip(required_cols, col_names):
        if col not in data.columns:
            raise ValueError(f"{name}='{col}' not found in data.")

    if cluster is not None and cluster not in data.columns:
        raise ValueError(f"cluster='{cluster}' not found in data.")

    if weightsname is not None and weightsname not in data.columns:
        raise ValueError(f"weightsname='{weightsname}' not found in data.")

    if xformla != "~1":
        try:
            covariate_vars = extract_vars_from_formula(xformla)
        except ValueError as e:
            raise ValueError(f"Invalid formula: {e}") from e

        for var in covariate_vars:
            if var not in data.columns:
                raise ValueError(f"Covariate '{var}' from formula not found in data.")

    _check_partition_uniqueness(data, idname, pname)
    _check_treatment_uniqueness(data, idname, gname)


def _apply_ddd_defaults(alp, boot, boot_type, n_boot, cluster, cband, est_method, inf_func):
    """Apply default values and handle parameter dependencies for DDD."""
    if alp > 0.10:
        warnings.warn(f"alp={alp} is high. Using alp=0.05.", stacklevel=3)
        alp = 0.05

    if boot and n_boot is None:
        warnings.warn("n_boot not specified. Using 999.", stacklevel=3)
        n_boot = 999

    if boot and not cband:
        warnings.warn("Setting cband=True for bootstrap.", stacklevel=3)
        cband = True

    if cluster is not None and not boot:
        warnings.warn(
            "Clustered SEs require bootstrap. Setting boot=True, cband=True.",
            stacklevel=3,
        )
        boot = True
        cband = True
        if n_boot is None:
            n_boot = 999

    if est_method not in ["dr", "reg", "ipw"]:
        raise ValueError(f"est_method must be 'dr', 'reg', or 'ipw', got '{est_method}'.")

    return {
        "alp": alp,
        "boot": boot,
        "boot_type": boot_type,
        "n_boot": n_boot if n_boot is not None else 999,
        "cluster": cluster,
        "cband": cband,
        "est_method": est_method,
        "inf_func": inf_func,
    }


def _check_partition_uniqueness(df: pl.DataFrame, idname, pname):
    """Check that partition is time-invariant within units."""
    partition_per_id = df.group_by(idname).agg(pl.col(pname).n_unique().alias("n_unique"))
    if (partition_per_id["n_unique"] > 1).any():
        raise ValueError(f"The value of {pname} must be the same across all periods for each unit.")


def _check_treatment_uniqueness(df: pl.DataFrame, idname, gname):
    """Check that treatment status is time-invariant within units."""
    treat_per_id = df.group_by(idname).agg(pl.col(gname).n_unique().alias("n_unique"))
    if (treat_per_id["n_unique"] > 1).any():
        raise ValueError(f"The value of {gname} must be the same across all periods for each unit.")


def _check_weights_uniqueness(df: pl.DataFrame, idname, weightsname):
    """Check that weights are time-invariant within units."""
    weights_per_id = df.group_by(idname).agg(pl.col(weightsname).n_unique().alias("n_unique"))
    if (weights_per_id["n_unique"] > 1).any():
        raise ValueError("Weights must be the same across all periods for each unit.")


def _check_covariates_time_invariant(df: pl.DataFrame, xformla, idname):
    """Check that covariates are time-invariant."""
    covariate_vars = extract_vars_from_formula(xformla)

    for var in covariate_vars:
        if var not in df.columns:
            continue
        var_per_id = df.group_by(idname).agg(pl.col(var).n_unique().alias("n_unique"))
        if (var_per_id["n_unique"] > 1).any():
            raise ValueError(f"Covariate '{var}' varies over time. Covariates must be time-invariant.")


def _check_cluster_time_invariant(df: pl.DataFrame, cluster, idname):
    """Check that cluster variable is time-invariant."""
    cluster_per_id = df.group_by(idname).agg(pl.col(cluster).n_unique().alias("n_unique"))
    if (cluster_per_id["n_unique"] > 1).any():
        raise ValueError("Cluster variable must be time-invariant within units.")


def _create_subgroups(treat, partition, treat_val):
    """Create subgroup assignments for DDD.

    Subgroup definitions:
    - 4: Treated AND Eligible (treat=g, partition=1)
    - 3: Treated BUT Ineligible (treat=g, partition=0)
    - 2: Eligible BUT Untreated (treat=0, partition=1)
    - 1: Untreated AND Ineligible (treat=0, partition=0)
    """
    is_treated = treat == treat_val
    is_eligible = partition == 1

    subgroup = np.where(
        is_treated & is_eligible,
        4,
        np.where(is_treated & ~is_eligible, 3, np.where(is_eligible, 2, 1)),
    )
    return subgroup


def _compute_subgroup_counts(df: pl.DataFrame, idname):
    """Compute counts per subgroup."""
    counts_df = df.group_by("_subgroup").agg(pl.col(idname).n_unique().alias("count"))
    return {int(row["_subgroup"]): int(row["count"]) for row in counts_df.iter_rows(named=True)}


def _validate_subgroup_sizes(subgroup_counts, min_size=5):
    """Validate that each subgroup has sufficient observations."""
    for sg, count in subgroup_counts.items():
        if count < min_size:
            raise ValueError(f"Subgroup {sg} has only {count} observations. Minimum required is {min_size}.")


def _extract_covariates(df: pl.DataFrame, xformla):
    """Extract and process covariates from data."""
    if xformla == "~1":
        n_units = len(df.filter(pl.col("_post") == 0))
        return np.empty((n_units, 0)), []

    covariate_vars = extract_vars_from_formula(xformla)

    df_pre = df.filter(pl.col("_post") == 0)
    cov_matrix = df_pre.select(covariate_vars).to_numpy().astype(float)

    if np.any(np.isnan(cov_matrix)):
        warnings.warn(
            "Missing values in covariates. Rows with NaN will cause issues.",
            stacklevel=3,
        )

    cov_matrix, kept_vars = _remove_collinear(cov_matrix, covariate_vars)

    return cov_matrix, kept_vars


def _remove_collinear(cov_matrix, var_names, tol=1e-6):
    """Remove collinear columns from covariate matrix using QR decomposition."""
    if cov_matrix.shape[1] == 0:
        return cov_matrix, var_names

    _, r, pivot = scipy.linalg.qr(cov_matrix, pivoting=True)

    diag_r = np.abs(np.diag(r))
    rank = np.sum(diag_r > tol * diag_r[0]) if len(diag_r) > 0 else 0

    keep_indices = pivot[:rank]
    keep_indices = np.sort(keep_indices)

    kept_vars = [var_names[i] for i in keep_indices]
    return cov_matrix[:, keep_indices], kept_vars
