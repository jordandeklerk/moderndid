"""Entry point for distributed DDD estimation."""

from __future__ import annotations

import logging

import numpy as np

from ._utils import get_or_create_client, validate_dask_input

log = logging.getLogger("moderndid.dask.backend")


def dask_ddd(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    pname=None,
    xformla=None,
    client=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    boot=False,
    biters=1000,
    cband=False,
    cluster=None,
    alpha=0.05,
    random_state=None,
    n_partitions=None,
):
    """Compute the distributed DDD estimator for Dask DataFrames.

    Automatically detects whether the data has multiple periods and dispatches
    to the appropriate distributed estimator.

    Parameters
    ----------
    data : dask.dataframe.DataFrame
        Dask DataFrame in long format.
    yname : str
        Outcome variable column name.
    tname : str
        Time period column name.
    idname : str or None
        Unit identifier column name.
    gname : str
        Treatment group column name.
    pname : str
        Partition/eligibility column name.
    xformla : str or None
        Covariate formula "~ x1 + x2".
    client : distributed.Client or None
        Dask client. Created automatically if None.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
        Which units to use as controls.
    base_period : {"universal", "varying"}, default "universal"
        Base period selection.
    est_method : {"dr", "reg", "ipw"}, default "dr"
        Estimation method.
    boot : bool, default False
        Whether to use multiplier bootstrap.
    biters : int, default 1000
        Number of bootstrap iterations.
    cband : bool, default False
        Whether to compute uniform confidence bands.
    cluster : str or None
        Cluster variable for clustered SEs.
    alpha : float, default 0.05
        Significance level.
    random_state : int or None
        Random seed for bootstrap.
    n_partitions : int or None
        Number of partitions. Defaults to number of workers.

    Returns
    -------
    DDDPanelResult or DDDMultiPeriodResult
        Same result types as the local estimators.
    """
    if gname is None:
        raise ValueError("gname is required.")
    if pname is None:
        raise ValueError("pname is required.")

    client = get_or_create_client(client)
    n_workers = len(client.scheduler_info()["workers"])
    log.info("dask_ddd: connected to %d workers", n_workers)

    required_cols = [yname, tname, gname, pname]
    if idname is not None:
        required_cols.append(idname)
    validate_dask_input(data, required_cols)

    multiple_periods = _detect_multiple_periods_dask(data, tname, gname)
    log.info("dask_ddd: multiple_periods=%s", multiple_periods)

    if multiple_periods:
        from moderndid.didtriple.utils import get_covariate_names

        from ._ddd_mp import dask_ddd_mp

        covariate_cols = get_covariate_names(xformla)

        return dask_ddd_mp(
            client=client,
            data=data,
            y_col=yname,
            time_col=tname,
            id_col=idname,
            group_col=gname,
            partition_col=pname,
            covariate_cols=covariate_cols,
            control_group=control_group,
            base_period=base_period,
            est_method=est_method,
            boot=boot,
            biters=biters,
            cband=cband,
            cluster=cluster,
            alpha=alpha,
            random_state=random_state,
            n_partitions=n_partitions,
        )

    # 2-period panel path: compute to numpy and use distributed panel estimator
    from moderndid.core.dataframe import to_polars
    from moderndid.core.preprocessing import preprocess_ddd_2periods
    from moderndid.didtriple.utils import add_intercept

    from ._ddd_panel import dask_ddd_panel

    df = to_polars(data.compute())

    ddd_data = preprocess_ddd_2periods(
        data=df,
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        pname=pname,
        xformla=xformla,
        est_method=est_method,
        weightsname=None,
        boot=boot,
        boot_type="multiplier",
        n_boot=biters,
        cluster=cluster,
        alp=alpha,
        inf_func=True,
    )

    covariates_with_intercept = add_intercept(ddd_data.covariates)

    return dask_ddd_panel(
        client=client,
        y1=ddd_data.y1,
        y0=ddd_data.y0,
        subgroup=ddd_data.subgroup,
        covariates=covariates_with_intercept,
        i_weights=ddd_data.weights,
        est_method=est_method,
        boot=boot,
        biters=biters,
        influence_func=True,
        alpha=alpha,
        random_state=random_state,
        n_partitions=n_partitions,
    )


def _detect_multiple_periods_dask(ddf, tname, gname):
    """Detect multiple periods from a Dask DataFrame.

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Input Dask DataFrame.
    tname : str
        Time column name.
    gname : str
        Group column name.

    Returns
    -------
    bool
        True if data has more than 2 time periods or treatment groups.
    """
    n_time = ddf[tname].nunique().compute()

    gvals = ddf[gname].unique().compute().values
    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time, n_groups) > 2
