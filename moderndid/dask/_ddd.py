"""Entry point for distributed DDD estimation."""

from __future__ import annotations

import logging

import numpy as np

from ._utils import get_or_create_client, validate_dask_input


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
    max_cohorts=None,
):
    r"""Compute the distributed triple difference-in-differences estimator.

    Distributed implementation of the DDD estimator from [1]_ for datasets
    that exceed single-machine memory. This function accepts a Dask DataFrame,
    automatically detects whether the data has two periods or multiple periods
    with staggered adoption, and dispatches to the appropriate distributed
    estimator.

    The underlying methodology is identical to :func:`~moderndid.ddd`: for each
    group-time cell :math:`(g,t)`, the estimator computes

    .. math::

        ATT(g,t) = \mathbb{E}[Y_{i,t}(g) - Y_{i,t}(\infty) \mid S_i = g, Q_i = 1],

    where :math:`S_i` is the period when treatment is enabled for unit
    :math:`i`'s group and :math:`Q_i \in \{0,1\}` indicates eligibility.
    Identification relies on a DDD conditional parallel trends assumption that
    allows differential trends between eligible and ineligible units, provided
    these differentials are stable across treatment-enabling groups.

    The distributed backend computes sufficient statistics (Gram matrices,
    cross-products) partition-by-partition and aggregates via tree-reduce,
    avoiding full materialization of the dataset on any single worker.

    Users do not need to call this function directly. Passing a Dask DataFrame
    to :func:`~moderndid.ddd` will automatically dispatch here.

    Parameters
    ----------
    data : dask.dataframe.DataFrame
        Panel data in long format as a Dask DataFrame. Each partition should
        contain complete observations (all columns) for a subset of units.
        Use ``dask.dataframe.read_parquet``, ``dask.dataframe.read_csv``, or
        ``dask.dataframe.from_pandas`` to construct the input.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the column containing time periods.
    idname : str, optional
        Name of the unit identifier column. Required for panel data.
    gname : str
        Name of the treatment group column. Should be 0 for never-treated
        units and a positive value indicating the first period when treatment
        is enabled for the unit's group.
    pname : str
        Name of the partition/eligibility column (1=eligible, 0=ineligible).
        This identifies which units within a treatment-enabling group are
        actually eligible to receive treatment.
    xformla : str, optional
        Formula for covariates in the form ``"~ x1 + x2"``. If None, only
        an intercept is used.
    client : distributed.Client, optional
        Dask distributed client. If None, a local client is created
        automatically via :func:`~moderndid.dask.get_or_create_client`.
    control_group : {"nevertreated", "notyettreated"}, default="nevertreated"
        Which units to use as controls. ``"nevertreated"`` uses only units
        that are never treated. ``"notyettreated"`` additionally includes
        units not yet treated by period :math:`t`.
    base_period : {"universal", "varying"}, default="universal"
        Base period selection for multi-period settings. ``"universal"`` uses
        the period immediately before the first treated period for all
        group-time cells. ``"varying"`` uses the period immediately before
        each group's treatment onset.
    est_method : {"dr", "reg", "ipw"}, default="dr"
        Estimation method. ``"dr"`` uses doubly robust estimation combining
        outcome regression and inverse probability weighting. ``"reg"`` uses
        outcome regression only. ``"ipw"`` uses inverse probability weighting
        only.
    boot : bool, default=False
        Whether to use the multiplier bootstrap for inference.
    biters : int, default=1000
        Number of bootstrap iterations. Only used when ``boot=True``.
    cband : bool, default=False
        Whether to compute uniform confidence bands that cover all group-time
        average treatment effects simultaneously. Only used when
        ``boot=True``.
    cluster : str, optional
        Name of the clustering variable for clustered standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    random_state : int, optional
        Random seed for reproducibility of bootstrap draws.
    n_partitions : int, optional
        Number of Dask partitions for distributing computation. If None,
        defaults to the total number of threads across all workers via
        :func:`~moderndid.dask.get_default_partitions`.
    max_cohorts : int or None, default None
        Maximum number of treatment cohorts to process in parallel.
        When ``None``, defaults to the number of Dask workers.

    Returns
    -------
    DDDPanelResult or DDDMultiPeriodResult
        For 2-period data, returns ``DDDPanelResult`` containing:

        - **att**: The DDD point estimate for the ATT
        - **se**: Standard error
        - **uci**, **lci**: Confidence interval bounds
        - **boots**: Bootstrap draws (if requested)
        - **att_inf_func**: Influence function
        - **did_atts**: Individual DiD ATT estimates
        - **subgroup_counts**: Number of units per subgroup
        - **args**: Estimation arguments

        For multi-period data, returns ``DDDMultiPeriodResult`` containing:

        - **att**: Array of ATT(g,t) point estimates
        - **se**: Standard errors for each ATT(g,t)
        - **uci**, **lci**: Confidence interval bounds
        - **groups**, **times**: Treatment cohort and time for each estimate
        - **glist**, **tlist**: Unique cohorts and periods
        - **inf_func_mat**: Influence function matrix
        - **n**: Number of units
        - **args**: Estimation arguments

    Examples
    --------
    For datasets that fit in memory, create a Dask DataFrame from an existing
    pandas or polars DataFrame:

    .. code-block:: python

        import dask.dataframe as dd
        from moderndid import ddd, gen_dgp_mult_periods

        dgp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
        ddf = dd.from_pandas(dgp["data"].to_pandas(), npartitions=4)

        result = ddd(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            est_method="dr",
        )

    For large datasets stored on disk, read directly into Dask:

    .. code-block:: python

        ddf = dd.read_parquet("large_panel/*.parquet")
        result = ddd(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
        )

    To connect to an existing Dask cluster instead of the default local one:

    .. code-block:: python

        from dask.distributed import Client
        from moderndid.dask import dask_ddd

        client = Client("scheduler-address:8786")
        result = dask_ddd(
            data=ddf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            client=client,
        )

    See Also
    --------
    ddd : Local (non-distributed) DDD estimator. Automatically dispatches
        to ``dask_ddd`` when passed a Dask DataFrame.
    get_or_create_client : Get or create a Dask distributed client.
    get_default_partitions : Compute default partition count from cluster.
    monitor_cluster : Monitor cluster memory and task statistics.

    References
    ----------

    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
        *Better Understanding Triple Differences Estimators.*
        arXiv preprint arXiv:2505.09942. https://arxiv.org/abs/2505.09942
    """
    if gname is None:
        raise ValueError("gname is required.")
    if pname is None:
        raise ValueError("pname is required.")

    client = get_or_create_client(client)
    logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)

    required_cols = [yname, tname, gname, pname]
    if idname is not None:
        required_cols.append(idname)
    validate_dask_input(data, required_cols)

    multiple_periods = _detect_multiple_periods_dask(data, tname, gname, client=client)

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
            max_cohorts=max_cohorts,
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


def _detect_multiple_periods_dask(ddf, tname, gname, client=None):
    """Detect whether data has more than 2 time periods or treatment groups."""
    if client is not None:
        t_fut = client.compute(ddf[tname].nunique())
        g_fut = client.compute(ddf[gname].unique())
        n_time, gvals = client.gather([t_fut, g_fut])
        gvals = gvals.values
    else:
        n_time = ddf[tname].nunique().compute()
        gvals = ddf[gname].unique().compute().values

    finite_gvals = [g for g in gvals if np.isfinite(g)]
    n_groups = len(finite_gvals)

    return max(n_time, n_groups) > 2
