"""Entry point for distributed DDD estimation via Spark."""

from __future__ import annotations

import logging

import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame

from moderndid.cupy.backend import _validate_backend_name
from moderndid.didtriple.utils import get_covariate_names

from ._ddd_mp import spark_ddd_mp
from ._utils import get_or_create_spark, validate_spark_input


def spark_ddd(
    data,
    yname,
    tname,
    idname=None,
    gname=None,
    pname=None,
    xformla=None,
    spark=None,
    control_group="nevertreated",
    base_period="universal",
    est_method="dr",
    weightsname=None,
    boot=False,
    biters=1000,
    cband=False,
    cluster=None,
    alpha=0.05,
    trim_level=0.995,
    panel=True,
    allow_unbalanced_panel=False,
    random_state=None,
    n_partitions=None,
    max_cohorts=None,
    backend=None,
):
    r"""Compute the distributed triple difference-in-differences estimator.

    Distributed implementation of the DDD estimator from [1]_ for datasets
    that exceed single-machine memory. This function accepts a PySpark
    DataFrame and dispatches to the distributed multi-period estimator.

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

    Users do not need to call this function directly. Passing a PySpark
    DataFrame to :func:`~moderndid.ddd` will automatically dispatch here.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Data in long format as a Spark DataFrame. Each partition should
        contain complete observations (all columns) for a subset of
        units or individuals.
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
    spark : pyspark.sql.SparkSession, optional
        Active Spark session. If None, a local session is created
        automatically via :func:`~moderndid.spark.get_or_create_spark`.
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
        Number of Spark partitions for distributing computation. If None,
        defaults to Spark's default parallelism via
        :func:`~moderndid.spark.get_default_partitions`.
    max_cohorts : int or None, default None
        Maximum number of treatment cohorts to process in parallel.
        When ``None``, defaults to the number of Spark executor cores.

    Returns
    -------
    DDDMultiPeriodResult
        Group-time DDD average treatment effect results containing:

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
    For datasets that fit in memory, create a Spark DataFrame from an existing
    pandas or polars DataFrame:

    .. code-block:: python

        from pyspark.sql import SparkSession
        from moderndid import ddd, gen_dgp_mult_periods

        spark = SparkSession.builder.master("local[*]").getOrCreate()
        dgp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
        sdf = spark.createDataFrame(dgp["data"].to_pandas())

        result = ddd(
            data=sdf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            est_method="dr",
        )

    For large datasets stored on disk, read directly into Spark:

    .. code-block:: python

        sdf = spark.read.parquet("large_panel/*.parquet")
        result = ddd(
            data=sdf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
        )

    To use an existing Spark session:

    .. code-block:: python

        from moderndid.spark import spark_ddd

        result = spark_ddd(
            data=sdf,
            yname="y",
            tname="time",
            idname="id",
            gname="group",
            pname="partition",
            spark=spark,
        )

    See Also
    --------
    ddd : Local (non-distributed) DDD estimator. Automatically dispatches
        to ``spark_ddd`` when passed a Spark DataFrame.
    get_or_create_spark : Get or create a Spark session.
    get_default_partitions : Compute default partition count from Spark.

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

    use_gpu = False
    if backend is not None:
        _validate_backend_name(backend)
        use_gpu = backend.lower() == "cupy"

    spark = get_or_create_spark(spark)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    is_spark = isinstance(data, SparkDataFrame)

    if not is_spark:
        if isinstance(data, pl.DataFrame):
            pdf = data.to_pandas()
            sdf = spark.createDataFrame(pdf)
        else:
            sdf = spark.createDataFrame(data)
    else:
        sdf = data

    required_cols = [yname, tname, gname, pname]
    if idname is not None:
        required_cols.append(idname)
    if weightsname is not None:
        required_cols.append(weightsname)
    validate_spark_input(sdf, required_cols)

    covariate_cols = get_covariate_names(xformla)

    return spark_ddd_mp(
        spark=spark,
        data=sdf,
        y_col=yname,
        time_col=tname,
        id_col=idname,
        group_col=gname,
        partition_col=pname,
        covariate_cols=covariate_cols,
        control_group=control_group,
        base_period=base_period,
        est_method=est_method,
        weightsname=weightsname,
        boot=boot,
        biters=biters,
        cband=cband,
        cluster=cluster,
        alpha=alpha,
        trim_level=trim_level,
        allow_unbalanced_panel=allow_unbalanced_panel,
        random_state=random_state,
        n_partitions=n_partitions,
        max_cohorts=max_cohorts,
        panel=panel,
        use_gpu=use_gpu,
    )
