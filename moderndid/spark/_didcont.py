"""Entry point for distributed continuous DiD estimation via Spark."""

from __future__ import annotations

import logging

import polars as pl

from ._utils import get_or_create_spark, validate_spark_input


def spark_cont_did(
    data,
    yname,
    tname,
    idname,
    gname=None,
    dname=None,
    xformla="~1",
    spark=None,
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
    **kwargs,
):
    r"""Compute distributed continuous treatment DiD via Spark.

    Distributed implementation of the continuous treatment DiD estimator
    from [1]_ for datasets that exceed single-machine memory. This function
    accepts a Spark DataFrame, collects it to the driver, and distributes
    the :math:`(g,t)` cell computations.

    The underlying methodology is identical to :func:`~moderndid.cont_did`.
    Each group-time cell's ATT and ACRT estimation is independent and can
    be parallelized.

    Users do not need to call this function directly. Passing a Spark
    DataFrame to :func:`~moderndid.cont_did` will automatically dispatch
    here.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Data in long format as a Spark DataFrame.
    yname : str
        Name of the outcome variable column.
    tname : str
        Name of the column containing time periods.
    idname : str
        Name of the unit identifier column.
    gname : str, optional
        Name of the treatment group column.
    dname : str
        Name of the continuous treatment (dose) column.
    xformla : str, default="~1"
        Formula for covariates.
    spark : pyspark.sql.SparkSession, optional
        Spark session. If None, an active session is used or created.
    target_parameter : {"level", "slope"}, default="level"
        Type of treatment effect.
    aggregation : {"dose", "eventstudy"}, default="dose"
        How to aggregate treatment effects.
    dose_est_method : {"parametric", "cck"}, default="parametric"
        Method for estimating dose-specific effects.
    dvals : array-like, optional
        Dose values at which to compute effects.
    degree : int, default=3
        B-spline degree.
    num_knots : int, default=0
        Number of interior knots.
    control_group : {"notyettreated", "nevertreated"}, default="notyettreated"
        Which units to use as controls.
    anticipation : int, default=0
        Number of anticipation periods.
    alp : float, default=0.05
        Significance level.
    boot : bool, default=False
        Whether to use bootstrap inference.
    boot_type : str, default="multiplier"
        Type of bootstrap.
    biters : int, default=1000
        Number of bootstrap iterations.
    base_period : {"varying", "universal"}, default="varying"
        Base period selection.
    random_state : int, optional
        Random seed for bootstrap.
    n_partitions : int, optional
        Not used (reserved for API consistency).

    Returns
    -------
    DoseResult or PTEResult
        Treatment effect results.

    See Also
    --------
    cont_did : Local (non-distributed) continuous DiD estimator.

    References
    ----------

    .. [1] Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. (2024).
           "Difference-in-differences with a continuous treatment."
    """
    spark = get_or_create_spark(spark)
    logging.getLogger("py4j").setLevel(logging.ERROR)

    from pyspark.sql import DataFrame as SparkDataFrame

    is_spark = isinstance(data, SparkDataFrame)

    if not is_spark:
        if isinstance(data, pl.DataFrame):
            pdf = data.to_pandas()
            sdf = spark.createDataFrame(pdf)
        else:
            sdf = spark.createDataFrame(data)
    else:
        sdf = data

    required_cols = [yname, tname, idname]
    if dname is not None:
        required_cols.append(dname)
    if gname is not None:
        required_cols.append(gname)
    if weightsname is not None:
        required_cols.append(weightsname)
    validate_spark_input(sdf, required_cols)

    from ._didcont_mp import spark_cont_did_mp

    return spark_cont_did_mp(
        spark=spark,
        data=sdf,
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
        **kwargs,
    )
