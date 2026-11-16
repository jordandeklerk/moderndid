"""Group-time average treatment effects of data held in a Spark DataFrame."""

import warnings
from functools import partial

import pyspark

from moderndid.core.preprocess import BasePeriod, ControlGroup, DIDConfig, EstimationMethod
from moderndid.core.preprocess.validators import check_columns
from moderndid.did.shards.driver import _check_random_state, fit_att_gt_on_shards
from moderndid.did.shards.layout import row_shard_from_arrow, shard_from_arrow
from moderndid.did.shards.row_driver import fit_att_gt_on_row_shards

from .executor import SparkExecutor
from .preprocess import _covariate_names, _release, build_row_table, build_unit_table, schema_frame


def spark_att_gt(
    data,
    yname,
    tname,
    idname,
    gname,
    xformla,
    weightsname,
    alpha,
    cband,
    boot,
    biters,
    clustervars,
    est_method,
    panel,
    allow_unbalanced_panel,
    control_group,
    anticipation,
    base_period,
    random_state,
    n_jobs,
):
    """Compute group-time average treatment effects of data held in a Spark DataFrame.

    The arguments follow :func:`~moderndid.att_gt`. A balanced panel from
    :func:`~moderndid.spark.preprocess.build_unit_table` goes to
    :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards`. Repeated
    cross-sections from :func:`~moderndid.spark.preprocess.build_row_table`
    and an unbalanced panel kept by ``allow_unbalanced_panel`` go to
    :func:`~moderndid.did.shards.row_driver.fit_att_gt_on_row_shards`. The fit
    needs pyspark 3.5 or later, a built-in ``est_method``, ``n_jobs=1``, and a
    ``random_state`` that
    :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards` accepts. Any
    other choice raises a ValueError before Spark runs a job.

    Without the bootstrap, the estimates and standard errors match those of
    the in-memory fit to rounding. A bootstrap fit draws different multipliers
    from an in-memory fit with the same ``random_state`` because a Spark fit
    keys each multiplier to its unit, its cluster, or the values of its row.

    Returns
    -------
    MPResult
        The result that :func:`~moderndid.att_gt` describes for a fit on a
        Spark DataFrame.
    """
    # The version is checked here because a Databricks runtime brings its own pyspark and skips the extra's floor.
    release = pyspark.__version__.split(".")[:2]
    if all(part.isdigit() for part in release) and tuple(int(part) for part in release) < (3, 5):
        raise ValueError(
            f"pyspark {pyspark.__version__} is installed. A fit of att_gt on a Spark DataFrame needs pyspark 3.5 or "
            "later. Upgrade pyspark or, on Databricks, choose Databricks Runtime 16.4 LTS or later. The Spark setup "
            "guide of the moderndid docs says which runtimes to choose."
        )
    if callable(est_method):
        name = getattr(est_method, "__name__", type(est_method).__name__)
        raise ValueError(
            f"est_method={name} is not supported for a Spark DataFrame. Pass est_method='dr', 'ipw', or 'reg'."
        )
    if n_jobs != 1:
        raise ValueError(
            f"n_jobs={n_jobs} is not supported for a Spark DataFrame. "
            "Pass n_jobs=1 and let the Spark executors run the fit in parallel."
        )
    _check_random_state(random_state, "a Spark DataFrame")

    check_columns(
        schema_frame(data),
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla,
        weightsname=weightsname,
        clustervars=clustervars,
    )
    if anticipation > 0:
        warnings.warn(
            f"anticipation = {anticipation}. Never-treated units (with group status 0 or inf) "
            "are assumed to never anticipate treatment. Anticipation only applies to eventually-treated units.",
            UserWarning,
        )

    config = DIDConfig(
        yname=yname,
        tname=tname,
        idname=idname,
        gname=gname,
        xformla=xformla if xformla is not None else "~1",
        panel=panel,
        allow_unbalanced_panel=allow_unbalanced_panel,
        control_group=ControlGroup(control_group),
        anticipation=anticipation,
        weightsname=weightsname,
        alpha=alpha,
        boot=boot,
        cband=cband,
        biters=biters,
        clustervars=clustervars if clustervars is not None else [],
        est_method=EstimationMethod(est_method),
        base_period=BasePeriod(base_period),
    )
    built = build_unit_table(data, config) if panel else build_row_table(data, config, draws=boot or cband)
    table, config, cohort_counts, n_obs, cluster_counts, whole_clusters, cached = built
    try:
        if clustervars is not None and idname in clustervars:
            clustervars = [name for name in clustervars if name != idname]

        estimation_params = {
            "yname": yname,
            "idname": idname,
            "control_group": control_group,
            "anticipation_periods": anticipation,
            "estimation_method": est_method,
            "bootstrap": boot,
            "uniform_bands": cband,
            "base_period": base_period,
            "panel": panel,
            "clustervars": clustervars,
            "cluster": None,
            "biters": biters,
            "random_state": random_state,
            "n_units": config.id_count,
            "n_obs": n_obs,
        }
        n_covariates = len(_covariate_names(config))
        # Preprocessing hands a panel in which some unit misses a period to the estimators of repeated cross-sections.
        # The returned configuration, not the caller's panel argument, picks the reader and the driver.
        if config.panel:
            read_shard = partial(shard_from_arrow, n_periods=config.time_periods_count, n_covariates=n_covariates)
            fit = fit_att_gt_on_shards
        else:
            read_shard = partial(row_shard_from_arrow, n_covariates=n_covariates)
            fit = fit_att_gt_on_row_shards
        executor = SparkExecutor(table, read_shard, whole_clusters=whole_clusters)
        result = fit(executor, config, cohort_counts, random_state, estimation_params, cluster_counts=cluster_counts)
    except BaseException as error:
        _release(cached, error, "table of units" if panel else "row table")
        raise
    _release(cached)
    return result
