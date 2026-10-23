"""Group-time average treatment effects of a panel held in a Spark DataFrame."""

import warnings

import pyspark

from moderndid.core.preprocess import BasePeriod, ControlGroup, DIDConfig, EstimationMethod
from moderndid.core.preprocess.validators import check_columns
from moderndid.did.shards.driver import _check_random_state, fit_att_gt_on_shards

from .executor import SparkExecutor
from .preprocess import _covariate_names, _release, build_unit_table, schema_frame


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
    """Compute group-time average treatment effects of a balanced panel held in a Spark DataFrame.

    The arguments follow :func:`~moderndid.att_gt`. Each step of the estimator
    runs in :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards` as a pass
    over the table of units that
    :func:`~moderndid.spark.preprocess.build_unit_table` builds. The rows stay
    on the Spark executors.

    Without the bootstrap, the estimates and standard errors agree with those
    of the in-memory fit to rounding. Since a Spark fit keys each bootstrap
    multiplier to its unit or cluster, a bootstrap fit draws different
    multipliers from an in-memory fit with the same ``random_state``.

    The fit needs pyspark 3.5 or later, ``panel=True``,
    ``allow_unbalanced_panel=False``, a built-in ``est_method``, and
    ``n_jobs=1``. The ``random_state`` must be None, a non-negative integer,
    a SeedSequence, a BitGenerator, or a Generator. Any other choice raises a
    ValueError before Spark runs a job.

    Returns
    -------
    MPResult
        The result that :func:`~moderndid.att_gt` describes. Its
        ``influence_moments`` field holds :class:`~moderndid.InfluenceMoments`
        in place of ``influence_func``, ``G``, and ``weights_ind``.
    """
    # Since a Databricks runtime brings its own pyspark and skips the extra's floor, the version is checked here.
    release = pyspark.__version__.split(".")[:2]
    if all(part.isdigit() for part in release) and tuple(int(part) for part in release) < (3, 5):
        raise ValueError(
            f"pyspark {pyspark.__version__} is installed. A fit of att_gt on a Spark DataFrame needs pyspark 3.5 or "
            "later. Upgrade pyspark or, on Databricks, choose Databricks Runtime 16.4 LTS or later. The Spark setup "
            "guide of the moderndid docs says which runtimes to choose."
        )
    if not panel:
        raise ValueError(
            "panel=False is not supported for a Spark DataFrame. Pass panel=True to fit the data as a balanced panel."
        )
    if allow_unbalanced_panel:
        raise ValueError(
            "allow_unbalanced_panel=True is not supported for a Spark DataFrame. "
            "Pass allow_unbalanced_panel=False to keep the units observed in every period."
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
    unit_table, config, cohort_counts, n_obs, cluster_counts, cached = build_unit_table(data, config)
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
        executor = SparkExecutor(unit_table, config.time_periods_count, len(_covariate_names(config)))
        result = fit_att_gt_on_shards(
            executor, config, cohort_counts, random_state, estimation_params, cluster_counts=cluster_counts
        )
    except BaseException as error:
        _release(cached, error)
        raise
    _release(cached)
    return result
