"""Preprocessing of a Spark DataFrame into the unit table of a sharded att_gt fit."""

import math
import operator
import warnings
from functools import reduce

import numpy as np
import polars as pl
from pyspark.errors import PySparkException
from pyspark.sql import functions as F
from pyspark.sql import types as T

from moderndid.core.preprocess import PreprocessDataBuilder
from moderndid.core.preprocess.constants import NEVER_TREATED_VALUE, ControlGroup, DataFormat
from moderndid.core.preprocess.models import ValidationResult
from moderndid.core.preprocess.transformers import ColumnSelector
from moderndid.core.preprocess.utils import extract_vars_from_formula
from moderndid.did.shards.layout import CLUSTER, COHORT, COVARIATE, OUTCOME, UNIT, WEIGHT


def schema_frame(data):
    """Return an empty polars DataFrame with the columns of a Spark DataFrame.

    Since the in-memory column checks need only the names of the columns and
    whether each one holds numbers, they run on it without a Spark job. A
    numeric column becomes Float64, a boolean column Boolean, and any other
    column String.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Data whose schema the result copies.

    Returns
    -------
    polars.DataFrame
        A DataFrame with no rows and the columns of the data in their order.
    """
    schema = {}
    for field in data.schema.fields:
        if isinstance(field.dataType, T.NumericType):
            schema[field.name] = pl.Float64
        elif isinstance(field.dataType, T.BooleanType):
            schema[field.name] = pl.Boolean
        else:
            schema[field.name] = pl.String
    return pl.DataFrame(schema=schema)


def build_unit_table(data, config):
    """Build the unit table that a Spark fit of att_gt reads.

    The steps mirror the in-memory preprocessing of a balanced panel in their
    order and with the same warnings and errors. Weights or covariates that are
    not numbers also raise a ValueError before any job runs. So does a decimal
    identifier or cluster with fraction digits or more than 18 digits. Apart
    from the column checks on the schema, the steps take one aggregation and
    one shuffle of the long data. No row of the data reaches the driver.

    The unit table follows the layout that
    :func:`~moderndid.did.shards.layout.shard_from_arrow` reads. In a clustered
    fit every partition holds all the units of its clusters. When the session
    refuses to persist the table of all units and to checkpoint it, a warning
    says that each pass of the fit rebuilds it.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Panel data in long format.
    config : DIDConfig
        Configuration of the fit with the options that att_gt passes.

    Returns
    -------
    tuple
        Tuple containing:

        - **unit_table**: Spark DataFrame with one row for each unit of the balanced panel
        - **config**: The configuration with its computed fields set as the in-memory preprocessing sets them
        - **cohort_counts**: Polars DataFrame of the number of units of each cohort in the columns ``cohort``
          and ``cohort_size`` with the cohorts in increasing order
        - **n_obs**: Number of rows of the balanced panel
        - **cluster_counts**: Pair of the number of clusters and a dict from each cohort to the number of
          clusters of its units, or None without a cluster variable besides the unit identifier
        - **cached**: Persisted table to unpersist once the fit ends, or None when the session refused to persist it
    """
    frame = schema_frame(data)
    PreprocessDataBuilder().with_data(frame).with_config(config).validate()

    names = ColumnSelector().transform(frame, config).columns
    types = {field.name: field.dataType for field in data.schema.fields}
    covariates = _covariate_names(config)
    cluster = next((name for name in config.clustervars if name != config.idname), None)
    weighted = config.weightsname is not None
    _check_value_types(config, covariates, types)
    casts = _decimal_casts(config, cluster, types)
    if casts:
        data = data.withColumns({name: _column(name).cast(dtype) for name, dtype in casts.items()})
        types.update(casts)

    flags = [_missing(name, types[name], keep_infinite=name == config.gname) for name in names]
    kept = ~reduce(operator.or_, flags)
    facts = _long_facts(data, config, flags, kept)

    if facts["kept"] < facts["rows"]:
        if facts["kept"] == 0:
            columns = [f"'{name}'" for name, count in zip(names, facts["missing"], strict=True) if count > 0]
            where = columns[0] if len(columns) == 1 else f"at least one of the columns {', '.join(columns)}"
            raise ValueError(f"Every row has a missing value in {where}. No data is left to estimate from.")
        warnings.warn(
            f"Dropped {facts['rows'] - facts['kept']} rows from original data due to missing values", UserWarning
        )

    groups = facts["groups"]
    last_start = max(period for period, _ in groups) + config.anticipation
    early_until = min(period for period, _ in groups) + config.anticipation
    encoded = [(period, NEVER_TREATED_VALUE if g == 0 or g > last_start else g) for period, g in groups]
    later = [(period, cohort) for period, cohort in encoded if cohort > early_until]
    remaining = sorted({cohort for _, cohort in later})
    periods = sorted({period for period, _ in later})
    latest = None
    if remaining and NEVER_TREATED_VALUE not in remaining:
        latest = remaining[-1]
        periods = [period for period in periods if period < latest - config.anticipation]
    coerced = latest if config.control_group == ControlGroup.NEVER_TREATED else None

    # Since the weights check raises only after the structural checks, a mean that is not positive must not stop the
    # shuffle that runs those checks first.
    mean_weight = facts["weight_sum"] / facts["kept"] if weighted else 1.0
    long = _kept_rows(data, config, kept, periods, mean_weight if mean_weight > 0 else 1.0, cluster, covariates, types)
    table = _unit_rows(
        long, len(periods), weighted, cluster is not None, len(covariates), (last_start, early_until, coerced)
    )
    # Since a refused persist is caught below, an analysis error in the table surfaces here first.
    _ = table.schema
    # Because the unit table only views this table of all units, unpersisting the unit table wouldn't free the cache.
    table, cached = _persist(table)
    try:
        rows = table.rollup(COHORT).agg(*_summary_aggregates(cluster is not None)).collect()
        total = next(row.asDict() for row in rows if row["total"] == 1)
        per_cohort = sorted(
            ((row[COHORT], row.asDict()) for row in rows if row["total"] == 0 and row["units"] > 0),
            key=lambda item: item[0],
        )

        errors = _structure_errors(total, long, config, cluster is not None)
        if errors:
            ValidationResult(is_valid=False, errors=errors).raise_if_invalid()

        if weighted and (facts["negative_weights"] > 0 or mean_weight <= 0):
            raise ValueError(f"The weights variable '{config.weightsname}' must be non-negative with a positive mean.")

        negative = [cohort for _, cohort in groups if cohort < 0]
        if negative:
            raise ValueError(
                f"gname = '{config.gname}' holds negative values such as {min(negative):g}. "
                "It must hold 0 for never-treated units and the first treated period for the others. "
                "Since 0 marks never-treated units, shift the periods so that the earliest one is positive."
            )

        if total["early"] > 0:
            warnings.warn(f"Dropped {total['early']} units that were already treated in the first period", UserWarning)
        if coerced is not None:
            warnings.warn(
                "No never-treated group is available. "
                "The last treated cohort is being coerced as 'never-treated' units.",
                UserWarning,
            )

        n_units = total["units"]
        if n_units < total["counted"]:
            warnings.warn(f"Dropped {total['counted'] - n_units} units while converting to balanced panel", UserWarning)
        if n_units == 0:
            raise ValueError(
                "All observations dropped while converting to balanced panel. "
                "Consider setting panel=False and/or revisiting 'idname'"
            )

        _update_config(config, periods, [cohort for cohort, _ in per_cohort], n_units)
        _check_groups(config, per_cohort, len(covariates))

        cohort_counts = pl.DataFrame(
            {
                "cohort": pl.Series([cohort for cohort, _ in per_cohort], dtype=pl.Float64),
                "cohort_size": pl.Series([counts["units"] for _, counts in per_cohort], dtype=pl.UInt32),
            }
        )
        cluster_counts = None
        if cluster is not None:
            cluster_counts = (total["clusters"], {cohort: counts["clusters"] for cohort, counts in per_cohort})

        layout = [UNIT, COHORT, OUTCOME] + [WEIGHT] * weighted + [COVARIATE] * bool(covariates)
        layout += [CLUSTER] * (cluster is not None)
        # Since a persisted table keeps all its shuffle partitions, the unit table merges them to about a megabyte
        # each, as adaptive execution would. That spares each pass a task per partition and keeps every cluster whole.
        unit_bytes = 8 * len(periods) * (1 + weighted + len(covariates)) + 64
        n_partitions = max(1, math.ceil(n_units * unit_bytes / 2**20))
        unit_table = table.filter(F.col("keep")).select(*layout).coalesce(n_partitions)
    except BaseException as error:
        # Since an interrupt or any failure before the fit takes over the table would leave it cached, every error
        # releases it here.
        _release(cached, error)
        raise
    return unit_table, config, cohort_counts, n_units * len(periods), cluster_counts, cached


def _column(name):
    """Return the column that a name refers to even when the name holds dots or spaces."""
    return F.col("`" + name.replace("`", "``") + "`")


def _covariate_names(config):
    """Return the covariates of the formula without the outcome as the in-memory tensors take them."""
    if not config.xformla or config.xformla == "~1":
        return []
    return [name for name in extract_vars_from_formula(config.xformla) if name != config.yname]


def _is_number(dtype):
    """Tell whether a Spark type holds values that cast to a double without error."""
    return isinstance(dtype, T.NumericType | T.BooleanType)


def _check_value_types(config, covariates, types):
    """Reject weights and covariates that are not numbers before any job casts them."""
    if config.weightsname is not None and not _is_number(types[config.weightsname]):
        raise ValueError(f"weightsname='{config.weightsname}' is not numeric. Convert the weights to numbers.")
    for name in covariates:
        if not _is_number(types[name]):
            raise ValueError(f"'{name}' in xformla is not numeric. Convert it to numbers or leave it out of xformla.")


def _decimal_casts(config, cluster, types):
    """Return the types that the fit reads its decimal columns as."""
    # Since the shards would key a decimal by its text and a double is exact only to 2**53, decimal ids and clusters
    # become 64-bit integers. A double holds the small values of decimal periods and cohorts exactly.
    casts = {name: T.DoubleType() for name in (config.tname, config.gname) if isinstance(types[name], T.DecimalType)}
    keys = {config.idname: (f"idname='{config.idname}'", "units")}
    if cluster is not None:
        keys[cluster] = (f"'{cluster}' in clustervars", "clusters")
    for name, (argument, members) in keys.items():
        dtype = types[name]
        if not isinstance(dtype, T.DecimalType):
            continue
        if dtype.scale != 0 or dtype.precision > 18:
            raise ValueError(
                f"{argument} is a {dtype.simpleString()} column. Since a Spark fit reads decimal identifiers and "
                "clusters as 64-bit integers, they must have no fraction digits and at most 18 digits. Cast the column "
                f"to bigint or code the {members} as integers first."
            )
        casts[name] = T.LongType()
    return casts


def _missing(name, dtype, keep_infinite):
    """Mark the rows whose value in a column counts as missing.

    A null counts as missing in every column. In a float column so do a NaN
    and an infinity. Since an infinite cohort marks a never-treated unit,
    ``keep_infinite`` lets the infinities stay.
    """
    column = _column(name)
    missing = column.isNull()
    if isinstance(dtype, T.FloatType | T.DoubleType):
        missing = missing | F.isnan(column)
        if not keep_infinite:
            missing = missing | (F.abs(column) == F.lit(float("inf")))
    return missing


def _long_facts(data, config, flags, kept):
    """Aggregate the long data by period and cohort into the counts that preprocessing needs.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Panel data in long format.
    config : DIDConfig
        Configuration of the fit.
    flags : list[pyspark.sql.Column]
        Missing-value marks of the selected columns in their order.
    kept : pyspark.sql.Column
        True for each row without a missing value.

    Returns
    -------
    dict
        Dictionary containing the counts:

        - **rows**: Number of rows
        - **kept**: Number of rows without a missing value
        - **missing**: Number of missing values in each selected column
        - **groups**: Sorted pairs of the period and the float cohort that the kept rows hold
        - **weight_sum**: Sum of the sampling weights of the kept rows
        - **negative_weights**: Number of kept rows with a negative sampling weight
    """
    aggregates = [F.count(F.lit(1)).alias("rows"), F.count(F.when(kept, 1)).alias("kept")]
    aggregates += [F.count(F.when(flag, 1)).alias(f"missing{index}") for index, flag in enumerate(flags)]
    if config.weightsname is not None:
        weight = _column(config.weightsname).cast("double")
        aggregates += [
            F.sum(F.when(kept, weight)).alias("weight_sum"),
            F.count(F.when(kept & (weight < 0), 1)).alias("negative_weights"),
        ]
    rows = data.groupBy(_column(config.tname).alias("t"), _column(config.gname).alias("g")).agg(*aggregates).collect()

    kept_rows = [row for row in rows if row["kept"] > 0]
    weighted = config.weightsname is not None
    return {
        "rows": sum(row["rows"] for row in rows),
        "kept": sum(row["kept"] for row in rows),
        "missing": [sum(row[f"missing{index}"] for row in rows) for index in range(len(flags))],
        "groups": sorted((row["t"], float(row["g"])) for row in kept_rows),
        "weight_sum": math.fsum(row["weight_sum"] for row in kept_rows) if weighted else 0.0,
        "negative_weights": sum(row["negative_weights"] for row in kept_rows) if weighted else 0,
    }


def _kept_rows(data, config, kept, periods, mean_weight, cluster, covariates, types):
    """Select the rows without missing values with each column under the name of its role.

    The columns take the names ``id``, ``t``, ``i``, ``g``, ``y``, ``w``,
    ``c``, and ``x0`` onward. The period index ``i`` is the position of the
    row's period in the sorted ``periods`` and -1 for any other period. The
    weights are divided by the positive ``mean_weight``. A float cluster of
    negative zero joins the cluster of zero.
    """
    period = _column(config.tname)
    index = F.array_position(F.array(*[F.lit(value) for value in periods]), period) - 1 if periods else F.lit(-1)
    columns = [
        _column(config.idname).alias("id"),
        period.alias("t"),
        index.alias("i"),
        _column(config.gname).cast("double").alias("g"),
        _column(config.yname).cast("double").alias("y"),
    ]
    if config.weightsname is not None:
        columns.append((_column(config.weightsname).cast("double") / F.lit(mean_weight)).alias("w"))
    if cluster is not None:
        key = _column(cluster)
        if isinstance(types[cluster], T.FloatType | T.DoubleType):
            key = key + F.lit(0.0)
        columns.append(key.alias("c"))
    columns += [_column(name).cast("double").alias(f"x{position}") for position, name in enumerate(covariates)]
    return data.filter(kept).select(*columns)


def _unit_rows(long, n_periods, weighted, clustered, n_covariates, cohorts):
    """Collect the rows of each unit into one row of the table of all units.

    Since the summary also counts the units that preprocessing drops, every unit
    with a kept row gets a row that records its checks. Only a unit that is not
    treated in the first period and holds all ``n_periods`` periods stays in the
    fit and keeps its outcomes, weights, and covariates in the order of the
    periods.

    The ``cohorts`` triple holds the latest start of a treated cohort, the latest
    cohort that counts as treated in the first period, and the cohort that
    becomes never-treated for lack of never-treated units or None. A cohort of 0
    or one that starts after the latest start becomes infinite.
    """
    last_start, early_until, coerced = cohorts
    fields = ["i", "y"] + ["w"] * weighted + [f"x{position}" for position in range(n_covariates)]
    # Since each task finalizes its own cluster sums, the shuffle hashes on the cluster alone and the later merge only
    # coalesces whole partitions. Any other repartition or a REBALANCE would split clusters and silently miscount them.
    grouped = long.repartition("c").groupBy("c", "id") if clustered else long.groupBy("id")
    units = grouped.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.min("g").alias("g_min"),
        F.max("g").alias("g_max"),
        F.size(F.collect_set("t")).alias("n_times"),
        F.collect_list(F.when(F.col("i") >= 0, F.struct(*fields))).alias("rows"),
    )

    raw = F.col("g_min")
    cohort = F.when((raw == 0) | (raw > F.lit(float(last_start))), F.lit(NEVER_TREATED_VALUE)).otherwise(raw)
    early = cohort <= F.lit(float(early_until))
    if coerced is not None:
        cohort = F.when(cohort == F.lit(coerced), F.lit(NEVER_TREATED_VALUE)).otherwise(cohort)
    present = F.size("rows")
    keep = ~early & (present == n_periods) if n_periods > 0 else F.lit(False)
    # Since array_sort orders the structs by their first field, the period index must stay first among the fields.
    ordered = F.array_sort("rows")

    columns = [
        F.col("id").alias(UNIT),
        cohort.alias(COHORT),
        F.when(keep, F.transform(ordered, lambda row: row["y"])).alias(OUTCOME),
    ]
    if weighted:
        columns.append(F.when(keep, F.transform(ordered, lambda row: row["w"])).alias(WEIGHT))
    if n_covariates > 0:
        values = F.transform(ordered, lambda row: F.array(*[row[f"x{position}"] for position in range(n_covariates)]))
        columns.append(F.when(keep, F.flatten(values)).alias(COVARIATE))
    if clustered:
        columns.append(F.col("c").alias(CLUSTER))
    columns += [
        keep.alias("keep"),
        early.alias("early"),
        (~early & (present > 0)).alias("counted"),
        F.col("n_rows"),
        (F.col("g_min") != F.col("g_max")).alias("varying"),
        (F.col("n_rows") > F.col("n_times")).alias("repeated"),
    ]
    return units.select(*columns)


def _persist(table):
    """Persist the table of all units or checkpoint it when the session refuses to persist it.

    The result pairs the table to read with the persisted table to unpersist
    once the fit ends or with None when nothing was persisted. When the session
    refuses the checkpoint too, a warning says that each pass rebuilds the table.
    """
    try:
        # Because PySpark's StorageLevel.MEMORY_AND_DISK constant is the serialized level, persist() takes no argument
        # and keeps its default MEMORY_AND_DISK_DESER.
        return table.persist(), table
    except PySparkException:
        # Since a session such as Databricks serverless refuses to persist, the checkpoint keeps the passes cheap.
        try:
            return table.localCheckpoint(), None
        except PySparkException:
            warnings.warn(
                "Spark refused to persist the table of units and to checkpoint it. "
                "Each pass of the fit rebuilds the table from the data.",
                UserWarning,
            )
            return table, None


def _release(cached, error=None):
    """Unpersist the table of all units without letting a failed release hide the error that ended the fit."""
    if cached is None:
        return
    try:
        cached.unpersist()
    except Exception as release_error:
        if error is None:
            raise
        error.add_note(f"Releasing the persisted table of units failed too: {release_error!r}")


def _summary_aggregates(clustered):
    """Return the aggregates of the summary that the rollup over cohorts computes."""
    aggregates = [
        F.count(F.when(F.col("keep"), 1)).alias("units"),
        F.max("n_rows").alias("max_rows"),
        F.count(F.when(F.col("varying"), 1)).alias("varying"),
        F.count(F.when(F.col("repeated"), 1)).alias("repeated"),
        F.count(F.when(F.col("early"), 1)).alias("early"),
        F.count(F.when(F.col("counted"), 1)).alias("counted"),
        F.grouping(COHORT).alias("total"),
    ]
    if clustered:
        aggregates += [
            F.count_distinct(F.when(F.col("keep"), F.col(CLUSTER))).alias("clusters"),
            F.count(F.lit(1)).alias("groups"),
            F.count_distinct(UNIT).alias("distinct_units"),
        ]
    return aggregates


def _structure_errors(total, long, config, clustered):
    """Return the message of each in-memory structural check that the kept rows fail in the order of the checks."""
    varying_cluster = clustered and total["groups"] != total["distinct_units"]
    if varying_cluster:
        # Since the units are grouped within clusters, a unit whose cluster varies splits into several rows. Its
        # checks then run on the kept rows grouped by unit alone.
        total = _unit_checks(long)

    errors = []
    if total["varying"] > 0:
        errors.append(
            "The value of gname (treatment variable) must be the same across all "
            "periods for each particular unit. The treatment must be irreversible."
        )
    if total["max_rows"] == 1:
        errors.append(
            "panel=True was specified, but no units appear in multiple time periods. "
            "Your data appears to be repeated cross-sections. "
            "Set panel=False to use the repeated cross-section estimator."
        )
    if total["repeated"] > 0:
        errors.append(_duplicate_error(long, config))
    if varying_cluster:
        errors.append(
            "DiD cannot handle time-varying cluster variables at the moment. Please check your cluster variable."
        )
    return errors


def _unit_checks(long):
    """Summarize the structural checks of the kept rows grouped by unit alone."""
    units = long.groupBy("id").agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.min("g").alias("g_min"),
        F.max("g").alias("g_max"),
        F.size(F.collect_set("t")).alias("n_times"),
    )
    row = units.agg(
        F.max("n_rows").alias("max_rows"),
        F.count(F.when(F.col("g_min") != F.col("g_max"), 1)).alias("varying"),
        F.count(F.when(F.col("n_rows") > F.col("n_times"), 1)).alias("repeated"),
    ).collect()[0]
    return row.asDict()


def _duplicate_error(long, config):
    """Describe the units observed more than once in a period by up to three of the repeated pairs."""
    pairs = long.groupBy("id", "t").agg(F.count(F.lit(1)).alias("n")).filter(F.col("n") > 1)
    n_pairs = pairs.count()
    shown = [f"({row['id']}, {row['t']})" for row in pairs.orderBy("id", "t").limit(3).collect()]
    listed = " and ".join(shown) if len(shown) < 3 else f"{', '.join(shown[:-1])}, and {shown[-1]}"
    if n_pairs <= 3:
        noun = "pair" if n_pairs == 1 else "pairs"
        where = f"the ({config.idname}, {config.tname}) {noun} {listed}"
    else:
        where = f"{n_pairs} ({config.idname}, {config.tname}) pairs, such as {listed}"
    return (
        "The value of idname must be unique (by tname). "
        "Some units are observed more than once in a period. "
        f"Rows repeat for {where}."
    )


def _update_config(config, periods, cohorts, n_units):
    """Set the computed fields of the configuration as the in-memory preprocessing sets them."""
    treated = [cohort for cohort in cohorts if np.isfinite(cohort)]
    # Without never-treated units, the latest cohort serves only as a comparison group under not-yet-treated controls.
    if config.control_group == ControlGroup.NOT_YET_TREATED and len(treated) == len(cohorts):
        treated = treated[:-1]
    config.time_periods = np.array(periods)
    config.time_periods_count = len(periods)
    config.treated_groups = np.array(treated)
    config.treated_groups_count = len(treated)
    config.id_count = n_units
    config.data_format = DataFormat.PANEL
    if len(periods) == 2:
        config.cband = False


def _check_groups(config, per_cohort, n_covariates):
    """Check the treated groups and warn about small cohorts as the in-memory preprocessing does."""
    if config.treated_groups_count == 0:
        raise ValueError(
            "No valid groups. The variable in 'gname' should be expressed as "
            "the time a unit is first treated (0 if never-treated)"
        )
    small = [cohort for cohort, counts in per_cohort if counts["units"] < n_covariates + 5]
    if small:
        group_list = ", ".join(np.format_float_positional(float(cohort), trim="-") for cohort in small)
        warnings.warn(
            f"Be aware that there are some small groups in your dataset.\nCheck groups: {group_list}", UserWarning
        )
        if NEVER_TREATED_VALUE in small and config.control_group == ControlGroup.NEVER_TREATED:
            raise ValueError(
                "The never-treated group is too small to serve as the comparison group. "
                "Pass control_group='notyettreated' to compare with units not yet treated."
            )
