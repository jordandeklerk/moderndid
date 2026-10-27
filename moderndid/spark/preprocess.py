"""Preprocessing of a Spark DataFrame into the table that a sharded att_gt fit reads."""

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
from moderndid.core.preprocess.constants import NEVER_TREATED_VALUE, ROW_ID_COLUMN, ControlGroup, DataFormat
from moderndid.core.preprocess.models import ValidationResult
from moderndid.core.preprocess.transformers import ColumnSelector
from moderndid.core.preprocess.utils import extract_vars_from_formula
from moderndid.did.shards.layout import (
    CLUSTER,
    COHORT,
    COPY,
    COVARIATE,
    OUTCOME,
    PERIOD,
    RAW_WEIGHT,
    UNIT,
    WEIGHT,
)


def schema_frame(data):
    """Return an empty polars DataFrame with the columns of a Spark DataFrame.

    The in-memory column checks need only the names of the columns and whether
    each one holds numbers. They run on this frame without a Spark job. A numeric
    column becomes Float64, a boolean column Boolean, and any other column String.

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
    """Build the unit table that a Spark fit of att_gt on a panel reads.

    The steps mirror the in-memory preprocessing of a panel in their order and
    with the same warnings and errors. Weights or covariates that are not
    numbers also raise a ValueError before any job runs. So does a decimal
    identifier or cluster with fraction digits or more than 18 digits.

    The table follows the layout that
    :func:`~moderndid.did.shards.layout.shard_from_arrow` reads. With
    ``config.allow_unbalanced_panel``, the fit keeps a unit with rows in only
    some of its periods. If it keeps any, the table takes the layout that
    :func:`~moderndid.did.shards.layout.row_shard_from_arrow` reads. As in
    memory, the configuration then has ``panel`` False.

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

        - **unit_table**: Spark DataFrame with one row for each unit of the fit
        - **config**: The configuration with its computed fields set as the in-memory preprocessing sets them
        - **cohort_counts**: Number of units of each cohort in the columns that
          :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards` reads
        - **n_obs**: Number of rows of the panel that the fit reads
        - **cluster_counts**: Cluster counts as :func:`~moderndid.did.shards.driver.fit_att_gt_on_shards` reads
          them, or None without a cluster variable besides the unit identifier
        - **whole_clusters**: Whether every partition holds all the units of its clusters, or False without a
          cluster variable besides the unit identifier
        - **cached**: Persisted table to unpersist once the fit ends, or None when the session refused to persist it
    """
    data, types, covariates, cluster, kept, facts = _prepare_long_data(data, config)
    weighted = config.weightsname is not None
    unbalanced = config.allow_unbalanced_panel

    groups = facts["groups"]
    last_start = max(period for period, _ in groups) + config.anticipation
    early_until = min(period for period, _ in groups) + config.anticipation
    periods, remaining, coerced = _fit_periods(groups, config, last_start, early_until)
    whole_clusters = _whole_clusters(cluster, facts["clusters"], remaining, periods)

    # The weights check raises only after the structural checks. A mean that is not positive must not stop the shuffle
    # that runs those checks first.
    mean_weight = facts["weight_sum"] / facts["kept"] if weighted else 1.0
    long = _kept_rows(data, config, kept, periods, mean_weight if mean_weight > 0 else 1.0, cluster, covariates, types)
    table = _unit_rows(
        long,
        len(periods),
        weighted,
        cluster is not None,
        whole_clusters,
        len(covariates),
        (last_start, early_until, coerced),
        unbalanced,
    )
    # Reading the schema raises an analysis error in the table here. The handler for a refused persist below would hide
    # it.
    _ = table.schema
    # The unit table only views this table of all units. Unpersisting the unit table wouldn't free the cache.
    table, cached = _persist(table)
    try:
        rows = table.rollup(COHORT).agg(*_summary_aggregates(cluster is not None, unbalanced)).collect()
        total = next(row.asDict() for row in rows if row["total"] == 1)
        per_cohort = sorted(
            ((row[COHORT], row.asDict()) for row in rows if row["total"] == 0 and row["units"] > 0),
            key=lambda item: item[0],
        )

        errors = _structure_errors(total, long, config, cluster is not None)
        if errors:
            ValidationResult(is_valid=False, errors=errors).raise_if_invalid()
        _check_weights_and_cohorts(config, facts, mean_weight)
        _warn_early_and_coerced(total["early"], "units", coerced)

        n_units = total["units"]
        if not unbalanced:
            if n_units < total["counted"]:
                warnings.warn(
                    f"Dropped {total['counted'] - n_units} units while converting to balanced panel", UserWarning
                )
            if n_units == 0:
                raise ValueError(
                    "All observations dropped while converting to balanced panel. "
                    "Consider setting panel=False and/or revisiting 'idname'"
                )
        # The in-memory fit switches a panel to the estimators of repeated cross-sections once some unit misses a
        # period.
        incomplete = unbalanced and total["complete"] < n_units
        if incomplete:
            config.panel = False
        data_format = DataFormat.REPEATED_CROSS_SECTION if incomplete else DataFormat.PANEL
        _update_config(config, periods, [cohort for cohort, _ in per_cohort], n_units, data_format)
        _check_groups(config, per_cohort, len(covariates), rows_per_period=incomplete)

        cohort_counts = _cohort_counts(per_cohort, "units", pl.UInt32)
        cluster_counts = None
        if cluster is not None:
            cluster_counts = (total["clusters"], {cohort: counts["clusters"] for cohort, counts in per_cohort})

        n_rows = total["rows"] if incomplete else n_units * len(periods)
        layout = [UNIT, COHORT] + [PERIOD] * incomplete + [OUTCOME] + [WEIGHT] * weighted
        layout += [COVARIATE] * bool(covariates) + [CLUSTER] * (cluster is not None)
        # A persisted table keeps all its shuffle partitions. Merging them to about a megabyte each, as adaptive
        # execution would, spares each pass a task per partition without splitting a cluster.
        row_bytes = 8 * (1 + weighted + len(covariates) + incomplete)
        n_partitions = max(1, math.ceil((n_rows * row_bytes + 64 * n_units) / 2**20))
        unit_table = table.filter(F.col("keep")).select(*layout).coalesce(n_partitions)
    except BaseException as error:
        # An interrupt or any failure before the fit takes over the table would leave it cached. Every error releases it
        # here.
        _release(cached, error)
        raise
    return unit_table, config, cohort_counts, n_rows, cluster_counts, whole_clusters, cached


def build_row_table(data, config, draws=False):
    """Build the row table that a Spark fit of att_gt on repeated cross-sections reads.

    As :func:`build_unit_table` does for a panel, the steps mirror the
    in-memory preprocessing of repeated cross-sections in their order and with
    the same warnings and errors. With an ``idname``, every row of an
    identifier that holds a row treated in the first period leaves the fit.

    The table follows the layout that
    :func:`~moderndid.did.shards.layout.row_shard_from_arrow` reads. For an
    unclustered fit that draws bootstrap multipliers, one shuffle numbers the
    copies of identical rows for :func:`~moderndid.did.shards.layout.row_keys`.
    For a clustered fit with many clusters, a shuffle by cluster keeps each
    cluster within one partition. Any other fit lays out its rows without a
    shuffle.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Repeated cross-sections in long format.
    config : DIDConfig
        Configuration of the fit with the options that att_gt passes.
    draws : bool, default=False
        Whether the fit draws bootstrap multipliers, as ``boot`` or ``cband``
        asks.

    Returns
    -------
    tuple
        Tuple containing:

        - **row_table**: Spark DataFrame with one row for each observation of the fit
        - **config**: The configuration with its computed fields set as the in-memory preprocessing sets them
        - **cohort_counts**: Number of rows of each cohort in the columns that
          :func:`~moderndid.did.shards.row_driver.fit_att_gt_on_row_shards` reads
        - **n_obs**: Number of rows that the fit reads
        - **cluster_counts**: Cluster counts as :func:`~moderndid.did.shards.row_driver.fit_att_gt_on_row_shards`
          reads them, or None without a cluster variable besides the unit identifier
        - **whole_clusters**: Whether every partition holds all the rows of its clusters, or False without a
          cluster variable besides the unit identifier
        - **cached**: Persisted table to unpersist once the fit ends, or None when the session refused to persist it
    """
    data, types, covariates, cluster, kept, facts = _prepare_long_data(data, config)
    weighted = config.weightsname is not None
    # A clustered fit draws one multiplier per cluster. Only an unclustered fit that draws needs row keys.
    keyed = draws and cluster is None

    groups = facts["groups"]
    observed = sorted({period for period, _ in groups})
    last_start = observed[-1] + config.anticipation
    early_until = observed[0] + config.anticipation
    # The in-memory fit raises the identifier error before the weights error. A mean weight that isn't positive
    # therefore falls back to a scale of one rather than stop the aggregation that checks the identifiers.
    mean_weight = facts["weight_sum"] / facts["kept"] if weighted else 1.0
    scale = mean_weight if mean_weight > 0 else 1.0
    long = _kept_rows(data, config, kept, None, scale, cluster, covariates, types, raw_weights=keyed and weighted)
    early = _cohort_column(F.col("g"), last_start) <= F.lit(float(early_until))
    some_early = any(_coded_cohort(g, last_start) <= early_until for _, g in groups)

    n_early = 0
    if config.idname is not None:
        n_early = _check_identifiers(long, early if some_early else None, len(observed))
    _check_weights_and_cohorts(config, facts, mean_weight)

    rows = long
    if config.idname is None:
        counts = facts["group_counts"]
        n_early = sum(count for (_, g), count in counts.items() if _coded_cohort(g, last_start) <= early_until)
        if n_early > 0:
            rows = long.filter(~early)
    elif n_early > 0:
        # A cohort can vary within an identifier of repeated cross-sections. The periods and cohorts left after dropping
        # the identifiers treated in the first period come from the remaining rows.
        rows = long.join(long.filter(early).select("id").distinct(), on="id", how="left_anti")
        groups = sorted((row["t"], float(row["g"])) for row in rows.select("t", "g").distinct().collect())

    periods, remaining, coerced = _fit_periods(groups, config, last_start, early_until)
    whole_clusters = _whole_clusters(cluster, facts["clusters"], remaining, periods)
    table = _row_rows(
        rows, periods, (last_start, coerced), weighted, keyed, cluster is not None, whole_clusters, len(covariates)
    )
    # Reading the schema raises an analysis error in the table here. The handler for a refused persist below would hide
    # it.
    _ = table.schema
    table, cached = _persist(table, "row table")
    try:
        aggregates = [F.count(F.lit(1)).alias("rows"), F.grouping(COHORT).alias("total")]
        if cluster is not None:
            aggregates.append(F.count_distinct(CLUSTER).alias("clusters"))
        summary = table.rollup(COHORT).agg(*aggregates).collect()
        # A rollup of no rows returns no grand total either. The default counts an empty table as no rows and no
        # clusters.
        total = next((row.asDict() for row in summary if row["total"] == 1), {"rows": 0, "clusters": 0})
        per_cohort = sorted(
            ((row[COHORT], row.asDict()) for row in summary if row["total"] == 0), key=lambda item: item[0]
        )

        _warn_early_and_coerced(n_early, "units" if config.idname is not None else "observations", coerced)
        # Each row is its own observation even when idname names units that rows share. The in-memory fit does the same.
        config.panel = False
        config.true_repeated_cross_sections = True
        config.idname = ROW_ID_COLUMN
        n_rows = total["rows"]
        cohorts = [cohort for cohort, _ in per_cohort]
        _update_config(config, periods, cohorts, n_rows, DataFormat.REPEATED_CROSS_SECTION)
        _check_groups(config, per_cohort, len(covariates), rows_per_period=True)

        # A cohort can hold more than 2**32 - 1 rows at Spark scale.
        cohort_counts = _cohort_counts(per_cohort, "rows", pl.UInt64)
        cluster_counts = None
        if cluster is not None:
            cluster_counts = (total["clusters"], {cohort: counts["clusters"] for cohort, counts in per_cohort})

        # Merging the persisted table's partitions to about a megabyte each spares every pass a task per partition. No
        # cluster splits because whole partitions merge.
        row_bytes = 8 * (1 + weighted + len(covariates)) + 64
        row_table = table.coalesce(max(1, math.ceil(n_rows * row_bytes / 2**20)))
    except BaseException as error:
        # An interrupt or any failure before the fit takes over the table would leave it cached. Every error releases it
        # here.
        _release(cached, error, "row table")
        raise
    return row_table, config, cohort_counts, n_rows, cluster_counts, whole_clusters, cached


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


def _prepare_long_data(data, config):
    """Run the column checks and the missing-data step of the in-memory preprocessing on the long data.

    The result holds the data with its decimal columns cast, the Spark type of
    each column, the names of the covariates, the name of the cluster column or
    None, the mark of the rows without a missing value, and the facts of
    :func:`_long_facts`.
    """
    frame = schema_frame(data)
    PreprocessDataBuilder().with_data(frame).with_config(config).validate()

    names = ColumnSelector().transform(frame, config).columns
    types = {field.name: field.dataType for field in data.schema.fields}
    covariates = _covariate_names(config)
    # att_gt never clusters on the unit identifier. config.idname still holds the caller's idname at this point.
    cluster = next((name for name in config.clustervars if name != config.idname), None)
    _check_value_types(config, covariates, types)
    casts = _decimal_casts(config, cluster, types)
    if casts:
        data = data.withColumns({name: _column(name).cast(dtype) for name, dtype in casts.items()})
        types.update(casts)

    flags = [_missing(name, types[name], keep_infinite=name == config.gname) for name in names]
    kept = ~reduce(operator.or_, flags)
    facts = _long_facts(data, config, flags, kept, cluster)

    if facts["kept"] < facts["rows"]:
        if facts["kept"] == 0:
            columns = [f"'{name}'" for name, count in zip(names, facts["missing"], strict=True) if count > 0]
            where = columns[0] if len(columns) == 1 else f"at least one of the columns {', '.join(columns)}"
            raise ValueError(f"Every row has a missing value in {where}. No data is left to estimate from.")
        warnings.warn(
            f"Dropped {facts['rows'] - facts['kept']} rows from original data due to missing values", UserWarning
        )
    return data, types, covariates, cluster, kept, facts


def _check_value_types(config, covariates, types):
    """Reject weights and covariates that are not numbers before any job casts them."""
    if config.weightsname is not None and not _is_number(types[config.weightsname]):
        raise ValueError(f"weightsname='{config.weightsname}' is not numeric. Convert the weights to numbers.")
    for name in covariates:
        if not _is_number(types[name]):
            raise ValueError(f"'{name}' in xformla is not numeric. Convert it to numbers or leave it out of xformla.")


def _decimal_casts(config, cluster, types):
    """Return the types that the fit reads its decimal columns as."""
    # Decimal ids and clusters become 64-bit integers because the shards would key a decimal by its text and a double is
    # exact only to 2**53. A double holds the small values of decimal periods and cohorts exactly.
    casts = {name: T.DoubleType() for name in (config.tname, config.gname) if isinstance(types[name], T.DecimalType)}
    keys = {}
    if config.idname is not None:
        keys[config.idname] = (f"idname='{config.idname}'", "units")
    if cluster is not None:
        keys[cluster] = (f"'{cluster}' in clustervars", "clusters")
    for name, (argument, members) in keys.items():
        dtype = types[name]
        if not isinstance(dtype, T.DecimalType):
            continue
        if dtype.scale != 0 or dtype.precision > 18:
            raise ValueError(
                f"{argument} is a {dtype.simpleString()} column. A Spark fit reads decimal identifiers and "
                "clusters as 64-bit integers. They must have no fraction digits and at most 18 digits. Cast the column "
                f"to bigint or code the {members} as integers first."
            )
        casts[name] = T.LongType()
    return casts


def _missing(name, dtype, keep_infinite):
    """Mark the rows whose value in a column counts as missing.

    An infinite cohort marks a never-treated unit. ``keep_infinite`` keeps the
    infinities of a float column from counting as missing.
    """
    column = _column(name)
    missing = column.isNull()
    if isinstance(dtype, T.FloatType | T.DoubleType):
        missing = missing | F.isnan(column)
        if not keep_infinite:
            missing = missing | (F.abs(column) == F.lit(float("inf")))
    return missing


def _long_facts(data, config, flags, kept, cluster):
    """Aggregate the long data by period and cohort into the counts that preprocessing needs.

    Parameters
    ----------
    data : pyspark.sql.DataFrame
        Data in long format.
    config : DIDConfig
        Configuration of the fit.
    flags : list[pyspark.sql.Column]
        Missing-value marks of the selected columns in their order.
    kept : pyspark.sql.Column
        True for each row without a missing value.
    cluster : str or None
        Name of the cluster column besides the unit identifier.

    Returns
    -------
    dict
        Dictionary containing the counts:

        - **rows**: Number of rows
        - **kept**: Number of rows without a missing value
        - **missing**: Number of missing values in each selected column
        - **groups**: Sorted pairs of the period and the float cohort that the kept rows hold
        - **group_counts**: Dict from each pair of ``groups`` to its number of kept rows
        - **weight_sum**: Sum of the sampling weights of the kept rows
        - **negative_weights**: Number of kept rows with a negative sampling weight
        - **clusters**: Approximate number of distinct clusters among the kept rows, or None without a cluster
          variable besides the unit identifier
    """
    aggregates = [F.count(F.lit(1)).alias("rows"), F.count(F.when(kept, 1)).alias("kept")]
    aggregates += [F.count(F.when(flag, 1)).alias(f"missing{index}") for index, flag in enumerate(flags)]
    if config.weightsname is not None:
        weight = _column(config.weightsname).cast("double")
        aggregates += [
            F.sum(F.when(kept, weight)).alias("weight_sum"),
            F.count(F.when(kept & (weight < 0), 1)).alias("negative_weights"),
        ]
    keys = [_column(config.tname).alias("t"), _column(config.gname).alias("g")]
    clusters = None
    if cluster is None:
        rows = data.groupBy(*keys).agg(*aggregates).collect()
    else:
        # A cluster such as a state holds units of several cohorts. Only the grand total, at grouping level 3, counts
        # each cluster once. An estimate is enough because the count only picks the layout of the unit table.
        aggregates += [
            F.approx_count_distinct(F.when(kept, _column(cluster))).alias("clusters"),
            F.grouping_id().alias("level"),
        ]
        levels = data.rollup(*keys).agg(*aggregates).collect()
        # A rollup of no rows returns no grand total either. The default counts no clusters for empty data.
        clusters = next((row["clusters"] for row in levels if row["level"] == 3), 0)
        rows = [row for row in levels if row["level"] == 0]

    kept_rows = [row for row in rows if row["kept"] > 0]
    weighted = config.weightsname is not None
    return {
        "rows": sum(row["rows"] for row in rows),
        "kept": sum(row["kept"] for row in rows),
        "missing": [sum(row[f"missing{index}"] for row in rows) for index in range(len(flags))],
        "groups": sorted((row["t"], float(row["g"])) for row in kept_rows),
        "group_counts": {(row["t"], float(row["g"])): row["kept"] for row in kept_rows},
        "weight_sum": math.fsum(row["weight_sum"] for row in kept_rows) if weighted else 0.0,
        "negative_weights": sum(row["negative_weights"] for row in kept_rows) if weighted else 0,
        "clusters": clusters,
    }


def _coded_cohort(cohort, last_start):
    """Return a float cohort as never-treated when it is 0 or starts after the latest start."""
    return NEVER_TREATED_VALUE if cohort == 0 or cohort > last_start else cohort


def _cohort_column(raw, last_start):
    """Return a cohort column that is never-treated where the cohort is 0 or starts after the latest start."""
    return F.when((raw == 0) | (raw > F.lit(float(last_start))), F.lit(NEVER_TREATED_VALUE)).otherwise(raw)


def _fit_periods(groups, config, last_start, early_until):
    """Return the periods and the cohorts of the fit and the cohort that becomes never-treated or None.

    When no unit is never-treated, the periods from ``anticipation`` periods
    before the latest cohort's start onward leave the fit here and in memory.
    """
    encoded = [(period, _coded_cohort(g, last_start)) for period, g in groups]
    later = [(period, cohort) for period, cohort in encoded if cohort > early_until]
    remaining = sorted({cohort for _, cohort in later})
    periods = sorted({period for period, _ in later})
    latest = None
    if remaining and NEVER_TREATED_VALUE not in remaining:
        latest = remaining[-1]
        periods = [period for period in periods if period < latest - config.anticipation]
    coerced = latest if config.control_group == ControlGroup.NEVER_TREATED else None
    return periods, remaining, coerced


def _whole_clusters(cluster, n_clusters, cohorts, periods):
    """Tell whether a clustered fit keeps every cluster within one partition."""
    # A unit's row of moments holds a column for each cell, each treated cohort, and the constant. This count bounds the
    # sums a task keeps per cluster.
    n_treated = sum(cohort != NEVER_TREATED_VALUE for cohort in cohorts)
    n_columns = n_treated * len(periods) + n_treated + 1
    return cluster is not None and not _spread_clusters(n_clusters, n_columns)


def _spread_clusters(n_clusters, n_columns):
    """Tell whether a clustered fit can spread each cluster over all the partitions."""
    # Each task then returns a row of n_columns sums for every cluster it touched. A limit of 2**17 sums keeps each
    # task's rows near a megabyte.
    return n_clusters * n_columns <= 2**17


def _period_index(period, periods):
    """Return the position of each row's period among the sorted periods or -1 for any other period."""
    if not periods:
        return F.lit(-1)
    # PySpark 3.5's array_position accepts only a literal value. A CASE WHEN chain numbers the periods instead.
    index = F.when(period == F.lit(periods[0]), 0)
    for position, value in enumerate(periods[1:], start=1):
        index = index.when(period == F.lit(value), position)
    return index.otherwise(-1)


def _kept_rows(data, config, kept, periods, mean_weight, cluster, covariates, types, raw_weights=False):
    """Select the rows without missing values with each column under the name of its role.

    The columns take the names ``id``, ``t``, ``i``, ``g``, ``y``, ``w``,
    ``r``, ``c``, and ``x0`` onward.
    """
    period = _column(config.tname)
    columns = [] if config.idname is None else [_column(config.idname).alias("id")]
    columns.append(period.alias("t"))
    if periods is not None:
        columns.append(_period_index(period, periods).alias("i"))
    columns += [
        _column(config.gname).cast("double").alias("g"),
        _column(config.yname).cast("double").alias("y"),
    ]
    if config.weightsname is not None:
        weight = _column(config.weightsname).cast("double")
        columns.append((weight / F.lit(mean_weight)).alias("w"))
        if raw_weights:
            columns.append(weight.alias("r"))
    if cluster is not None:
        key = _column(cluster)
        if isinstance(types[cluster], T.FloatType | T.DoubleType):
            # A float cluster of -0.0 joins the cluster of 0.0 once 0.0 is added.
            key = key + F.lit(0.0)
        columns.append(key.alias("c"))
    columns += [_column(name).cast("double").alias(f"x{position}") for position, name in enumerate(covariates)]
    return data.filter(kept).select(*columns)


def _unit_rows(long, n_periods, weighted, clustered, whole_clusters, n_covariates, cohorts, unbalanced=False):
    """Collect the rows of each unit into one row of the table of all units.

    The ``cohorts`` triple holds the latest start of a treated cohort, the
    latest cohort treated in the first period, and the coerced cohort or None.
    """
    last_start, early_until, coerced = cohorts
    fields = ["i", "y"] + ["w"] * weighted + [f"x{position}" for position in range(n_covariates)]
    if whole_clusters:
        # Each task then finalizes its own cluster sums. The shuffle hashes on the cluster alone and the later merge
        # only coalesces whole partitions. Any other repartition or a REBALANCE would silently miscount clusters.
        grouped = long.repartition("c").groupBy("c", "id")
    elif clustered:
        # Hashing on the cluster alone would run a fit of few clusters as a few unequal tasks. Hashing on the unit too
        # spreads each cluster's units over the partitions. The driver then finalizes the cluster sums.
        grouped = long.groupBy("c", "id")
    else:
        grouped = long.groupBy("id")
    units = grouped.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.min("g").alias("g_min"),
        F.max("g").alias("g_max"),
        F.size(F.collect_set("t")).alias("n_times"),
        F.collect_list(F.when(F.col("i") >= 0, F.struct(*fields))).alias("rows"),
    )

    cohort = _cohort_column(F.col("g_min"), last_start)
    early = cohort <= F.lit(float(early_until))
    if coerced is not None:
        cohort = F.when(cohort == F.lit(coerced), F.lit(NEVER_TREATED_VALUE)).otherwise(cohort)
    present = F.size("rows")
    complete = present == n_periods if n_periods > 0 else F.lit(False)
    keep = ~early & ((present > 0) if unbalanced else complete)
    # array_sort orders the structs by their first field. The period index must stay first among the fields.
    ordered = F.array_sort("rows")

    columns = [F.col("id").alias(UNIT), cohort.alias(COHORT)]
    if unbalanced:
        columns.append(F.when(keep, F.transform(ordered, lambda row: row["i"])).alias(PERIOD))
    columns.append(F.when(keep, F.transform(ordered, lambda row: row["y"])).alias(OUTCOME))
    if weighted:
        columns.append(F.when(keep, F.transform(ordered, lambda row: row["w"])).alias(WEIGHT))
    if n_covariates > 0:
        values = F.transform(ordered, lambda row: F.array(*[row[f"x{position}"] for position in range(n_covariates)]))
        columns.append(F.when(keep, F.flatten(values)).alias(COVARIATE))
    if clustered:
        columns.append(F.col("c").alias(CLUSTER))
    # Every unit keeps a row of its checks because the summary also counts the units that preprocessing drops.
    columns += [
        keep.alias("keep"),
        early.alias("early"),
        (~early & (present > 0)).alias("counted"),
        F.col("n_rows"),
        (F.col("g_min") != F.col("g_max")).alias("varying"),
        (F.col("n_rows") > F.col("n_times")).alias("repeated"),
    ]
    if unbalanced:
        columns += [present.alias("present"), complete.alias("complete")]
    return units.select(*columns)


def _row_rows(rows, periods, cohorts, weighted, keyed, clustered, whole_clusters, n_covariates):
    """Select the rows of the fit's periods in the layout of the row table.

    The ``cohorts`` pair holds the first and last entries of the triple of
    :func:`_unit_rows`.
    """
    last_start, coerced = cohorts
    cohort = _cohort_column(F.col("g"), last_start)
    if coerced is not None:
        cohort = F.when(cohort == F.lit(coerced), F.lit(NEVER_TREATED_VALUE)).otherwise(cohort)
    covariates = [f"x{position}" for position in range(n_covariates)]
    columns = [_period_index(F.col("t"), periods).alias(PERIOD), cohort.alias(COHORT), F.col("y").alias(OUTCOME)]
    columns += [F.col("w").alias(WEIGHT)] * weighted + [F.col("r").alias(RAW_WEIGHT)] * (weighted and keyed)
    columns += [F.col(name) for name in covariates] + [F.col("c").alias(CLUSTER)] * clustered
    selected = rows.select(*columns).filter(F.col(PERIOD) >= 0)

    copies = []
    if keyed:
        # A row's key hashes the values in these columns plus its copy index. Grouping identical rows lets each copy
        # take its own index. Adding 0.0 groups -0.0 with 0.0 as the key does.
        floats = [COHORT, OUTCOME] + [RAW_WEIGHT] * weighted + covariates
        keys = [F.col(PERIOD)] + [(F.col(name) + F.lit(0.0)).alias(name) for name in floats]
        # Every row of a group carries the same normalized weight because it depends only on the raw weight.
        carried = [F.first(WEIGHT).alias(WEIGHT)] * weighted
        selected = selected.groupBy(*keys).agg(F.count(F.lit(1)).alias("n"), *carried)
        copies = [F.explode(F.sequence(F.lit(0), F.col("n") - 1)).alias(COPY)]

    layout = [F.col(PERIOD), F.col(COHORT), F.col(OUTCOME)]
    layout += [F.col(WEIGHT)] * weighted + [F.col(RAW_WEIGHT)] * (weighted and keyed)
    if n_covariates > 0:
        layout.append(F.array(*[F.col(name) for name in covariates]).alias(COVARIATE))
    layout += [F.col(CLUSTER)] * clustered
    table = selected.select(*layout, *copies)
    if whole_clusters:
        # Each task finalizes the sums of its own clusters. Hashing on the cluster alone keeps each cluster in one
        # partition. The later merge only coalesces whole partitions.
        table = table.repartition(CLUSTER)
    return table


def _persist(table, name="table of units"):
    """Persist a table or checkpoint it when the session refuses to persist it.

    The result pairs the table to read with the persisted table to unpersist
    once the fit ends or with None when nothing was persisted. The warning for
    a session that refuses both calls the table by ``name``.
    """
    try:
        # PySpark's StorageLevel.MEMORY_AND_DISK constant is the serialized level. persist() takes no argument and keeps
        # its default MEMORY_AND_DISK_DESER.
        return table.persist(), table
    except PySparkException:
        # A session such as Databricks serverless refuses to persist. The checkpoint then keeps the passes cheap.
        try:
            return table.localCheckpoint(), None
        except PySparkException:
            warnings.warn(
                f"Spark refused to persist the {name} and to checkpoint it. "
                "Each pass of the fit rebuilds the table from the data.",
                UserWarning,
            )
            return table, None


def _release(cached, error=None, name="table of units"):
    """Unpersist a cached table without letting a failed release hide the error that ended the fit."""
    if cached is None:
        return
    try:
        cached.unpersist()
    except Exception as release_error:
        if error is None:
            raise
        error.add_note(f"Releasing the persisted {name} failed too: {release_error!r}")


def _summary_aggregates(clustered, unbalanced=False):
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
    if unbalanced:
        aggregates += [
            F.sum(F.when(F.col("keep"), F.col("present"))).alias("rows"),
            F.count(F.when(F.col("keep") & F.col("complete"), 1)).alias("complete"),
        ]
    return aggregates


def _structure_errors(total, long, config, clustered):
    """Return the message of each in-memory structural check that the kept rows fail in the order of the checks."""
    varying_cluster = clustered and total["groups"] != total["distinct_units"]
    if varying_cluster:
        # Grouping the units within clusters splits a unit whose cluster varies into several rows. Its checks then run
        # on the kept rows grouped by unit alone.
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


def _check_identifiers(long, early, n_periods):
    """Warn when the identifiers of repeated cross-sections look like a panel and count the early-treated ones.

    The count is the number of identifiers with a row that ``early`` marks, or
    0 when ``early`` is None.
    """
    aggregates = [F.count(F.lit(1)).alias("n_rows")]
    totals = [F.max("n_rows").alias("max_rows")]
    if early is not None:
        aggregates.append(F.max(F.when(early, 1)).alias("early"))
        totals.append(F.count("early").alias("early"))
    row = long.groupBy("id").agg(*aggregates).agg(*totals).collect()[0]
    if row["max_rows"] == n_periods and n_periods > 1:
        warnings.warn(
            "panel=False was specified, but units appear across all time periods. "
            "Your data appears to be panel data. "
            "Consider setting panel=True to use the panel estimator.",
            UserWarning,
        )
    return 0 if early is None else row["early"]


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


def _check_weights_and_cohorts(config, facts, mean_weight):
    """Raise the errors of the in-memory steps that normalize the weights and code the cohorts in their order."""
    if config.weightsname is not None and (facts["negative_weights"] > 0 or mean_weight <= 0):
        raise ValueError(f"The weights variable '{config.weightsname}' must be non-negative with a positive mean.")

    negative = [cohort for _, cohort in facts["groups"] if cohort < 0]
    if negative:
        raise ValueError(
            f"gname = '{config.gname}' holds negative values such as {min(negative):g}. "
            "It must hold 0 for never-treated units and the first treated period for the others. "
            "Shift the periods so that the earliest one is positive."
        )


def _warn_early_and_coerced(n_early, members, coerced):
    """Warn about the early-treated units or rows and the coerced cohort as the in-memory steps do."""
    if n_early > 0:
        warnings.warn(f"Dropped {n_early} {members} that were already treated in the first period", UserWarning)
    if coerced is not None:
        warnings.warn(
            "No never-treated group is available. The last treated cohort is being coerced as 'never-treated' units.",
            UserWarning,
        )


def _update_config(config, periods, cohorts, n_units, data_format):
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
    config.data_format = data_format
    if len(periods) == 2:
        config.cband = False


def _check_groups(config, per_cohort, n_covariates, rows_per_period=False):
    """Check the treated groups and warn about small cohorts as the in-memory preprocessing does.

    With ``rows_per_period``, a cohort's size is its number of rows divided by
    the number of periods rather than its number of units.
    """
    if config.treated_groups_count == 0:
        raise ValueError(
            "No valid groups. The variable in 'gname' should be expressed as "
            "the time a unit is first treated (0 if never-treated)"
        )
    # For an unbalanced panel and repeated cross-sections, the in-memory check sizes a cohort as its rows over the
    # number of periods. This check does the same.
    sizes = [
        (cohort, counts["rows"] / config.time_periods_count if rows_per_period else counts["units"])
        for cohort, counts in per_cohort
    ]
    small = [cohort for cohort, size in sizes if size < n_covariates + 5]
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


def _cohort_counts(per_cohort, measure, dtype):
    """Return the polars DataFrame of the size of each cohort in increasing order of the cohorts."""
    return pl.DataFrame(
        {
            "cohort": pl.Series([cohort for cohort, _ in per_cohort], dtype=pl.Float64),
            "cohort_size": pl.Series([counts[measure] for _, counts in per_cohort], dtype=dtype),
        }
    )
