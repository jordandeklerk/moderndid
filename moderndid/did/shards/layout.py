"""Arrow layouts of the tables that shards read."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from moderndid.core.bootstrap import canonical_keys

from .container import RowShard, Shard

UNIT = "unit"
PERIOD = "period"
COHORT = "cohort"
OUTCOME = "y"
WEIGHT = "w"
RAW_WEIGHT = "raw_w"
COVARIATE = "x"
CLUSTER = "cluster"
COPY = "copy"


def shard_from_arrow(batch, n_periods, n_covariates):
    """Read a shard from rows of the unit table.

    The unit table holds one unit of a balanced panel in each row. Its list
    columns ``y`` and ``w`` hold one value for each of the T sorted periods.
    Entry :math:`jp + c` of the list column ``x`` is covariate :math:`c` in
    period :math:`j`. The table holds ``w`` only in a weighted fit, ``x`` only
    in a fit with covariates, and ``cluster`` only in a fit with a cluster
    variable besides the unit identifier.

    Parameters
    ----------
    batch : pyarrow.RecordBatch or pyarrow.Table
        Rows of the unit table.
    n_periods : int
        Number of time periods T.
    n_covariates : int
        Number of covariates p, not counting the intercept.

    Returns
    -------
    Shard
        The m units of the rows:

        - **unit_keys**: Canonical keys of the ``unit`` column
        - **cohort**: The ``cohort`` column as floats
        - **y**: The ``y`` lists as a matrix of shape (m, T)
        - **weights**: The ``w`` lists as a matrix of shape (m, T), or None without that column
        - **covariates**: The ``x`` lists as an array of shape (m, T, p), or None without covariates
        - **cluster_keys**: Canonical keys of the ``cluster`` column, or None without that column
    """
    n_rows = batch.num_rows
    names = batch.schema.names

    y = _list_values(batch.column(OUTCOME), n_rows, n_periods, OUTCOME)
    weights = _list_values(batch.column(WEIGHT), n_rows, n_periods, WEIGHT) if WEIGHT in names else None
    covariates = None
    if COVARIATE in names and n_covariates > 0:
        flat = _list_values(batch.column(COVARIATE), n_rows, n_periods * n_covariates, COVARIATE)
        covariates = flat.reshape(n_rows, n_periods, n_covariates)
    cluster_keys = canonical_keys(_column_values(batch.column(CLUSTER))) if CLUSTER in names else None

    return Shard(
        unit_keys=canonical_keys(_column_values(batch.column(UNIT))),
        cohort=_column_values(batch.column(COHORT)).astype(np.float64),
        y=y,
        weights=weights,
        covariates=covariates,
        cluster_keys=cluster_keys,
    )


def row_shard_from_arrow(batch, n_covariates):
    """Read a row shard from the row table or the unit table of an unbalanced panel.

    The row table holds one observation of repeated cross-sections in each
    row. The unit table holds one unit in each row. Its ``unit`` column of
    identifiers tells it from the row table. Its list columns ``period``,
    ``y``, ``w``, and ``x`` run over the unit's rows in period order.

    Both tables hold ``w`` only in a weighted fit, ``x`` only in a fit with
    covariates, and ``cluster`` only in a fit with a cluster variable besides
    the unit identifier. In a fit that draws a multiplier for each row, the row
    table also holds the ``copy`` index and, when weighted, the raw weight
    ``raw_w`` that :func:`row_keys` hashes.

    Parameters
    ----------
    batch : pyarrow.RecordBatch or pyarrow.Table
        Rows of the row table or of the unit table.
    n_covariates : int
        Number of covariates p, not counting the intercept.

    Returns
    -------
    RowShard
        The m rows of the batch in u units:

        - **period**: The ``period`` column, flattened for the unit table
        - **cohort**: The ``cohort`` column as floats, repeated over the rows of each unit for the unit table
        - **y**: The ``y`` column, flattened for the unit table
        - **weights**: The ``w`` column, flattened for the unit table, or None without that column
        - **covariates**: The ``x`` lists of p values for each row as a matrix of shape (m, p), or None without
          covariates
        - **unit_keys**: Canonical keys of the ``unit`` column, :func:`row_keys` of rows with ``copy``, or None
        - **units**: Position of each row's unit for the unit table, or None for the row table
        - **cluster_keys**: Canonical keys of the ``cluster`` column, or None without that column
    """
    if UNIT in batch.schema.names:
        return _unit_table_shard(batch, n_covariates)
    return _row_table_shard(batch, n_covariates)


def row_keys(period, cohort, y, raw_weights, covariates, copy):
    r"""Hash the values of each row and its copy index into a 64-bit key.

    The 64-bit words :math:`w_k` of a row are its period, cohort, outcome, raw
    weight, each covariate, and copy index in that order. Each float :math:`x`
    enters as the bits of :math:`x + 0.0`. The key is the splitmix64 chain
    from :math:`h_0 = \mathrm{0x9E3779B97F4A7C15}` over these words.

    .. math::

        h_k = \operatorname{mix}(h_{k-1} \oplus w_k).

    Every product in :math:`\operatorname{mix}(z)` is taken modulo
    :math:`2^{64}`.

    .. math::

        z_1 &= z \oplus (z \gg 30), \\
        z_2 &= z_1 \cdot \mathrm{0xBF58476D1CE4E5B9}, \\
        z_3 &= z_2 \oplus (z_2 \gg 27), \\
        z_4 &= z_3 \cdot \mathrm{0x94D049BB133111EB}, \\
        \operatorname{mix}(z) &= z_4 \oplus (z_4 \gg 31).

    Parameters
    ----------
    period : ndarray
        Index of each row's period among the sorted periods, of shape (m,).
    cohort : ndarray
        Period in which each row's unit is first treated or infinity for a
        never-treated unit, of shape (m,).
    y : ndarray
        Outcome of each row, of shape (m,).
    raw_weights : ndarray, optional
        Sampling weight of each row before normalization, of shape (m,). None
        leaves the weight out of the key.
    covariates : ndarray, optional
        Covariates of each row without the intercept, of shape (m, p). None
        leaves them out of the key.
    copy : ndarray
        Index of each row among the rows that share all the other values,
        counted from 0, of shape (m,).

    Returns
    -------
    ndarray
        Key of each row as unsigned 64-bit integers, of shape (m,).
    """
    floats = [cohort, y]
    if raw_weights is not None:
        floats.append(raw_weights)
    if covariates is not None:
        floats.extend(covariates.T)

    words = [np.asarray(period, dtype=np.int64).astype(np.uint64)]
    # Rows that differ only in the sign of a zero share a key because adding 0.0 turns -0.0 into 0.0.
    words.extend((np.asarray(value, dtype=np.float64) + 0.0).view(np.uint64) for value in floats)
    words.append(np.asarray(copy, dtype=np.int64).astype(np.uint64))

    keys = np.full(len(words[0]), 0x9E3779B97F4A7C15, dtype=np.uint64)
    for word in words:
        keys = _mix(keys ^ word)
    return keys


def _column_values(column):
    """Return a flat Arrow column as a NumPy array."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return column.to_numpy(zero_copy_only=False)


def _list_values(column, n_rows, width, name):
    """Return a list column whose lists all hold the same number of values as a matrix."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    values = column.flatten().to_numpy(zero_copy_only=False)
    if column.null_count > 0 or len(values) != n_rows * width:
        raise RuntimeError(
            f"The unit table column {name!r} must hold {width} values in every row. "
            "Rebuild the unit table from the panel."
        )
    return values.reshape(n_rows, width).astype(np.float64, copy=False)


def _row_table_shard(batch, n_covariates):
    """Read a shard from rows of the row table as units of one row each."""
    n_rows = batch.num_rows
    names = batch.schema.names

    period = _column_values(batch.column(PERIOD)).astype(np.int64)
    cohort = _column_values(batch.column(COHORT)).astype(np.float64)
    y = _column_values(batch.column(OUTCOME)).astype(np.float64)
    weights = _column_values(batch.column(WEIGHT)).astype(np.float64) if WEIGHT in names else None
    covariates = None
    if COVARIATE in names and n_covariates > 0:
        flat = _list_values_by_row(batch.column(COVARIATE), np.full(n_rows, n_covariates), COVARIATE, "row table")
        covariates = flat.reshape(n_rows, n_covariates).astype(np.float64, copy=False)
    cluster_keys = canonical_keys(_column_values(batch.column(CLUSTER))) if CLUSTER in names else None

    unit_keys = None
    if COPY in names:
        raw_weights = _column_values(batch.column(RAW_WEIGHT)).astype(np.float64) if WEIGHT in names else None
        copy = _column_values(batch.column(COPY)).astype(np.int64)
        unit_keys = row_keys(period, cohort, y, raw_weights, covariates, copy)

    return RowShard(
        period=period,
        cohort=cohort,
        y=y,
        weights=weights,
        covariates=covariates,
        unit_keys=unit_keys,
        cluster_keys=cluster_keys,
    )


def _unit_table_shard(batch, n_covariates):
    """Read a shard from rows of the unit table by flattening the lists of each unit."""
    names = batch.schema.names
    counts = _list_lengths(batch.column(PERIOD))

    period = _list_values_by_row(batch.column(PERIOD), counts, PERIOD, "unit table").astype(np.int64, copy=False)
    y = _list_values_by_row(batch.column(OUTCOME), counts, OUTCOME, "unit table").astype(np.float64, copy=False)
    weights = None
    if WEIGHT in names:
        weights = _list_values_by_row(batch.column(WEIGHT), counts, WEIGHT, "unit table").astype(np.float64, copy=False)
    covariates = None
    if COVARIATE in names and n_covariates > 0:
        flat = _list_values_by_row(batch.column(COVARIATE), counts * n_covariates, COVARIATE, "unit table")
        covariates = flat.reshape(len(period), n_covariates).astype(np.float64, copy=False)
    cluster_keys = canonical_keys(_column_values(batch.column(CLUSTER))) if CLUSTER in names else None

    return RowShard(
        period=period,
        cohort=np.repeat(_column_values(batch.column(COHORT)).astype(np.float64), counts),
        y=y,
        weights=weights,
        covariates=covariates,
        unit_keys=canonical_keys(_column_values(batch.column(UNIT))),
        units=np.repeat(np.arange(batch.num_rows, dtype=np.int64), counts),
        cluster_keys=cluster_keys,
    )


def _list_lengths(column):
    """Return the number of values in each list of a list column with -1 for a null list."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return pc.list_value_length(column).fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)


def _list_values_by_row(column, widths, name, table):
    """Return the values of a list column whose lists hold the given numbers of values."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.null_count > 0 or not np.array_equal(_list_lengths(column), widths):
        raise RuntimeError(
            f"The {table} column {name!r} must hold a list of the right length in every row. Rebuild the {table}."
        )
    return column.flatten().to_numpy(zero_copy_only=False)


def _mix(z):
    """Return the splitmix64 finalizer of an array of unsigned 64-bit integers."""
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))
