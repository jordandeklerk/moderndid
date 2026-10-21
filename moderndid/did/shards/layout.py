"""Arrow layout of the unit table that shards read."""

import numpy as np
import pyarrow as pa

from moderndid.core.bootstrap import canonical_keys

from .container import Shard

UNIT = "unit"
COHORT = "cohort"
OUTCOME = "y"
WEIGHT = "w"
COVARIATE = "x"
CLUSTER = "cluster"


def shard_from_arrow(batch, n_periods, n_covariates):
    """Read a shard from rows of the unit table.

    The unit table has one row for each unit. Its column ``unit`` holds the unit
    identifier and ``cohort`` the period in which the unit is first treated or
    infinity if it is never treated. The list columns ``y`` and ``w`` hold the
    outcome and the normalized sampling weight in each of the T sorted periods.
    Entry :math:`jp + c` of the list column ``x`` is covariate :math:`c` in
    period :math:`j`. The ``w`` column appears only in a weighted fit and the
    ``x`` column only in a fit with covariates. A fit with a cluster variable
    besides the unit identifier adds a ``cluster`` column that holds each unit's
    cluster.

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
