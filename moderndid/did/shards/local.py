"""In-memory execution of the shard engine for tests and as the reference executor."""

import numpy as np
import polars as pl

from moderndid.core.bootstrap import canonical_keys
from moderndid.core.preprocess import WEIGHTS_COLUMN
from moderndid.cupy.backend import to_numpy
from moderndid.did.compute_att_gt import _unit_index

from .container import RowShard, Shard
from .layout import row_keys
from .partials import combine_partials, empty_partial, finalize_keyed


class LocalExecutor:
    """Run the passes of a fit on shards held in memory.

    A cluster may span several shards because each pass finalizes the keyed
    sums only after it combines every shard. The executor counts its passes in
    ``n_passes``.

    Parameters
    ----------
    shards : list[Shard] or list[RowShard]
        The shards of the data.
    """

    def __init__(self, shards):
        self.shards = list(shards)
        self.n_passes = 0

    def run(self, kernel, state):
        """Apply a kernel to every shard and combine the statistics.

        Parameters
        ----------
        kernel : callable
            Function of a shard and the state that returns a Partial.
        state : NamedTuple
            State of the pass.

        Returns
        -------
        Partial
            Statistics of all the shards with the keyed sums finalized.
        """
        self.n_passes += 1
        partial = empty_partial()
        for shard in self.shards:
            partial = combine_partials(partial, kernel(shard, state))
        return finalize_keyed(partial)


def shards_from_did_data(data, n_shards, random_state=None):
    """Split the units of a preprocessed balanced panel into random shards.

    Each unit goes whole to one shard. See the
    :ref:`guide to fits on shards <dev-shards>` for what a kernel receives from
    the shards.

    Parameters
    ----------
    data : DIDData
        Preprocessed balanced panel.
    n_shards : int
        Number of shards. Must be positive.
    random_state : int, Generator, optional
        Seed or NumPy ``Generator`` for the assignment of units to shards.

    Returns
    -------
    list[Shard]
        Shards whose sizes differ by at most one unit.
    """
    _check_shard_count(n_shards)
    config = data.config
    if not config.panel or data.outcomes_tensor is None:
        raise ValueError(
            "shards_from_did_data needs a balanced panel. Build the shards of repeated cross-sections and "
            "unbalanced panels with row_shards_from_did_data."
        )

    units = data.time_invariant_data
    unit_keys = canonical_keys(units[config.idname].to_numpy())
    cohorts = units[config.gname].to_numpy().astype(float)
    outcomes = np.column_stack([to_numpy(y) for y in data.outcomes_tensor])
    weights = None
    if config.weightsname is not None:
        weights = np.column_stack([to_numpy(w) for w in data.weights_tensor])
    covariates = None
    if config.xformla not in (None, "~1"):
        # Every period's design starts with the intercept. The shards keep only the covariates after it.
        covariates = np.stack([to_numpy(x)[:, 1:] for x in data.covariates_tensor], axis=1)
    cluster_keys = None
    clustervars = [name for name in config.clustervars if name != config.idname]
    if clustervars:
        cluster_keys = canonical_keys(units[clustervars[0]].to_numpy())

    order = np.random.default_rng(random_state).permutation(len(unit_keys))
    return [
        Shard(
            unit_keys=unit_keys[rows],
            cohort=cohorts[rows],
            y=outcomes[rows],
            weights=None if weights is None else weights[rows],
            covariates=None if covariates is None else covariates[rows],
            cluster_keys=None if cluster_keys is None else cluster_keys[rows],
        )
        for rows in np.array_split(order, int(n_shards))
    ]


def row_shards_from_did_data(data, n_shards, random_state=None, keyed=True, cluster=None):
    """Split preprocessed repeated cross-sections or an unbalanced panel into random shards.

    A row of repeated cross-sections takes its key from
    :func:`~moderndid.did.shards.layout.row_keys`. A unit of an unbalanced
    panel goes whole to one shard under the canonical key of its identifier.

    Parameters
    ----------
    data : DIDData
        Preprocessed repeated cross-sections or unbalanced panel.
    n_shards : int
        Number of shards. Must be positive.
    random_state : int, Generator, optional
        Seed or NumPy ``Generator`` for the assignment of units to shards.
    keyed : bool, default=True
        Whether the rows of repeated cross-sections get keys. The units of an
        unbalanced panel always get keys.
    cluster : str, optional
        Name of the cluster variable besides the unit identifier, one of the
        ``clustervars`` of the preprocessed data.

    Returns
    -------
    list[RowShard]
        Shards whose sizes differ by at most one unit.
    """
    _check_shard_count(n_shards)
    config = data.config
    if config.panel:
        raise ValueError(
            "row_shards_from_did_data needs repeated cross-sections or an unbalanced panel. Preprocess the data "
            "with panel=False, or with allow_unbalanced_panel=True on a panel in which some unit misses a period."
        )
    if cluster is not None and cluster not in config.clustervars:
        raise ValueError(
            f"cluster={cluster!r} is not valid. Must be one of the clustervars of the preprocessed data, "
            f"{config.clustervars!r}."
        )

    frame = data.data
    period = np.searchsorted(config.time_periods, frame[config.tname].to_numpy()).astype(np.int64)
    cohort = frame[config.gname].to_numpy().astype(np.float64)
    y = frame[config.yname].to_numpy().astype(np.float64)
    weights = None
    if config.weightsname is not None:
        weights = frame[WEIGHTS_COLUMN].to_numpy().astype(np.float64)
    covariates = None
    design = to_numpy(data.covariates_matrix)
    # The shards store the covariates after the intercept that starts every row's design. A formula such as "~ 1" leaves
    # none, as in the row table of a Spark fit.
    if design.ndim == 2 and design.shape[1] > 1:
        covariates = np.asarray(design[:, 1:], dtype=np.float64)

    rng = np.random.default_rng(random_state)

    if config.true_repeated_cross_sections:
        unit_keys = None
        if keyed:
            raw_weights = None
            if config.weightsname is not None:
                raw_weights = frame[config.weightsname].to_numpy().astype(np.float64)
            copy = _copy_indices(period, cohort, y, raw_weights, covariates)
            unit_keys = row_keys(period, cohort, y, raw_weights, covariates, copy)
        cluster_keys = None if cluster is None else canonical_keys(frame[cluster].to_numpy())

        return [
            RowShard(
                period=period[rows],
                cohort=cohort[rows],
                y=y[rows],
                weights=None if weights is None else weights[rows],
                covariates=None if covariates is None else covariates[rows],
                unit_keys=None if unit_keys is None else unit_keys[rows],
                cluster_keys=None if cluster_keys is None else cluster_keys[rows],
            )
            for rows in np.array_split(rng.permutation(len(period)), int(n_shards))
        ]

    units = data.time_invariant_data
    unit_index = _unit_index(data)
    unit_keys = canonical_keys(units[config.idname].to_numpy())
    cluster_keys = None if cluster is None else canonical_keys(units[cluster].to_numpy())

    # A shard keeps each unit's rows together in period order.
    sorted_rows = np.lexsort((period, unit_index))
    sizes = np.bincount(unit_index, minlength=len(unit_keys))
    starts = np.cumsum(sizes) - sizes

    shards = []
    for members in np.array_split(rng.permutation(len(unit_keys)), int(n_shards)):
        counts = sizes[members]
        offsets = np.cumsum(counts) - counts
        rows = sorted_rows[np.arange(counts.sum()) + np.repeat(starts[members] - offsets, counts)]
        shards.append(
            RowShard(
                period=period[rows],
                cohort=cohort[rows],
                y=y[rows],
                weights=None if weights is None else weights[rows],
                covariates=None if covariates is None else covariates[rows],
                unit_keys=unit_keys[members],
                units=np.repeat(np.arange(len(members), dtype=np.int64), counts),
                cluster_keys=None if cluster_keys is None else cluster_keys[members],
            )
        )
    return shards


def _check_shard_count(n_shards):
    """Raise a ValueError unless the number of shards is a positive integer."""
    if isinstance(n_shards, bool) or not isinstance(n_shards, int | np.integer) or n_shards < 1:
        raise ValueError(f"n_shards={n_shards!r} is not valid. Must be a positive integer.")


def _copy_indices(period, cohort, y, raw_weights, covariates):
    """Return the zero-based copy index of each row among the rows that share all of its values."""
    # Adding 0.0 maps -0.0 to 0.0 so rows that differ only in the sign of a zero count as copies.
    columns = [period, cohort + 0.0, y + 0.0]
    if raw_weights is not None:
        columns.append(raw_weights + 0.0)
    if covariates is not None:
        columns.extend(covariates.T + 0.0)
    frame = pl.DataFrame({str(k): column for k, column in enumerate(columns)})
    return frame.select(pl.int_range(pl.len()).over(frame.columns).alias("copy"))["copy"].to_numpy()
