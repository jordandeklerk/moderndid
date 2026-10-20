"""In-memory execution of the shard engine for tests and as the reference executor."""

import numpy as np

from moderndid.core.bootstrap import canonical_keys
from moderndid.cupy.backend import to_numpy

from .container import Shard
from .partials import combine_partials, empty_partial, finalize_keyed


class LocalExecutor:
    """Run the passes of a fit on shards held in memory.

    Since each pass finalizes the keyed sums only after it combines every
    shard, a cluster may span several shards. The executor counts its passes
    in ``n_passes``.

    Parameters
    ----------
    shards : list[Shard]
        The shards of the panel.
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

    Each unit goes whole to one shard and keeps every period of its outcome
    and of any normalized weights and covariates the fit uses.

    See the :ref:`architecture guide <architecture-shards>` for what a kernel
    receives from the shards.

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
    if isinstance(n_shards, bool) or not isinstance(n_shards, int | np.integer) or n_shards < 1:
        raise ValueError(f"n_shards={n_shards!r} is not valid. Must be a positive integer.")
    config = data.config
    if not config.panel or config.allow_unbalanced_panel or data.outcomes_tensor is None:
        raise ValueError(
            "shards_from_did_data needs a balanced panel. Preprocess the data with panel=True and "
            "allow_unbalanced_panel=False."
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
        # Since every period's design starts with the intercept, the shards keep only the covariates after it.
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
