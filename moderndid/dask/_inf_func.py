"""Distributed influence function computation and variance estimation."""

from __future__ import annotations

import numpy as np

from moderndid.distributed._inf_func import (  # noqa: F401
    _compute_did,
    _compute_inf_func,
    compute_did_distributed,
)

from ._gram import _sum_gram_pair, tree_reduce


def compute_variance_distributed(client, inf_func_partitions, n_total, k):
    r"""Compute variance from distributed influence function partitions.

    Computes :math:`V = (1/n) \, \Psi^\top \Psi` without materializing
    the full matrix on one node. Each worker computes its local Gram
    matrix :math:`\Psi_i^\top \Psi_i` and results are tree-reduced.
    Standard errors are :math:`\sqrt{\operatorname{diag}(V) / n}`.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client.
    inf_func_partitions : list of ndarray
        Per-partition influence function arrays.
    n_total : int
        Total number of observations.
    k : int
        Number of influence function columns.

    Returns
    -------
    se : ndarray of shape :math:`(k,)`
        Standard errors.
    """

    def _local_gram(chunk):
        return chunk.T @ chunk, np.zeros(chunk.shape[1]), chunk.shape[0]

    scattered = client.scatter(inf_func_partitions)
    futures = [client.submit(_local_gram, pf) for pf in scattered]
    PtP, _, _ = tree_reduce(client, futures, _sum_gram_pair)
    V = PtP / n_total
    return np.sqrt(np.diag(V) / n_total)
