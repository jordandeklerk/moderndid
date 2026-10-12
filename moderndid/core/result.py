"""Shared helpers for DiD result containers."""

from __future__ import annotations

import numpy as np

from moderndid.core.maketables import cluster_label, n_from_first_dim, vcov_info_from_bootstrap


def extract_vcov_info(params, *, bootstrap_key="bootstrap", cluster_key="cluster"):
    """Build the vcov-info dict of maketables from an estimation params dict.

    Parameters
    ----------
    params : dict
        Estimation settings of a result.
    bootstrap_key : str, default="bootstrap"
        Key of the flag that says whether the standard errors come from a bootstrap.
    cluster_key : str, default="cluster"
        Key of the cluster variable, or of the cluster of each unit.

    Returns
    -------
    dict
        - **vcov_type**: "bootstrap" or "analytical"
        - **clustervar**: Variable the standard errors cluster by as :func:`extract_clustervar` finds it, or None
          without a variable name
    """
    clustervar = extract_clustervar(params, cluster_key=cluster_key)
    return vcov_info_from_bootstrap(
        is_bootstrap=bool(params.get(bootstrap_key, False)),
        cluster=None if clustervar is True else clustervar,
    )


def extract_clustervar(params, *, cluster_key="cluster"):
    """Return the name of the variable the standard errors cluster by.

    A name stored under ``cluster_key`` comes back as it is. True there stands
    for clusters that have no variable name. The results of att_gt and aggte
    store the cluster of each unit there instead and keep the names in
    ``"clustervars"``. Since clusters of one unit each leave the standard
    errors at the unit level, those names count only when some cluster holds
    several units. Without anything under ``cluster_key``, the names in
    ``"clustervars"`` count as they are.

    Parameters
    ----------
    params : dict
        Estimation settings of a result.
    cluster_key : str, default="cluster"
        Key of the cluster variable, or of the cluster of each unit.

    Returns
    -------
    str, bool, or None
        The cluster variable with several names joined by "+", True for
        clusters without a variable name, or None without clusters.
    """
    cluster = params.get(cluster_key)
    if isinstance(cluster, str | bool):
        return cluster or None
    names = params.get("clustervars")
    if cluster is not None and len(np.unique(cluster)) == len(cluster):
        names = None
    return cluster_label(names) or None


def extract_n_obs(
    *influence_arrays: object,
    params: dict | None = None,
    keys: tuple[str, ...] = ("n_obs", "n_units"),
) -> int | None:
    """Extract observation count from *params* or influence-function shapes.

    Checks the keys of *params* first, in order. Without a match it falls
    back to the first dimension of each influence array.
    """
    if params is not None:
        for key in keys:
            n = params.get(key)
            if n is not None:
                return int(n)
    for arr in influence_arrays:
        n = n_from_first_dim(arr)
        if n is not None:
            return n
    return None
