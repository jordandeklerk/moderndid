"""Shared helpers for DiD result containers."""

from __future__ import annotations

from moderndid.core.maketables import n_from_first_dim, vcov_info_from_bootstrap


def extract_vcov_info(
    params: dict,
    *,
    bootstrap_key: str = "bootstrap",
    cluster_key: str = "cluster",
) -> dict[str, str | None]:
    """Build vcov-info dict from an estimation params dict.

    Reads the bootstrap flag and cluster variable from *params*,
    falling back to ``"clustervars"`` when *cluster_key* is absent.
    """
    cluster = params.get(cluster_key)
    if cluster is None:
        cluster = params.get("clustervars")
    return vcov_info_from_bootstrap(
        is_bootstrap=bool(params.get(bootstrap_key, False)),
        cluster=cluster,
    )


def extract_n_obs(
    *influence_arrays: object,
    params: dict | None = None,
    keys: tuple[str, ...] = ("n_obs", "n_units"),
) -> int | None:
    """Extract observation count from *params* or influence-function shapes.

    Checks *params* dict keys first (in order), then falls back to the
    first dimension of each influence array.
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
