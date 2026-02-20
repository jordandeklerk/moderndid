"""Shared input validation for distributed DDD backends."""

from __future__ import annotations

import numpy as np


def _validate_inputs(y1, y0, subgroup, covariates, i_weights):
    """Validate and preprocess input arrays."""
    y1 = np.asarray(y1).flatten()
    y0 = np.asarray(y0).flatten()
    subgroup = np.asarray(subgroup).flatten()
    n_units = len(subgroup)

    if len(y1) != n_units or len(y0) != n_units:
        raise ValueError("y1, y0, and subgroup must have the same length.")

    if covariates is None:
        covariates = np.ones((n_units, 1))
    else:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

    if covariates.shape[0] != n_units:
        raise ValueError("covariates must have the same number of rows as subgroup.")

    if i_weights is None:
        i_weights = np.ones(n_units)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if len(i_weights) != n_units:
            raise ValueError("i_weights must have the same length as subgroup.")
        if np.any(i_weights < 0):
            raise ValueError("i_weights must be non-negative.")

    i_weights = i_weights / np.mean(i_weights)

    unique_subgroups = set(int(v) for v in np.unique(subgroup))
    if not unique_subgroups.issubset({1, 2, 3, 4}):
        raise ValueError(f"subgroup must contain only values 1, 2, 3, 4. Got {unique_subgroups}.")
    if 4 not in unique_subgroups:
        raise ValueError("subgroup must contain at least one unit in subgroup 4.")

    return y1, y0, subgroup, covariates, i_weights, n_units
