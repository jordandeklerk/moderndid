"""Shared nuisance estimation types and helpers for distributed backends."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DistPScoreResult(NamedTuple):
    """Result from distributed propensity score estimation."""

    propensity_scores: np.ndarray
    hessian_matrix: np.ndarray | None
    keep_ps: np.ndarray


class DistOutcomeRegResult(NamedTuple):
    """Result from distributed outcome regression."""

    delta_y: np.ndarray
    or_delta: np.ndarray
    reg_coeff: np.ndarray | None


def _build_partitions_for_subset(X, W, y, n_partitions):
    """Split arrays into roughly equal partitions."""
    splits = np.array_split(np.arange(len(y)), n_partitions)
    return [(X[idx], W[idx], y[idx]) for idx in splits if len(idx) > 0]


def _compute_pscore_null(subgroup, comp_subgroup):
    """Compute null propensity scores for REG method."""
    mask = (subgroup == 4) | (subgroup == comp_subgroup)
    n_sub = int(np.sum(mask))
    return DistPScoreResult(
        propensity_scores=np.ones(n_sub),
        hessian_matrix=None,
        keep_ps=np.ones(n_sub, dtype=bool),
    )


def _compute_outcome_regression_null(y1, y0, subgroup, comp_subgroup):
    """Compute null outcome regression for IPW method."""
    mask = (subgroup == 4) | (subgroup == comp_subgroup)
    delta_y = (y1 - y0)[mask]
    or_delta = np.zeros(len(delta_y))
    return DistOutcomeRegResult(delta_y=delta_y, or_delta=or_delta, reg_coeff=None)
