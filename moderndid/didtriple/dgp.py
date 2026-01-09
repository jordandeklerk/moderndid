"""Synthetic data generation for DDD estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["gen_dgp_2periods", "generate_simple_ddd_data"]

# Theoretical moments
_MEAN_Z1 = np.exp(0.25 / 2)
_SD_Z1 = np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
_MEAN_Z2 = 10.0
_SD_Z2 = 0.54164
_MEAN_Z3 = 0.21887
_SD_Z3 = 0.04453
_MEAN_Z4 = 402.0
_SD_Z4 = 56.63891


def gen_dgp_2periods(
    n,
    dgp_type,
    random_state=None,
) -> dict:
    """Generate synthetic panel data for 2-period DDD estimation.

    Four subgroups are created based on treatment and partition status:

    - Subgroup 4: Treated AND Eligible (state=1, partition=1)
    - Subgroup 3: Treated BUT Ineligible (state=1, partition=0)
    - Subgroup 2: Eligible BUT Untreated (state=0, partition=1)
    - Subgroup 1: Untreated AND Ineligible (state=0, partition=0)

    Parameters
    ----------
    n : int, default=5000
        Number of units to simulate.
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pd.DataFrame in long format with columns [id, state, partition,
          time, y, cov1, cov2, cov3, cov4, cluster]
        - *true_att*: True ATT (always 0)
        - *oracle_att*: Oracle ATT from potential outcomes
        - *efficiency_bound*: Theoretical efficiency bound
    """
    if dgp_type not in [1, 2, 3, 4]:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")

    rng = np.random.default_rng(random_state)
    att = 0.0

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    x = np.column_stack([x1, x2, x3, x4])
    z = _transform_covariates(x)

    w1 = np.array([-1.0, 0.5, -0.25, -0.1])
    w2 = np.array([-0.5, 2.0, 0.5, -0.2])
    w3 = np.array([3.0, -1.5, 0.75, -0.3])
    b1 = np.array([27.4, 13.7, 13.7, 13.7])
    b2 = np.array([6.85, 3.43, 3.43, 3.43])

    if dgp_type == 1:
        ps_covars, or_covars = z, z
        efficiency_bound = 32.82
    elif dgp_type == 2:
        ps_covars, or_covars = x, z
        efficiency_bound = 32.52
    elif dgp_type == 3:
        ps_covars, or_covars = z, x
        efficiency_bound = 32.82
    else:
        ps_covars, or_covars = x, x
        efficiency_bound = 32.52

    fps1 = _fps(0.2, w1, ps_covars)
    fps2 = _fps(0.2, w2, ps_covars)
    fps3 = _fps(0.05, w3, ps_covars)
    freg1 = _freg(b1, or_covars)
    freg0 = _freg(b2, or_covars)

    # Multinomial logit for subgroup assignment
    exp_f1 = np.exp(fps1)
    exp_f2 = np.exp(fps2)
    exp_f3 = np.exp(fps3)
    sum_exp_f = exp_f1 + exp_f2 + exp_f3

    p1 = exp_f1 / (1 + sum_exp_f)
    p2 = exp_f2 / (1 + sum_exp_f)
    p4 = 1 / (1 + sum_exp_f)

    u = rng.uniform(size=n)
    pa = np.zeros(n, dtype=int)
    pa[u <= p1] = 1
    pa[(u > p1) & (u <= p1 + p2)] = 2
    pa[(u > p1 + p2) & (u <= 1 - p4)] = 3
    pa[u > 1 - p4] = 4

    # PA=3,4 -> treated; PA=2,4 -> eligible
    state = np.where((pa == 3) | (pa == 4), 1, 0)
    partition = np.where((pa == 2) | (pa == 4), 1, 0)

    unobs_het = state * partition * freg1 + (1 - state) * partition * freg0
    or_lin = state * freg1 + (1 - state) * freg0
    v = rng.normal(loc=unobs_het, scale=1.0)

    y0 = or_lin + v + rng.standard_normal(n)
    y10 = or_lin + v + rng.standard_normal(n) + or_lin
    y11 = or_lin + v + rng.standard_normal(n) + or_lin + att

    treated_eligible = state * partition
    if np.sum(treated_eligible) > 0:
        oracle_att = (np.sum(treated_eligible * y11) - np.sum(treated_eligible * y10)) / np.sum(treated_eligible)
    else:
        oracle_att = np.nan

    y1 = treated_eligible * y11 + (1 - treated_eligible) * y10
    clusters = rng.integers(1, 51, size=n)

    df_t1 = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": 1,
            "y": y0,
            "cov1": z[:, 0],
            "cov2": z[:, 1],
            "cov3": z[:, 2],
            "cov4": z[:, 3],
            "cluster": clusters,
        }
    )

    df_t2 = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": 2,
            "y": y1,
            "cov1": z[:, 0],
            "cov2": z[:, 1],
            "cov3": z[:, 2],
            "cov4": z[:, 3],
            "cluster": clusters,
        }
    )

    df = pd.concat([df_t1, df_t2], ignore_index=True)
    df = df.sort_values(["id", "time"]).reset_index(drop=True)

    return {
        "data": df,
        "true_att": att,
        "oracle_att": oracle_att,
        "efficiency_bound": efficiency_bound,
    }


def generate_simple_ddd_data(
    n,
    att,
    random_state=None,
) -> pd.DataFrame:
    """Generate simple DDD panel data with a known treatment effect.

    Parameters
    ----------
    n : int, default=500
        Number of units to simulate.
    att : float, default=5.0
        True average treatment effect on the treated.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:

        - *id*: Unit identifier
        - *state*: Treatment indicator (1=treated, 0=control)
        - *partition*: Eligibility indicator (1=eligible, 0=ineligible)
        - *time*: Time period (1=pre, 2=post)
        - *y*: Outcome variable
        - *x1*, *x2*: Covariates
    """
    rng = np.random.default_rng(random_state)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    state = rng.binomial(1, 0.5, n)
    partition = rng.binomial(1, 0.5, n)
    alpha_i = rng.standard_normal(n)

    y0 = 2 + 5 * state - 2 * partition + 0.5 * x1 + 0.3 * x2 + 4 * state * partition + alpha_i + rng.standard_normal(n)

    y1 = (
        2
        + 5 * state
        - 2 * partition
        + 3
        + 0.5 * x1
        + 0.3 * x2
        + 4 * state * partition
        + 2 * state
        + 3 * partition
        + att * state * partition
        + alpha_i
        + rng.standard_normal(n)
    )

    df_t1 = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": 1,
            "y": y0,
            "x1": x1,
            "x2": x2,
        }
    )

    df_t2 = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": 2,
            "y": y1,
            "x1": x1,
            "x2": x2,
        }
    )

    df = pd.concat([df_t1, df_t2], ignore_index=True)
    df = df.sort_values(["id", "time"]).reset_index(drop=True)

    return df


def _transform_covariates(x: np.ndarray) -> np.ndarray:
    """Transform X to Z via nonlinear functions for doubly robust testing."""
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    z1_tilde = np.exp(x1 / 2)
    z2_tilde = x2 / (1 + np.exp(x1)) + 10
    z3_tilde = (x1 * x3 / 25 + 0.6) ** 3
    z4_tilde = (x1 + x4 + 20) ** 2

    z1 = (z1_tilde - _MEAN_Z1) / _SD_Z1
    z2 = (z2_tilde - _MEAN_Z2) / _SD_Z2
    z3 = (z3_tilde - _MEAN_Z3) / _SD_Z3
    z4 = (z4_tilde - _MEAN_Z4) / _SD_Z4

    return np.column_stack([z1, z2, z3, z4])


def _fps(psi: float, coefs: np.ndarray, xvars: np.ndarray) -> np.ndarray:
    """Compute propensity score index: psi * (X @ coefs)."""
    return psi * (xvars @ coefs)


def _freg(coefs: np.ndarray, xvars: np.ndarray) -> np.ndarray:
    """Compute outcome regression index: 210 + X @ coefs."""
    return 210 + xvars @ coefs
