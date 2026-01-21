"""Synthetic data generation for DDD estimators."""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["gen_dgp_2periods", "gen_dgp_mult_periods", "generate_simple_ddd_data"]

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
    panel=True,
    random_state=None,
) -> dict:
    """Generate synthetic data for 2-period DDD estimation.

    Four subgroups are created based on treatment and partition status:

    - Subgroup 4: Treated AND Eligible (state=1, partition=1)
    - Subgroup 3: Treated BUT Ineligible (state=1, partition=0)
    - Subgroup 2: Eligible BUT Untreated (state=0, partition=1)
    - Subgroup 1: Untreated AND Ineligible (state=0, partition=0)

    Parameters
    ----------
    n : int, default=5000
        Number of units to simulate. For panel data, this is the total number of
        units observed in both periods. For repeated cross-section data, this is
        the number of observations per period.
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    panel : bool, default=True
        If True, generate panel data where each unit is observed in both periods.
        If False, generate repeated cross-section data where different units are
        sampled in each period.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pl.DataFrame in long format with columns [id, state, partition,
          time, y, cov1, cov2, cov3, cov4, cluster]
        - *true_att*: True ATT (always 0)
        - *oracle_att*: Oracle ATT from potential outcomes
        - *efficiency_bound*: Theoretical efficiency bound
    """
    if dgp_type not in [1, 2, 3, 4]:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")

    rng = np.random.default_rng(random_state)
    att = 0.0

    w1 = np.array([-1.0, 0.5, -0.25, -0.1])
    w2 = np.array([-0.5, 2.0, 0.5, -0.2])
    w3 = np.array([3.0, -1.5, 0.75, -0.3])
    b1 = np.array([27.4, 13.7, 13.7, 13.7])
    b2 = np.array([6.85, 3.43, 3.43, 3.43])

    if dgp_type == 1:
        efficiency_bound = 32.82
    elif dgp_type == 2:
        efficiency_bound = 32.52
    elif dgp_type == 3:
        efficiency_bound = 32.82
    else:
        efficiency_bound = 32.52

    if panel:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        fps1 = _fps(0.2, w1, ps_covars)
        fps2 = _fps(0.2, w2, ps_covars)
        fps3 = _fps(0.05, w3, ps_covars)
        freg1 = _freg(b1, or_covars)
        freg0 = _freg(b2, or_covars)

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

        df_t1 = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "state": state,
                "partition": partition,
                "time": np.ones(n, dtype=int),
                "y": y0,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df_t2 = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "state": state,
                "partition": partition,
                "time": np.full(n, 2, dtype=int),
                "y": y1,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df = pl.concat([df_t1, df_t2])
        df = df.sort(["id", "time"])

    else:
        df_list = []
        oracle_att = np.nan
        id_offset = 0

        for t in [1, 2]:
            x1 = rng.standard_normal(n)
            x2 = rng.standard_normal(n)
            x3 = rng.standard_normal(n)
            x4 = rng.standard_normal(n)
            x = np.column_stack([x1, x2, x3, x4])
            z = _transform_covariates(x)

            if dgp_type == 1:
                ps_covars, or_covars = z, z
            elif dgp_type == 2:
                ps_covars, or_covars = x, z
            elif dgp_type == 3:
                ps_covars, or_covars = z, x
            else:
                ps_covars, or_covars = x, x

            fps1 = _fps(0.2, w1, ps_covars)
            fps2 = _fps(0.2, w2, ps_covars)
            fps3 = _fps(0.05, w3, ps_covars)
            freg1 = _freg(b1, or_covars)
            freg0 = _freg(b2, or_covars)

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

            state = np.where((pa == 3) | (pa == 4), 1, 0)
            partition = np.where((pa == 2) | (pa == 4), 1, 0)

            unobs_het = state * partition * freg1 + (1 - state) * partition * freg0
            or_lin = state * freg1 + (1 - state) * freg0
            v = rng.normal(loc=unobs_het, scale=1.0)

            if t == 1:
                y = or_lin + v + rng.standard_normal(n)
            else:
                treated_eligible = state * partition
                y10 = or_lin + v + rng.standard_normal(n) + or_lin
                y11 = or_lin + v + rng.standard_normal(n) + or_lin + att
                y = treated_eligible * y11 + (1 - treated_eligible) * y10

                if np.sum(treated_eligible) > 0:
                    oracle_att = (np.sum(treated_eligible * y11) - np.sum(treated_eligible * y10)) / np.sum(
                        treated_eligible
                    )

            clusters = rng.integers(1, 51, size=n)

            df_t = pl.DataFrame(
                {
                    "id": np.arange(id_offset + 1, id_offset + n + 1),
                    "state": state,
                    "partition": partition,
                    "time": np.full(n, t, dtype=int),
                    "y": y,
                    "cov1": z[:, 0],
                    "cov2": z[:, 1],
                    "cov3": z[:, 2],
                    "cov4": z[:, 3],
                    "cluster": clusters,
                }
            )
            df_list.append(df_t)
            id_offset += n

        df = pl.concat(df_list)

    return {
        "data": df,
        "true_att": att,
        "oracle_att": oracle_att,
        "efficiency_bound": efficiency_bound,
    }


def gen_dgp_mult_periods(
    n: int,
    dgp_type: int = 1,
    panel: bool = True,
    random_state=None,
) -> dict:
    """Generate data with staggered treatment adoption for multi-period DDD.

    Generates data where units adopt treatment at different times across
    three periods. The DGP has 3 timing groups (cohort=0 never treated, 2=treated
    at period 2, 3=treated at period 3) and two partitions (eligible/ineligible).

    Parameters
    ----------
    n : int
        Number of units to simulate. For panel data, this is the total number of
        units observed in all periods. For repeated cross-section data, this is
        the number of observations per period.
    dgp_type : {1, 2, 3, 4}, default=1
        Controls nuisance function specification:

        - 1: Both propensity score and outcome regression use Z (both correct)
        - 2: Propensity score uses X, outcome regression uses Z (OR correct)
        - 3: Propensity score uses Z, outcome regression uses X (PS correct)
        - 4: Both use X (both misspecified when estimating with Z)

    panel : bool, default=True
        If True, generate panel data where each unit is observed in all periods.
        If False, generate repeated cross-section data where different units are
        sampled in each period.
    random_state : int, Generator, or None, default=None
        Controls randomness for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:

        - *data*: pl.DataFrame in long format with columns [id, group, partition,
          time, y, cov1, cov2, cov3, cov4, cluster]
        - *data_wide*: pl.DataFrame in wide format with one row per unit (only for panel=True)
        - *es_0_oracle*: Oracle event-study parameter at event time 0
        - *prob_g2_p1*: Proportion of units with cohort=2 and eligibility
        - *prob_g3_p1*: Proportion of units with cohort=3 and eligibility
    """
    if dgp_type not in [1, 2, 3, 4]:
        raise ValueError(f"dgp_type must be 1, 2, 3, or 4, got {dgp_type}")

    rng = np.random.default_rng(random_state)
    xsi_ps = 0.4

    w1 = np.array([-1.0, 0.5, -0.25, -0.1])
    w2 = np.array([-0.5, 1.0, -0.1, -0.25])
    w3 = np.array([-0.25, 0.1, -1.0, -0.1])
    b1 = np.array([27.4, 13.7, 13.7, 13.7])

    index_att_g2 = 10
    index_att_g3 = 25

    if panel:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        pi_2a = np.exp(_fps2(xsi_ps, w1, ps_covars, 1.25))
        pi_2b = np.exp(_fps2(-xsi_ps, w1, ps_covars, -0.5))
        pi_3a = np.exp(_fps2(xsi_ps, w2, ps_covars, 2.0))
        pi_3b = np.exp(_fps2(-xsi_ps, w2, ps_covars, -1.25))
        pi_0a = np.exp(_fps2(xsi_ps, w3, ps_covars, -0.5))

        sum_pi = 1 + pi_2a + pi_2b + pi_3a + pi_3b + pi_0a
        pi_2a = pi_2a / sum_pi
        pi_2b = pi_2b / sum_pi
        pi_3a = pi_3a / sum_pi
        pi_3b = pi_3b / sum_pi
        pi_0a = pi_0a / sum_pi
        pi_0b = 1 - (pi_2a + pi_2b + pi_3a + pi_3b + pi_0a)

        probs_pscore = np.column_stack([pi_2a, pi_2b, pi_3a, pi_3b, pi_0a, pi_0b])
        group_types = np.array([rng.choice(6, p=probs_pscore[i]) + 1 for i in range(n)])

        partition = np.isin(group_types, [1, 3, 5]).astype(int)
        cohort = np.where(
            np.isin(group_types, [1, 2]),
            2,
            np.where(np.isin(group_types, [3, 4]), 3, 0),
        )

        index_lin = _freg(b1, or_covars)
        index_partition = partition * index_lin
        index_unobs_het = cohort * index_lin + index_partition
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10

        baseline_t1 = index_lin + index_partition + v
        y_t1 = baseline_t1 + rng.standard_normal(n)

        baseline_t2 = baseline_t1 + index_pt_violation + index_trend
        y_t2_never = baseline_t2 + rng.standard_normal(n)
        y_t2_g2 = baseline_t2 + rng.standard_normal(n) + index_att_g2 * partition

        baseline_t3 = baseline_t1 + 2 * index_trend + 2 * index_pt_violation
        y_t3_never = baseline_t3 + rng.standard_normal(n)
        y_t3_g2 = baseline_t3 + rng.standard_normal(n) + 2 * index_att_g2 * partition
        y_t3_g3 = baseline_t3 + rng.standard_normal(n) + index_att_g3 * partition

        y_t2 = np.where((cohort == 2) & (partition == 1), y_t2_g2, y_t2_never)
        y_t3 = np.where(
            (cohort == 2) & (partition == 1),
            y_t3_g2,
            np.where((cohort == 3) & (partition == 1), y_t3_g3, y_t3_never),
        )

        mask_g2_p1 = group_types == 1
        mask_g3_p1 = group_types == 3

        if np.sum(mask_g2_p1) > 0:
            att_g2_t2_unf = (np.sum(mask_g2_p1 * y_t2_g2) - np.sum(mask_g2_p1 * y_t2_never)) / np.sum(mask_g2_p1)
        else:
            att_g2_t2_unf = np.nan

        if np.sum(mask_g3_p1) > 0:
            att_g3_t3_unf = (np.sum(mask_g3_p1 * y_t3_g3) - np.sum(mask_g3_p1 * y_t3_never)) / np.sum(mask_g3_p1)
        else:
            att_g3_t3_unf = np.nan

        prob_g2_p1 = np.mean(pi_2a / (pi_2a + pi_3a))
        prob_g3_p1 = np.mean(pi_3a / (pi_2a + pi_3a))
        es_0_oracle = att_g2_t2_unf * prob_g2_p1 + att_g3_t3_unf * prob_g3_p1

        clusters = rng.integers(1, 51, size=n)

        data_wide = pl.DataFrame(
            {
                "id": np.arange(1, n + 1),
                "group": cohort,
                "partition": partition,
                "y_t1": y_t1,
                "y_t2": y_t2,
                "y_t3": y_t3,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )

        df_list = []
        for t, y_vals in enumerate([y_t1, y_t2, y_t3], start=1):
            df_t = pl.DataFrame(
                {
                    "id": np.arange(1, n + 1),
                    "group": cohort,
                    "partition": partition,
                    "time": np.full(n, t, dtype=int),
                    "y": y_vals,
                    "cov1": z[:, 0],
                    "cov2": z[:, 1],
                    "cov3": z[:, 2],
                    "cov4": z[:, 3],
                    "cluster": clusters,
                }
            )
            df_list.append(df_t)

        data = pl.concat(df_list)
        data = data.sort(["id", "time"])

        return {
            "data": data,
            "data_wide": data_wide,
            "es_0_oracle": es_0_oracle,
            "prob_g2_p1": prob_g2_p1,
            "prob_g3_p1": prob_g3_p1,
        }

    df_list = []
    id_offset = 0
    all_pi_2a = []
    all_pi_3a = []

    for t in [1, 2, 3]:
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        x4 = rng.standard_normal(n)
        x = np.column_stack([x1, x2, x3, x4])
        z = _transform_covariates(x)

        if dgp_type == 1:
            ps_covars, or_covars = z, z
        elif dgp_type == 2:
            ps_covars, or_covars = x, z
        elif dgp_type == 3:
            ps_covars, or_covars = z, x
        else:
            ps_covars, or_covars = x, x

        pi_2a = np.exp(_fps2(xsi_ps, w1, ps_covars, 1.25))
        pi_2b = np.exp(_fps2(-xsi_ps, w1, ps_covars, -0.5))
        pi_3a = np.exp(_fps2(xsi_ps, w2, ps_covars, 2.0))
        pi_3b = np.exp(_fps2(-xsi_ps, w2, ps_covars, -1.25))
        pi_0a = np.exp(_fps2(xsi_ps, w3, ps_covars, -0.5))

        sum_pi = 1 + pi_2a + pi_2b + pi_3a + pi_3b + pi_0a
        pi_2a = pi_2a / sum_pi
        pi_2b = pi_2b / sum_pi
        pi_3a = pi_3a / sum_pi
        pi_3b = pi_3b / sum_pi
        pi_0a = pi_0a / sum_pi
        pi_0b = 1 - (pi_2a + pi_2b + pi_3a + pi_3b + pi_0a)

        all_pi_2a.extend(pi_2a)
        all_pi_3a.extend(pi_3a)

        probs_pscore = np.column_stack([pi_2a, pi_2b, pi_3a, pi_3b, pi_0a, pi_0b])
        group_types = np.array([rng.choice(6, p=probs_pscore[i]) + 1 for i in range(n)])

        partition = np.isin(group_types, [1, 3, 5]).astype(int)
        cohort = np.where(
            np.isin(group_types, [1, 2]),
            2,
            np.where(np.isin(group_types, [3, 4]), 3, 0),
        )

        index_lin = _freg(b1, or_covars)
        index_partition = partition * index_lin
        index_unobs_het = cohort * index_lin + index_partition
        index_trend = index_lin

        v = rng.normal(loc=index_unobs_het, scale=1.0)
        index_pt_violation = v / 10

        baseline = index_lin + index_partition + v

        if t == 1:
            y = baseline + rng.standard_normal(n)
        elif t == 2:
            baseline_t2 = baseline + index_pt_violation + index_trend
            y_never = baseline_t2 + rng.standard_normal(n)
            y_treated = baseline_t2 + rng.standard_normal(n) + index_att_g2 * partition
            y = np.where((cohort == 2) & (partition == 1), y_treated, y_never)
        else:
            baseline_t3 = baseline + 2 * index_trend + 2 * index_pt_violation
            y_never = baseline_t3 + rng.standard_normal(n)
            y_g2 = baseline_t3 + rng.standard_normal(n) + 2 * index_att_g2 * partition
            y_g3 = baseline_t3 + rng.standard_normal(n) + index_att_g3 * partition
            y = np.where(
                (cohort == 2) & (partition == 1),
                y_g2,
                np.where((cohort == 3) & (partition == 1), y_g3, y_never),
            )

        clusters = rng.integers(1, 51, size=n)

        df_t = pl.DataFrame(
            {
                "id": np.arange(id_offset + 1, id_offset + n + 1),
                "group": cohort,
                "partition": partition,
                "time": np.full(n, t, dtype=int),
                "y": y,
                "cov1": z[:, 0],
                "cov2": z[:, 1],
                "cov3": z[:, 2],
                "cov4": z[:, 3],
                "cluster": clusters,
            }
        )
        df_list.append(df_t)
        id_offset += n

    data = pl.concat(df_list)

    all_pi_2a = np.array(all_pi_2a)
    all_pi_3a = np.array(all_pi_3a)
    prob_g2_p1 = np.mean(all_pi_2a / (all_pi_2a + all_pi_3a))
    prob_g3_p1 = np.mean(all_pi_3a / (all_pi_2a + all_pi_3a))

    return {
        "data": data,
        "data_wide": None,
        "es_0_oracle": np.nan,
        "prob_g2_p1": prob_g2_p1,
        "prob_g3_p1": prob_g3_p1,
    }


def generate_simple_ddd_data(
    n,
    att,
    random_state=None,
) -> pl.DataFrame:
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
    pl.DataFrame
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

    df_t1 = pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": np.ones(n, dtype=int),
            "y": y0,
            "x1": x1,
            "x2": x2,
        }
    )

    df_t2 = pl.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "state": state,
            "partition": partition,
            "time": np.full(n, 2, dtype=int),
            "y": y1,
            "x1": x1,
            "x2": x2,
        }
    )

    df = pl.concat([df_t1, df_t2])
    df = df.sort(["id", "time"])

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
    """Compute propensity score index."""
    return psi * (xvars @ coefs)


def _fps2(psi: float, coefs: np.ndarray, xvars: np.ndarray, c: float) -> np.ndarray:
    """Compute propensity score index with constant."""
    return psi * (c + xvars @ coefs)


def _freg(coefs: np.ndarray, xvars: np.ndarray) -> np.ndarray:
    """Compute outcome regression index."""
    return 210 + xvars @ coefs
