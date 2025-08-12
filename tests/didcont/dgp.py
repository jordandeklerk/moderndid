"""Data generating process for continuous treatment DiD."""

import numpy as np
import pandas as pd


def simulate_contdid_data(
    n=5000,
    num_time_periods=4,
    num_groups=None,
    p_group=None,
    p_untreated=None,
    dose_linear_effect=0,
    dose_quadratic_effect=0,
    seed=42,
):
    """Simulate panel data for DiD with a continuous treatment.

    Parameters
    ----------
    n : int
        The number of cross-sectional units.
    num_time_periods : int
        The number of time periods.
    num_groups : int, optional
        The number of groups. Defaults to the number of time periods.
        Groups consist of a never-treated group and groups that become
        treated in every period starting in the second period.
    p_group : list, optional
        A vector of probabilities that a unit will be in a particular treated group.
        Defaults to equal probabilities.
    p_untreated : float, optional
        The probability that a unit will be in the never-treated group.
        Defaults to 1/num_groups.
    dose_linear_effect : float
        The linear effect of the treatment.
    dose_quadratic_effect : float
        The quadratic effect of the treatment.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A balanced panel data frame with the following columns:

        - id: unit id
        - time_period: time period
        - Y: outcome
        - G: unit's group
        - D: amount of the treatment
    """
    rng = np.random.default_rng(seed)

    if num_groups is None:
        num_groups = num_time_periods

    time_periods = np.arange(1, num_time_periods + 1)
    groups = np.concatenate(([0], time_periods[1:]))

    if p_untreated is None:
        p_untreated = 1 / num_groups

    if p_group is None:
        p_group_len = num_groups - 1
        p_group = np.repeat((1 - p_untreated) / p_group_len, p_group_len)

    p = np.concatenate(([p_untreated], p_group))
    p /= p.sum()

    group = rng.choice(groups, n, replace=True, p=p)
    dose = rng.uniform(0, 1, n)

    eta = rng.normal(loc=group, scale=1, size=n)
    time_effects = np.arange(1, num_time_periods + 1)
    y0_t = time_effects + eta[:, np.newaxis] + rng.normal(size=(n, num_time_periods))

    y1_t = (
        dose_linear_effect * dose[:, np.newaxis]
        + dose_quadratic_effect * (dose**2)[:, np.newaxis]
        + time_effects
        + eta[:, np.newaxis]
        + rng.normal(size=(n, num_time_periods))
    )

    post_matrix = (group[:, np.newaxis] <= time_periods) & (group[:, np.newaxis] != 0)

    y = post_matrix * y1_t + (1 - post_matrix) * y0_t

    df = pd.DataFrame(y, columns=[f"Y_{t}" for t in time_periods])
    df["id"] = np.arange(1, n + 1)
    df["G"] = group
    df["D"] = dose

    df_long = pd.melt(
        df,
        id_vars=["id", "G", "D"],
        value_vars=[f"Y_{t}" for t in time_periods],
        var_name="time_period",
        value_name="Y",
    )

    df_long["time_period"] = df_long["time_period"].str.replace("Y_", "").astype(int)
    df_long.loc[df_long["G"] == 0, "D"] = 0
    df_long.loc[df_long["time_period"] < df_long["G"], "D"] = 0

    return df_long.sort_values(["id", "time_period"]).reset_index(drop=True)
