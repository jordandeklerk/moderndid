"""Scalable data generating process for staggered DiD benchmarking."""

from __future__ import annotations

import numpy as np
import polars as pl


class StaggeredDIDDGP:
    """Data generating process for staggered difference-in-differences designs.

    Generates long-format panel data compatible with :func:`moderndid.att_gt` for
    benchmarking computational performance. Supports configurable number of units,
    time periods, treatment timing groups, and covariates.

    Parameters
    ----------
    n_units : int, default=1000
        Number of cross-sectional units.
    n_periods : int, default=5
        Number of time periods.
    n_groups : int, default=3
        Number of distinct treatment timing groups (excluding never-treated).
    n_covariates : int, default=0
        Number of covariates to include (x1, x2, ..., xN).
    treatment_effect : float, default=1.0
        Base treatment effect magnitude.
    never_treated_fraction : float, default=0.3
        Fraction of units that are never treated.
    random_seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_units: int = 1000,
        n_periods: int = 5,
        n_groups: int = 3,
        n_covariates: int = 0,
        treatment_effect: float = 1.0,
        never_treated_fraction: float = 0.3,
        random_seed: int = 42,
    ):
        self.n_units = n_units
        self.n_periods = n_periods
        self.n_groups = n_groups
        self.n_covariates = n_covariates
        self.treatment_effect = treatment_effect
        self.never_treated_fraction = never_treated_fraction
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def set_seed(self, seed: int | None = None) -> None:
        """Set the random seed."""
        if seed is not None:
            self.random_seed = seed
        self._rng = np.random.default_rng(self.random_seed)

    def generate_data(self) -> dict:
        """Generate synthetic staggered DiD panel data.

        Returns
        -------
        dict
            Dictionary containing:

            - 'df': pl.DataFrame with columns [id, time, y, first_treat, x1, ..., xN]
            - 'true_att': dict mapping (group, time) to true ATT value
            - 'n_observations': total number of observations
            - 'n_units': number of units
            - 'n_periods': number of time periods
            - 'groups': list of treatment timing groups
        """
        self.set_seed()

        time_periods = np.arange(1, self.n_periods + 1)

        max_start_period = min(self.n_periods, self.n_periods)
        group_periods = np.linspace(2, max_start_period, self.n_groups, dtype=int)
        group_periods = np.unique(group_periods)

        n_never_treated = int(self.n_units * self.never_treated_fraction)
        n_treated = self.n_units - n_never_treated

        units_per_group = n_treated // len(group_periods)
        remainder = n_treated % len(group_periods)

        group_assignments = []
        for i, g in enumerate(group_periods):
            n_in_group = units_per_group + (1 if i < remainder else 0)
            group_assignments.extend([g] * n_in_group)

        group_assignments.extend([0] * n_never_treated)
        self._rng.shuffle(group_assignments)
        first_treat = np.array(group_assignments)

        unit_ids = np.arange(1, self.n_units + 1)
        unit_effects = self._rng.normal(0, 1, self.n_units)
        time_effects = self._rng.normal(0, 0.5, self.n_periods)

        covariates = {}
        covariate_effects = np.zeros(self.n_units)
        for j in range(self.n_covariates):
            x_j = self._rng.normal(0, 1, self.n_units)
            covariates[f"x{j + 1}"] = x_j
            covariate_effects += 0.5 * x_j

        ids = []
        times = []
        ys = []
        first_treats = []
        covariate_cols = {f"x{j + 1}": [] for j in range(self.n_covariates)}

        true_att = {}

        for i, unit_id in enumerate(unit_ids):
            g = first_treat[i]

            for t_idx, t in enumerate(time_periods):
                ids.append(unit_id)
                times.append(t)
                first_treats.append(g)

                for j in range(self.n_covariates):
                    covariate_cols[f"x{j + 1}"].append(covariates[f"x{j + 1}"][i])

                y = unit_effects[i] + time_effects[t_idx] + covariate_effects[i]
                y += self._rng.normal(0, 0.5)

                if g > 0 and t >= g:
                    time_since_treat = t - g
                    att_gt = self.treatment_effect * (1 + 0.1 * time_since_treat)
                    y += att_gt

                    if (g, t) not in true_att:
                        true_att[(g, t)] = att_gt

                ys.append(y)

        data = {
            "id": ids,
            "time": times,
            "y": ys,
            "first_treat": first_treats,
        }
        data.update(covariate_cols)

        df = pl.DataFrame(data)
        df = df.sort(["id", "time"])

        return {
            "df": df,
            "true_att": true_att,
            "n_observations": len(df),
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "groups": sorted([g for g in np.unique(first_treat) if g > 0]),
        }
