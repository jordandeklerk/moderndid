"""Data generating processes for testing."""

import abc

import numpy as np
import pandas as pd


class BaseDGP(abc.ABC):
    """Base abstract class for all data generating processes."""

    def __init__(self, random_seed=42):
        """Initialize the data generating process.

        Parameters
        ----------
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.set_seed()

    @abc.abstractmethod
    def generate_data(self, *args, **kwargs):
        """Generate synthetic data from the DGP.

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with the generated data
            - Additional elements specific to the DGP
        """

    def set_seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed : int, optional
            If provided, sets a new seed. Otherwise, uses the one from initialization.
        """
        if seed is not None:
            self.random_seed = seed
        np.random.seed(self.random_seed)


class SantAnnaZhaoDRDiD(BaseDGP):
    """Data Generating Process for Sant'Anna and Zhao (2020)'s DGP 1 (RC Case)."""

    def __init__(
        self,
        n_units: int = 1000,
        treatment_fraction: float = 0.5,
        common_support_strength: float = 0.75,
        random_seed: int = 1234,
    ):
        """Initialize the Sant'Anna and Zhao DRDiD data generating process.

        Parameters
        ----------
        n_units : int, default=1000
            Total number of units (sample size)
        treatment_fraction : float, default=0.5
            Proportion of units in the post-treatment period
        common_support_strength : float, default=0.75
            Propensity score index (strength of common support)
        random_seed : int, default=1234
            Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.n_units = n_units
        self.treatment_fraction = treatment_fraction
        self.common_support_strength = common_support_strength
        self.mean_z1 = np.exp(0.25 / 2)
        self.sd_z1 = np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
        self.mean_z2 = 10
        self.sd_z2 = 0.54164
        self.mean_z3 = 0.21887
        self.sd_z3 = 0.04453
        self.mean_z4 = 402
        self.sd_z4 = 56.63891

    def generate_data(self, *args, att: float = 0.0, **kwargs):
        """Generate synthetic data following Sant'Anna and Zhao (2020)'s DGP 1.

        Parameters
        ----------
        att : float, default=0.0
            Average treatment effect on the treated.
            In the original paper this is set to 0 for simulation purposes.

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with the generated data in long format
            - 'true_att': True average treatment effect on the treated
            - Additional elements specific to the DGP
        """
        if args:
            raise ValueError(f"Unexpected positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

        self.set_seed()

        x1 = np.random.normal(loc=0, scale=1, size=self.n_units)
        x2 = np.random.normal(loc=0, scale=1, size=self.n_units)
        x3 = np.random.normal(loc=0, scale=1, size=self.n_units)
        x4 = np.random.normal(loc=0, scale=1, size=self.n_units)

        z1 = np.exp(x1 / 2)
        z2 = x2 / (1 + np.exp(x1)) + 10
        z3 = np.power(x1 * x3 / 25 + 0.6, 3)
        z4 = np.power(x1 + x4 + 20, 2)

        z1 = (z1 - self.mean_z1) / self.sd_z1
        z2 = (z2 - self.mean_z2) / self.sd_z2
        z3 = (z3 - self.mean_z3) / self.sd_z3
        z4 = (z4 - self.mean_z4) / self.sd_z4

        x = np.column_stack([x1, x2, x3, x4])
        z = np.column_stack([z1, z2, z3, z4])

        pi = 1 / (1 + np.exp(-self.common_support_strength * (-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4)))
        d = (np.random.uniform(size=self.n_units) <= pi).astype(int)

        index_lin = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)
        index_unobs_het = d * index_lin
        index_att = att

        index_trend = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)

        v = np.random.normal(loc=index_unobs_het, scale=1, size=self.n_units)
        y0 = index_lin + v + np.random.normal(size=self.n_units)
        y10 = index_lin + v + np.random.normal(size=self.n_units) + index_trend
        y11 = index_lin + v + np.random.normal(size=self.n_units) + index_trend + index_att

        y1 = d * y11 + (1 - d) * y10
        post = (np.random.uniform(size=self.n_units) <= self.treatment_fraction).astype(int)
        y = post * y1 + (1 - post) * y0
        id = np.arange(1, self.n_units + 1)

        df = pd.DataFrame({"id": id, "post": post, "y": y, "d": d, "x1": z1, "x2": z2, "x3": z3, "x4": z4})
        df = df.sort_values(by="id").reset_index(drop=True)
        true_att = att

        return {
            "df": df,
            "true_att": true_att,
            "raw_covariates": x,
            "transformed_covariates": z,
            "propensity_scores": pi,
            "potential_outcomes_pre": {"y0": y0},
            "potential_outcomes_post": {"y0": y10, "y1": y11},
        }
