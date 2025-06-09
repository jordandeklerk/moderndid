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


class DiD(BaseDGP):
    """Data generating process for difference-in-differences designs."""

    def __init__(
        self,
        n_units=100,
        n_time=5,
        treatment_time=3,
        n_features=2,
        random_seed=42,
    ):
        """Initialize the DiD data generating process.

        Parameters
        ----------
        n_units : int, default=100
            Number of units (e.g., individuals, firms, countries).
        n_time : int, default=5
            Number of time periods.
        treatment_time : int, default=3
            Time period when treatment starts (0-indexed).
        n_features : int, default=2
            Number of covariates to generate.
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.n_units = n_units
        self.n_time = n_time
        self.treatment_time = treatment_time
        self.n_features = n_features

    def generate_data(
        self,
        *args,
        treatment_fraction=0.5,
        unit_effect_scale=0.5,
        time_effect_scale=0.3,
        noise_scale=0.2,
        effect_base=1.0,
        effect_coef=0.5,
        effect_feature_idx=1,
        confounding_strength=0.5,
        **kwargs,
    ):
        """Generate synthetic panel data for difference-in-differences analysis.

        Parameters
        ----------
        treatment_fraction : float, default=0.5
            Fraction of units to be treated.
        unit_effect_scale : float, default=0.5
            Scale parameter for unit-specific effects.
        time_effect_scale : float, default=0.3
            Scale parameter for time-specific effects.
        noise_scale : float, default=0.2
            Scale parameter for random noise.
        effect_base : float, default=1.0
            Base treatment effect.
        effect_coef : float, default=0.5
            Coefficient for heterogeneous treatment effects.
        effect_feature_idx : int, default=1
            Index of the feature used for heterogeneous treatment effects.
        confounding_strength : float, default=0.5
            Strength of confounding (correlation between features and treatment).

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with panel data
            - 'treatment_effects': numpy array of true treatment effects
            - 'unit_effects': numpy array of unit-specific effects
            - 'time_effects': numpy array of time-specific effects
            - 'features': numpy array of features
        """
        if args:
            raise ValueError(f"Unexpected positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
        self.set_seed()

        unit_ids = np.arange(self.n_units)
        time_ids = np.arange(self.n_time)

        features = np.random.normal(size=(self.n_units, self.n_features))

        # Treatment assignment with potential confounding
        random_component = np.random.normal(size=self.n_units, scale=1.0)
        propensity = confounding_strength * features[:, 0] + random_component
        treatment = np.zeros(self.n_units, dtype=bool)
        n_treated = int(self.n_units * treatment_fraction)
        if n_treated > 0:
            treatment[np.argsort(propensity)[-n_treated:]] = True

        unit_effects = confounding_strength * features[:, 0] + np.random.normal(
            size=self.n_units, scale=unit_effect_scale
        )
        time_effects = time_effect_scale * np.arange(self.n_time) + np.random.normal(
            size=self.n_time, scale=time_effect_scale / 2
        )

        true_effects = effect_base
        if self.n_features > 0 and effect_feature_idx < self.n_features:
            true_effects = true_effects + effect_coef * features[:, effect_feature_idx]
        else:
            true_effects = np.full(self.n_units, effect_base)

        outcomes = np.zeros((self.n_units, self.n_time))
        for i in range(self.n_units):
            for t in range(self.n_time):
                outcomes[i, t] = unit_effects[i] + time_effects[t]

                if treatment[i] and t >= self.treatment_time:
                    outcomes[i, t] += true_effects[i]

                outcomes[i, t] += np.random.normal(scale=noise_scale)

        data = []
        for i in unit_ids:
            for t in time_ids:
                row = {
                    "unit_id": i,
                    "time_id": t,
                    "treatment": treatment[i] and t >= self.treatment_time,
                }

                for j in range(self.n_features):
                    row[f"X{j + 1}"] = features[i, j]

                row["outcome"] = outcomes[i, t]
                data.append(row)

        df = pd.DataFrame(data)
        return {
            "df": df,
            "treatment_effects": true_effects[treatment] if np.any(treatment) else np.array([]),
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "features": features,
        }

    def generate_staggered_adoption(
        self,
        treatment_times=None,
        treatment_fraction=0.5,
        unit_effect_scale=0.5,
        time_effect_scale=0.3,
        noise_scale=0.2,
        effect_base=1.0,
        effect_coef=0.5,
        effect_feature_idx=1,
        dynamic_effects=None,
        confounding_strength=0.5,
    ):
        """Generate synthetic panel data with staggered treatment adoption.

        Parameters
        ----------
        treatment_times : list or None, default=None
            List of possible treatment times. If None, uses [2, 3, 4].
        treatment_fraction : float, default=0.5
            Fraction of units to ever be treated.
        unit_effect_scale : float, default=0.5
            Scale parameter for unit-specific effects.
        time_effect_scale : float, default=0.3
            Scale parameter for time-specific effects.
        noise_scale : float, default=0.2
            Scale parameter for random noise.
        effect_base : float, default=1.0
            Base treatment effect.
        effect_coef : float, default=0.5
            Coefficient for heterogeneous treatment effects.
        effect_feature_idx : int, default=1
            Index of the feature used for heterogeneous treatment effects.
        dynamic_effects : list or None, default=None
            List of coefficients for dynamic treatment effects.
            For example, [0.5, 1.0, 1.2] would mean the effect is 0.5 in the
            first period, 1.0 in the second, and 1.2 in the third.
            If None, uses a constant effect.
        confounding_strength : float, default=0.5
            Strength of confounding (correlation between features and treatment).

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with panel data
            - 'treatment_effects': numpy array of true treatment effects
            - 'unit_effects': numpy array of unit-specific effects
            - 'time_effects': numpy array of time-specific effects
            - 'features': numpy array of features
            - 'first_treatment': numpy array indicating when each unit was first treated
        """
        self.set_seed()

        if treatment_times is None:
            max_time = min(4, self.n_time - 1)
            if max_time < 2 and self.n_time > 1:
                treatment_times = [self.n_time - 1] if self.n_time - 1 >= 0 else []
            elif max_time < 2 and self.n_time <= 1:
                treatment_times = []
            else:
                treatment_times = list(range(2, max_time + 1))

        unit_ids = np.arange(self.n_units)
        time_ids = np.arange(self.n_time)

        features = np.random.normal(size=(self.n_units, self.n_features))

        random_component = np.random.normal(size=self.n_units, scale=1.0)
        propensity = confounding_strength * features[:, 0] + random_component

        n_treated = int(self.n_units * treatment_fraction)
        ever_treated = np.zeros(self.n_units, dtype=bool)
        if n_treated > 0:
            ever_treated[np.argsort(propensity)[-n_treated:]] = True

        first_treatment = np.full(self.n_units, -1)
        if len(treatment_times) > 0:
            for i in range(self.n_units):
                if ever_treated[i]:
                    first_treatment[i] = np.random.choice(treatment_times)
        else:
            ever_treated.fill(False)

        unit_effects = confounding_strength * features[:, 0] + np.random.normal(
            size=self.n_units, scale=unit_effect_scale
        )
        time_effects = time_effect_scale * np.arange(self.n_time) + np.random.normal(
            size=self.n_time, scale=time_effect_scale / 2
        )

        true_effects_base = effect_base
        if self.n_features > 0 and effect_feature_idx < self.n_features:
            true_effects = true_effects_base + effect_coef * features[:, effect_feature_idx]
        else:
            true_effects = np.full(self.n_units, effect_base)

        if dynamic_effects is None:
            dynamic_effects = [1.0]

        outcomes = np.zeros((self.n_units, self.n_time))
        for i in range(self.n_units):
            for t in range(self.n_time):
                outcomes[i, t] = unit_effects[i] + time_effects[t]

                if first_treatment[i] >= 0 and t >= first_treatment[i]:
                    periods_treated = t - first_treatment[i]
                    if periods_treated < len(dynamic_effects):
                        effect_multiplier = dynamic_effects[periods_treated]
                    else:
                        effect_multiplier = dynamic_effects[-1]

                    outcomes[i, t] += true_effects[i] * effect_multiplier

                outcomes[i, t] += np.random.normal(scale=noise_scale)

        data = []
        for i in unit_ids:
            for t in time_ids:
                row = {
                    "unit_id": i,
                    "time_id": t,
                    "treatment": first_treatment[i] >= 0 and t >= first_treatment[i],
                    "first_treated": first_treatment[i] if first_treatment[i] >= 0 else np.nan,
                }

                for j in range(self.n_features):
                    row[f"X{j + 1}"] = features[i, j]

                row["outcome"] = outcomes[i, t]
                data.append(row)

        df = pd.DataFrame(data)
        return {
            "df": df,
            "treatment_effects": true_effects[ever_treated] if np.any(ever_treated) else np.array([]),
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "features": features,
            "first_treatment": first_treatment,
            "dynamic_effects": dynamic_effects,
        }


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
