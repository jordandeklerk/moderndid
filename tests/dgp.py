"""Data generating process."""

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


class DiDDGP(BaseDGP):
    """Data generating process for difference-in-differences designs.

    Creates synthetic panel data with treatment effects for DiD analysis.
    The data generating process includes unit-specific effects, time-specific effects,
    treatment effects (potentially heterogeneous), and random noise.
    """

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
        treatment[np.argsort(propensity)[-n_treated:]] = True

        unit_effects = confounding_strength * features[:, 0] + np.random.normal(
            size=self.n_units, scale=unit_effect_scale
        )
        time_effects = time_effect_scale * np.arange(self.n_time) + np.random.normal(
            size=self.n_time, scale=time_effect_scale / 2
        )

        # Treatment effects (potentially heterogeneous)
        true_effects = effect_base + effect_coef * features[:, effect_feature_idx]

        # Outcomes
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
            "treatment_effects": true_effects[treatment],
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
            treatment_times = list(range(2, max_time + 1))

        unit_ids = np.arange(self.n_units)
        time_ids = np.arange(self.n_time)

        features = np.random.normal(size=(self.n_units, self.n_features))

        random_component = np.random.normal(size=self.n_units, scale=1.0)
        propensity = confounding_strength * features[:, 0] + random_component

        # Determine which units will ever be treated
        n_treated = int(self.n_units * treatment_fraction)
        ever_treated = np.zeros(self.n_units, dtype=bool)
        ever_treated[np.argsort(propensity)[-n_treated:]] = True

        first_treatment = np.full(self.n_units, -1)  # -1 means never treated
        for i in range(self.n_units):
            if ever_treated[i]:
                first_treatment[i] = np.random.choice(treatment_times)

        unit_effects = confounding_strength * features[:, 0] + np.random.normal(
            size=self.n_units, scale=unit_effect_scale
        )
        time_effects = time_effect_scale * np.arange(self.n_time) + np.random.normal(
            size=self.n_time, scale=time_effect_scale / 2
        )

        true_effects = effect_base + effect_coef * features[:, effect_feature_idx]

        if dynamic_effects is None:
            dynamic_effects = [1.0]

        outcomes = np.zeros((self.n_units, self.n_time))
        for i in range(self.n_units):
            for t in range(self.n_time):
                outcomes[i, t] = unit_effects[i] + time_effects[t]

                # Add treatment effect if unit is treated at time t
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
            "treatment_effects": true_effects[ever_treated],
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "features": features,
            "first_treatment": first_treatment,
            "dynamic_effects": dynamic_effects,
        }


class SyntheticControlsDGP(BaseDGP):
    """Data generating process for synthetic control designs.

    Creates synthetic panel data where one or more units are treated after
    a pre-treatment period, and their outcomes are compared against a
    synthetically constructed control unit from a pool of untreated units.
    The data includes unit-specific effects, time-specific effects,
    effects of covariates, and a treatment effect.
    """

    def __init__(
        self,
        n_treated_units=1,
        n_control_units=20,
        n_time_pre=10,
        n_time_post=5,
        n_features=3,
        random_seed=42,
    ):
        """Initialize the Synthetic Controls data generating process.

        Parameters
        ----------
        n_treated_units : int, default=1
            Number of units that receive treatment.
        n_control_units : int, default=20
            Number of units in the control pool.
        n_time_pre : int, default=10
            Number of time periods before treatment.
        n_time_post : int, default=5
            Number of time periods after treatment starts.
        n_features : int, default=3
            Number of covariates to generate.
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.n_treated_units = n_treated_units
        self.n_control_units = n_control_units
        self.n_time_pre = n_time_pre
        self.n_time_post = n_time_post
        self.n_features = n_features

        if self.n_treated_units <= 0:
            raise ValueError("n_treated_units must be positive.")
        if self.n_control_units <= 0:
            raise ValueError("n_control_units must be positive.")
        if self.n_time_pre <= 0:
            raise ValueError("n_time_pre must be positive.")
        if self.n_time_post <= 0:
            raise ValueError("n_time_post must be positive.")

    def generate_data(
        self,
        *args,
        feature_effect_scale=1.0,
        unit_effect_scale=0.5,
        time_effect_scale=0.3,
        noise_scale=0.2,
        treatment_effect_value=2.0,
        confounding_strength=0.3,
        **kwargs,
    ):
        """Generate synthetic panel data for synthetic control analysis.

        Parameters
        ----------
        feature_effect_scale : float, default=1.0
            Scale parameter for the effects of features on the outcome.
            Feature weights are drawn from N(0, feature_effect_scale^2).
        unit_effect_scale : float, default=0.5
            Scale parameter for the random component of unit-specific effects.
        time_effect_scale : float, default=0.3
            Scale parameter for time-specific effects.
        noise_scale : float, default=0.2
            Scale parameter for random noise in the outcome.
        treatment_effect_value : float, default=2.0
            The magnitude of the treatment effect.
        confounding_strength : float, default=0.3
            Strength of confounding. Unit effects are generated as
            `confounding_strength * features[:, 0] + N(0, unit_effect_scale^2)`.
            If n_features is 0, this parameter has no effect.

        Returns
        -------
        dict
            Dictionary containing:
            - 'df': pandas DataFrame with panel data.
            - 'true_treatment_effect': The true treatment effect value (scalar).
            - 'unit_effects': numpy array of unit-specific effects.
            - 'time_effects': numpy array of time-specific effects.
            - 'features': numpy array of features (n_total_units x n_features).
            - 'feature_weights': numpy array of weights for features.
            - 'n_treated_units': Number of treated units.
            - 'n_control_units': Number of control units.
            - 'n_time_pre': Number of pre-treatment periods.
            - 'n_time_post': Number of post-treatment periods.
        """
        if args:
            raise ValueError(f"Unexpected positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
        self.set_seed()

        n_total_units = self.n_treated_units + self.n_control_units
        n_total_time = self.n_time_pre + self.n_time_post

        unit_ids = np.arange(n_total_units)
        time_ids = np.arange(n_total_time)

        # Features
        features = np.random.normal(size=(n_total_units, self.n_features))

        # Unit-specific effects
        confounded_part = np.zeros(n_total_units)
        if self.n_features > 0 and confounding_strength != 0:
            confounded_part = confounding_strength * features[:, 0]

        random_part_unit_effects = np.random.normal(size=n_total_units, scale=unit_effect_scale)
        unit_effects = confounded_part + random_part_unit_effects

        # Time-specific effects
        time_effects_trend = time_effect_scale * np.arange(n_total_time)
        time_effects_noise = np.random.normal(size=n_total_time, scale=time_effect_scale / 2)
        time_effects = time_effects_trend + time_effects_noise

        # Feature weights
        feature_weights = np.random.normal(loc=0.0, scale=feature_effect_scale, size=self.n_features)

        # Outcomes
        outcomes = np.zeros((n_total_units, n_total_time))
        for i in range(n_total_units):
            for t in range(n_total_time):
                outcome_val = unit_effects[i] + time_effects[t]

                if self.n_features > 0:
                    outcome_val += np.dot(features[i, :], feature_weights)

                # Add treatment effect
                is_treated_unit = i < self.n_treated_units
                is_post_period = t >= self.n_time_pre
                if is_treated_unit and is_post_period:
                    outcome_val += treatment_effect_value

                outcome_val += np.random.normal(scale=noise_scale)
                outcomes[i, t] = outcome_val

        data = []
        for i in unit_ids:
            for t in time_ids:
                row = {
                    "unit_id": i,
                    "time_id": t,
                    "outcome": outcomes[i, t],
                    "is_treated_unit": i < self.n_treated_units,
                    "is_post_period": t >= self.n_time_pre,
                }
                for j in range(self.n_features):
                    row[f"X{j + 1}"] = features[i, j]
                data.append(row)

        df = pd.DataFrame(data)

        return {
            "df": df,
            "true_treatment_effect": treatment_effect_value,
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "features": features,
            "feature_weights": feature_weights,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "n_time_pre": self.n_time_pre,
            "n_time_post": self.n_time_post,
        }
