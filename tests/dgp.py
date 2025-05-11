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


class SyntheticControl(BaseDGP):
    """Data generating process for synthetic control designs."""

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

        features = np.random.normal(size=(n_total_units, self.n_features))

        confounded_part = np.zeros(n_total_units)
        if self.n_features > 0 and confounding_strength != 0:
            confounded_part = confounding_strength * features[:, 0]

        random_part_unit_effects = np.random.normal(size=n_total_units, scale=unit_effect_scale)
        unit_effects = confounded_part + random_part_unit_effects

        time_effects_trend = time_effect_scale * np.arange(n_total_time)
        time_effects_noise = np.random.normal(size=n_total_time, scale=time_effect_scale / 2)
        time_effects = time_effects_trend + time_effects_noise

        feature_weights = np.array([])
        if self.n_features > 0:
            feature_weights = np.random.normal(loc=0.0, scale=feature_effect_scale, size=self.n_features)

        outcomes = np.zeros((n_total_units, n_total_time))
        for i in range(n_total_units):
            for t in range(n_total_time):
                outcome_val = unit_effects[i] + time_effects[t]

                if self.n_features > 0:
                    outcome_val += np.dot(features[i, :], feature_weights)

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


class DRDiDSC(BaseDGP):
    """Data Generating Process for Doubly Robust DiD Synthetic Control methods."""

    def __init__(
        self,
        n_units: int = 100,
        n_time_pre: int = 10,
        n_time_post: int = 5,
        n_features: int = 3,
        n_latent_factors: int = 2,
        random_seed: int = 42,
    ):
        """Initialize the DRDiDSC data generating process.

        Parameters
        ----------
        n_units : int, default=100
            Total number of units.
        n_time_pre : int, default=10
            Number of time periods before treatment.
        n_time_post : int, default=5
            Number of time periods after treatment starts.
        n_features : int, default=3
            Number of time-invariant covariates to generate.
        n_latent_factors : int, default=2
            Number of latent factors in the outcome model.
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.n_units = n_units
        self.n_time_pre = n_time_pre
        self.n_time_post = n_time_post
        self.n_total_time = n_time_pre + n_time_post
        self.treatment_time = n_time_pre
        self.n_features = n_features
        self.n_latent_factors = n_latent_factors

    def generate_data(
        self,
        *args,
        feature_mean: float | np.ndarray = 0.0,
        feature_cov: np.ndarray | None = None,
        prop_score_coefs: np.ndarray | None = None,
        prop_score_intercept: float = 0.0,
        treatment_fraction_target: float = 0.2,
        factor_loadings_scale: float = 0.5,
        time_factors_scale: float = 0.5,
        outcome_coefs: np.ndarray | None = None,
        unit_fe_scale: float = 0.3,
        time_fe_scale: float = 0.2,
        time_trend_strength: float = 0.05,
        noise_scale: float = 0.1,
        att_base: float = 1.0,
        att_hetero_coefs: np.ndarray | None = None,
        att_dynamic_coeffs: list[float] | np.ndarray | None = None,
        **kwargs,
    ):
        """Generate synthetic panel data for DR-DiD-SC analysis.

        Parameters
        ----------
        feature_mean : float or ndarray, default=0.0
            Mean for covariate generation. Scalar or array of shape (n_features,).
        feature_cov : ndarray or None, default=None
            Covariance matrix for covariate generation, shape (n_features, n_features).
            If None, defaults to identity matrix.
        prop_score_coefs : ndarray or None, default=None
            Coefficients for features in the propensity score model, shape (n_features,).
            If None and n_features > 0, defaults to [1.0, 0, ...].
        prop_score_intercept : float, default=0.0
            Intercept for the propensity score model.
        treatment_fraction_target : float, default=0.2
            Target fraction of units to be treated.
        factor_loadings_scale : float, default=0.5
            Scale (std dev) for random factor loadings L_i.
        time_factors_scale : float, default=0.5
            Scale (std dev) for random time factors F_t.
        outcome_coefs : ndarray or None, default=None
            Coefficients for features in the Y(0) outcome model, shape (n_features,).
            If None and n_features > 0, defaults to [0.5, 0, ...].
        unit_fe_scale : float, default=0.3
            Scale (std dev) for random unit fixed effects alpha_i.
        time_fe_scale : float, default=0.2
            Scale (std dev) for random component of time fixed effects delta_t.
        time_trend_strength : float, default=0.05
             Coefficient for a linear time trend component in delta_t.
        noise_scale : float, default=0.1
            Scale (std dev) for idiosyncratic error term epsilon_it.
        att_base : float, default=1.0
            Base average treatment effect on the treated.
        att_hetero_coefs : ndarray or None, default=None
            Coefficients for feature-based heterogeneity in ATT, shape (n_features,).
            If None, no heterogeneity based on X.
        att_dynamic_coeffs : list[float], ndarray or None, default=None
            Coefficients for dynamic treatment effects relative to treatment_time.
            e.g., [1.0, 1.2, 0.8] means effect is scaled by 1.0 in period T0, 1.2 in T0+1, etc.
            If None, effect is constant as per att_base and att_hetero_coefs.

        Returns
        -------
        dict
            Dictionary containing detailed components of the DGP.
        """
        if args:
            raise ValueError(f"Unexpected positional arguments: {args}")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")
        self.set_seed()

        _feature_mean_vec: np.ndarray
        if isinstance(feature_mean, int | float):
            _feature_mean_vec = np.full(self.n_features, float(feature_mean))
        else:
            _feature_mean_vec = np.asarray(feature_mean, dtype=float)
            if _feature_mean_vec.shape != (self.n_features,):
                raise ValueError("feature_mean must be scalar or (n_features,)")

        _feature_cov_matrix: np.ndarray
        if feature_cov is None:
            _feature_cov_matrix = np.eye(self.n_features)
        else:
            _feature_cov_matrix = np.asarray(feature_cov, dtype=float)
            if _feature_cov_matrix.shape != (self.n_features, self.n_features):
                raise ValueError("feature_cov must be (n_features, n_features)")

        X: np.ndarray
        if self.n_features > 0:
            X = np.random.multivariate_normal(_feature_mean_vec, _feature_cov_matrix, self.n_units)
        else:
            X = np.empty((self.n_units, 0))

        _prop_score_coefs_actual = np.array([], dtype=float)
        linear_propensity: np.ndarray
        if self.n_features > 0:
            if prop_score_coefs is None:
                _prop_score_coefs_actual = np.zeros(self.n_features)
                _prop_score_coefs_actual[0] = 1.0
            else:
                _prop_score_coefs_actual = np.asarray(prop_score_coefs, dtype=float)
                if _prop_score_coefs_actual.shape != (self.n_features,):
                    raise ValueError("prop_score_coefs must be (n_features,)")
            linear_propensity = X @ _prop_score_coefs_actual + prop_score_intercept
        else:
            linear_propensity = np.full(self.n_units, prop_score_intercept, dtype=float)

        true_prop_scores = 1 / (1 + np.exp(-linear_propensity))

        D = np.zeros(self.n_units, dtype=bool)
        if self.n_units > 0:
            n_treated_target = int(self.n_units * treatment_fraction_target)
            n_treated_actual = n_treated_target

            if n_treated_target >= self.n_units and self.n_units > 1:
                n_treated_actual = self.n_units - 1
            elif n_treated_target >= self.n_units and self.n_units <= 1:
                n_treated_actual = 0

            if n_treated_actual < 0:
                n_treated_actual = 0

            if n_treated_actual > 0:
                sorted_indices = np.argsort(true_prop_scores)
                D[sorted_indices[-n_treated_actual:]] = True

        L = np.random.normal(scale=factor_loadings_scale, size=(self.n_units, self.n_latent_factors))
        F = np.random.normal(scale=time_factors_scale, size=(self.n_total_time, self.n_latent_factors))
        factor_model_component = L @ F.T

        _outcome_coefs_actual = np.array([], dtype=float)
        direct_covariate_effect: np.ndarray
        if self.n_features > 0:
            if outcome_coefs is None:
                _outcome_coefs_actual = np.zeros(self.n_features)
                _outcome_coefs_actual[0] = 0.5
            else:
                _outcome_coefs_actual = np.asarray(outcome_coefs, dtype=float)
                if _outcome_coefs_actual.shape != (self.n_features,):
                    raise ValueError("outcome_coefs must be (n_features,)")
            direct_covariate_effect = (X @ _outcome_coefs_actual)[:, np.newaxis]
        else:
            direct_covariate_effect = np.zeros((self.n_units, 1))

        alpha_i = np.random.normal(scale=unit_fe_scale, size=self.n_units)[:, np.newaxis]

        time_rng = np.arange(self.n_total_time)
        delta_t_random_component = np.random.normal(scale=time_fe_scale, size=self.n_total_time)
        delta_t_trend_component = time_trend_strength * time_rng
        delta_t = (delta_t_random_component + delta_t_trend_component)[np.newaxis, :]

        epsilon_it = np.random.normal(scale=noise_scale, size=(self.n_units, self.n_total_time))
        Y0_it = direct_covariate_effect + factor_model_component + alpha_i + delta_t + epsilon_it

        _att_hetero_coefs_actual = np.array([], dtype=float)
        hetero_component_for_tau = np.zeros((self.n_units, 1))
        if self.n_features > 0 and att_hetero_coefs is not None:
            _att_hetero_coefs_actual = np.asarray(att_hetero_coefs, dtype=float)
            if _att_hetero_coefs_actual.shape != (self.n_features,):
                raise ValueError("att_hetero_coefs must be (n_features,)")
            hetero_component_for_tau = (X @ _att_hetero_coefs_actual)[:, np.newaxis]

        unit_base_att = att_base + hetero_component_for_tau

        actual_tau_it = np.zeros((self.n_units, self.n_total_time))
        _att_dynamic_coeffs_actual: np.ndarray
        if att_dynamic_coeffs is None:
            _att_dynamic_coeffs_actual = np.array([1.0])
        else:
            _att_dynamic_coeffs_actual = np.asarray(att_dynamic_coeffs, dtype=float)

        for i in range(self.n_units):
            if D[i]:
                for t in range(self.treatment_time, self.n_total_time):
                    time_since_treatment = t - self.treatment_time
                    dyn_coeff_idx = min(time_since_treatment, len(_att_dynamic_coeffs_actual) - 1)
                    dynamic_multiplier = _att_dynamic_coeffs_actual[dyn_coeff_idx]
                    actual_tau_it[i, t] = unit_base_att[i, 0] * dynamic_multiplier

        Y_it = Y0_it + actual_tau_it

        unit_ids_col = np.repeat(np.arange(self.n_units), self.n_total_time)
        time_ids_col = np.tile(np.arange(self.n_total_time), self.n_units)

        df_data = {
            "unit_id": unit_ids_col,
            "time_id": time_ids_col,
            "outcome": Y_it.ravel(),
            "is_treated_unit": np.repeat(D, self.n_total_time),
        }

        treatment_active_in_period = np.zeros((self.n_units, self.n_total_time), dtype=bool)
        for i in range(self.n_units):
            if D[i]:
                treatment_active_in_period[i, self.treatment_time :] = True
        df_data["treatment_in_period"] = treatment_active_in_period.ravel()

        for j in range(self.n_features):
            df_data[f"X{j + 1}"] = np.repeat(X[:, j], self.n_total_time)
        df = pd.DataFrame(df_data)

        true_att_values = actual_tau_it[D, self.treatment_time :]
        avg_true_att = np.mean(true_att_values) if true_att_values.size > 0 else np.nan

        return {
            "df": df,
            "X": X,
            "Y0_it": Y0_it,
            "actual_tau_it": actual_tau_it,
            "true_prop_scores": true_prop_scores,
            "D_treatment_assignment": D,
            "L_factor_loadings": L,
            "F_time_factors": F,
            "alpha_i_unit_effects": alpha_i.flatten(),
            "delta_t_time_effects": delta_t.flatten(),
            "beta_prop_propensity_coefs": _prop_score_coefs_actual,
            "beta_outcome_outcome_coefs": _outcome_coefs_actual,
            "att_hetero_coefs": _att_hetero_coefs_actual,
            "avg_true_att": avg_true_att,
            "n_treated_units_actual": np.sum(D),
            "n_control_units_actual": self.n_units - np.sum(D),
            "treatment_time": self.treatment_time,
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
