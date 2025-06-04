"""Base classes for bootstrap inference."""

import warnings
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn.linear_model import LogisticRegression

from .propensity_estimators import ipt_pscore


class PropensityScoreMethod(Enum):
    """Enumeration of propensity score estimation methods."""

    LOGISTIC = "logistic"
    IPT = "ipt"


class BaseBootstrap(ABC):
    """Abstract base class for bootstrap estimators."""

    def __init__(self, n_bootstrap: int = 1000, trim_level: float = 0.995, random_state: int | None = None):
        """Initialize bootstrap estimator.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations. Default is 1000.
        trim_level : float
            Maximum propensity score value for control units to avoid extreme weights.
            Default is 0.995.
        random_state : int, RandomState instance or None
            Controls the random number generation for reproducibility.
        """
        if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")
        if not isinstance(trim_level, int | float) or not 0 < trim_level <= 1:
            raise ValueError("trim_level must be a number between 0 (exclusive) and 1 (inclusive).")

        self.n_bootstrap = n_bootstrap
        self.trim_level = trim_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    @abstractmethod
    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Additional arguments specific to the estimator.

        Returns
        -------
        float
            The bootstrap estimate for this iteration.
        """
        pass  # pylint: disable=unnecessary-pass

    def _generate_weights(self, i_weights: np.ndarray) -> np.ndarray:
        """Generate bootstrap weights using exponential distribution.

        Parameters
        ----------
        i_weights : ndarray
            Original observation weights.

        Returns
        -------
        ndarray
            Bootstrap weights.
        """
        n_units = len(i_weights)
        v = self.rng.exponential(scale=1.0, size=n_units)
        return i_weights * v

    def _apply_trimming(self, ps: np.ndarray, d: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply trimming to control units based on propensity scores.

        Parameters
        ----------
        ps : ndarray
            Propensity scores.
        d : ndarray
            Treatment indicators.
        weights : ndarray
            Current weights.

        Returns
        -------
        ndarray
            Trimmed weights.
        """
        trim_ps_mask = np.ones_like(ps, dtype=bool)
        control_mask = d == 0
        trim_ps_mask[control_mask] = ps[control_mask] < self.trim_level

        weights_trimmed = weights.copy()
        weights_trimmed[~trim_ps_mask] = 0
        return weights_trimmed

    def _validate_results(self, estimates: np.ndarray) -> np.ndarray:
        """Check for failures and issue warnings.

        Parameters
        ----------
        bootstrap_estimates : ndarray
            Array of bootstrap estimates.

        Returns
        -------
        ndarray
            The same array, after issuing appropriate warnings.
        """
        n_failed = np.sum(np.isnan(estimates))
        if n_failed > 0:
            warnings.warn(
                f"{n_failed} out of {self.n_bootstrap} bootstrap iterations failed and resulted "
                "in NaN. "
                "This might be due to issues in propensity score estimation, outcome regression, "
                "or the AIPW calculation itself (e.g. perfect prediction, collinearity, "
                "small effective sample sizes after weighting/trimming).",
                UserWarning,
            )
        if n_failed > self.n_bootstrap * 0.1:
            warnings.warn(
                f"More than 10% ({n_failed}/{self.n_bootstrap}) of bootstrap iterations failed. "
                "Results may be unreliable.",
                UserWarning,
            )
        return estimates.copy()

    @staticmethod
    def _estimate_propensity_scores(
        d: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        method: PropensityScoreMethod = PropensityScoreMethod.LOGISTIC,
    ) -> np.ndarray:
        """Estimate propensity scores using specified method.

        Parameters
        ----------
        d : ndarray
            Treatment indicators.
        x : ndarray
            Covariate matrix.
        weights : ndarray
            Sample weights.
        method : PropensityScoreMethod
            Method to use for propensity score estimation.

        Returns
        -------
        ndarray
            Estimated propensity scores.
        """
        if method == PropensityScoreMethod.LOGISTIC:
            ps_model = LogisticRegression(solver="lbfgs", max_iter=10000)
            ps_model.fit(x, d, sample_weight=weights)
            return ps_model.predict_proba(x)[:, 1]

        if method == PropensityScoreMethod.IPT:
            return ipt_pscore(d, x, weights)

        raise ValueError(f"Unknown propensity score method: {method}")

    def _run_bootstrap_iterations(self, i_weights: np.ndarray, **kwargs) -> np.ndarray:
        """Run all bootstrap iterations.

        Parameters
        ----------
        i_weights : ndarray
            Original observation weights.
        **kwargs
            Additional arguments to pass to _compute_single_bootstrap.

        Returns
        -------
        ndarray
            Array of bootstrap estimates.
        """
        bootstrap_estimates = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            b_weights = self._generate_weights(i_weights)

            try:
                bootstrap_estimates[b] = self._compute_single_bootstrap(b_weights, **kwargs)
            except (ValueError, np.linalg.LinAlgError) as e:
                warnings.warn(f"Bootstrap iteration {b} failed: {e}", UserWarning)
                bootstrap_estimates[b] = np.nan
                continue

        return self._validate_results(bootstrap_estimates)

    @abstractmethod
    def fit(self, *args, **kwargs) -> np.ndarray:
        """Compute bootstrap estimates.

        Returns
        -------
        ndarray
            Array of bootstrap estimates.
        """
        pass  # pylint: disable=unnecessary-pass


class PanelBootstrap(BaseBootstrap):
    """Bootstrap estimator for panel data."""

    def fit(  # pylint: disable=arguments-differ
        self,
        delta_y: np.ndarray,
        d: np.ndarray,
        x: np.ndarray,
        i_weights: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute bootstrap estimates for panel data.

        Parameters
        ----------
        delta_y : ndarray
            Outcome difference (post - pre).
        d : ndarray
            Treatment indicators.
        x : ndarray
            Covariate matrix.
        i_weights : ndarray
            Initial weights.
        **kwargs
            Additional arguments for specific estimators.

        Returns
        -------
        ndarray
            Array of bootstrap estimates.
        """
        _validate_inputs(
            {"delta_y": delta_y, "d": d, "i_weights": i_weights},
            x,
            self.n_bootstrap,
            self.trim_level,
            check_intercept=True,
        )

        return self._run_bootstrap_iterations(
            i_weights,
            delta_y=delta_y,
            d=d,
            x=x,
            **kwargs,
        )


class RepeatedCrossSectionBootstrap(BaseBootstrap):
    """Bootstrap estimator for repeated cross-section data."""

    def fit(  # pylint: disable=arguments-differ
        self,
        y: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        x: np.ndarray,
        i_weights: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Compute bootstrap estimates for repeated cross-section data.

        Parameters
        ----------
        y : ndarray
            Outcome variable.
        t : ndarray
            Time period indicators (0 for pre, 1 for post).
        d : ndarray
            Treatment indicators.
        x : ndarray
            Covariate matrix.
        i_weights : ndarray
            Initial weights.
        **kwargs
            Additional arguments for specific estimators.

        Returns
        -------
        ndarray
            Array of bootstrap estimates.
        """
        _validate_inputs(
            {"y": y, "t": t, "d": d, "i_weights": i_weights},
            x,
            self.n_bootstrap,
            self.trim_level,
            check_intercept=True,
        )

        return self._run_bootstrap_iterations(
            i_weights,
            y=y,
            t=t,
            d=d,
            x=x,
            **kwargs,
        )


def _validate_inputs(arrays_dict, x, n_bootstrap, trim_level, check_intercept=False):
    """Validate inputs for bootstrap functions."""
    for name, arr in arrays_dict.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")

    if x is not None:
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a NumPy array if provided.")
        if x.ndim != 2:
            raise ValueError("x must be a 2-dimensional array if provided.")

    for name, arr in arrays_dict.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional.")

    first_array = next(iter(arrays_dict.values()))
    n_units = first_array.shape[0]

    for name, arr in arrays_dict.items():
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations.")

    if x is not None and x.shape[0] != n_units:
        raise ValueError("If provided, x must have the same number of observations as other arrays.")

    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")

    if not 0 < trim_level <= 1:
        raise ValueError("trim_level must be between 0 (exclusive) and 1 (inclusive).")

    if check_intercept and x is not None and not np.all(x[:, 0] == 1.0):
        warnings.warn(
            "The first column of the covariate matrix 'x' does not appear to be an intercept "
            "(all ones). "
            "IPT propensity score estimation typically requires an intercept.",
            UserWarning,
        )

    return n_units
