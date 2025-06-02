"""Base classes for bootstrap inference."""

import warnings
from abc import ABC, abstractmethod

import numpy as np


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
                f"{n_failed} out of {self.n_bootstrap} bootstrap iterations failed and resulted in NaN. "
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

    @abstractmethod
    def fit(self, **kwargs) -> np.ndarray:
        """Compute bootstrap estimates.

        Returns
        -------
        ndarray
            Array of bootstrap estimates.
        """
        pass  # pylint: disable=unnecessary-pass


def _validate_inputs(arrays_dict, x, n_bootstrap, trim_level, check_intercept=False):
    """Validate inputs for bootstrap functions."""
    for name, arr in arrays_dict.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")

    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array.")

    for name, arr in arrays_dict.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional.")

    if x.ndim != 2:
        raise ValueError("x must be a 2-dimensional array.")

    first_array = next(iter(arrays_dict.values()))
    n_units = first_array.shape[0]

    for name, arr in arrays_dict.items():
        if arr.shape[0] != n_units:
            raise ValueError("All arrays must have the same number of observations.")

    if x.shape[0] != n_units:
        raise ValueError("All arrays must have the same number of observations.")

    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")

    if not 0 < trim_level < 1:
        raise ValueError("trim_level must be between 0 and 1.")

    if check_intercept and not np.all(x[:, 0] == 1.0):
        warnings.warn(
            "The first column of the covariate matrix 'x' does not appear to be an intercept (all ones). "
            "IPT propensity score estimation typically requires an intercept.",
            UserWarning,
        )

    return n_units
