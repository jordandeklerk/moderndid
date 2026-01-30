"""Result containers."""

from typing import NamedTuple

import numpy as np
import polars as pl


class EffectsResult(NamedTuple):
    """Container for treatment effects at each horizon.

    Attributes
    ----------
    horizons : ndarray
        Event horizons (1, 2, ..., effects).
    estimates : ndarray
        Point estimates at each horizon.
    std_errors : ndarray
        Standard errors at each horizon.
    ci_lower : ndarray
        Lower confidence interval bounds.
    ci_upper : ndarray
        Upper confidence interval bounds.
    n_switchers : ndarray
        Number of switchers at each horizon.
    n_observations : ndarray
        Number of observations at each horizon.
    """

    horizons: np.ndarray
    estimates: np.ndarray
    std_errors: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_switchers: np.ndarray
    n_observations: np.ndarray

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        return pl.DataFrame(
            {
                "Horizon": self.horizons,
                "Estimate": self.estimates,
                "Std. Error": self.std_errors,
                "CI Lower": self.ci_lower,
                "CI Upper": self.ci_upper,
                "N Switchers": self.n_switchers,
                "N Obs": self.n_observations,
            }
        )


class PlacebosResult(NamedTuple):
    """Container for placebo effects at each pre-treatment horizon.

    Attributes
    ----------
    horizons : ndarray
        Pre-treatment horizons (-1, -2, ..., -placebo).
    estimates : ndarray
        Point estimates at each horizon.
    std_errors : ndarray
        Standard errors at each horizon.
    ci_lower : ndarray
        Lower confidence interval bounds.
    ci_upper : ndarray
        Upper confidence interval bounds.
    n_switchers : ndarray
        Number of switchers at each horizon.
    n_observations : ndarray
        Number of observations at each horizon.
    """

    horizons: np.ndarray
    estimates: np.ndarray
    std_errors: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_switchers: np.ndarray
    n_observations: np.ndarray

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        return pl.DataFrame(
            {
                "Horizon": self.horizons,
                "Estimate": self.estimates,
                "Std. Error": self.std_errors,
                "CI Lower": self.ci_lower,
                "CI Upper": self.ci_upper,
                "N Switchers": self.n_switchers,
                "N Obs": self.n_observations,
            }
        )


class ATEResult(NamedTuple):
    """Container for average total effect.

    Attributes
    ----------
    estimate : float
        Point estimate of the average total effect.
    std_error : float
        Standard error of the estimate.
    ci_lower : float
        Lower confidence interval bound.
    ci_upper : float
        Upper confidence interval bound.
    """

    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float


class HeterogeneityResult(NamedTuple):
    """Container for heterogeneous effects analysis.

    Attributes
    ----------
    horizon : int
        Effect horizon analyzed.
    covariates : list[str]
        Covariate names.
    estimates : np.ndarray
        Coefficient estimates for each covariate.
    std_errors : np.ndarray
        Standard errors for each coefficient.
    t_stats : np.ndarray
        T-statistics for each coefficient.
    ci_lower : np.ndarray
        Lower confidence interval bounds.
    ci_upper : np.ndarray
        Upper confidence interval bounds.
    n_obs : int
        Number of observations in the regression.
    f_pvalue : float
        P-value from joint F-test that all covariate coefficients are zero.
    """

    horizon: int
    covariates: list[str]
    estimates: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_obs: int
    f_pvalue: float

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        return pl.DataFrame(
            {
                "Horizon": [self.horizon] * len(self.covariates),
                "Covariate": self.covariates,
                "Estimate": self.estimates,
                "Std. Error": self.std_errors,
                "t-stat": self.t_stats,
                "CI Lower": self.ci_lower,
                "CI Upper": self.ci_upper,
                "N": [self.n_obs] * len(self.covariates),
                "F p-value": [self.f_pvalue] * len(self.covariates),
            }
        )


class DIDInterResult(NamedTuple):
    """Container for DIDInter estimation results.

    Attributes
    ----------
    effects : EffectsResult
        Treatment effects for each post-treatment horizon.
    placebos : PlacebosResult, optional
        Placebo effects for each pre-treatment horizon.
    ate : ATEResult, optional
        Average total effect across all horizons.
    n_units : int
        Total number of units in the sample.
    n_switchers : int
        Number of switchers in the sample.
    n_never_switchers : int
        Number of never-switchers in the sample.
    ci_level : float
        Confidence level used for intervals (e.g., 95.0).
    effects_equal_test : dict, optional
        Test for equality of effects across horizons.
    placebo_joint_test : dict, optional
        Joint test that all placebo effects are zero.
    influence_effects : ndarray, optional
        Influence function for effects.
    influence_placebos : ndarray, optional
        Influence function for placebos.
    heterogeneity : list[HeterogeneityResult], optional
        Heterogeneous effects analysis results for each horizon.
    estimation_params : dict
        Parameters used for estimation.
    """

    effects: EffectsResult
    placebos: PlacebosResult | None = None
    ate: ATEResult | None = None
    n_units: int = 0
    n_switchers: int = 0
    n_never_switchers: int = 0
    ci_level: float = 95.0
    effects_equal_test: dict | None = None
    placebo_joint_test: dict | None = None
    influence_effects: np.ndarray | None = None
    influence_placebos: np.ndarray | None = None
    heterogeneity: list | None = None
    estimation_params: dict = {}
