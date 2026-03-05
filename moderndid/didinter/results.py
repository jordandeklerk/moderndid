"""Result containers."""

from typing import NamedTuple

import polars as pl

import numpy as np


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

    #: Event horizons (1, 2, ..., effects).
    horizons: np.ndarray
    #: Point estimates at each horizon.
    estimates: np.ndarray
    #: Standard errors at each horizon.
    std_errors: np.ndarray
    #: Lower confidence interval bounds.
    ci_lower: np.ndarray
    #: Upper confidence interval bounds.
    ci_upper: np.ndarray
    #: Number of switchers at each horizon.
    n_switchers: np.ndarray
    #: Number of observations at each horizon.
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

    #: Pre-treatment horizons (-1, -2, ..., -placebo).
    horizons: np.ndarray
    #: Point estimates at each horizon.
    estimates: np.ndarray
    #: Standard errors at each horizon.
    std_errors: np.ndarray
    #: Lower confidence interval bounds.
    ci_lower: np.ndarray
    #: Upper confidence interval bounds.
    ci_upper: np.ndarray
    #: Number of switchers at each horizon.
    n_switchers: np.ndarray
    #: Number of observations at each horizon.
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
    n_observations : float
        Total observations contributing to the ATE.
    n_switchers : float
        Total switchers contributing to the ATE.
    """

    #: Point estimate of the average total effect.
    estimate: float
    #: Standard error of the estimate.
    std_error: float
    #: Lower confidence interval bound.
    ci_lower: float
    #: Upper confidence interval bound.
    ci_upper: float
    #: Total observations contributing to the ATE.
    n_observations: float = 0.0
    #: Total switchers contributing to the ATE.
    n_switchers: float = 0.0


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

    #: Effect horizon analyzed.
    horizon: int
    #: Covariate names.
    covariates: list[str]
    #: Coefficient estimates for each covariate.
    estimates: np.ndarray
    #: Standard errors for each coefficient.
    std_errors: np.ndarray
    #: T-statistics for each coefficient.
    t_stats: np.ndarray
    #: Lower confidence interval bounds.
    ci_lower: np.ndarray
    #: Upper confidence interval bounds.
    ci_upper: np.ndarray
    #: Number of observations in the regression.
    n_obs: int
    #: P-value from joint F-test that all covariate coefficients are zero.
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
    vcov_warnings : list
        Variance-covariance warnings.
    """

    #: Treatment effects for each post-treatment horizon.
    effects: EffectsResult
    #: Placebo effects for each pre-treatment horizon.
    placebos: PlacebosResult | None = None
    #: Average total effect across all horizons.
    ate: ATEResult | None = None
    #: Total number of units in the sample.
    n_units: int = 0
    #: Number of switchers in the sample.
    n_switchers: int = 0
    #: Number of never-switchers in the sample.
    n_never_switchers: int = 0
    #: Confidence level used for intervals.
    ci_level: float = 95.0
    #: Test for equality of effects across horizons.
    effects_equal_test: dict | None = None
    #: Joint test that all placebo effects are zero.
    placebo_joint_test: dict | None = None
    #: Influence function for effects.
    influence_effects: np.ndarray | None = None
    #: Influence function for placebos.
    influence_placebos: np.ndarray | None = None
    #: Heterogeneous effects analysis results.
    heterogeneity: list | None = None
    #: Parameters used for estimation.
    estimation_params: dict = {}
    #: Variance-covariance warnings.
    vcov_warnings: list = []
