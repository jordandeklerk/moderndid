"""Result containers for intertemporal DiD estimators."""

from typing import NamedTuple

import numpy as np
import polars as pl

from moderndid.core.maketables import build_coef_table, se_type_label, vcov_info_from_bootstrap


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

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

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

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        names: list[str] = []
        estimates: list[float] = []
        se: list[float] = []
        ci95l: list[float] = []
        ci95u: list[float] = []

        if self.ate is not None:
            names.append("ATE")
            estimates.append(float(self.ate.estimate))
            se.append(float(self.ate.std_error))
            ci95l.append(float(self.ate.ci_lower))
            ci95u.append(float(self.ate.ci_upper))

        for horizon, estimate, std_error, lower, upper in zip(
            self.effects.horizons,
            self.effects.estimates,
            self.effects.std_errors,
            self.effects.ci_lower,
            self.effects.ci_upper,
            strict=False,
        ):
            names.append(f"Effect h={int(horizon)}")
            estimates.append(float(estimate))
            se.append(float(std_error))
            ci95l.append(float(lower))
            ci95u.append(float(upper))

        if self.placebos is not None:
            for horizon, estimate, std_error, lower, upper in zip(
                self.placebos.horizons,
                self.placebos.estimates,
                self.placebos.std_errors,
                self.placebos.ci_lower,
                self.placebos.ci_upper,
                strict=False,
            ):
                names.append(f"Placebo h={int(horizon)}")
                estimates.append(float(estimate))
                se.append(float(std_error))
                ci95l.append(float(lower))
                ci95u.append(float(upper))

        return build_coef_table(names, estimates, se, ci95l=ci95l, ci95u=ci95u)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            if self.n_units > 0:
                return int(self.n_units)
            if len(self.effects.n_observations) > 0:
                return int(np.nanmax(self.effects.n_observations))
            return None
        if key == "n_switchers":
            if self.n_switchers > 0:
                return int(self.n_switchers)
            if len(self.effects.n_switchers) > 0:
                return int(np.nanmax(self.effects.n_switchers))
            return None
        if key == "n_never_switchers":
            return int(self.n_never_switchers) if self.n_never_switchers > 0 else None
        if key == "se_type":
            cluster = self.estimation_params.get("cluster")
            return "Clustered" if cluster else se_type_label(False)
        if key == "placebo_joint_pvalue":
            if self.placebo_joint_test is None:
                return None
            return self.placebo_joint_test.get("p_value")
        if key == "effects_equal_pvalue":
            if self.effects_equal_test is None:
                return None
            return self.effects_equal_test.get("p_value")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "Intertemporal ATT"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Intertemporal DiD output does not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        cluster = self.estimation_params.get("cluster")
        return vcov_info_from_bootstrap(
            is_bootstrap=False,
            cluster=cluster,
            clustered_label="clustered",
        )

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "n_switchers": "Switchers",
            "n_never_switchers": "Never-switchers",
            "placebo_joint_pvalue": "Joint placebo p-value",
            "effects_equal_pvalue": "Equal effects p-value",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["N", "n_switchers", "n_never_switchers", "se_type"]
        if self.placebo_joint_test is not None:
            keys.append("placebo_joint_pvalue")
        if self.effects_equal_test is not None:
            keys.append("effects_equal_pvalue")
        return keys
