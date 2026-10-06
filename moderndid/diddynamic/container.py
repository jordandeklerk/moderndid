"""Result container for the dynamic covariate balancing estimator."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl

from moderndid.core.maketables import build_single_coef_table
from moderndid.core.result import extract_vcov_info


class DynBalancingResult(NamedTuple):
    r"""Container for dynamic covariate balancing treatment effect estimates.

    Stores point estimates, variances, and diagnostic information produced
    by the dynamic covariate balancing estimator.  The average treatment
    effect is defined as

    .. math::

        \text{ATE} = \mu_1 - \mu_2

    where :math:`\mu_1` and :math:`\mu_2` are the potential outcome
    estimates under the two treatment histories *ds1* and *ds2*.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables.  See :ref:`publication_tables`.

    Attributes
    ----------
    att : float
        The ATE point estimate (:math:`\mu_1 - \mu_2`).
    var_att : float
        Variance of the ATE.
    mu1 : float
        Potential outcome estimate under *ds1*.
    mu2 : float
        Potential outcome estimate under *ds2*.
    var_mu1 : float
        Variance of *mu1*.
    var_mu2 : float
        Variance of *mu2*.
    robust_quantile : float
        Chi-squared critical value of the ATE when the estimator ran with
        ``robust_quantile=True``, and the Gaussian critical value otherwise.
    gaussian_quantile : float
        Gaussian critical value for inference.
    gammas : dict
        Weights per treatment history (keys ``'ds1'``, ``'ds2'``). They are
        balancing weights for ``balancing='dcb'`` and inverse probability
        weights otherwise.
    coefficients : dict
        LASSO coefficients per treatment history. Empty unless
        ``balancing='dcb'``.
    imbalances : dict
        Standardized covariate imbalance of the weights per treatment history
        (keys ``'ds1'``, ``'ds2'``). Each value is a DataFrame with columns
        ``period``, ``covariate``, and ``imbalance``. The imbalance compares
        the weighted covariate mean of a period with that of the period
        before, or with the plain mean in the first period, in units of the
        covariate's standard deviation.
    estimation_params : dict
        Standard moderndid metadata (observation count, variable names, etc.).
    """

    #: The ATE point estimate.
    att: float
    #: Variance of the ATE.
    var_att: float
    #: Potential outcome estimate under ds1.
    mu1: float
    #: Potential outcome estimate under ds2.
    mu2: float
    #: Variance of mu1.
    var_mu1: float
    #: Variance of mu2.
    var_mu2: float
    #: Chi-squared critical value, or the Gaussian one when robust quantiles are off.
    robust_quantile: float
    #: Gaussian critical value.
    gaussian_quantile: float
    #: Weights per treatment history.
    gammas: dict
    #: LASSO coefficients per treatment history (DCB only).
    coefficients: dict
    #: Standardized covariate imbalance per treatment history.
    imbalances: dict
    #: Standard moderndid metadata.
    estimation_params: dict = {}

    @property
    def se(self) -> float:
        """Standard error of the ATE estimate."""
        return math.sqrt(self.var_att)

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATE", self.att, self.se)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            n_obs = self.estimation_params.get("n_obs")
            if n_obs is not None:
                return int(n_obs)
            return None
        if key == "se_type":
            return "Analytical"
        if key == "balancing":
            raw = self.estimation_params.get("balancing")
            if raw is None:
                return None
            return raw.upper().replace("_", "-")
        if key == "method":
            return self.estimation_params.get("method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return self.estimation_params.get("yname", "")

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Dynamic balancing results do not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return extract_vcov_info(self.estimation_params)

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {
            "balancing": "Balancing",
            "method": "Method",
        }

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type", "balancing"]


class DynBalancingHistoryResult(NamedTuple):
    r"""Container for treatment effects estimated over varying history lengths.

    Stores a summary table of ATEs, variances, and critical values for
    each treatment history length, plus the individual
    :class:`DynBalancingResult` objects for per-lag diagnostics.

    Attributes
    ----------
    summary : polars.DataFrame
        One row per history length with columns ``period_length``,
        ``att``, ``var_att``, ``mu1``, ``var_mu1``, ``mu2``,
        ``var_mu2``, ``robust_quantile``, ``gaussian_quantile``.
    results : list[DynBalancingResult]
        Individual estimation results, ordered by ascending history length.
    """

    summary: pl.DataFrame
    results: list


class DynBalancingHetResult(NamedTuple):
    r"""Container for treatment effects estimated across different final periods.

    Stores a summary table of ATEs, variances, and critical values for
    each final period, plus the individual :class:`DynBalancingResult`
    objects for per-period diagnostics.

    Attributes
    ----------
    summary : polars.DataFrame
        One row per final period with columns ``final_period``,
        ``att``, ``var_att``, ``mu1``, ``var_mu1``, ``mu2``,
        ``var_mu2``, ``robust_quantile``, ``gaussian_quantile``.
    results : list[DynBalancingResult]
        Individual estimation results, ordered by ascending final period.
    """

    summary: pl.DataFrame
    results: list


class IPWResult(NamedTuple):
    """Result of inverse probability weighting for one treatment history.

    Attributes
    ----------
    mu_hat : float
        Estimated potential outcome under the target treatment history.
    variance : float
        Estimated variance of ``mu_hat``.
    gammas : ndarray
        Weight matrix of shape ``(n, T)``. Column ``t`` holds the normalized
        inverse probability weights of the units that follow the target
        history through period ``t``.
    predictions : ndarray
        Matrix of shape ``(n, T)`` with the outcome projection of each period
        that enters the estimate. Without an outcome model every entry equals
        ``mu_hat``.
    not_nas : list[ndarray]
        Row indices that enter each period.
    """

    mu_hat: float
    variance: float
    gammas: np.ndarray
    predictions: np.ndarray
    not_nas: list


class DCBResult(NamedTuple):
    """Result of DCB weight estimation.

    Attributes
    ----------
    mu_hat : float
        Estimated potential outcome under target treatment history.
    gammas : np.ndarray
        Weight matrix of shape ``(n, T)`` with per-period balancing weights.
    predictions : np.ndarray
        Prediction matrix of shape ``(n, T)`` from the coefficient stage.
    not_nas : list[np.ndarray]
        Valid row indices per period.
    coef_t : list[np.ndarray]
        Coefficient vectors per period.
    bias : float
        Debiasing correction, ``nan`` if debiasing was not requested.
    """

    mu_hat: float
    gammas: np.ndarray
    predictions: np.ndarray
    not_nas: list[np.ndarray]
    coef_t: list[np.ndarray]
    bias: float


class CoefficientResult(NamedTuple):
    """Per-period coefficient estimates and predictions.

    Attributes
    ----------
    coef_t : list[ndarray]
        Coefficient vectors per period, each with shape ``(1 + p,)``
        where the first element is the intercept.
    pred_t : list[ndarray]
        Prediction vectors per period on the clean covariate matrix.
    covariates_nonna : list[ndarray]
        Covariate matrices per period with NaN rows removed.
    not_nas : list[ndarray]
        Integer arrays of valid row indices per period.
    model_effect : list[float]
        Last treatment coefficient per period. Empty for ``lasso_subsample``.
    """

    coef_t: list[np.ndarray]
    pred_t: list[np.ndarray]
    covariates_nonna: list[np.ndarray]
    not_nas: list[np.ndarray]
    model_effect: list[float]


class QuantileResult(NamedTuple):
    """Critical values for confidence interval construction.

    Attributes
    ----------
    robust_quantile_ate : float
        Chi-squared-based critical value for ATE inference.
    gaussian_quantile_ate : float
        Gaussian critical value for ATE inference.
    robust_quantile_mu : float
        Chi-squared-based critical value for potential outcome inference.
    gaussian_quantile_mu : float
        Gaussian critical value for potential outcome inference.
    """

    robust_quantile_ate: float
    gaussian_quantile_ate: float
    robust_quantile_mu: float
    gaussian_quantile_mu: float
