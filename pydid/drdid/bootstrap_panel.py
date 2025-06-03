"""Panel data bootstrap estimator classes."""

import warnings

import numpy as np

from .base_bootstrap import PanelBootstrap, PropensityScoreMethod
from .propensity_estimators import aipw_did_panel, std_ipw_panel, twfe_panel
from .wols import wols_panel


class ImprovedDRDiDPanel(PanelBootstrap):
    """Improved doubly robust DiD estimator for panel data using IPT propensity scores."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using IPT propensity scores.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Contains delta_y, d, x from parent fit method.

        Returns
        -------
        float
            Bootstrap estimate for this iteration.
        """
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.IPT)

        if np.any(ps_b[d == 0] == 1.0):
            warnings.warn(
                "Propensity score is 1 for some control units, cannot compute IPW.",
                UserWarning,
            )
            return np.nan

        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        # Compute outcome regression predictions
        wols_result = wols_panel(delta_y, d, x, ps_b, b_weights_trimmed)
        out_reg = wols_result.out_reg

        return aipw_did_panel(delta_y, d, ps_b, out_reg, b_weights_trimmed)


class IPWPanel(PanelBootstrap):
    """IPW-only estimator for panel data."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using IPW only."""
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.IPT)

        if np.any(ps_b[d == 0] == 1.0):
            warnings.warn(
                "Propensity score is 1 for some control units, cannot compute IPW.",
                UserWarning,
            )
            return np.nan

        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        w_treat = b_weights_trimmed[d == 1]
        n1 = np.sum(w_treat)

        w_cont = b_weights_trimmed[d == 0] * ps_b[d == 0] / (1 - ps_b[d == 0])
        n0 = np.sum(w_cont)

        if n1 == 0 or n0 == 0:
            return np.nan

        att_treat = np.sum(w_treat * delta_y[d == 1]) / n1
        att_cont = np.sum(w_cont * delta_y[d == 0]) / n0

        return att_treat - att_cont


class StandardizedIPWPanel(PanelBootstrap):
    """Standardized IPW estimator for panel data."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using standardized IPW."""
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.IPT)

        if np.any(ps_b[d == 0] == 1.0):
            warnings.warn(
                "Propensity score is 1 for some control units, cannot compute IPW.",
                UserWarning,
            )
            return np.nan

        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        return std_ipw_panel(delta_y, d, ps_b, b_weights_trimmed)


class TraditionalDRDiDPanel(PanelBootstrap):
    """Traditional doubly robust DiD estimator using logistic regression."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using traditional DR-DiD."""
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.LOGISTIC)

        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        # Compute outcome regression predictions
        wols_result = wols_panel(delta_y, d, x, ps_b, b_weights_trimmed)
        out_reg = wols_result.out_reg

        return aipw_did_panel(delta_y, d, ps_b, out_reg, b_weights_trimmed)


class RegressionPanel(PanelBootstrap):
    """Regression-only estimator for panel data (no propensity scores)."""

    def __init__(
        self,
        n_bootstrap: int = 1000,
        random_state: int | None = None,
    ):
        """Initialize regression panel estimator.

        Note: trim_level is set to 1.0 since no propensity scores are used.
        """
        super().__init__(n_bootstrap, trim_level=1.0, random_state=random_state)

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using regression only."""
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        try:
            X = np.column_stack([x, d])
            W = np.diag(b_weights)

            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ delta_y

            if np.linalg.cond(XtWX) > 1e12:
                return np.nan

            beta = np.linalg.solve(XtWX, XtWy)

            return float(beta[-1])

        except (np.linalg.LinAlgError, ValueError):
            return np.nan


class TWFEPanel(PanelBootstrap):
    """Two-way fixed effects estimator for panel data."""

    def __init__(
        self,
        n_bootstrap: int = 1000,
        random_state: int | None = None,
    ):
        """Initialize TWFE panel estimator.

        Note: trim_level is set to 1.0 since no propensity scores are used.
        """
        super().__init__(n_bootstrap, trim_level=1.0, random_state=random_state)

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using TWFE."""
        delta_y = kwargs["delta_y"]
        d = kwargs["d"]
        x = kwargs["x"]

        return twfe_panel(delta_y, d, x, b_weights)
