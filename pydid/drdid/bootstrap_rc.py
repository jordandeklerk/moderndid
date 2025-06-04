"""Repeated cross-section bootstrap estimator classes using logistic regression."""

import numpy as np

from .base_bootstrap import PropensityScoreMethod, RepeatedCrossSectionBootstrap
from .propensity_estimators import aipw_did_rc_imp1, aipw_did_rc_imp2, ipw_did_rc
from .wols import wols_rc


class ImprovedDRDiDRC1(RepeatedCrossSectionBootstrap):
    """Improved DR-DiD for repeated cross-sections with control-only outcome regression."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Contains y, t, d, x from parent fit method.

        Returns
        -------
        float
            Bootstrap estimate for this iteration.
        """
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.LOGISTIC)
        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        or_control_pre_b = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=True, treat=False).out_reg
        or_control_post_b = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=False, treat=False).out_reg

        or_reg_b = np.zeros_like(y)
        or_reg_b[(d == 0) & (t == 0)] = or_control_pre_b[(d == 0) & (t == 0)]
        or_reg_b[(d == 0) & (t == 1)] = or_control_post_b[(d == 0) & (t == 1)]
        or_reg_b[(d == 1) & (t == 0)] = or_control_pre_b[(d == 1) & (t == 0)]
        or_reg_b[(d == 1) & (t == 1)] = or_control_post_b[(d == 1) & (t == 1)]

        return aipw_did_rc_imp1(y, t, d, ps_b, or_reg_b, b_weights_trimmed)


class ImprovedDRDiDRC2(RepeatedCrossSectionBootstrap):
    """Locally efficient DR-DiD for repeated cross-sections."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration."""
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.LOGISTIC)
        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        out_y_treat_post = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=False, treat=True).out_reg
        out_y_treat_pre = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=True, treat=True).out_reg
        out_y_cont_post = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=False, treat=False).out_reg
        out_y_cont_pre = wols_rc(y, t, d, x, ps_b, b_weights_trimmed, pre=True, treat=False).out_reg

        return aipw_did_rc_imp2(
            y, t, d, ps_b, out_y_treat_post, out_y_treat_pre, out_y_cont_post, out_y_cont_pre, b_weights_trimmed
        )


class TraditionalDRDiDRC(RepeatedCrossSectionBootstrap):
    """Traditional DR-DiD for repeated cross-sections."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration."""
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.LOGISTIC)
        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        control_mean = np.mean(y[d == 0]) if np.sum(d == 0) > 0 else 0.0
        or_reg_simple = np.full_like(y, control_mean)

        return aipw_did_rc_imp1(y, t, d, ps_b, or_reg_simple, b_weights_trimmed)


class IPWRepeatedCrossSection(RepeatedCrossSectionBootstrap):
    """IPW-only estimator for repeated cross-sections."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using IPW.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Contains y, t, d, x from parent fit method.

        Returns
        -------
        float
            Bootstrap estimate for this iteration.
        """
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        ps_b = self._estimate_propensity_scores(d, x, b_weights, PropensityScoreMethod.LOGISTIC)
        ps_b = np.clip(ps_b, 1e-6, 1 - 1e-6)

        b_weights_trimmed = self._apply_trimming(ps_b, d, b_weights)

        return ipw_did_rc(y, t, d, ps_b, b_weights_trimmed)


class RegressionDiDRC(RepeatedCrossSectionBootstrap):
    """Regression-based robust DiD for repeated cross-sections."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using regression adjustment.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Contains y, t, d, x from parent fit method.

        Returns
        -------
        float
            Bootstrap estimate for this iteration.
        """
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        reg_control_pre = wols_rc(y, t, d, x, np.full_like(y, 0.5, dtype=float), b_weights, pre=True, treat=False)
        reg_control_post = wols_rc(y, t, d, x, np.full_like(y, 0.5, dtype=float), b_weights, pre=False, treat=False)

        out_reg_pre = reg_control_pre.out_reg
        out_reg_post = reg_control_post.out_reg

        # Compute OR estimator with regression adjustment
        treated_post = (d == 1) & (t == 1)
        treated_pre = (d == 1) & (t == 0)

        att_b = np.sum(b_weights[treated_post] * y[treated_post]) / np.sum(b_weights[treated_post])
        att_b -= np.sum(b_weights[treated_pre] * y[treated_pre]) / np.sum(b_weights[treated_pre])

        treated = d == 1
        att_b -= np.sum(b_weights[treated] * (out_reg_post[treated] - out_reg_pre[treated])) / np.sum(
            b_weights[treated]
        )

        return att_b


class TWFERepeatedCrossSection(RepeatedCrossSectionBootstrap):
    """Two-Way Fixed Effects (TWFE) DiD estimator for repeated cross-sections."""

    def _compute_single_bootstrap(self, b_weights: np.ndarray, **kwargs) -> float:
        """Compute a single bootstrap iteration using TWFE regression.

        Parameters
        ----------
        b_weights : ndarray
            Bootstrap weights for this iteration.
        **kwargs
            Contains y, t, d, x from parent fit method.

        Returns
        -------
        float
            Bootstrap estimate for this iteration.
        """
        y = kwargs["y"]
        t = kwargs["t"]
        d = kwargs["d"]
        x = kwargs["x"]

        # If there are no treated units, the treatment effect is 0.
        if np.sum(d) == 0:
            return 0.0

        n_obs = len(y)
        intercept = np.ones((n_obs, 1))
        post = t.reshape(-1, 1)
        treat = d.reshape(-1, 1)
        post_treat = (t * d).reshape(-1, 1)

        if x is not None:
            if np.all(x[:, 0] == 1.0):
                x_no_intercept = x[:, 1:]
            else:
                x_no_intercept = x
            design_matrix = np.hstack([intercept, post, treat, post_treat, x_no_intercept])
        else:
            design_matrix = np.hstack([intercept, post, treat, post_treat])

        sqrt_weights = np.sqrt(b_weights)
        y_weighted = y * sqrt_weights
        x_weighted = design_matrix * sqrt_weights[:, np.newaxis]

        try:
            xtx = x_weighted.T @ x_weighted
            xty = x_weighted.T @ y_weighted
            coefficients = np.linalg.solve(xtx, xty)  # noqa: N806
            return coefficients[3]
        except np.linalg.LinAlgError:
            return np.nan
