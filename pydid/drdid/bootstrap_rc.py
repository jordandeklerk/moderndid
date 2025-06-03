"""Repeated cross-section bootstrap estimator classes using logistic regression."""

import numpy as np

from .base_bootstrap import PropensityScoreMethod, RepeatedCrossSectionBootstrap
from .propensity_estimators import aipw_did_rc_imp1, aipw_did_rc_imp2
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
