"""Result container for the dynamic covariate balancing estimator."""

from __future__ import annotations

import math
from typing import NamedTuple

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
        Robust chi-squared critical value for inference.
    gaussian_quantile : float
        Gaussian critical value for inference.
    gammas : dict
        Balancing weights per treatment history (keys ``'ds1'``, ``'ds2'``).
    coefficients : dict
        LASSO coefficients per treatment history.
    imbalances : dict
        Covariate imbalance measures.
    estimation_params : dict
        Standard moderndid metadata (observation count, variable names, etc.).

    References
    ----------
    .. [1] Viviano, D. and Bradic, J. (2026). "Dynamic covariate balancing:
       estimating treatment effects over time with potential local projections."
       *Biometrika*, asag016. https://doi.org/10.1093/biomet/asag016
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
    #: Robust chi-squared critical value.
    robust_quantile: float
    #: Gaussian critical value.
    gaussian_quantile: float
    #: Balancing weights per treatment history.
    gammas: dict
    #: LASSO coefficients per treatment history.
    coefficients: dict
    #: Covariate imbalance measures.
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
