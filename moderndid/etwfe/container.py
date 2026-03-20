"""Result containers for ETWFE and EMFX output."""

from typing import NamedTuple

import numpy as np
from scipy import stats

from moderndid.core.maketables import (
    build_coef_table_with_ci,
    make_effect_names,
    make_group_time_names,
)


class EtwfeResult(NamedTuple):
    """Container for ETWFE regression output.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Returned by :func:`~moderndid.etwfe.estimator.etwfe`. Stores the saturated
    TWFE regression coefficients and variance-covariance matrix needed by
    :func:`~moderndid.etwfe.emfx.emfx` for aggregation.

    Attributes
    ----------
    coefficients : ndarray
        Coefficient estimates for each cohort x time interaction term.
    std_errors : ndarray
        Standard errors for each coefficient.
    vcov : ndarray
        Variance-covariance matrix of the interaction coefficients.
    coef_names : list[str]
        Names for each coefficient (from pyfixest).
    gt_pairs : list[tuple[float, float]]
        (group, time) pair for each coefficient.
    n_obs : int
        Number of observations used in estimation.
    n_units : int
        Number of unique cross-sectional units.
    r_squared : float or None
        R-squared of the regression.
    adj_r_squared : float or None
        Adjusted R-squared of the regression.
    data : object or None
        Preprocessed DataFrame (used by emfx for cell counts).
    config : object or None
        EtwfeConfig used for estimation.
    estimation_params : dict
        Additional estimation parameters.
    """

    #: Coefficient estimates for each cohort x time interaction term.
    coefficients: np.ndarray
    #: Standard errors for each coefficient.
    std_errors: np.ndarray
    #: Variance-covariance matrix of the interaction coefficients.
    vcov: np.ndarray
    #: Names for each coefficient from pyfixest.
    coef_names: list
    #: (group, time) pair for each coefficient.
    gt_pairs: list
    #: Number of observations used in estimation.
    n_obs: int
    #: Number of unique cross-sectional units.
    n_units: int
    #: R-squared of the regression.
    r_squared: float | None = None
    #: Adjusted R-squared of the regression.
    adj_r_squared: float | None = None
    #: Preprocessed DataFrame (used by emfx for aggregation).
    data: object = None
    #: EtwfeConfig used for estimation.
    config: object = None
    #: Estimation parameters (yname, cgroup, formula, etc.).
    estimation_params: dict = {}

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        alpha = float(self.estimation_params.get("alpha", 0.05))
        names = make_group_time_names(
            [g for g, _ in self.gt_pairs],
            [t for _, t in self.gt_pairs],
            prefix="ATT",
        )
        return build_coef_table_with_ci(names, self.coefficients, self.std_errors, alpha=alpha)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return self.n_obs
        if key == "n_units":
            return self.n_units
        if key == "R2":
            return self.r_squared
        if key == "R2_adj":
            return self.adj_r_squared
        if key == "se_type":
            return self.estimation_params.get("vcov_type", "CRV1")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "ETWFE"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """Return fixed-effects formula string."""
        return self.estimation_params.get("fe_spec")

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return {
            "vcov_type": self.estimation_params.get("vcov_type", "CRV1"),
            "clustervar": self.estimation_params.get("clustervar"),
        }

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {"n_units": "Units", "R2": "R-squared", "R2_adj": "Adj. R-squared"}

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["N", "n_units", "se_type"]
        if self.r_squared is not None:
            keys.append("R2")
        return keys


class EmfxResult(NamedTuple):
    """Container for aggregated ETWFE marginal effects.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.

    Returned by :func:`~moderndid.etwfe.emfx.emfx`.

    Attributes
    ----------
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error for the overall ATT.
    aggregation_type : str
        Type of aggregation: ``"simple"``, ``"group"``, ``"calendar"``,
        or ``"event"``.
    event_times : ndarray or None
        Event times, groups, or calendar times for non-simple aggregations.
    att_by_event : ndarray or None
        ATT estimates for each aggregation level.
    se_by_event : ndarray or None
        Standard errors for each aggregation level.
    ci_lower : ndarray or None
        Lower confidence interval bounds.
    ci_upper : ndarray or None
        Upper confidence interval bounds.
    critical_value : float
        Critical value used for confidence intervals.
    n_obs : int
        Number of observations in the original regression.
    estimation_params : dict
        Additional estimation parameters.
    """

    #: Overall average treatment effect on the treated.
    overall_att: float
    #: Standard error for the overall ATT.
    overall_se: float
    #: Type of aggregation: "simple", "group", "calendar", or "event".
    aggregation_type: str
    #: Event times, groups, or calendar times for non-simple aggregations.
    event_times: np.ndarray | None = None
    #: ATT estimates for each aggregation level.
    att_by_event: np.ndarray | None = None
    #: Standard errors for each aggregation level.
    se_by_event: np.ndarray | None = None
    #: Lower confidence interval bounds.
    ci_lower: np.ndarray | None = None
    #: Upper confidence interval bounds.
    ci_upper: np.ndarray | None = None
    #: Critical value used for confidence intervals.
    critical_value: float = 1.96
    #: Number of observations in the original regression.
    n_obs: int = 0
    #: Estimation parameters (alpha, vcov_type, etc.).
    estimation_params: dict = {}

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        alpha = float(self.estimation_params.get("alpha", 0.05))
        z_crit = stats.norm.ppf(1 - alpha / 2)

        names = ["Overall ATT"]
        estimates = [self.overall_att]
        se = [self.overall_se]

        if self.event_times is not None and self.att_by_event is not None and self.se_by_event is not None:
            prefix = {
                "event": "Event",
                "group": "Group",
                "calendar": "Time",
            }.get(self.aggregation_type, "Effect")
            names.extend(make_effect_names(self.event_times, prefix=prefix))
            estimates.extend(np.asarray(self.att_by_event, dtype=float).tolist())
            se.extend(np.asarray(self.se_by_event, dtype=float).tolist())

        crit = np.full(len(names), z_crit)
        return build_coef_table_with_ci(names, estimates, se, alpha=alpha, critical_values=crit)

    def __maketables_stat__(self, key: str) -> int | float | str | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return self.n_obs
        if key == "aggregation":
            return self.aggregation_type
        if key == "se_type":
            return self.estimation_params.get("vcov_type", "Delta method")
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.estimation_params.get("yname", "EMFX"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """EMFX output does not report fixed-effects formulas."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return {
            "vcov_type": "Delta method",
            "clustervar": self.estimation_params.get("clustervar"),
        }

    @property
    def __maketables_stat_labels__(self) -> dict[str, str]:
        """Return custom labels for model-level statistics."""
        return {"aggregation": "Aggregation"}

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        keys = ["N", "aggregation", "se_type"]
        return keys
