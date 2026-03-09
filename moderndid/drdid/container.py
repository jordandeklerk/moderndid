"""Result containers for doubly robust DiD estimators."""

from typing import Any, NamedTuple

import numpy as np

from moderndid.core.maketables import (
    build_single_coef_table,
    n_from_shape,
    se_type_label,
    vcov_info_from_bootstrap,
)

from .format import print_did_result


class DRDIDResult(NamedTuple):
    """Result from the doubly robust DiD estimator.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.
    """

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    call_params: dict[str, Any]
    args: dict[str, Any]

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATT", self.att, self.se, ci95l=self.lci, ci95u=self.uci)

    def __maketables_stat__(self, key: str) -> int | str | float | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return n_from_shape(self.call_params.get("data_shape"))
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.call_params.get("yname", "Outcome"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """DR-DiD wrappers do not estimate fixed effects models."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return vcov_info_from_bootstrap(is_bootstrap=bool(self.args.get("boot", False)))

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type"]


DRDIDResult = print_did_result(DRDIDResult)


class IPWDIDResult(NamedTuple):
    """Result from the inverse propensity weighted DiD estimator.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.
    """

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    call_params: dict[str, Any]
    args: dict[str, Any]

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATT", self.att, self.se, ci95l=self.lci, ci95u=self.uci)

    def __maketables_stat__(self, key: str) -> int | str | float | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return n_from_shape(self.call_params.get("data_shape"))
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.call_params.get("yname", "Outcome"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """IPW wrappers do not estimate fixed effects models."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return vcov_info_from_bootstrap(is_bootstrap=bool(self.args.get("boot", False)))

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type"]


IPWDIDResult = print_did_result(IPWDIDResult)


class ORDIDResult(NamedTuple):
    """Result from the outcome regression DiD estimator.

    This class implements the ``maketables`` plug-in interface for
    publication-quality tables. See :ref:`publication_tables`.
    """

    att: float
    se: float
    uci: float
    lci: float
    boots: np.ndarray | None
    att_inf_func: np.ndarray | None
    call_params: dict[str, Any]
    args: dict[str, Any]

    @property
    def __maketables_coef_table__(self):
        """Return canonical coefficient table for maketables."""
        return build_single_coef_table("ATT", self.att, self.se, ci95l=self.lci, ci95u=self.uci)

    def __maketables_stat__(self, key: str) -> int | str | float | None:
        """Return model-level statistics for maketables."""
        if key == "N":
            return n_from_shape(self.call_params.get("data_shape"))
        if key == "se_type":
            return se_type_label(bool(self.args.get("boot", False)))
        return None

    @property
    def __maketables_depvar__(self) -> str:
        """Return dependent variable label for maketables."""
        return str(self.call_params.get("yname", "Outcome"))

    @property
    def __maketables_fixef_string__(self) -> str | None:
        """OR wrappers do not estimate fixed effects models."""
        return None

    @property
    def __maketables_vcov_info__(self) -> dict[str, str | None]:
        """Return variance-covariance metadata."""
        return vcov_info_from_bootstrap(is_bootstrap=bool(self.args.get("boot", False)))

    @property
    def __maketables_default_stat_keys__(self) -> list[str]:
        """Default model-level stats to display in ETable."""
        return ["N", "se_type"]


ORDIDResult = print_did_result(ORDIDResult)
