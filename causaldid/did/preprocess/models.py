"""Data models for DiD preprocessing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from causaldid.utils import extract_vars_from_formula

from .constants import (
    BasePeriod,
    ControlGroup,
    DataFormat,
    EstimationMethod,
)


@dataclass
class DIDConfig:
    """Configuration for DiD analysis."""

    yname: str
    tname: str
    gname: str

    idname: str | None = None
    xformla: str = "~1"
    panel: bool = True
    allow_unbalanced_panel: bool = True
    control_group: ControlGroup = ControlGroup.NEVER_TREATED
    anticipation: int = 0
    weightsname: str | None = None
    alp: float = 0.05
    bstrap: bool = False
    cband: bool = False
    biters: int = 1000
    clustervars: list[str] = field(default_factory=list)
    est_method: EstimationMethod = EstimationMethod.DOUBLY_ROBUST
    base_period: BasePeriod = BasePeriod.VARYING
    print_details: bool = True
    faster_mode: bool = False
    pl: bool = False
    cores: int = 1

    true_repeated_cross_sections: bool = False
    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    treated_groups: np.ndarray = field(default_factory=lambda: np.array([]))
    treated_groups_count: int = 0
    id_count: int = 0
    data_format: DataFormat = DataFormat.PANEL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {k: v.value if isinstance(v, Enum) else v for k, v in self.__dict__.items()}


@dataclass
class DIDData:
    """Container for preprocessed DiD data."""

    data: pd.DataFrame
    time_invariant_data: pd.DataFrame
    weights: np.ndarray

    cohort_counts: pd.DataFrame
    period_counts: pd.DataFrame
    crosstable_counts: pd.DataFrame

    outcomes_tensor: list[np.ndarray] | None = None
    covariates_matrix: np.ndarray | None = None
    covariates_tensor: list[np.ndarray] | None = None
    cluster: np.ndarray | None = None

    config: DIDConfig = field(default_factory=DIDConfig)

    @property
    def is_panel(self) -> bool:
        """Check if data is in panel format."""
        return self.config.data_format == DataFormat.PANEL

    @property
    def is_balanced_panel(self) -> bool:
        """Check if panel is balanced."""
        return self.is_panel and self.outcomes_tensor is not None

    @property
    def has_covariates(self) -> bool:
        """Check if covariates are included."""
        return self.covariates_matrix is not None or self.covariates_tensor is not None

    def get_covariate_names(self) -> list[str]:
        """Extract covariate names from formula."""
        if self.config.xformla == "~1" or self.config.xformla is None:
            return []

        vars = extract_vars_from_formula(self.config.xformla)
        return [v for v in vars if v != self.config.yname]


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            error_msg = "\n".join(self.errors)
            raise ValueError(f"Validation failed:\n{error_msg}")

    def print_warnings(self) -> None:
        """Print all warnings."""
        import warnings

        for warning in self.warnings:
            warnings.warn(warning, UserWarning, stacklevel=2)
