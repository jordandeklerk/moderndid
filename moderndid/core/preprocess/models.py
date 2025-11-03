"""Data models for preprocessed data containers."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from moderndid.utils import extract_vars_from_formula

from .config import BasePreprocessConfig, ContDIDConfig, DIDConfig
from .constants import DataFormat


@dataclass
class PreprocessedData:
    """Preprocessed data."""

    data: pd.DataFrame
    time_invariant_data: pd.DataFrame
    weights: np.ndarray

    cohort_counts: pd.DataFrame
    period_counts: pd.DataFrame
    crosstable_counts: pd.DataFrame

    config: BasePreprocessConfig
    cluster: np.ndarray | None = None


@dataclass
class DIDData(PreprocessedData):
    """DID data."""

    outcomes_tensor: list[np.ndarray] | None = None
    covariates_matrix: np.ndarray | None = None
    covariates_tensor: list[np.ndarray] | None = None

    config: DIDConfig = field(default_factory=DIDConfig)

    @property
    def is_panel(self) -> bool:
        """Check if data is panel."""
        return self.config.data_format == DataFormat.PANEL

    @property
    def is_balanced_panel(self) -> bool:
        """Check if data is balanced panel."""
        return self.is_panel and self.outcomes_tensor is not None

    @property
    def has_covariates(self) -> bool:
        """Check if data has covariates."""
        return self.covariates_matrix is not None or self.covariates_tensor is not None

    def get_covariate_names(self) -> list[str]:
        """Get covariate names."""
        if self.config.xformla == "~1" or self.config.xformla is None:
            return []
        vars_list = extract_vars_from_formula(self.config.xformla)
        return [v for v in vars_list if v != self.config.yname]


@dataclass
class ContDIDData(PreprocessedData):
    """ContDID data."""

    time_map: dict = field(default_factory=dict)
    original_time_periods: np.ndarray = field(default_factory=lambda: np.array([]))

    config: ContDIDConfig = field(default_factory=ContDIDConfig)

    @property
    def is_panel(self) -> bool:
        """Check if data is panel."""
        return self.config.panel

    @property
    def has_dose(self) -> bool:
        """Check if data has dose."""
        return self.config.dname is not None

    @property
    def has_covariates(self) -> bool:
        """Check if data has covariates."""
        return self.config.xformla != "~1" and self.config.xformla is not None

    def get_covariate_names(self) -> list[str]:
        """Get covariate names."""
        if not self.has_covariates:
            return []
        vars_list = extract_vars_from_formula(self.config.xformla)
        return [v for v in vars_list if v != self.config.yname]

    def map_time_to_original(self, time_idx: int | np.ndarray) -> int | np.ndarray:
        """Map time to original."""
        if self.time_map:
            reverse_map = {v: k for k, v in self.time_map.items()}
            if isinstance(time_idx, np.ndarray):
                return np.array([reverse_map[t] for t in time_idx])
            return reverse_map[time_idx]
        return time_idx


@dataclass
class ValidationResult:
    """Validation result."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise if invalid."""
        if not self.is_valid:
            error_msg = "\n".join(self.errors)
            raise ValueError(f"Validation failed:\n{error_msg}")

    def _warnings(self) -> None:
        """Warnings."""
        import warnings

        for warning in self.warnings:
            warnings.warn(warning, UserWarning, stacklevel=2)
