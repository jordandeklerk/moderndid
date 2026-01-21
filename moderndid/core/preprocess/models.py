"""Data models for preprocessed data containers."""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config import (
    BasePreprocessConfig,
    ContDIDConfig,
    DDDConfig,
    DIDConfig,
    TwoPeriodDIDConfig,
)
from .constants import DataFormat
from .utils import extract_vars_from_formula


@dataclass
class PreprocessedData:
    """Preprocessed data."""

    data: pl.DataFrame
    time_invariant_data: pl.DataFrame
    weights: np.ndarray

    cohort_counts: pl.DataFrame
    period_counts: pl.DataFrame
    crosstable_counts: pl.DataFrame

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
class TwoPeriodDIDData:
    """Two-period DiD data."""

    y1: np.ndarray | None = None
    y0: np.ndarray | None = None
    y: np.ndarray | None = None
    post: np.ndarray | None = None
    D: np.ndarray = field(default_factory=lambda: np.array([]))
    covariates: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    covariate_names: list[str] = field(default_factory=list)
    n_units: int = 0
    n_obs: int = 0
    config: TwoPeriodDIDConfig | None = None

    @property
    def is_panel(self) -> bool:
        """Check if data is panel."""
        return self.y1 is not None and self.y0 is not None


@dataclass
class DDDData:
    """DDD data."""

    y1: np.ndarray = field(default_factory=lambda: np.array([]))
    y0: np.ndarray = field(default_factory=lambda: np.array([]))
    treat: np.ndarray = field(default_factory=lambda: np.array([]))
    partition: np.ndarray = field(default_factory=lambda: np.array([]))
    subgroup: np.ndarray = field(default_factory=lambda: np.array([]))
    covariates: np.ndarray = field(default_factory=lambda: np.array([]))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster: np.ndarray | None = None
    n_units: int = 0
    subgroup_counts: dict = field(default_factory=dict)
    covariate_names: list[str] = field(default_factory=list)
    config: DDDConfig = field(default_factory=DDDConfig)

    @property
    def has_covariates(self) -> bool:
        """Check if data has covariates beyond intercept."""
        return self.covariates.size > 0 and self.covariates.shape[1] > 0

    @property
    def has_cluster(self) -> bool:
        """Check if data has cluster variable."""
        return self.cluster is not None

    def get_covariate_names(self) -> list[str]:
        """Get covariate names."""
        return self.covariate_names


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
