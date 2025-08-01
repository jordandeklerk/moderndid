"""Tensor creation factories for DiD preprocessing."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd

from causaldid.utils import extract_vars_from_formula

from .constants import WEIGHTS_COLUMN, DataFormat
from .models import DIDConfig


class TensorFactory(Protocol):
    """Protocol for tensor factories."""

    def create_tensors(self, data: pd.DataFrame, config: DIDConfig) -> dict[str, np.ndarray | list[np.ndarray] | None]:
        """Create tensors from data."""


class BaseTensorFactory(ABC):
    """Base class for tensor factories."""

    @abstractmethod
    def create_outcomes_tensor(self, data: pd.DataFrame, config: DIDConfig) -> list[np.ndarray] | None:
        """Create outcomes tensor."""

    @abstractmethod
    def create_covariates_tensor(self, data: pd.DataFrame, config: DIDConfig) -> list[np.ndarray] | np.ndarray | None:
        """Create covariates tensor or matrix."""

    @staticmethod
    def create_time_invariant_data(data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Extract time-invariant data (one row per unit)."""
        time_invariant_cols = [config.idname, config.gname, WEIGHTS_COLUMN]

        if config.clustervars:
            time_invariant_cols.extend(config.clustervars)

        # Add time-invariant covariates if any
        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            formula_vars = [v for v in formula_vars if v != config.yname]

            # Check which variables are time-invariant
            for var in formula_vars:
                if var in data.columns:
                    var_counts = data.groupby(config.idname)[var].nunique()
                    if (var_counts == 1).all():
                        time_invariant_cols.append(var)

        time_invariant_cols = list(dict.fromkeys(time_invariant_cols))

        return (
            data.groupby(config.idname)
            .first()[[col for col in time_invariant_cols if col != config.idname]]
            .reset_index()
        )

    @staticmethod
    def create_summary_tables(
        data: pd.DataFrame, time_invariant_data: pd.DataFrame, config: DIDConfig
    ) -> dict[str, pd.DataFrame]:
        """Create summary tables for cohorts and periods."""
        # Cohort counts
        cohort_counts = time_invariant_data.groupby(config.gname).size().reset_index(name="cohort_size")
        cohort_counts.columns = ["cohort", "cohort_size"]

        # Period counts
        period_counts = data.groupby(config.tname).size().reset_index(name="period_size")
        period_counts.columns = ["period", "period_size"]

        # Crosstable counts
        crosstable = data.groupby([config.tname, config.gname]).size().reset_index(name="count")
        crosstable.columns = ["period", "cohort", "count"]
        crosstable_counts = crosstable.pivot(index="period", columns="cohort", values="count").fillna(0)

        return {
            "cohort_counts": cohort_counts,
            "period_counts": period_counts,
            "crosstable_counts": crosstable_counts,
        }

    @staticmethod
    def extract_cluster_variable(time_invariant_data: pd.DataFrame, config: DIDConfig) -> np.ndarray | None:
        """Extract cluster variable if specified."""
        if config.clustervars and len(config.clustervars) > 0:
            return time_invariant_data[config.clustervars[0]].values
        return None

    @staticmethod
    def extract_weights(time_invariant_data: pd.DataFrame) -> np.ndarray:
        """Extract normalized weights."""
        return time_invariant_data[WEIGHTS_COLUMN].values


class PanelTensorFactory(BaseTensorFactory):
    """Factory for balanced panel data tensors."""

    def create_outcomes_tensor(self, data: pd.DataFrame, config: DIDConfig) -> list[np.ndarray]:
        """Create list of outcome arrays, one per time period."""
        outcomes_tensor = []

        for i in range(len(config.time_periods)):
            start_idx = i * config.id_count
            end_idx = (i + 1) * config.id_count
            outcomes_tensor.append(data[config.yname].iloc[start_idx:end_idx].values)

        return outcomes_tensor

    def create_covariates_tensor(self, data: pd.DataFrame, config: DIDConfig) -> list[np.ndarray]:
        """Create list of covariate matrices, one per time period."""
        if config.xformla == "~1" or config.xformla is None:
            return [np.ones((config.id_count, 1)) for _ in range(config.time_periods_count)]

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        covariates_tensor = []
        for i in range(config.time_periods_count):
            start_idx = i * config.id_count
            end_idx = (i + 1) * config.id_count
            period_data = data.iloc[start_idx:end_idx]

            X = np.column_stack([np.ones(len(period_data))] + [period_data[v].values for v in formula_vars])
            covariates_tensor.append(X)

        return covariates_tensor


class UnbalancedPanelTensorFactory(BaseTensorFactory):
    """Factory for unbalanced panel data."""

    def create_outcomes_tensor(self, data: pd.DataFrame, config: DIDConfig) -> None:
        """No outcomes tensor for unbalanced panels."""
        return None

    def create_covariates_tensor(self, data: pd.DataFrame, config: DIDConfig) -> np.ndarray:
        """Create single covariate matrix using time-invariant data."""
        time_invariant_data = self.__class__.create_time_invariant_data(data, config)

        if config.xformla == "~1" or config.xformla is None:
            return np.ones((len(time_invariant_data), 1))

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        # Filter to only time-invariant variables
        available_vars = []
        for var in formula_vars:
            if var in time_invariant_data.columns:
                available_vars.append(var)
            else:
                # Try to get from original data if truly time-invariant
                var_counts = data.groupby(config.idname)[var].nunique()
                if (var_counts == 1).all():
                    time_invariant_data[var] = data.groupby(config.idname)[var].first().values
                    available_vars.append(var)

        X = np.column_stack(
            [np.ones(len(time_invariant_data))] + [time_invariant_data[v].values for v in available_vars]
        )

        return X


class RepeatedCrossSectionTensorFactory(BaseTensorFactory):
    """Factory for repeated cross-section data."""

    def create_outcomes_tensor(self, data: pd.DataFrame, config: DIDConfig) -> None:
        """No outcomes tensor for repeated cross-sections."""
        return None

    def create_covariates_tensor(self, data: pd.DataFrame, config: DIDConfig) -> np.ndarray:
        """Create covariate matrix from full data."""
        if config.xformla == "~1" or config.xformla is None:
            return np.ones((len(data), 1))

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        X = np.column_stack([np.ones(len(data))] + [data[v].values for v in formula_vars])

        return X


class TensorFactorySelector:
    """Selects appropriate tensor factory based on data format."""

    @staticmethod
    def get_factory(config: DIDConfig) -> BaseTensorFactory:
        """Get appropriate tensor factory."""
        if config.data_format == DataFormat.PANEL:
            return PanelTensorFactory()
        if config.data_format == DataFormat.UNBALANCED_PANEL:
            return UnbalancedPanelTensorFactory()
        if config.data_format == DataFormat.REPEATED_CROSS_SECTION:
            return RepeatedCrossSectionTensorFactory()
        raise ValueError(f"Unknown data format: {config.data_format}")

    @classmethod
    def create_tensors(
        cls, data: pd.DataFrame, config: DIDConfig
    ) -> dict[str, np.ndarray | list[np.ndarray] | pd.DataFrame | None]:
        """Create all tensors using appropriate factory."""
        factory = cls.get_factory(config)

        time_invariant_data = factory.create_time_invariant_data(data, config)
        summary_tables = factory.create_summary_tables(data, time_invariant_data, config)

        outcomes_tensor = factory.create_outcomes_tensor(data, config)
        covariates_tensor = factory.create_covariates_tensor(data, config)

        cluster = factory.extract_cluster_variable(time_invariant_data, config)
        weights = factory.extract_weights(time_invariant_data)

        if isinstance(covariates_tensor, list):
            covariates_matrix = None
        else:
            covariates_matrix = covariates_tensor
            covariates_tensor = None

        return {
            "data": data,
            "time_invariant_data": time_invariant_data,
            "outcomes_tensor": outcomes_tensor,
            "covariates_matrix": covariates_matrix,
            "covariates_tensor": covariates_tensor,
            "cluster": cluster,
            "weights": weights,
            **summary_tables,
        }
