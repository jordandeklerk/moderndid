"""Tensor creation factories for DiD preprocessing."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .config import DIDConfig
from .constants import WEIGHTS_COLUMN, DataFormat
from .utils import extract_vars_from_formula


class TensorFactory(Protocol):
    """Protocol for tensor factories."""

    def create_tensors(self, data: DataFrame, config: DIDConfig) -> dict[str, np.ndarray | list[np.ndarray] | None]:
        """Create tensors from data."""


class BaseTensorFactory(ABC):
    """Base class for tensor factories."""

    @abstractmethod
    def create_outcomes_tensor(self, data: pl.DataFrame, config: DIDConfig) -> list[np.ndarray] | None:
        """Create outcomes tensor."""

    @abstractmethod
    def create_covariates_tensor(self, data: pl.DataFrame, config: DIDConfig) -> list[np.ndarray] | np.ndarray | None:
        """Create covariates tensor or matrix."""

    @staticmethod
    def create_time_invariant_data(data: DataFrame, config: DIDConfig) -> pl.DataFrame:
        """Extract time-invariant data."""
        df = to_polars(data)
        time_invariant_cols = [config.idname, config.gname, WEIGHTS_COLUMN]

        if config.clustervars:
            time_invariant_cols.extend(config.clustervars)

        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            formula_vars = [v for v in formula_vars if v != config.yname]

            for var in formula_vars:
                if var in df.columns:
                    var_counts = df.group_by(config.idname).agg(pl.col(var).n_unique().alias("n_unique"))
                    if (var_counts["n_unique"] == 1).all():
                        time_invariant_cols.append(var)

        time_invariant_cols = list(dict.fromkeys(time_invariant_cols))

        cols_to_select = [col for col in time_invariant_cols if col in df.columns]
        return (
            df.group_by(config.idname, maintain_order=True)
            .first()
            .select([config.idname] + [c for c in cols_to_select if c != config.idname])
        )

    @staticmethod
    def create_summary_tables(
        data: DataFrame, time_invariant_data: pl.DataFrame, config: DIDConfig
    ) -> dict[str, pl.DataFrame]:
        """Create summary tables."""
        df = to_polars(data)

        cohort_counts = (
            time_invariant_data.group_by(config.gname, maintain_order=True)
            .len()
            .rename({config.gname: "cohort", "len": "cohort_size"})
        )

        period_counts = (
            df.group_by(config.tname, maintain_order=True).len().rename({config.tname: "period", "len": "period_size"})
        )

        crosstable = (
            df.group_by([config.tname, config.gname], maintain_order=True)
            .len()
            .rename({config.tname: "period", config.gname: "cohort", "len": "count"})
        )
        crosstable_counts = crosstable.pivot(index="period", on="cohort", values="count").fill_null(0)

        return {
            "cohort_counts": cohort_counts,
            "period_counts": period_counts,
            "crosstable_counts": crosstable_counts,
        }

    @staticmethod
    def extract_cluster_variable(time_invariant_data: pl.DataFrame, config: DIDConfig) -> np.ndarray | None:
        """Extract cluster variable if specified."""
        if config.clustervars and len(config.clustervars) > 0:
            return time_invariant_data[config.clustervars[0]].to_numpy()
        return None

    @staticmethod
    def extract_weights(time_invariant_data: pl.DataFrame) -> np.ndarray:
        """Extract normalized weights."""
        return time_invariant_data[WEIGHTS_COLUMN].to_numpy()


class PanelTensorFactory(BaseTensorFactory):
    """Factory for balanced panel data tensors."""

    def create_outcomes_tensor(self, data: pl.DataFrame, config: DIDConfig) -> list[np.ndarray]:
        """Create list of outcome arrays, one per time period."""
        df = to_polars(data)
        outcomes_tensor = []

        for i in range(len(config.time_periods)):
            start_idx = i * config.id_count
            end_idx = (i + 1) * config.id_count
            outcomes_tensor.append(df[config.yname].slice(start_idx, end_idx - start_idx).to_numpy())

        return outcomes_tensor

    def create_covariates_tensor(self, data: pl.DataFrame, config: DIDConfig) -> list[np.ndarray]:
        """Create list of covariate matrices, one per time period."""
        df = to_polars(data)

        if config.xformla == "~1" or config.xformla is None:
            return [np.ones((config.id_count, 1)) for _ in range(config.time_periods_count)]

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        covariates_tensor = []
        for i in range(config.time_periods_count):
            start_idx = i * config.id_count
            end_idx = (i + 1) * config.id_count
            period_data = df.slice(start_idx, end_idx - start_idx)

            X = np.column_stack([np.ones(len(period_data))] + [period_data[v].to_numpy() for v in formula_vars])
            covariates_tensor.append(X)

        return covariates_tensor


class UnbalancedPanelTensorFactory(BaseTensorFactory):
    """Factory for unbalanced panel data."""

    def create_outcomes_tensor(self, data: pl.DataFrame, config: DIDConfig) -> None:
        """No outcomes tensor for unbalanced panels."""
        return None

    def create_covariates_tensor(self, data: pl.DataFrame, config: DIDConfig) -> np.ndarray:
        """Create single covariate matrix using time-invariant data."""
        df = to_polars(data)
        time_invariant_data = self.__class__.create_time_invariant_data(df, config)

        if config.xformla == "~1" or config.xformla is None:
            return np.ones((len(time_invariant_data), 1))

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        available_vars = []
        for var in formula_vars:
            if var in time_invariant_data.columns:
                available_vars.append(var)
            else:
                var_counts = df.group_by(config.idname).agg(pl.col(var).n_unique().alias("n_unique"))
                if (var_counts["n_unique"] == 1).all():
                    first_vals = df.group_by(config.idname).first().select([config.idname, var])
                    time_invariant_data = time_invariant_data.join(first_vals, on=config.idname, how="left")
                    available_vars.append(var)

        X = np.column_stack(
            [np.ones(len(time_invariant_data))] + [time_invariant_data[v].to_numpy() for v in available_vars]
        )

        return X


class RepeatedCrossSectionTensorFactory(BaseTensorFactory):
    """Factory for repeated cross-section data."""

    def create_outcomes_tensor(self, data: pl.DataFrame, config: DIDConfig) -> None:
        """No outcomes tensor for repeated cross-sections."""
        return None

    def create_covariates_tensor(self, data: pl.DataFrame, config: DIDConfig) -> np.ndarray:
        """Create covariate matrix from full data."""
        df = to_polars(data)

        if config.xformla == "~1" or config.xformla is None:
            return np.ones((len(df), 1))

        formula_vars = extract_vars_from_formula(config.xformla)
        formula_vars = [v for v in formula_vars if v != config.yname]

        X = np.column_stack([np.ones(len(df))] + [df[v].to_numpy() for v in formula_vars])

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
        cls, data: DataFrame, config: DIDConfig
    ) -> dict[str, np.ndarray | list[np.ndarray] | pl.DataFrame | None]:
        """Create all tensors using appropriate factory."""
        df = to_polars(data)
        factory = cls.get_factory(config)

        time_invariant_data = factory.create_time_invariant_data(df, config)
        summary_tables = factory.create_summary_tables(df, time_invariant_data, config)

        outcomes_tensor = factory.create_outcomes_tensor(df, config)
        covariates_tensor = factory.create_covariates_tensor(df, config)

        cluster = factory.extract_cluster_variable(time_invariant_data, config)
        weights = factory.extract_weights(time_invariant_data)

        if isinstance(covariates_tensor, list):
            covariates_matrix = None
        else:
            covariates_matrix = covariates_tensor
            covariates_tensor = None

        return {
            "data": df,
            "time_invariant_data": time_invariant_data,
            "outcomes_tensor": outcomes_tensor,
            "covariates_matrix": covariates_matrix,
            "covariates_tensor": covariates_tensor,
            "cluster": cluster,
            "weights": weights,
            **summary_tables,
        }
