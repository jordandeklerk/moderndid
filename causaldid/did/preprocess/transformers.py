"""Data transformation classes for DiD preprocessing."""

import warnings
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd

from causaldid.utils import extract_vars_from_formula

from .constants import (
    NEVER_TREATED_VALUE,
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    ControlGroup,
    DataFormat,
)
from .models import DIDConfig


class DataTransformer(Protocol):
    """Protocol for data transformers."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Transform data according to configuration."""


class BaseTransformer(ABC):
    """Base class for data transformers."""

    @abstractmethod
    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Transform data according to configuration."""


class ColumnSelector(BaseTransformer):
    """Selects relevant columns based on configuration."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Select only relevant columns."""
        cols_to_keep = [config.yname, config.tname, config.gname]

        if config.idname:
            cols_to_keep.append(config.idname)

        if config.weightsname:
            cols_to_keep.append(config.weightsname)

        if config.clustervars:
            cols_to_keep.extend(config.clustervars)

        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            formula_vars = [v for v in formula_vars if v != config.yname]
            cols_to_keep.extend(formula_vars)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None]

        return data[cols_to_keep].copy()


class MissingDataHandler(BaseTransformer):
    """Handles missing data by dropping rows with NaN values."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Drop rows with missing values."""
        n_orig = len(data)
        data_clean = data.dropna()
        n_new = len(data_clean)

        if n_orig > n_new:
            warnings.warn(f"Dropped {n_orig - n_new} rows from original data due to missing values")

        return data_clean


class WeightNormalizer(BaseTransformer):
    """Adds and normalizes sampling weights."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Add normalized weights column."""
        data = data.copy()

        if config.weightsname is None:
            weights = np.ones(len(data))
        else:
            weights = data[config.weightsname].values

        weights = weights / weights.mean()
        data[WEIGHTS_COLUMN] = weights

        return data


class TreatmentEncoder(BaseTransformer):
    """Encodes treatment groups according to DiD conventions."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Encode treatment groups (0 -> infinity for never-treated)."""
        data = data.copy()

        data[config.gname] = data[config.gname].astype(float)
        data.loc[data[config.gname] == 0, config.gname] = NEVER_TREATED_VALUE

        tlist = sorted(data[config.tname].unique())
        max_treatment_time = max(tlist)
        data.loc[data[config.gname] > max_treatment_time, config.gname] = NEVER_TREATED_VALUE

        return data


class EarlyTreatmentFilter(BaseTransformer):
    """Filters out units treated before the first valid period."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Filter early-treated units."""
        data = data.copy()

        tlist = sorted(data[config.tname].unique())
        first_period = min(tlist)

        treated_early = data[config.gname] <= first_period + config.anticipation

        if config.idname:
            early_units = data.loc[treated_early, config.idname].unique()
            n_early = len(early_units)
            if n_early > 0:
                warnings.warn(f"Dropped {n_early} units that were already treated in the first period")
                data = data[~data[config.idname].isin(early_units)]
        else:
            n_early = treated_early.sum()
            if n_early > 0:
                warnings.warn(f"Dropped {n_early} observations that were already treated in the first period")
                data = data[~treated_early]

        return data


class NeverTreatedHandler(BaseTransformer):
    """Handles cases with no never-treated units."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Handle missing never-treated group."""
        data = data.copy()

        glist = sorted(data[config.gname].unique())

        if NEVER_TREATED_VALUE in glist:
            return data

        finite_glist = [g for g in glist if np.isfinite(g)]
        if not finite_glist:
            return data

        latest_g = max(finite_glist)
        cutoff_t = latest_g - config.anticipation

        if config.control_group == ControlGroup.NEVER_TREATED:
            warnings.warn(
                "No never-treated group is available. "
                "The last treated cohort is being coerced as 'never-treated' units."
            )
            data = data[data[config.tname] < cutoff_t].copy()
            data.loc[data[config.gname] == latest_g, config.gname] = NEVER_TREATED_VALUE
        else:
            data = data[data[config.tname] < cutoff_t].copy()

        return data


class PanelBalancer(BaseTransformer):
    """Balances panel data by keeping only complete units."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Balance panel by keeping only units observed in all periods."""
        if not config.panel or config.allow_unbalanced_panel or not config.idname:
            return data

        data = data.copy()
        tlist = sorted(data[config.tname].unique())
        n_periods = len(tlist)

        unit_counts = data.groupby(config.idname).size()
        complete_units = unit_counts[unit_counts == n_periods].index

        n_old = data[config.idname].nunique()
        data = data[data[config.idname].isin(complete_units)].copy()
        n_new = data[config.idname].nunique()

        if n_new < n_old:
            warnings.warn(f"Dropped {n_old - n_new} units while converting to balanced panel")

        if len(data) == 0:
            raise ValueError(
                "All observations dropped while converting to balanced panel. "
                "Consider setting panel=False and/or revisiting 'idname'"
            )

        return data


class RepeatedCrossSectionHandler(BaseTransformer):
    """Handles repeated cross-section data structure."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Add row IDs for repeated cross-sections."""
        if config.panel:
            return data

        data = data.copy()

        if config.idname is None:
            config.true_repeated_cross_sections = True

        if config.true_repeated_cross_sections:
            data = data.reset_index(drop=True)
            data[ROW_ID_COLUMN] = data.index
            config.idname = ROW_ID_COLUMN
        else:
            data[ROW_ID_COLUMN] = data[config.idname]

        return data


class DataSorter(BaseTransformer):
    """Sorts data for consistent processing."""

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Sort data by time, group, and id."""
        sort_cols = [config.tname, config.gname]

        if config.idname:
            sort_cols.append(config.idname)

        return data.sort_values(sort_cols).copy()


class ConfigUpdater:
    """Updates configuration with computed values from data."""

    @staticmethod
    def update(data: pd.DataFrame, config: DIDConfig) -> None:
        """Update config with computed values."""
        tlist = sorted(data[config.tname].unique())
        glist = sorted(data[config.gname].unique())

        glist_finite = [g for g in glist if np.isfinite(g)]

        if config.idname:
            n_units = data[config.idname].nunique()
        else:
            n_units = len(data)

        config.time_periods = np.array(tlist)
        config.time_periods_count = len(tlist)
        config.treated_groups = np.array(glist_finite)
        config.treated_groups_count = len(glist_finite)
        config.id_count = n_units

        if config.panel and config.allow_unbalanced_panel:
            unit_counts = data.groupby(config.idname).size()
            is_balanced = (unit_counts == len(tlist)).all()
            if is_balanced:
                config.data_format = DataFormat.PANEL
            else:
                config.data_format = DataFormat.UNBALANCED_PANEL
        elif config.panel:
            config.data_format = DataFormat.PANEL
        else:
            config.data_format = DataFormat.REPEATED_CROSS_SECTION

        if len(tlist) == 2:
            config.cband = False


class DataTransformerPipeline:
    """Pipeline of data transformers."""

    def __init__(self, transformers: list[BaseTransformer] | None = None):
        """Initialize with list of transformers."""
        self.transformers = transformers or self._get_default_transformers()

    @staticmethod
    def _get_default_transformers() -> list[BaseTransformer]:
        """Get default transformation pipeline."""
        return [
            ColumnSelector(),
            MissingDataHandler(),
            WeightNormalizer(),
            TreatmentEncoder(),
            EarlyTreatmentFilter(),
            NeverTreatedHandler(),
            PanelBalancer(),
            RepeatedCrossSectionHandler(),
            DataSorter(),
        ]

    def transform(self, data: pd.DataFrame, config: DIDConfig) -> pd.DataFrame:
        """Apply all transformations in sequence."""
        for transformer in self.transformers:
            data = transformer.transform(data, config)

        ConfigUpdater.update(data, config)

        return data
