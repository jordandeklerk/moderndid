"""Data transformation classes for preprocessing."""

import warnings
from typing import Protocol

import numpy as np
import pandas as pd

from .base import BaseTransformer
from .config import BasePreprocessConfig, ContDIDConfig, DIDConfig, TwoPeriodDIDConfig
from .constants import (
    NEVER_TREATED_VALUE,
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    ControlGroup,
    DataFormat,
)
from .utils import extract_vars_from_formula


class DataTransformer(Protocol):
    """Data transformer."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""


class ColumnSelector(BaseTransformer):
    """Column selector."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
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

        if isinstance(config, ContDIDConfig) and config.dname:
            cols_to_keep.append(config.dname)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None]

        return data[cols_to_keep].copy()


class MissingDataHandler(BaseTransformer):
    """Missing data."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        n_orig = len(data)
        data_clean = data.dropna()
        n_new = len(data_clean)

        if n_orig > n_new:
            if isinstance(config, TwoPeriodDIDConfig) and config.panel:
                raise ValueError(
                    f"Missing values found in panel data. Dropped {n_orig - n_new} rows. "
                    "Panel data requires complete observations for all time periods. "
                    "Please handle missing values before preprocessing."
                )
            warnings.warn(f"Dropped {n_orig - n_new} rows from original data due to missing values")

        return data_clean


class WeightNormalizer(BaseTransformer):
    """Weight normalizer."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        data = data.copy()

        if config.weightsname is None:
            weights = np.ones(len(data))
        else:
            weights = data[config.weightsname].values

        weights = weights / weights.mean()
        data[WEIGHTS_COLUMN] = weights

        return data


class DataSorter(BaseTransformer):
    """Data sorter."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        sort_cols = [config.tname, config.gname]

        if config.idname:
            sort_cols.append(config.idname)

        return data.sort_values(sort_cols).copy()


class TreatmentEncoder(BaseTransformer):
    """Treatment encoder."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        data = data.copy()

        data[config.gname] = data[config.gname].astype(float)
        data.loc[data[config.gname] == 0, config.gname] = NEVER_TREATED_VALUE

        tlist = sorted(data[config.tname].unique())
        max_treatment_time = max(tlist)
        data.loc[data[config.gname] > max_treatment_time, config.gname] = NEVER_TREATED_VALUE

        return data


class EarlyTreatmentFilter(BaseTransformer):
    """Early treatment filter."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
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


class ControlGroupCreator(BaseTransformer):
    """Control group creator."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDConfig):
            return data

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
    """Panel balancer."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
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
    """Repeated cross section handler."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
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


class TimePeriodRecoder(BaseTransformer):
    """Time period recoder."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return data

        data = data.copy()
        original_periods = sorted(data[config.tname].unique())
        time_map = {t: i + 1 for i, t in enumerate(original_periods)}

        data[config.tname] = data[config.tname].map(time_map)
        config.time_map = time_map

        return data


class EarlyTreatmentGroupFilter(BaseTransformer):
    """Early treatment group filter."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return data

        data = data.copy()

        glist = sorted([g for g in data[config.gname].unique() if np.isfinite(g)])
        tlist = sorted(data[config.tname].unique())

        if not glist:
            return data

        min_valid_group = config.required_pre_periods + config.anticipation + min(tlist)

        groups_to_drop = [g for g in glist if g < min_valid_group]

        if groups_to_drop:
            warnings.warn(
                f"Dropped {len(groups_to_drop)} groups treated before period {min_valid_group} "
                f"(required_pre_periods={config.required_pre_periods}, anticipation={config.anticipation})"
            )
            data = data[~data[config.gname].isin(groups_to_drop)].copy()

        return data


class DoseValidator(BaseTransformer):
    """Dose validator."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig) or not config.dname:
            return data

        data = data.copy()

        never_treated = data[config.gname] == NEVER_TREATED_VALUE
        if never_treated.any():
            data.loc[never_treated, config.dname] = 0

        treated_units = (data[config.gname] != NEVER_TREATED_VALUE) & np.isfinite(data[config.gname])
        post_treatment = data[config.tname] >= data[config.gname]
        no_dose = (data[config.dname] == 0) | data[config.dname].isna()

        invalid_obs = treated_units & post_treatment & no_dose
        n_invalid = invalid_obs.sum()

        if n_invalid > 0:
            warnings.warn(f"Dropped {n_invalid} post-treatment observations with missing or zero dose values")
            data = data[~invalid_obs].copy()

        return data


class ConfigUpdater:
    """Config updater."""

    @staticmethod
    def update(data: pd.DataFrame, config: BasePreprocessConfig) -> None:
        """Update config."""
        if isinstance(config, TwoPeriodDIDConfig):
            tlist = sorted(data[config.tname].unique())
            treat_list = sorted(data[config.treat_col].unique())

            if config.idname:
                n_units = data[config.idname].nunique()
            else:
                n_units = len(data)

            config.time_periods = np.array(tlist)
            config.time_periods_count = len(tlist)
            config.treated_groups = np.array(treat_list)
            config.treated_groups_count = len(treat_list)
            config.id_count = n_units
            return

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


class PrePostColumnSelector(BaseTransformer):
    """Pre-post column selector."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return data

        cols_to_keep = [config.yname, config.tname, config.treat_col]

        if config.idname:
            cols_to_keep.append(config.idname)

        if config.weightsname:
            cols_to_keep.append(config.weightsname)

        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            formula_vars = [v for v in formula_vars if v != config.yname]
            cols_to_keep.extend(formula_vars)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None]

        return data[cols_to_keep].copy()


class PrePostCovariateProcessor(BaseTransformer):
    """Pre-post covariate processor."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return data

        import formulaic as fml

        covariates_formula = config.xformla if config.xformla else "~1"

        try:
            model_matrix_result = fml.model_matrix(
                covariates_formula,
                data,
                output="pandas",
            )
            covariates_df = model_matrix_result

            if hasattr(model_matrix_result, "model_spec") and model_matrix_result.model_spec:
                original_cov_names = [var for var in model_matrix_result.model_spec.variables if var != "1"]
            else:
                original_cov_names = []
                warnings.warn("Could not retrieve model_spec from formulaic output.", UserWarning)

        except Exception as e:
            raise ValueError(f"Error processing covariates_formula '{covariates_formula}' with formulaic: {e}") from e

        cols_to_drop = [name for name in original_cov_names if name in data.columns]
        data_processed = pd.concat([data.drop(columns=cols_to_drop), covariates_df], axis=1)

        return data_processed


class PrePostPanelBalancer(BaseTransformer):
    """Pre-post panel balancer."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig) or not config.panel or not config.idname:
            return data

        n_times = data[config.tname].nunique()
        obs_counts = data.groupby(config.idname).size()
        ids_to_keep = obs_counts[obs_counts == n_times].index

        if len(ids_to_keep) < len(obs_counts):
            warnings.warn("Panel data is unbalanced. Dropping units with incomplete observations.", UserWarning)

        return data[data[config.idname].isin(ids_to_keep)].copy()


class PrePostInvarianceChecker(BaseTransformer):
    """Pre-post invariance checker."""

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pd.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig) or not config.panel or not config.idname:
            return data

        time_periods = sorted(data[config.tname].unique())
        if len(time_periods) != 2:
            return data

        pre_period, post_period = time_periods

        pre_df = data[data[config.tname] == pre_period].set_index(config.idname)
        post_df = data[data[config.tname] == post_period].set_index(config.idname)

        common_ids = pre_df.index.intersection(post_df.index)
        pre_df = pre_df.loc[common_ids]
        post_df = post_df.loc[common_ids]

        if not pre_df[config.treat_col].equals(post_df[config.treat_col]):
            raise ValueError(f"Treatment indicator ('{config.treat_col}') must be time-invariant in panel data.")

        if WEIGHTS_COLUMN in pre_df.columns and WEIGHTS_COLUMN in post_df.columns:
            if not pre_df[WEIGHTS_COLUMN].equals(post_df[WEIGHTS_COLUMN]):
                raise ValueError("Weights must be time-invariant in panel data.")

        return data


class DataTransformerPipeline:
    """Data transformer pipeline."""

    def __init__(self, transformers: list[BaseTransformer] | None = None):
        """Initialize data transformer pipeline."""
        self.transformers = transformers or []

    @staticmethod
    def get_did_pipeline() -> "DataTransformerPipeline":
        """Get DID pipeline."""
        return DataTransformerPipeline(
            [
                ColumnSelector(),
                MissingDataHandler(),
                WeightNormalizer(),
                TreatmentEncoder(),
                EarlyTreatmentFilter(),
                ControlGroupCreator(),
                PanelBalancer(),
                RepeatedCrossSectionHandler(),
                DataSorter(),
            ]
        )

    @staticmethod
    def get_cont_did_pipeline() -> "DataTransformerPipeline":
        """Get ContDID pipeline."""
        return DataTransformerPipeline(
            [
                ColumnSelector(),
                MissingDataHandler(),
                WeightNormalizer(),
                TimePeriodRecoder(),
                EarlyTreatmentGroupFilter(),
                DoseValidator(),
                PanelBalancer(),
                DataSorter(),
            ]
        )

    @staticmethod
    def get_two_period_pipeline() -> "DataTransformerPipeline":
        """Get two-period pipeline."""
        return DataTransformerPipeline(
            [
                PrePostColumnSelector(),
                MissingDataHandler(),
                PrePostCovariateProcessor(),
                WeightNormalizer(),
                PrePostPanelBalancer(),
                PrePostInvarianceChecker(),
            ]
        )

    def transform(self, data: pd.DataFrame, config: BasePreprocessConfig) -> pd.DataFrame:
        """Transform data."""
        for transformer in self.transformers:
            data = transformer.transform(data, config)

        ConfigUpdater.update(data, config)

        return data
