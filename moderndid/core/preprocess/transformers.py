"""Data transformation classes for preprocessing."""

import warnings
from typing import Protocol

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .base import BaseTransformer
from .config import BasePreprocessConfig, ContDIDConfig, DIDConfig, DIDInterConfig, TwoPeriodDIDConfig
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

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""


class ColumnSelector(BaseTransformer):
    """Column selector."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
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

        return df.select(cols_to_keep)


class MissingDataHandler(BaseTransformer):
    """Missing data."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
        n_orig = len(df)
        data_clean = df.drop_nulls()
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

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        if config.weightsname is None:
            weights = np.ones(len(df))
        else:
            weights = df[config.weightsname].to_numpy()

        weights = weights / weights.mean()
        return df.with_columns(pl.Series(name=WEIGHTS_COLUMN, values=weights))


class DataSorter(BaseTransformer):
    """Data sorter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
        sort_cols = [config.tname, config.gname]

        idname = getattr(config, "idname", None)
        if idname:
            sort_cols.append(idname)

        return df.sort(sort_cols)


class TreatmentEncoder(BaseTransformer):
    """Treatment encoder."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        df = df.with_columns(pl.col(config.gname).cast(pl.Float64))

        df = df.with_columns(
            pl.when(pl.col(config.gname) == 0)
            .then(pl.lit(NEVER_TREATED_VALUE))
            .otherwise(pl.col(config.gname))
            .alias(config.gname)
        )

        tlist = sorted(df[config.tname].unique().to_list())
        max_treatment_time = max(tlist)
        df = df.with_columns(
            pl.when(pl.col(config.gname) > max_treatment_time)
            .then(pl.lit(NEVER_TREATED_VALUE))
            .otherwise(pl.col(config.gname))
            .alias(config.gname)
        )

        return df


class EarlyTreatmentFilter(BaseTransformer):
    """Early treatment filter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        tlist = sorted(df[config.tname].unique().to_list())
        first_period = min(tlist)

        treated_early_mask = pl.col(config.gname) <= first_period + config.anticipation

        if config.idname:
            early_units = df.filter(treated_early_mask)[config.idname].unique().to_list()
            n_early = len(early_units)
            if n_early > 0:
                warnings.warn(f"Dropped {n_early} units that were already treated in the first period")
                df = df.filter(~pl.col(config.idname).is_in(early_units))
        else:
            n_early = df.filter(treated_early_mask).height
            if n_early > 0:
                warnings.warn(f"Dropped {n_early} observations that were already treated in the first period")
                df = df.filter(~treated_early_mask)

        return df


class ControlGroupCreator(BaseTransformer):
    """Control group creator."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDConfig):
            return to_polars(data)

        df = to_polars(data)

        glist = sorted(df[config.gname].unique().to_list())

        if NEVER_TREATED_VALUE in glist:
            return df

        finite_glist = [g for g in glist if np.isfinite(g)]
        if not finite_glist:
            return df

        latest_g = max(finite_glist)
        cutoff_t = latest_g - config.anticipation

        if config.control_group == ControlGroup.NEVER_TREATED:
            warnings.warn(
                "No never-treated group is available. "
                "The last treated cohort is being coerced as 'never-treated' units."
            )
            df = df.filter(pl.col(config.tname) < cutoff_t)
            df = df.with_columns(
                pl.when(pl.col(config.gname) == latest_g)
                .then(pl.lit(NEVER_TREATED_VALUE))
                .otherwise(pl.col(config.gname))
                .alias(config.gname)
            )
        else:
            df = df.filter(pl.col(config.tname) < cutoff_t)

        return df


class PanelBalancer(BaseTransformer):
    """Panel balancer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not config.panel or config.allow_unbalanced_panel or not config.idname:
            return to_polars(data)

        df = to_polars(data)
        tlist = sorted(df[config.tname].unique().to_list())
        n_periods = len(tlist)

        unit_counts = df.group_by(config.idname).len()
        complete_units = unit_counts.filter(pl.col("len") == n_periods)[config.idname].to_list()

        n_old = df[config.idname].n_unique()
        df = df.filter(pl.col(config.idname).is_in(complete_units))
        n_new = df[config.idname].n_unique()

        if n_new < n_old:
            warnings.warn(f"Dropped {n_old - n_new} units while converting to balanced panel")

        if len(df) == 0:
            raise ValueError(
                "All observations dropped while converting to balanced panel. "
                "Consider setting panel=False and/or revisiting 'idname'"
            )

        return df


class RepeatedCrossSectionHandler(BaseTransformer):
    """Repeated cross section handler."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if config.panel:
            return to_polars(data)

        df = to_polars(data)

        if config.idname is None:
            config.true_repeated_cross_sections = True

        if config.true_repeated_cross_sections:
            df = df.with_row_index(name=ROW_ID_COLUMN)
            config.idname = ROW_ID_COLUMN
        else:
            df = df.with_columns(pl.col(config.idname).alias(ROW_ID_COLUMN))

        return df


class TimePeriodRecoder(BaseTransformer):
    """Time period recoder."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return to_polars(data)

        df = to_polars(data)
        original_periods = sorted(df[config.tname].unique().to_list())
        time_map = {t: i + 1 for i, t in enumerate(original_periods)}

        df = df.with_columns(pl.col(config.tname).replace(time_map).alias(config.tname))
        config.time_map = time_map

        return df


class EarlyTreatmentGroupFilter(BaseTransformer):
    """Early treatment group filter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return to_polars(data)

        df = to_polars(data)

        glist = sorted([g for g in df[config.gname].unique().to_list() if np.isfinite(g)])
        tlist = sorted(df[config.tname].unique().to_list())

        if not glist:
            return df

        min_valid_group = config.required_pre_periods + config.anticipation + min(tlist)

        groups_to_drop = [g for g in glist if g < min_valid_group]

        if groups_to_drop:
            warnings.warn(
                f"Dropped {len(groups_to_drop)} groups treated before period {min_valid_group} "
                f"(required_pre_periods={config.required_pre_periods}, anticipation={config.anticipation})"
            )
            df = df.filter(~pl.col(config.gname).is_in(groups_to_drop))

        return df


class DoseValidatorTransformer(BaseTransformer):
    """Dose validator transformer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig) or not config.dname:
            return to_polars(data)

        df = to_polars(data)

        df = df.with_columns(
            pl.when(pl.col(config.gname) == NEVER_TREATED_VALUE)
            .then(pl.lit(0))
            .otherwise(pl.col(config.dname))
            .alias(config.dname)
        )

        invalid_mask = (
            (pl.col(config.gname) != NEVER_TREATED_VALUE)
            & pl.col(config.gname).is_finite()
            & (pl.col(config.tname) >= pl.col(config.gname))
            & ((pl.col(config.dname) == 0) | pl.col(config.dname).is_null())
        )

        n_invalid = df.filter(invalid_mask).height

        if n_invalid > 0:
            warnings.warn(f"Dropped {n_invalid} post-treatment observations with missing or zero dose values")
            df = df.filter(~invalid_mask)

        return df


class ConfigUpdater:
    """Config updater."""

    @staticmethod
    def update(data: DataFrame, config: BasePreprocessConfig) -> None:
        """Update config."""
        df = to_polars(data)

        if isinstance(config, TwoPeriodDIDConfig):
            tlist = sorted(df[config.tname].unique().to_list())
            treat_list = sorted(df[config.treat_col].unique().to_list())

            if config.idname:
                n_units = df[config.idname].n_unique()
            else:
                n_units = len(df)

            config.time_periods = np.array(tlist)
            config.time_periods_count = len(tlist)
            config.treated_groups = np.array(treat_list)
            config.treated_groups_count = len(treat_list)
            config.id_count = n_units
            return

        tlist = sorted(df[config.tname].unique().to_list())
        glist = sorted(df[config.gname].unique().to_list())

        glist_finite = [g for g in glist if np.isfinite(g)]

        if config.idname:
            n_units = df[config.idname].n_unique()
        else:
            n_units = len(df)

        config.time_periods = np.array(tlist)
        config.time_periods_count = len(tlist)
        config.treated_groups = np.array(glist_finite)
        config.treated_groups_count = len(glist_finite)
        config.id_count = n_units

        if config.panel and config.allow_unbalanced_panel:
            unit_counts = df.group_by(config.idname).len()
            is_balanced = (unit_counts["len"] == len(tlist)).all()
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

    def transform(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return to_polars(data)

        df = to_polars(data)
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

        return df.select(cols_to_keep)


class PrePostCovariateProcessor(BaseTransformer):
    """Pre-post covariate processor."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return to_polars(data)

        import formulaic as fml

        df = to_polars(data)
        covariates_formula = config.xformla if config.xformla else "~1"

        try:
            model_matrix_result = fml.model_matrix(covariates_formula, df)
            covariates_pl = model_matrix_result.__wrapped__

            if hasattr(model_matrix_result, "model_spec") and model_matrix_result.model_spec:
                original_cov_names = [var for var in model_matrix_result.model_spec.variables if var != "1"]
            else:
                original_cov_names = []
                warnings.warn("Could not retrieve model_spec from formulaic output.", UserWarning)

        except Exception as e:
            raise ValueError(f"Error processing covariates_formula '{covariates_formula}' with formulaic: {e}") from e

        cols_to_drop = [name for name in original_cov_names if name in df.columns]
        cols_to_keep = [col for col in df.columns if col not in cols_to_drop]

        return pl.concat([df.select(cols_to_keep), covariates_pl], how="horizontal")


class PrePostPanelBalancer(BaseTransformer):
    """Pre-post panel balancer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig) or not config.panel or not config.idname:
            return to_polars(data)

        df = to_polars(data)
        n_times = df[config.tname].n_unique()
        obs_counts = df.group_by(config.idname).len()
        ids_to_keep = obs_counts.filter(pl.col("len") == n_times)[config.idname].to_list()

        if len(ids_to_keep) < obs_counts.height:
            warnings.warn("Panel data is unbalanced. Dropping units with incomplete observations.", UserWarning)

        return df.filter(pl.col(config.idname).is_in(ids_to_keep))


class PrePostInvarianceChecker(BaseTransformer):
    """Pre-post invariance checker."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig) or not config.panel or not config.idname:
            return to_polars(data)

        df = to_polars(data)
        time_periods = sorted(df[config.tname].unique().to_list())
        if len(time_periods) != 2:
            return df

        pre_period, post_period = time_periods

        pre_df = df.filter(pl.col(config.tname) == pre_period).sort(config.idname)
        post_df = df.filter(pl.col(config.tname) == post_period).sort(config.idname)

        pre_ids = set(pre_df[config.idname].to_list())
        post_ids = set(post_df[config.idname].to_list())
        common_ids = list(pre_ids.intersection(post_ids))

        pre_df = pre_df.filter(pl.col(config.idname).is_in(common_ids)).sort(config.idname)
        post_df = post_df.filter(pl.col(config.idname).is_in(common_ids)).sort(config.idname)

        if not pre_df[config.treat_col].equals(post_df[config.treat_col]):
            raise ValueError(f"Treatment indicator ('{config.treat_col}') must be time-invariant in panel data.")

        if WEIGHTS_COLUMN in pre_df.columns and WEIGHTS_COLUMN in post_df.columns:
            if not pre_df[WEIGHTS_COLUMN].equals(post_df[WEIGHTS_COLUMN]):
                raise ValueError("Weights must be time-invariant in panel data.")

        return df


class DIDInterColumnSelector(BaseTransformer):
    """DIDInter column selector."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)
        cols_to_keep = [config.yname, config.tname, config.gname, config.dname]

        if config.weightsname:
            cols_to_keep.append(config.weightsname)

        if config.cluster:
            cols_to_keep.append(config.cluster)

        if config.controls:
            cols_to_keep.extend(config.controls)

        if config.trends_nonparam:
            cols_to_keep.extend(config.trends_nonparam)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None and col in df.columns]

        df = df.select(cols_to_keep)

        for col in [config.tname, config.gname, config.dname]:
            if col in df.columns and df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        return df


class SwitcherIdentifier(BaseTransformer):
    """Identify switchers."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)
        df = df.sort([config.gname, config.tname])

        df = df.with_columns((pl.col(config.dname) - pl.col(config.dname).shift(1).over(config.gname)).alias("_d_diff"))

        first_switch = (
            df.filter((pl.col("_d_diff") != 0) & pl.col("_d_diff").is_not_null())
            .group_by(config.gname)
            .agg(pl.col(config.tname).min().alias("F_g"))
        )

        df = df.join(first_switch, on=config.gname, how="left")
        df = df.with_columns(pl.col("F_g").fill_null(float("inf")))

        base_treatment = (
            df.filter(pl.col(config.tname) == pl.col(config.tname).min().over(config.gname))
            .select([config.gname, pl.col(config.dname).alias("d_sq")])
            .unique()
        )
        df = df.join(base_treatment, on=config.gname, how="left")

        switch_direction = (
            df.filter(pl.col("_d_diff").is_not_null() & (pl.col("_d_diff") != 0))
            .group_by(config.gname)
            .agg(pl.col("_d_diff").first().alias("_first_diff"))
        )
        df = df.join(switch_direction, on=config.gname, how="left")

        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf"))
            .then(0)
            .when(pl.col("_first_diff") > 0)
            .then(1)
            .when(pl.col("_first_diff") < 0)
            .then(-1)
            .otherwise(0)
            .alias("S_g")
        )

        df = df.drop(["_d_diff", "_first_diff"])

        return df


class ContinuousTreatmentProcessor(BaseTransformer):
    """Process continuous treatment for DIDInter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        if config.continuous <= 0:
            return to_polars(data)

        df = to_polars(data)
        dname = config.dname
        degree_pol = config.continuous

        df = df.with_columns(
            [
                pl.col(dname).alias(f"{dname}_orig"),
                pl.col("d_sq").alias("d_sq_orig"),
            ]
        )

        for p in range(1, degree_pol + 1):
            df = df.with_columns((pl.col("d_sq") ** p).alias(f"d_sq_{p}"))

        df = df.with_columns(
            pl.when(pl.col("S_g").is_not_null() & (pl.col("S_g") != 0) & (pl.col("F_g") <= pl.col(config.tname)))
            .then(pl.col("S_g").cast(pl.Float64))
            .when(pl.col("S_g").is_null())
            .then(pl.lit(None))
            .otherwise(0.0)
            .alias(f"{dname}_binarized")
        )

        T_max = int(df[config.tname].max())
        time_fe_controls = []

        for t in range(2, T_max + 1):
            for p in range(1, degree_pol + 1):
                fe_col = f"time_fe_{t}_bt{p}"
                df = df.with_columns(((pl.col(config.tname) >= t).cast(pl.Float64) * pl.col(f"d_sq_{p}")).alias(fe_col))
                time_fe_controls.append(fe_col)

        if config.controls is None:
            config.controls = []
        config.controls = list(config.controls) + time_fe_controls

        df = df.with_columns(pl.col(f"{dname}_binarized").alias(dname))
        df = df.drop(f"{dname}_binarized")

        return df


class DIDInterPanelBalancer(BaseTransformer):
    """DIDInter panel balancer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        if config.allow_unbalanced_panel:
            return to_polars(data)

        df = to_polars(data)
        tlist = sorted(df[config.tname].unique().to_list())
        n_periods = len(tlist)

        unit_counts = df.group_by(config.gname).len()
        complete_units = unit_counts.filter(pl.col("len") == n_periods)[config.gname].to_list()

        n_old = df[config.gname].n_unique()
        df = df.filter(pl.col(config.gname).is_in(complete_units))
        n_new = df[config.gname].n_unique()

        if n_new < n_old:
            warnings.warn(f"Dropped {n_old - n_new} units while converting to balanced panel")

        if len(df) == 0:
            raise ValueError(
                "All observations dropped while converting to balanced panel. "
                "Consider setting allow_unbalanced_panel=True."
            )

        return df


class DIDInterConfigUpdater:
    """DIDInter config updater."""

    @staticmethod
    def update(data: DataFrame, config: DIDInterConfig) -> None:
        """Update config."""
        df = to_polars(data)

        tlist = sorted(df[config.tname].unique().to_list())
        n_groups = df[config.gname].n_unique()

        config.time_periods = np.array(tlist)
        config.time_periods_count = len(tlist)
        config.n_groups = n_groups
        config.id_count = n_groups

        T_max = int(df[config.tname].max())
        T_min = int(df[config.tname].min())

        switchers = df.filter(pl.col("F_g") != float("inf"))
        if len(switchers) > 0:
            max_effects = (
                switchers.group_by(config.gname)
                .agg((pl.lit(T_max) - pl.col("F_g") + 1).max().alias("max_exp"))
                .select("max_exp")
                .min()
                .item()
            )
            max_placebo = (
                switchers.group_by(config.gname)
                .agg((pl.col("F_g") - pl.lit(T_min) - 1).max().alias("max_pre"))
                .select("max_pre")
                .min()
                .item()
            )
            config.max_effects_available = int(max_effects) if max_effects is not None else 0
            config.max_placebo_available = int(max_placebo) if max_placebo is not None else 0
        else:
            config.max_effects_available = 0
            config.max_placebo_available = 0

        if config.effects > config.max_effects_available:
            warnings.warn(
                f"Requested effects={config.effects} but only {config.max_effects_available} "
                f"post-treatment periods available. Using effects={config.max_effects_available}.",
                UserWarning,
                stacklevel=4,
            )
            config.effects = config.max_effects_available

        if config.placebo > config.max_placebo_available:
            warnings.warn(
                f"Requested placebo={config.placebo} but only {config.max_placebo_available} "
                f"pre-treatment periods available. Using placebo={config.max_placebo_available}.",
                UserWarning,
                stacklevel=4,
            )
            config.placebo = config.max_placebo_available

        if config.allow_unbalanced_panel:
            unit_counts = df.group_by(config.gname).len()
            is_balanced = (unit_counts["len"] == len(tlist)).all()
            if is_balanced:
                config.data_format = DataFormat.PANEL
            else:
                config.data_format = DataFormat.UNBALANCED_PANEL
        else:
            config.data_format = DataFormat.PANEL


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
                TreatmentEncoder(),
                TimePeriodRecoder(),
                EarlyTreatmentGroupFilter(),
                DoseValidatorTransformer(),
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

    @staticmethod
    def get_didinter_pipeline() -> "DataTransformerPipeline":
        """Get DIDInter pipeline."""
        return DataTransformerPipeline(
            [
                DIDInterColumnSelector(),
                MissingDataHandler(),
                WeightNormalizer(),
                SwitcherIdentifier(),
                ContinuousTreatmentProcessor(),
                DIDInterPanelBalancer(),
                DataSorter(),
            ]
        )

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
        for transformer in self.transformers:
            df = transformer.transform(df, config)

        if isinstance(config, DIDInterConfig):
            DIDInterConfigUpdater.update(df, config)
        else:
            ConfigUpdater.update(df, config)

        return df
