"""Data transformation classes for preprocessing."""

import warnings
from typing import Protocol

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .base import BaseTransformer
from .config import BasePreprocessConfig, ContDIDConfig, DDDConfig, DIDConfig, DIDInterConfig, TwoPeriodDIDConfig
from .constants import (
    NEVER_TREATED_VALUE,
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    ControlGroup,
    DataFormat,
)
from .utils import create_ddd_subgroups, extract_vars_from_formula, make_balanced_panel, validate_subgroup_sizes


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

        if isinstance(config, DIDInterConfig):
            df = df.with_columns(
                [
                    pl.col(config.dname).mean().over(config.gname).alias("_mean_D"),
                    pl.col(config.yname).mean().over(config.gname).alias("_mean_Y"),
                ]
            )
            df = df.filter(pl.col("_mean_D").is_not_null() & pl.col("_mean_Y").is_not_null())
            df = df.drop(["_mean_D", "_mean_Y"])
            return df

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

        weights = np.ones(len(df)) if config.weightsname is None else df[config.weightsname].to_numpy()

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

            n_units = df[config.idname].n_unique() if config.idname else len(df)

            config.time_periods = np.array(tlist)
            config.time_periods_count = len(tlist)
            config.treated_groups = np.array(treat_list)
            config.treated_groups_count = len(treat_list)
            config.id_count = n_units
            return

        tlist = sorted(df[config.tname].unique().to_list())
        glist = sorted(df[config.gname].unique().to_list())

        glist_finite = [g for g in glist if np.isfinite(g)]

        n_units = df[config.idname].n_unique() if config.idname else len(df)

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

        if (
            WEIGHTS_COLUMN in pre_df.columns
            and WEIGHTS_COLUMN in post_df.columns
            and not pre_df[WEIGHTS_COLUMN].equals(post_df[WEIGHTS_COLUMN])
        ):
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

        if config.xformla and config.xformla != "~1":
            covariate_names = extract_vars_from_formula(config.xformla)
            cols_to_keep.extend(covariate_names)

        if config.trends_nonparam:
            cols_to_keep.extend(config.trends_nonparam)

        if config.predict_het:
            cols_to_keep.extend(config.predict_het[0])

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

        base_treatment_pre = (
            df.filter(pl.col(config.tname) == pl.col(config.tname).min().over(config.gname))
            .select([config.gname, pl.col(config.dname).alias("_d_sq_pre")])
            .unique()
        )
        df = df.join(base_treatment_pre, on=config.gname, how="left")
        df = df.with_columns((pl.col(config.dname) - pl.col("_d_sq_pre")).alias("_diff_from_sq"))

        first_switch_pre = (
            df.filter((pl.col("_d_diff") != 0) & pl.col("_d_diff").is_not_null())
            .group_by(config.gname)
            .agg(pl.col(config.tname).min().alias("_F_g_pre"))
        )
        df = df.join(first_switch_pre, on=config.gname, how="left")

        t_max_per_unit = df.group_by(config.gname).agg(pl.col(config.tname).max().alias("_T_max_unit"))
        df = df.join(t_max_per_unit, on=config.gname, how="left")

        df = df.with_columns(
            pl.when(pl.col("_F_g_pre").is_not_null())
            .then(pl.col("_T_max_unit") - pl.col("_F_g_pre") + 1)
            .otherwise(pl.lit(0.0))
            .alias("L_g")
        )
        df = df.drop(["_F_g_pre", "_T_max_unit"])

        df = df.with_columns(
            [
                pl.when((pl.col("_diff_from_sq") > 0) & pl.col(config.dname).is_not_null())
                .then(1)
                .otherwise(0)
                .cum_sum()
                .clip(upper_bound=1)
                .over(config.gname)
                .alias("_ever_strict_increase"),
                pl.when((pl.col("_diff_from_sq") < 0) & pl.col(config.dname).is_not_null())
                .then(1)
                .otherwise(0)
                .cum_sum()
                .clip(upper_bound=1)
                .over(config.gname)
                .alias("_ever_strict_decrease"),
            ]
        )

        if not config.keep_bidirectional_switchers:
            df = df.filter(~((pl.col("_ever_strict_increase") == 1) & (pl.col("_ever_strict_decrease") == 1)))

        df = df.drop(["_ever_strict_increase", "_ever_strict_decrease", "_d_sq_pre", "_diff_from_sq"])

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

        switch_treatment = (
            df.filter(pl.col(config.tname) == pl.col("F_g"))
            .select([config.gname, pl.col(config.dname).alias("d_fg")])
            .unique()
        )
        df = df.join(switch_treatment, on=config.gname, how="left")

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

        if config.drop_missing_preswitch:
            min_treat_time = (
                df.filter(pl.col(config.dname).is_not_null())
                .group_by(config.gname)
                .agg(pl.col(config.tname).min().alias("_min_treat_time"))
            )
            df = df.join(min_treat_time, on=config.gname, how="left")

            df = df.filter(
                ~(
                    (pl.col("_min_treat_time") < pl.col("F_g"))
                    & (pl.col(config.tname) >= pl.col("_min_treat_time"))
                    & (pl.col(config.tname) < pl.col("F_g"))
                    & pl.col(config.dname).is_null()
                )
            )
            df = df.drop("_min_treat_time")

        df = df.drop(["_d_diff", "_first_diff"])

        return df


class SwitcherFilter(BaseTransformer):
    """Filter units based on switchers parameter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        if config.switchers == "in":
            valid_units = df.filter((pl.col("S_g") == 1) | (pl.col("S_g") == 0))[config.gname].unique().to_list()
            df = df.filter(pl.col(config.gname).is_in(valid_units))
        elif config.switchers == "out":
            valid_units = df.filter((pl.col("S_g") == -1) | (pl.col("S_g") == 0))[config.gname].unique().to_list()
            df = df.filter(pl.col(config.gname).is_in(valid_units))

        return df


class FgVariationFilter(BaseTransformer):
    """Filter out baseline treatment groups with no variation in F_g."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        group_cols = ["d_sq"]
        if config.trends_nonparam:
            group_cols.extend(config.trends_nonparam)

        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf")).then(0).otherwise(pl.col("F_g")).alias("_F_g_for_std")
        )

        df = df.with_columns(pl.col("_F_g_for_std").std().over(group_cols).round(3).alias("_var_F_g"))

        df = df.filter(pl.col("_var_F_g") > 0)
        df = df.drop(["_var_F_g", "_F_g_for_std"])

        return df


class ControlsTimeFilter(BaseTransformer):
    """Filter to cells with at least one never-switcher as control."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        df = df.with_columns(pl.when(pl.col("F_g") == float("inf")).then(1).otherwise(0).alias("_never_change_d"))

        ctrl_group = [config.tname, "d_sq"]
        if config.trends_nonparam:
            ctrl_group.extend(config.trends_nonparam)

        df = df.with_columns(pl.col("_never_change_d").max().over(ctrl_group).alias("_controls_time"))

        df = df.filter(pl.col("_controls_time") > 0)
        df = df.drop(["_never_change_d", "_controls_time"])

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
        degree_pol = config.continuous

        df = df.with_columns(pl.col("d_sq").alias("d_sq_orig"))

        for p in range(1, degree_pol + 1):
            df = df.with_columns((pl.col("d_sq_orig") ** p).alias(f"d_sq_{p}"))

        df = df.with_columns(pl.lit(0).alias("d_sq"))

        mapping_df = (
            df.select("d_sq")
            .filter(pl.col("d_sq").is_not_null())
            .unique()
            .sort("d_sq")
            .with_row_index("d_sq_int", offset=1)
            .select(["d_sq", "d_sq_int"])
        )
        df = df.join(mapping_df, on="d_sq", how="left")

        return df


class DIDInterPanelBalancer(BaseTransformer):
    """DIDInter panel balancer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        groups = df.select(config.gname).unique()
        times = df.select(config.tname).unique()

        full_index = groups.join(times, how="cross")

        df = full_index.join(df, on=[config.gname, config.tname], how="left")

        time_invariant_cols = ["F_g", "d_sq", "S_g", "L_g"]
        for col in time_invariant_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).mean().over(config.gname).alias(col))

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
                .max()
                .item()
            )
            max_placebo = (
                switchers.group_by(config.gname)
                .agg((pl.col("F_g") - pl.lit(T_min) - 1).max().alias("max_pre"))
                .select("max_pre")
                .max()
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


class TrendsLinTransformer(BaseTransformer):
    """Apply first-differencing transformation for linear trends."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        if not config.trends_lin:
            return df

        t_min = df[config.tname].min()
        df = df.filter(pl.col("F_g") != t_min + 1)
        df = df.sort([config.gname, config.tname])

        df = df.with_columns(
            (pl.col(config.yname) - pl.col(config.yname).shift(1).over(config.gname)).alias(config.yname)
        )

        if config.xformla and config.xformla != "~1":
            covariate_names = extract_vars_from_formula(config.xformla)
            for ctrl in covariate_names:
                df = df.with_columns((pl.col(ctrl) - pl.col(ctrl).shift(1).over(config.gname)).alias(ctrl))

        df = df.filter(pl.col(config.tname) != t_min)

        return df


class DIDInterDataPreparer(BaseTransformer):
    """Prepare data for DIDInter computation."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)
        gname = config.gname
        tname = config.tname
        yname = config.yname
        dname = config.dname

        df = df.sort([gname, tname])
        df = df.with_columns(pl.lit(1.0).alias("weight_gt"))

        if config.weightsname:
            df = df.with_columns(pl.col(config.weightsname).fill_null(0).alias("weight_gt"))

        df = df.with_columns(
            pl.when(pl.col(yname).is_null() | pl.col(dname).is_null())
            .then(0.0)
            .otherwise(pl.col("weight_gt"))
            .alias("weight_gt")
        )

        first_obs = df.group_by(gname).agg(pl.col(tname).min().alias("_first_t")).select([gname, "_first_t"])
        df = df.join(first_obs, on=gname, how="left")
        df = df.with_columns((pl.col(tname) == pl.col("_first_t")).cast(pl.Int64).alias("first_obs_by_gp"))
        df = df.drop("_first_t")

        t_max_by_group = df.group_by(gname).agg(pl.col(tname).max().alias("t_max_by_group"))
        df = df.join(t_max_by_group, on=gname, how="left")

        group_cols = ["d_sq"]
        if config.trends_nonparam:
            group_cols.extend(config.trends_nonparam)

        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf"))
            .then(pl.col("t_max_by_group") + 1)
            .otherwise(pl.col("F_g"))
            .alias("_F_g_trunc")
        )
        df = df.with_columns((pl.col("_F_g_trunc").max().over(group_cols) - 1).alias("T_g"))
        df = df.drop("_F_g_trunc")

        return df


class DDDColumnSelector(BaseTransformer):
    """DDD column selector."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Select relevant columns for DDD preprocessing."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)
        cols_to_keep = [config.yname, config.tname, config.idname, config.gname, config.pname]

        if config.cluster:
            cols_to_keep.append(config.cluster)

        if config.weightsname:
            cols_to_keep.append(config.weightsname)

        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            cols_to_keep.extend(formula_vars)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None and col in df.columns]

        return df.select(cols_to_keep)


class DDDWeightProcessor(BaseTransformer):
    """DDD weight processor."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Extract/create weights, validate, normalize, add as WEIGHTS_COLUMN."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)

        if config.weightsname is not None:
            weights = df[config.weightsname].to_numpy().astype(float)
            if np.any(np.isnan(weights)):
                raise ValueError("Missing values in weights column.")
            weights_per_id = df.group_by(config.idname).agg(pl.col(config.weightsname).n_unique().alias("n_unique"))
            if (weights_per_id["n_unique"] > 1).any():
                raise ValueError("Weights must be the same across all periods for each unit.")
        else:
            weights = np.ones(len(df))

        weights = weights / np.mean(weights)
        return df.with_columns(pl.Series(name=WEIGHTS_COLUMN, values=weights))


class DDDPanelBalancer(BaseTransformer):
    """DDD panel balancer."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Sort and balance the panel."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)
        df = df.sort([config.idname, config.tname])
        df = make_balanced_panel(df, config.idname, config.tname)

        if len(df) == 0:
            raise ValueError("No observations remain after creating balanced panel.")

        return df


class DDDPostIndicatorCreator(BaseTransformer):
    """DDD post indicator creator."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Create _post column indicating post-treatment period."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)
        tlist = np.sort(df[config.tname].unique().to_numpy())
        return df.with_columns((pl.col(config.tname) == tlist[1]).cast(pl.Int64).alias("_post"))


class DDDCovariateInvarianceChecker(BaseTransformer):
    """DDD covariate invariance checker."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Check that covariates are time-invariant."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)

        if config.xformla == "~1":
            return df

        covariate_vars = extract_vars_from_formula(config.xformla)

        for var in covariate_vars:
            if var not in df.columns:
                continue
            var_per_id = df.group_by(config.idname).agg(pl.col(var).n_unique().alias("n_unique"))
            if (var_per_id["n_unique"] > 1).any():
                raise ValueError(f"Covariate '{var}' varies over time. Covariates must be time-invariant.")

        return df


class DDDSubgroupCreator(BaseTransformer):
    """DDD subgroup creator."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Assign 4-group subgroups and validate sizes."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)
        glist = np.sort(df[config.gname].unique().to_numpy())
        treat_val = glist[1]

        subgroup = create_ddd_subgroups(df[config.gname].to_numpy(), df[config.pname].to_numpy(), treat_val)
        df = df.with_columns(pl.Series("_subgroup", subgroup))

        counts_df = df.group_by("_subgroup").agg(pl.col(config.idname).n_unique().alias("count"))
        subgroup_counts = {int(row["_subgroup"]): int(row["count"]) for row in counts_df.iter_rows(named=True)}
        validate_subgroup_sizes(subgroup_counts)

        return df


class DDDConfigUpdater:
    """DDD config updater."""

    @staticmethod
    def update(data: DataFrame, config: DDDConfig) -> None:
        """Update DDD config with computed values."""
        df = to_polars(data)

        tlist = np.sort(df[config.tname].unique().to_numpy())
        config.time_periods = tlist
        config.time_periods_count = len(tlist)

        n_units = df.filter(pl.col("_post") == 0).height
        config.n_units = n_units


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
                SwitcherFilter(),
                FgVariationFilter(),
                ControlsTimeFilter(),
                ContinuousTreatmentProcessor(),
                DIDInterPanelBalancer(),
                TrendsLinTransformer(),
                DIDInterDataPreparer(),
                DataSorter(),
            ]
        )

    @staticmethod
    def get_ddd_pipeline() -> "DataTransformerPipeline":
        """Get DDD pipeline."""
        return DataTransformerPipeline(
            [
                DDDColumnSelector(),
                MissingDataHandler(),
                DDDWeightProcessor(),
                DDDPanelBalancer(),
                DDDPostIndicatorCreator(),
                DDDCovariateInvarianceChecker(),
                DDDSubgroupCreator(),
            ]
        )

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
        for transformer in self.transformers:
            df = transformer.transform(df, config)

        if isinstance(config, DDDConfig):
            DDDConfigUpdater.update(df, config)
        elif isinstance(config, DIDInterConfig):
            DIDInterConfigUpdater.update(df, config)
        else:
            ConfigUpdater.update(df, config)

        return df
