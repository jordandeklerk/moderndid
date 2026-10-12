"""Data transformation classes for preprocessing."""

import warnings
from typing import Protocol

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .base import BaseTransformer
from .config import (
    BasePreprocessConfig,
    ContDIDConfig,
    DDDConfig,
    DIDConfig,
    DIDInterConfig,
    DynBalancingConfig,
    EtwfeConfig,
    TwoPeriodDIDConfig,
)
from .constants import (
    NEVER_TREATED_VALUE,
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    ControlGroup,
    DataFormat,
)
from .utils import (
    create_ddd_subgroups,
    extract_vars_from_formula,
    get_formula_columns,
    get_transformed_terms,
    make_balanced_panel,
    nonfinite_to_null,
    validate_subgroup_sizes,
)
from .validators import _ddd_subgroup_error, _missing_column_errors, _weights_error

try:
    import formulaic
except ImportError:
    formulaic = None


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
    """Drop the rows that have missing values.

    A null, a NaN, and an infinity all count as missing. The same data
    therefore loses the same rows whether polars or pandas holds it. Since an
    infinity marks never-treated units in the cohort column, it stays there.
    The intertemporal estimator keeps rows with a missing outcome or
    treatment and drops the rows and groups that :meth:`drop_didinter_rows`
    describes. The dynamic balancing estimator drops only the rows that
    :meth:`drop_dyn_balancing_rows` describes.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if isinstance(config, DynBalancingConfig):
            return self.drop_dyn_balancing_rows(data, config)

        # The intertemporal estimator's gname names groups. Every other gname holds cohorts.
        cohort = getattr(config, "gname", None)
        keep_infinite = [] if isinstance(config, DIDInterConfig) or cohort is None else [cohort]
        df = nonfinite_to_null(data, keep_infinite=keep_infinite)

        if isinstance(config, DIDInterConfig):
            df, messages = self.drop_didinter_rows(df, config)
            for message in messages:
                warnings.warn(message)
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
            if n_new == 0:
                columns = [f"'{name}'" for name in df.columns if df[name].null_count() > 0]
                where = columns[0] if len(columns) == 1 else f"at least one of the columns {', '.join(columns)}"
                raise ValueError(f"Every row has a missing value in {where}. No data is left to estimate from.")
            warnings.warn(f"Dropped {n_orig - n_new} rows from original data due to missing values")

        return data_clean

    @staticmethod
    def drop_didinter_rows(data, config):
        """Drop the rows and groups that the intertemporal estimator cannot use.

        Rows missing a control have no adjusted outcome. Since the clustered variance needs the
        cluster of every row, rows missing the cluster leave as well. Both kinds of rows leave before
        baselines and switch dates are set. Groups whose treatment or outcome is missing in every
        remaining row leave next.

        Parameters
        ----------
        data : DataFrame
            Panel before preprocessing.
        config : DIDInterConfig
            Configuration that names the controls and the cluster.

        Returns
        -------
        df : polars.DataFrame
            The panel without those rows and groups.
        messages : list of str
            One warning message for each kind of row dropped. The list is empty when no row is dropped.
        """
        df = to_polars(data)
        messages = []
        if config.xformla and config.xformla != "~1":
            n_orig = len(df)
            df = df.drop_nulls(subset=extract_vars_from_formula(config.xformla))
            if len(df) < n_orig:
                messages.append(f"Dropped {n_orig - len(df)} rows from original data due to missing covariates")
        if config.cluster:
            n_orig = len(df)
            df = df.drop_nulls(subset=config.cluster)
            if len(df) < n_orig:
                messages.append(
                    f"Dropped {n_orig - len(df)} rows from original data due to a missing cluster in '{config.cluster}'"
                )
        df = df.with_columns(
            [
                pl.col(config.dname).mean().over(config.gname).alias(".mean_D"),
                pl.col(config.yname).mean().over(config.gname).alias(".mean_Y"),
            ]
        )
        df = df.filter(pl.col(".mean_D").is_not_null() & pl.col(".mean_Y").is_not_null())
        return df.drop([".mean_D", ".mean_Y"]), messages

    @staticmethod
    def drop_dyn_balancing_rows(data, config):
        """Drop the rows that miss the unit, the period, or the cluster.

        A null, a NaN, and an infinity all count as missing. Every later step
        reads the unit and the period, and the variance reads the cluster. A
        unit that loses a period this way leaves at the history check.

        Missing outcomes, treatments, and covariates stay for the later steps
        to handle. An infinite outcome, treatment, or covariate counts as
        missing and becomes a NaN.

        Parameters
        ----------
        data : DataFrame
            Panel before preprocessing.
        config : DynBalancingConfig
            Configuration that names the unit, period, and cluster columns.

        Returns
        -------
        pl.DataFrame
            The panel without those rows.
        """
        df = to_polars(data)
        key_columns = list(dict.fromkeys([config.idname, config.tname, *(config.clustervars or [])]))
        keys = nonfinite_to_null(df.select(key_columns))
        missing = keys.select(pl.any_horizontal(pl.all().is_null())).to_series()
        n_missing = int(missing.sum())

        if n_missing > 0:
            if n_missing == len(df):
                quoted = [f"'{name}'" for name in key_columns]
                where = " or ".join(quoted) if len(quoted) < 3 else f"{', '.join(quoted[:-1])}, or {quoted[-1]}"
                raise ValueError(f"Every row has a missing value in {where}. No data is left to estimate from.")
            warnings.warn(f"Dropped {n_missing} rows from original data due to missing values")
            df = df.filter(~missing)

        # Since the later steps already read a NaN as missing, an infinity becomes a NaN.
        float_columns = [name for name, dtype in df.schema.items() if dtype.is_float() and name not in key_columns]
        return df.with_columns(
            pl.when(pl.col(name).is_infinite()).then(float("nan")).otherwise(pl.col(name)).alias(name)
            for name in float_columns
        )


class WeightNormalizer(BaseTransformer):
    """Check the sampling weights and divide them by their mean.

    The weights must be non-negative with a positive mean. Since the step runs
    after the missing-data step, the check covers only the rows that stay.
    Without ``weightsname`` every row gets weight 1.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        if config.weightsname is not None:
            error = _weights_error(df[config.weightsname], config.weightsname)
            if error is not None:
                raise ValueError(error)

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
    """Code the cohort of never-treated units as infinity.

    A cohort of 0 or infinity marks never-treated units. So does a start
    after the last period. For att_gt, a cohort that starts at most
    ``anticipation`` periods after the last period stays treated. Its units
    already react inside the panel. A negative cohort raises an error.

    Parameters
    ----------
    cohort_argument : str, default "gname"
        Name that the error gives the argument of the cohort column.
    """

    def __init__(self, cohort_argument="gname"):
        self.cohort_argument = cohort_argument

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)
        cohort = pl.col(config.gname)
        df = df.with_columns(cohort.cast(pl.Float64))

        negative = df.filter(cohort < 0)[config.gname]
        if len(negative) > 0:
            raise ValueError(
                f"{self.cohort_argument} = '{config.gname}' holds negative values such as {negative.min():g}. "
                "It must hold 0 for never-treated units and the first treated period for the others. "
                "Since 0 marks never-treated units, shift the periods so that the earliest one is positive."
            )

        last_start = df[config.tname].max()
        # Since cont_did places every cohort at an observed period, a start after the panel stays never-treated there.
        if isinstance(config, DIDConfig):
            last_start += config.anticipation

        return df.with_columns(
            pl.when((cohort == 0) | (cohort > last_start))
            .then(pl.lit(NEVER_TREATED_VALUE))
            .otherwise(cohort)
            .alias(config.gname)
        )


class EarlyTreatmentFilter(BaseTransformer):
    """Early treatment filter."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        tlist = sorted(df[config.tname].unique().to_list())
        first_period = min(tlist)

        # Because the triple difference config has no anticipation, its units react from their first treated period.
        treated_early_mask = pl.col(config.gname) <= first_period + getattr(config, "anticipation", 0)

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
    """Keep the periods that have comparison units when no unit is never treated.

    From the start of the latest cohort's treatment, less anticipation, no unit
    is untreated. Those periods leave the data. With never-treated controls, the
    latest cohort becomes the never-treated group. With not-yet-treated controls,
    it stays in the data as a comparison group that :class:`ConfigUpdater` leaves
    out of the treated groups.
    """

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
    """Panel balancer.

    Parameters
    ----------
    option : str, default "panel=False"
        Option that the error suggests when no unit is observed in every period.
    id_argument : str, default "idname"
        Name that the error gives the argument of the unit column.
    """

    def __init__(self, option="panel=False", id_argument="idname"):
        self.option = option
        self.id_argument = id_argument

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
                f"Consider setting {self.option} and/or revisiting '{self.id_argument}'"
            )

        return df


class RepeatedCrossSectionHandler(BaseTransformer):
    """Repeated cross section handler."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        df = to_polars(data)

        # Since panel=False asks for the repeated cross section estimators, every row is its own observation,
        # even when idname names a unit that several rows share.
        if not config.panel:
            config.true_repeated_cross_sections = True
            config.idname = ROW_ID_COLUMN
            return df.with_row_index(name=ROW_ID_COLUMN)

        if config.allow_unbalanced_panel and config.idname:
            unit_counts = df.group_by(config.idname).len()
            # Since an unbalanced panel has no outcome tensor, it takes the repeated cross section estimators.
            # They still sum each unit's influence function through the row id.
            if (unit_counts["len"] != df[config.tname].n_unique()).any():
                config.panel = False
                return df.with_columns(pl.col(config.idname).alias(ROW_ID_COLUMN))

        return df


class TimePeriodRecoder(BaseTransformer):
    """Recode the time and group columns to period positions."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return to_polars(data)

        df = to_polars(data)
        original_periods = sorted(df[config.tname].unique().to_list())
        time_map = {t: i + 1 for i, t in enumerate(original_periods)}

        # Since groups are compared with periods, they take the same positions. A start between two observed
        # periods has no position. Moving it to a neighboring period would shift the group's event times.
        groups = df[config.gname].to_numpy().astype(float)
        finite = np.isfinite(groups)
        between = finite & (groups > original_periods[0]) & ~np.isin(groups, original_periods)
        if np.any(between):
            between_groups = np.unique(groups[between])
            starts = ", ".join(f"{g:g}" for g in between_groups)
            which = f"group {starts}" if len(between_groups) == 1 else f"groups {starts}"
            raise ValueError(
                f"Treatment starts between observed periods for {which}. Each group must be an observed period "
                "or 0. Recode such a group to the observed period from which its effects should count, such as "
                "the first period that observes it as treated."
            )
        groups[finite] = np.searchsorted(original_periods, groups[finite], side="left") + 1

        df = df.with_columns(
            pl.col(config.tname).replace(time_map).alias(config.tname),
            pl.Series(config.gname, groups),
        )
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
            period_labels = {index: period for period, index in (config.time_map or {}).items()}
            warnings.warn(
                f"Dropped {len(groups_to_drop)} groups treated before period "
                f"{period_labels.get(min_valid_group, min_valid_group)} "
                f"(required_pre_periods={config.required_pre_periods}, anticipation={config.anticipation})"
            )
            df = df.filter(~pl.col(config.gname).is_in(groups_to_drop))

        return df


class ContDIDControlGroupFilter(BaseTransformer):
    """Keep the periods that have untreated comparison units."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, ContDIDConfig):
            return to_polars(data)

        df = to_polars(data)
        groups = df[config.gname]
        finite_groups = groups.filter(groups.is_finite())
        if groups.is_infinite().any() or finite_groups.is_empty():
            return df

        if config.control_group == ControlGroup.NEVER_TREATED:
            raise ValueError(
                "control_group='nevertreated' needs never-treated units. The data has none. "
                "Use control_group='notyettreated' instead."
            )

        # Without never-treated units, nobody is untreated from the start of the last cohort's
        # treatment (less anticipation) on.
        cutoff = finite_groups.max() - config.anticipation
        kept = df.filter(pl.col(config.tname) < cutoff)
        n_kept_periods = kept[config.tname].n_unique()
        if n_kept_periods < 2:
            raise ValueError(
                "The data has no never-treated units. Since fewer than two periods come before the last "
                "cohort starts treatment, no period has untreated units to compare with."
            )

        period_labels = {position: period for period, position in (config.time_map or {}).items()}
        # Without a kept period in which some cohort is treated, no effect is left to estimate.
        if not (kept[config.gname] <= kept[config.tname].max()).any():
            raise ValueError(
                "The data has no never-treated units. Since no cohort starts treatment before period "
                f"{period_labels.get(cutoff, cutoff)}, no treated period has untreated units to compare with."
            )

        warnings.warn(
            "The data has no never-treated units. Since no unit is untreated from period "
            f"{period_labels.get(cutoff, cutoff)} on, those periods were dropped."
        )
        return kept


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

        if not config.idname:
            return df

        # Each cell reads a unit's dose from a single period, even a cell before treatment starts. A dose
        # recorded as 0 until then takes its value at the start of treatment.
        start_dose = (
            df.filter(pl.col(config.gname).is_finite() & (pl.col(config.tname) >= pl.col(config.gname)))
            .group_by(config.idname)
            .agg(pl.col(config.dname).sort_by(config.tname).first().alias(".start_dose"))
        )
        df = (
            df.join(start_dose, on=config.idname, how="left")
            .with_columns(
                pl.when((pl.col(config.tname) < pl.col(config.gname)) & (pl.col(config.dname) == 0))
                .then(pl.col(".start_dose").fill_null(0))
                .otherwise(pl.col(config.dname))
                .alias(config.dname)
            )
            .drop(".start_dose")
        )

        # A dose that moved over time would give the cells different treatments.
        dose_range = df.group_by(config.idname).agg(
            (pl.col(config.dname).max() - pl.col(config.dname).min()).alias("range"),
            pl.col(config.dname).abs().max().alias("scale"),
        )
        n_varying = dose_range.filter(pl.col("range") > 1e-8 * (1 + pl.col("scale"))).height
        if n_varying > 0:
            raise ValueError(
                f"Each unit's dose in '{config.dname}' must stay the same over time. Before treatment starts, "
                f"the dose may also be recorded as 0. The dose changes over time for {n_varying} units."
            )

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
        # Without never-treated units, no unit is untreated once the latest cohort starts treatment.
        # That cohort is therefore only a comparison group and gets no cells of its own.
        no_never_treated = len(glist_finite) == len(glist)
        if isinstance(config, DIDConfig) and config.control_group == ControlGroup.NOT_YET_TREATED and no_never_treated:
            glist_finite = glist_finite[:-1]

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

        # With two periods there is one cell. A dose-response band still spans the dose grid.
        dose_band = isinstance(config, ContDIDConfig) and config.aggregation == "dose"
        if len(tlist) == 2 and not dose_band:
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
            if get_transformed_terms(config.xformla):
                # Since PrePostCovariateProcessor evaluates transformed terms itself, only the columns they read stay.
                formula_vars = get_formula_columns(config.xformla, df.columns)
            else:
                formula_vars = extract_vars_from_formula(config.xformla)
                missing = _missing_column_errors(df.columns, {"xformla": formula_vars})
                if missing:
                    raise ValueError("\n".join(missing))
            formula_vars = [v for v in formula_vars if v != config.yname]
            cols_to_keep.extend(formula_vars)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None]

        return df.select(cols_to_keep)


class PrePostCovariateProcessor(BaseTransformer):
    """Build the intercept and covariate columns of the two-period estimators.

    A formula of numeric or boolean columns gives an intercept plus those
    columns, as for every other estimator. The formulaic package evaluates
    any other formula, such as one with a transformed term like ``I(x**2)``
    or with a column of strings or categories. Since formulaic comes with the
    optional extras only, such a formula raises an error on a base install.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return to_polars(data)

        df = to_polars(data)

        if not config.xformla or config.xformla == "~1":
            return df.with_columns(pl.lit(1.0).alias("Intercept"))

        transformed = get_transformed_terms(config.xformla)
        if transformed:
            return self._evaluate_formula(
                df,
                config.xformla,
                f"xformla term '{transformed[0]}' requires formulaic. Install it with uv add formulaic or "
                "pip install formulaic. You can also add the transformed covariate to the data as its own column.",
            )

        # Since the builder never reads the design columns as covariates, they keep their own types.
        design = {config.yname, config.tname, config.treat_col, config.idname, config.weightsname}
        names = [name for name in extract_vars_from_formula(config.xformla) if name not in design]
        categorical = [name for name in names if not (df.schema[name].is_numeric() or df.schema[name] == pl.Boolean)]
        if categorical:
            return self._evaluate_formula(
                df,
                config.xformla,
                f"xformla column '{categorical[0]}' holds strings or categories. Expanding it into indicator "
                "columns requires formulaic. Install it with uv add formulaic or pip install formulaic. You can "
                "also add the indicator columns to the data yourself.",
            )

        others = [col for col in df.columns if col not in names]
        return df.select(*others, pl.lit(1.0).alias("Intercept"), *(pl.col(name).cast(pl.Float64) for name in names))

    @staticmethod
    def _evaluate_formula(df, formula, install_hint):
        """Build the intercept and covariate columns with formulaic."""
        if formulaic is None:
            raise ImportError(install_hint)

        try:
            model_matrix_result = formulaic.model_matrix(formula, df)
            covariates_pl = model_matrix_result.__wrapped__

            if hasattr(model_matrix_result, "model_spec") and model_matrix_result.model_spec:
                original_cov_names = [var for var in model_matrix_result.model_spec.variables if var != "1"]
            else:
                original_cov_names = []
                warnings.warn("Could not retrieve model_spec from formulaic output.", UserWarning)

        except Exception as e:
            raise ValueError(f"Error processing covariates_formula '{formula}' with formulaic: {e}") from e

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


class DIDInterTimeRanker(BaseTransformer):
    """Replace each period by its rank among the observed periods.

    The estimator moves through periods one step at a time, for example from the period before a
    group's first switch to the periods after it. Ranking puts consecutive observed periods one step
    apart however the periods are coded. The sorted original periods go to ``config.time_periods``.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)
        config.time_periods = np.sort(df[config.tname].drop_nulls().unique().to_numpy())
        return df.with_columns(pl.col(config.tname).rank("dense").cast(pl.Int64))


class SwitcherIdentifier(BaseTransformer):
    """Find the baseline treatment and the first switch of each group.

    The step adds the baseline treatment ``d_sq``, its dense rank ``d_sq_int``, the first switch
    period ``F_g``, the treatment ``d_fg`` in that period, the switch direction ``S_g``, and the
    number of periods ``L_g`` from the first switch through the group's last period. Control pools
    group on ``d_sq_int`` because integer ranks compare exactly.

    The joins keep the row order of the data. The flags of bidirectional switchers and the
    direction of the first switch read each group's periods in time order.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)
        df = df.sort([config.gname, config.tname])

        df = df.with_columns((pl.col(config.dname) - pl.col(config.dname).shift(1).over(config.gname)).alias(".d_diff"))

        base_treatment_pre = (
            df.filter(pl.col(config.tname) == pl.col(config.tname).min().over(config.gname))
            .select([config.gname, pl.col(config.dname).alias(".d_sq_pre")])
            .unique()
        )
        df = df.join(base_treatment_pre, on=config.gname, how="left", maintain_order="left")
        df = df.with_columns((pl.col(config.dname) - pl.col(".d_sq_pre")).alias(".diff_from_sq"))

        first_switch_pre = (
            df.filter((pl.col(".d_diff") != 0) & pl.col(".d_diff").is_not_null())
            .group_by(config.gname)
            .agg(pl.col(config.tname).min().alias(".F_g_pre"))
        )
        df = df.join(first_switch_pre, on=config.gname, how="left", maintain_order="left")

        t_max_per_unit = df.group_by(config.gname).agg(pl.col(config.tname).max().alias(".T_max_unit"))
        df = df.join(t_max_per_unit, on=config.gname, how="left", maintain_order="left")

        df = df.with_columns(
            pl.when(pl.col(".F_g_pre").is_not_null())
            .then(pl.col(".T_max_unit") - pl.col(".F_g_pre") + 1)
            .otherwise(pl.lit(0.0))
            .alias("L_g")
        )
        df = df.drop([".F_g_pre", ".T_max_unit"])

        df = df.with_columns(
            [
                pl.when((pl.col(".diff_from_sq") > 0) & pl.col(config.dname).is_not_null())
                .then(1)
                .otherwise(0)
                .cum_sum()
                .clip(upper_bound=1)
                .over(config.gname)
                .alias(".ever_strict_increase"),
                pl.when((pl.col(".diff_from_sq") < 0) & pl.col(config.dname).is_not_null())
                .then(1)
                .otherwise(0)
                .cum_sum()
                .clip(upper_bound=1)
                .over(config.gname)
                .alias(".ever_strict_decrease"),
            ]
        )

        if not config.keep_bidirectional_switchers:
            df = df.filter(~((pl.col(".ever_strict_increase") == 1) & (pl.col(".ever_strict_decrease") == 1)))

        df = df.drop([".ever_strict_increase", ".ever_strict_decrease", ".d_sq_pre", ".diff_from_sq"])

        first_switch = (
            df.filter((pl.col(".d_diff") != 0) & pl.col(".d_diff").is_not_null())
            .group_by(config.gname)
            .agg(pl.col(config.tname).min().alias("F_g"))
        )

        df = df.join(first_switch, on=config.gname, how="left", maintain_order="left")
        df = df.with_columns(pl.col("F_g").fill_null(float("inf")))

        base_treatment = (
            df.filter(pl.col(config.tname) == pl.col(config.tname).min().over(config.gname))
            .select([config.gname, pl.col(config.dname).alias("d_sq")])
            .unique()
        )
        df = df.join(base_treatment, on=config.gname, how="left", maintain_order="left")

        switch_treatment = (
            df.filter(pl.col(config.tname) == pl.col("F_g"))
            .select([config.gname, pl.col(config.dname).alias("d_fg")])
            .unique()
        )
        df = df.join(switch_treatment, on=config.gname, how="left", maintain_order="left")

        switch_direction = (
            df.filter(pl.col(".d_diff").is_not_null() & (pl.col(".d_diff") != 0))
            .group_by(config.gname)
            .agg(pl.col(".d_diff").first().alias(".first_diff"))
        )
        df = df.join(switch_direction, on=config.gname, how="left", maintain_order="left")

        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf"))
            .then(0)
            .when(pl.col(".first_diff") > 0)
            .then(1)
            .when(pl.col(".first_diff") < 0)
            .then(-1)
            .otherwise(0)
            .alias("S_g")
        )

        if config.drop_missing_preswitch:
            min_treat_time = (
                df.filter(pl.col(config.dname).is_not_null())
                .group_by(config.gname)
                .agg(pl.col(config.tname).min().alias(".min_treat_time"))
            )
            df = df.join(min_treat_time, on=config.gname, how="left", maintain_order="left")

            df = df.filter(
                ~(
                    (pl.col(".min_treat_time") < pl.col("F_g"))
                    & (pl.col(config.tname) >= pl.col(".min_treat_time"))
                    & (pl.col(config.tname) < pl.col("F_g"))
                    & pl.col(config.dname).is_null()
                )
            )
            df = df.drop(".min_treat_time")

        df = df.drop([".d_diff", ".first_diff"])

        return df.with_columns(pl.col("d_sq").rank("dense").cast(pl.Int64).alias("d_sq_int"))


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
            pl.when(pl.col("F_g") == float("inf")).then(0).otherwise(pl.col("F_g")).alias(".F_g_for_std")
        )

        df = df.with_columns(pl.col(".F_g_for_std").std().over(group_cols).round(3).alias(".var_F_g"))

        df = df.filter(pl.col(".var_F_g") > 0)
        df = df.drop([".var_F_g", ".F_g_for_std"])

        return df


class ControlsTimeFilter(BaseTransformer):
    """Keep the periods in which a baseline treatment still has a group that has not switched."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        # A group is a control until its first switch. A period without a never-switcher still has
        # controls when some group with the same baseline treatment switches later.
        df = df.with_columns((pl.col("F_g") > pl.col(config.tname)).cast(pl.Int64).alias(".not_yet_switched"))

        ctrl_group = [config.tname, "d_sq"]
        if config.trends_nonparam:
            ctrl_group.extend(config.trends_nonparam)

        df = df.with_columns(pl.col(".not_yet_switched").max().over(ctrl_group).alias(".controls_time"))

        df = df.filter(pl.col(".controls_time") > 0)
        df = df.drop([".not_yet_switched", ".controls_time"])

        return df


class ContinuousTreatmentProcessor(BaseTransformer):
    """Pool the groups of a continuous baseline treatment into one baseline.

    Since few groups share a value of a continuous baseline treatment, groups with the same
    baseline cannot serve as each other's controls. With ``continuous`` set to a degree :math:`p`,
    the baseline goes to ``.d_sq_orig`` and its powers up to :math:`p` go to ``.d_sq_1``, ...,
    ``.d_sq_p``. Setting ``d_sq`` to 0 and its rank ``d_sq_int`` to 1 for every group lets all
    groups compare with each other. :class:`ContinuousTreatmentBinarizer` later adds the controls
    that let the outcome trends depend on the baseline.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig) or config.continuous <= 0:
            return to_polars(data)

        df = to_polars(data).with_columns(pl.col("d_sq").cast(pl.Float64).alias(".d_sq_orig"))
        df = df.with_columns(
            (pl.col(".d_sq_orig") ** power).alias(f".d_sq_{power}") for power in range(1, config.continuous + 1)
        )
        return df.with_columns(pl.lit(0.0).alias("d_sq"), pl.lit(1, dtype=pl.Int64).alias("d_sq_int"))


class ContinuousTreatmentBinarizer(BaseTransformer):
    """Turn a continuous treatment into the signed indicator of having switched.

    The original treatment goes to ``.d_orig``. The treatment becomes ``S_g`` from ``F_g`` on
    and 0 before. ``d_fg`` and the treatment paths then track the switch and its direction. The
    original treatment and baseline still measure the size of each switch for the normalized
    effects and the average total effect.

    For every period :math:`j` after the first and every power :math:`k` up to ``continuous``, the
    step also adds the control ``.baseline_trend_{j}_{k}``. It interacts the indicator of period
    :math:`j` or later with the :math:`k`-th power of the baseline treatment. The first differences
    of these controls let the outcome evolution of each period depend on a polynomial in the
    baseline treatment.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig) or config.continuous <= 0:
            return to_polars(data)

        tname = config.tname
        df = to_polars(data)
        switched = pl.col("F_g") != float("inf")
        df = df.with_columns(
            pl.col(config.dname).alias(".d_orig"),
            pl.when(switched)
            .then(pl.col("S_g") * (pl.col(tname) >= pl.col("F_g")).cast(pl.Float64))
            .otherwise(None)
            .alias(config.dname),
            pl.when(switched).then(pl.col("S_g")).otherwise(pl.col("d_sq")).cast(pl.Float64).alias("d_fg"),
        )

        periods = sorted(df[tname].unique().to_list())
        return df.with_columns(
            ((pl.col(tname) >= period).cast(pl.Float64) * pl.col(f".d_sq_{power}")).alias(
                f".baseline_trend_{period}_{power}"
            )
            for period in periods[1:]
            for power in range(1, config.continuous + 1)
        )


class DIDInterPanelBalancer(BaseTransformer):
    """Give every group a row in every period.

    The added rows copy the group's switch columns from its observed rows.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DIDInterConfig):
            return to_polars(data)

        df = to_polars(data)

        groups = df.select(config.gname).unique()
        times = df.select(config.tname).unique()

        full_index = groups.join(times, how="cross")

        df = full_index.join(df, on=[config.gname, config.tname], how="left")

        # A mean of copies of a value such as 0.1 can land a rounding error away from it and differ
        # between groups. Copying the observed value keeps each baseline in one control pool.
        group_columns = {
            "F_g": pl.Float64,
            "d_sq": pl.Float64,
            "d_sq_int": pl.Int64,
            "S_g": pl.Float64,
            "L_g": pl.Float64,
        }
        return df.with_columns(
            pl.col(col).drop_nulls().first().over(config.gname).cast(dtype)
            for col, dtype in group_columns.items()
            if col in df.columns
        )


class DIDInterConfigUpdater:
    """DIDInter config updater."""

    @staticmethod
    def update(data: DataFrame, config: DIDInterConfig) -> None:
        """Update config."""
        df = to_polars(data)

        max_effects, max_placebo = DIDInterConfigUpdater.estimable_horizons(df, config)
        if max_effects == 0:
            kind = {"in": " whose treatment increases", "out": " whose treatment decreases"}.get(config.switchers, "")
            raise ValueError(
                f"No effect can be estimated because no switching group{kind} has a group with the same baseline "
                "treatment that has not switched yet."
            )

        tlist = sorted(df[config.tname].unique().to_list())
        n_groups = df[config.gname].n_unique()

        # The data holds period ranks. Rank r stands for the r-th period that DIDInterTimeRanker recorded.
        config.time_periods = np.asarray(config.time_periods)[np.array(tlist, dtype=int) - 1]
        config.time_periods_count = len(tlist)
        config.n_groups = n_groups
        config.id_count = n_groups
        config.max_effects_available = max_effects
        config.max_placebo_available = max_placebo

        if config.effects > max_effects:
            warnings.warn(
                f"Requested effects={config.effects} but effects can only be estimated up to horizon {max_effects}. "
                f"Using effects={max_effects}.",
                UserWarning,
                stacklevel=4,
            )
            config.effects = max_effects

        placebo_cap = min(max_placebo, config.effects)
        if config.placebo > placebo_cap:
            if placebo_cap < max_placebo:
                reason = "the number of placebos cannot exceed the number of effects"
            else:
                reason = f"placebos can only be estimated up to horizon {max_placebo}"
            warnings.warn(
                f"Requested placebo={config.placebo} but {reason}. Using placebo={placebo_cap}.",
                UserWarning,
                stacklevel=4,
            )
            config.placebo = placebo_cap

        if config.allow_unbalanced_panel:
            unit_counts = df.group_by(config.gname).len()
            is_balanced = (unit_counts["len"] == len(tlist)).all()
            if is_balanced:
                config.data_format = DataFormat.PANEL
            else:
                config.data_format = DataFormat.UNBALANCED_PANEL
        else:
            config.data_format = DataFormat.PANEL

    @staticmethod
    def switcher_directions(config):
        """Get the treatment directions of the switchers that the estimates cover.

        With ``switchers="in"`` or ``"out"`` only one direction is estimated. The groups that switch the
        other way stay in the data as controls until they switch.

        Parameters
        ----------
        config : DIDInterConfig
            Configuration whose ``switchers`` names the directions.

        Returns
        -------
        list of int
            The values of ``S_g`` to estimate, 1 for treatment increases and -1 for decreases.
        """
        return {"in": [1], "out": [-1]}.get(config.switchers, [1, -1])

    @staticmethod
    def estimable_horizons(data, config):
        """Count the effects and placebos that the switchers in the data reach.

        An effect needs a group with the same baseline treatment that has not switched yet. Since
        ``T_g`` is the last period with such a group, a switcher reaches horizon l when
        ``F_g - 1 + l <= T_g``. A placebo at horizon l also needs the period ``F_g - 1 - l``. Only
        the switchers that reach the effect at horizon l enter that placebo.

        Parameters
        ----------
        data : DataFrame
            Preprocessed panel with the columns ``F_g``, ``S_g``, and ``T_g``.
        config : DIDInterConfig
            Configuration whose ``switchers`` sets the switchers that count.

        Returns
        -------
        max_effects : int
            Number of effects that some switcher reaches, 0 when none does.
        max_placebo : int
            Number of placebos that some switcher reaches.
        """
        df = to_polars(data)
        # Since S_g holds floats, is_in needs the directions as floats too.
        directions = [float(d) for d in DIDInterConfigUpdater.switcher_directions(config)]
        switchers = df.filter((pl.col("F_g") != float("inf")) & pl.col("S_g").is_in(directions))
        if switchers.is_empty():
            return 0, 0

        effect_horizons = pl.col("T_g") - pl.col("F_g") + 1
        # Since periods are ranks from 1, F_g - 2 periods come before the last period before the switch.
        placebo_horizons = pl.min_horizontal(effect_horizons, pl.col("F_g") - 2)
        max_effects, max_placebo = switchers.select(
            effect_horizons.max().alias("effects"), placebo_horizons.max().alias("placebo")
        ).row(0)
        # Since the differenced outcome of trends_lin starts one period later, one placebo fewer fits.
        return int(max_effects), max(int(max_placebo) - int(config.trends_lin), 0)


class TrendsLinTransformer(BaseTransformer):
    """Apply first-differencing transformation for linear trends.

    The outcome in levels stays in ``.outcome_levels`` for the heterogeneity regressions.
    """

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
            pl.col(config.yname).alias(".outcome_levels"),
            (pl.col(config.yname) - pl.col(config.yname).shift(1).over(config.gname)).alias(config.yname),
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

        first_obs = df.group_by(gname).agg(pl.col(tname).min().alias(".first_t")).select([gname, ".first_t"])
        df = df.join(first_obs, on=gname, how="left")
        df = df.with_columns((pl.col(tname) == pl.col(".first_t")).cast(pl.Int64).alias("first_obs_by_gp"))
        df = df.drop(".first_t")

        t_max_by_group = df.group_by(gname).agg(pl.col(tname).max().alias("t_max_by_group"))
        df = df.join(t_max_by_group, on=gname, how="left")

        group_cols = ["d_sq"]
        if config.trends_nonparam:
            group_cols.extend(config.trends_nonparam)

        df = df.with_columns(
            pl.when(pl.col("F_g") == float("inf"))
            .then(pl.col("t_max_by_group") + 1)
            .otherwise(pl.col("F_g"))
            .alias(".F_g_trunc")
        )
        df = df.with_columns((pl.col(".F_g_trunc").max().over(group_cols) - 1).alias("T_g"))
        df = df.drop(".F_g_trunc")

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

        df = df.select(cols_to_keep)
        # Since an infinite cohort marks a never-treated unit, the two-period steps read it as the 0 they expect.
        if df.schema[config.gname].is_float():
            never_treated = pl.col(config.gname) == float("inf")
            df = df.with_columns(pl.when(never_treated).then(0.0).otherwise(pl.col(config.gname)).alias(config.gname))
        return df


class DDDWeightProcessor(BaseTransformer):
    """DDD weight processor."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Extract/create weights, validate, normalize, add as WEIGHTS_COLUMN."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)

        if config.weightsname is not None:
            error = _weights_error(df[config.weightsname], config.weightsname)
            if error is not None:
                raise ValueError(error)
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
        n_old = df[config.idname].n_unique()
        df = make_balanced_panel(df, config.idname, config.tname)

        if len(df) == 0:
            raise ValueError("No observations remain after creating balanced panel.")

        n_dropped = n_old - df[config.idname].n_unique()
        if n_dropped > 0:
            warnings.warn(f"Dropped {n_dropped} units while converting to balanced panel")

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
        """Assign 4-group subgroups and check that each has enough units."""
        if not isinstance(config, DDDConfig):
            return to_polars(data)

        df = to_polars(data)
        glist = np.sort(df[config.gname].unique().to_numpy())
        treat_val = glist[1]

        subgroup = create_ddd_subgroups(df[config.gname].to_numpy(), df[config.pname].to_numpy(), treat_val)
        df = df.with_columns(pl.Series("_subgroup", subgroup))

        subgroup_error = _ddd_subgroup_error(subgroup, config.gname, config.pname)
        if subgroup_error is not None:
            raise ValueError(subgroup_error)

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


class EtwfeConfigUpdater:
    """ETWFE config updater."""

    @staticmethod
    def update(data: DataFrame, config: EtwfeConfig) -> None:
        """Update ETWFE config with computed values."""
        df = to_polars(data)

        tlist = np.sort(df[config.tname].unique().to_numpy())
        glist = np.sort(df[config.gname].unique().to_numpy())

        config.time_periods = tlist
        config.time_periods_count = len(tlist)
        config.treated_groups = glist
        config.treated_groups_count = len(glist)

        if config.idname:
            config.n_units = df[config.idname].n_unique()
        else:
            config.n_units = len(df)
        config.n_obs = len(df)


class DynBalancingColumnSelector(BaseTransformer):
    """Dynamic balancing column selector."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DynBalancingConfig):
            return to_polars(data)

        df = to_polars(data)
        cols_to_keep = [config.yname, config.tname, config.idname, config.treatment_name]

        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            formula_vars = [v for v in formula_vars if v != config.yname]
            cols_to_keep.extend(formula_vars)

        if config.fixed_effects:
            cols_to_keep.extend(config.fixed_effects)

        if config.clustervars:
            cols_to_keep.extend(config.clustervars)

        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        cols_to_keep = [col for col in cols_to_keep if col is not None and col in df.columns]

        return df.select(cols_to_keep)


class DynBalancingPanelBalancer(BaseTransformer):
    """Keep the units with a full treatment history and an observed final outcome.

    Rows outside the history window are gone by this step. Missing values
    there never drop a unit. Missing covariates stay in place as NaN for the
    estimator to handle period by period.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DynBalancingConfig):
            return to_polars(data)

        df = to_polars(data)
        idname = config.idname
        noun = "stacked unit histories" if config.pooled else "units"
        n_start = df[idname].n_unique()

        treatment = pl.col(config.treatment_name)
        df = df.filter(treatment.is_not_null() & treatment.cast(pl.Float64).is_not_nan())
        counts = df.group_by(idname).len()
        complete_ids = counts.filter(pl.col("len") == len(config.ds1))[idname]
        n_incomplete = n_start - len(complete_ids)
        if n_incomplete > 0:
            warnings.warn(
                f"Dropped {n_incomplete} {noun} that are not observed in every period of the treatment history.",
                stacklevel=2,
            )

        outcome = pl.col(config.yname)
        observed_ids = df.filter(
            (pl.col(config.tname) == config.final_period)
            & pl.col(idname).is_in(complete_ids.implode())
            & outcome.is_not_null()
            & outcome.cast(pl.Float64).is_not_nan()
        )[idname]
        n_missing_outcome = len(complete_ids) - len(observed_ids)
        if n_missing_outcome > 0:
            warnings.warn(f"Dropped {n_missing_outcome} {noun} with a missing final-period outcome.", stacklevel=2)

        df = df.filter(pl.col(idname).is_in(observed_ids.implode()))

        if len(df) == 0:
            raise ValueError("No observations remain after creating balanced panel.")

        return df


class DynBalancingPooler(BaseTransformer):
    """Check the treatment-history window and stack earlier windows when pooling.

    The history covers the ``len(ds1)`` periods that end at ``final_period``.
    With ``pooled=True`` every complete window that ends between
    ``initial_period`` and ``final_period`` becomes a separate unit history.
    Its periods are shifted to line up with the last window. The default
    ``initial_period`` is the earliest period at which a full window ends.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DynBalancingConfig):
            return to_polars(data)

        df = to_polars(data)
        tname = config.tname
        idname = config.idname
        length_treatment = len(config.ds1)

        time_periods = sorted(df[tname].drop_nulls().unique().to_list())
        if config.final_period is None:
            config.final_period = int(time_periods[-1])
        final_period = config.final_period
        if final_period not in time_periods:
            raise ValueError(f"final_period={final_period} is not in the data time periods.")

        window = list(range(final_period - length_treatment + 1, final_period + 1))
        absent = [p for p in window if p not in time_periods]
        if absent:
            raise ValueError(
                f"The treatment history needs periods {window[0]} to {final_period}. Period "
                f"{absent[0]} is not in the data. Time periods must be consecutive integers. "
                "The history may also be too long for the available periods."
            )

        if not config.pooled:
            if config.initial_period is not None:
                warnings.warn("initial_period only applies when pooled=True. It is ignored.", stacklevel=2)
            return df

        if config.initial_period is None:
            config.initial_period = int(time_periods[0]) + length_treatment - 1
        elif config.initial_period not in time_periods:
            raise ValueError(f"initial_period={config.initial_period} is not in the data time periods.")
        elif config.initial_period > final_period:
            raise ValueError(
                f"initial_period={config.initial_period} must not be later than final_period={final_period}."
            )
        num_periods = final_period - config.initial_period

        ids = df[idname]
        unit_text = pl.col(idname).cast(pl.Utf8)
        # Since a whole-number float id names its units like the same integer, the stacked units sort in the same
        # order whatever the id type.
        if ids.dtype.is_float() and ((ids == ids.round()) & (ids.abs() < 2.0**53)).all():
            unit_text = pl.col(idname).cast(pl.Int64).cast(pl.Utf8)

        if num_periods <= 0:
            warnings.warn(
                "pooled=True has no effect because no earlier treatment-history window ends "
                "between initial_period and final_period. Falling back to non-pooled estimation.",
                stacklevel=2,
            )
            df = df.with_columns(
                pl.col(tname).alias("new_Time"),
                unit_text.alias("new_name"),
            )
            config.idname = "new_name"
            if config.fixed_effects and tname in config.fixed_effects:
                config.fixed_effects = ["new_Time" if fe == tname else fe for fe in config.fixed_effects]
            return df

        # Since polars 2.0 rejects is_in of a float column against integers and sums UInt64 and Int64 into Int128,
        # periods are compared and shifted as Int64 or Float64.
        period_type = pl.Int64 if df.schema[tname].is_integer() else pl.Float64
        period = pl.col(tname).cast(period_type)

        # Build lag offsets and the time windows they require
        lags = pl.DataFrame({"_lag": list(range(1, num_periods + 1))})
        lags = lags.with_columns(
            pl.col("_lag")
            .map_elements(
                lambda j: list(range(final_period - j - length_treatment + 1, final_period - j + 1)),
                return_dtype=pl.List(pl.Int64),
            )
            .alias("_needed")
        )

        # Cross-join every unit-period row with every lag
        df_with_lag = df.join(lags, how="cross")

        # Keep only rows whose time falls within the needed window for that lag
        df_with_lag = df_with_lag.filter(period.is_in(pl.col("_needed").cast(pl.List(period_type))))

        # Filter to units that have complete windows for each lag
        complete_keys = (
            df_with_lag.group_by([idname, "_lag"])
            .agg(pl.len().alias("_cnt"))
            .filter(pl.col("_cnt") == length_treatment)
            .select([idname, "_lag"])
        )
        df_with_lag = df_with_lag.join(complete_keys, on=[idname, "_lag"])

        if df_with_lag.height > 0:
            pseudo = df_with_lag.with_columns(
                pl.col(tname).alias("new_Time"),
                (unit_text + "l" + pl.col("_lag").cast(pl.Utf8) + "unit").alias("new_name"),
                (period + pl.col("_lag")).alias(tname),
            ).drop("_lag", "_needed")

            original = df.with_columns(
                pl.col(tname).alias("new_Time"),
                unit_text.alias("new_name"),
                period.alias(tname),
            )
            pooled_df = pl.concat([original, pseudo.select(original.columns)], how="vertical_relaxed")
        else:
            warnings.warn(
                "pooled=True produced no pseudo-observations. Check that the "
                "panel has enough periods relative to the treatment history length.",
                stacklevel=2,
            )
            original = df.with_columns(
                pl.col(tname).alias("new_Time"),
                unit_text.alias("new_name"),
            )
            pooled_df = original

        config.idname = "new_name"
        if config.fixed_effects and tname in config.fixed_effects:
            config.fixed_effects = ["new_Time" if fe == tname else fe for fe in config.fixed_effects]
        return pooled_df


class DynBalancingFixedEffectDummifier(BaseTransformer):
    """Build one dummy per fixed-effect level among the rows that enter estimation.

    This step runs after the window filter and the panel balancer. A level
    that appears only outside the estimation sample would give an all-zero
    column that still counts toward the number of balanced covariates. A null,
    NaN, or infinite value leaves every dummy of its row missing.
    """

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DynBalancingConfig):
            return to_polars(data)

        if not config.fixed_effects:
            return to_polars(data)

        df = to_polars(data)

        for fe_col in config.fixed_effects:
            # Since a missing value names no level, every dummy of its row stays missing like a missing covariate.
            levels = nonfinite_to_null(df.select(fe_col)).to_series()
            for val in sorted(levels.drop_nulls().unique().to_list()):
                df = df.with_columns((levels == val).cast(pl.Int32).alias(f"{fe_col}_{val}"))

        return df


class DynBalancingPeriodFilter(BaseTransformer):
    """Keep the ``len(ds1)`` periods that end at ``final_period``."""

    def transform(self, data: DataFrame, config: BasePreprocessConfig) -> pl.DataFrame:
        """Transform data."""
        if not isinstance(config, DynBalancingConfig):
            return to_polars(data)

        df = to_polars(data)
        first_period = config.final_period - len(config.ds1) + 1
        return df.filter(pl.col(config.tname).is_between(first_period, config.final_period))


class DynBalancingConfigUpdater:
    """Dynamic balancing config updater."""

    @staticmethod
    def update(data: DataFrame, config: DynBalancingConfig) -> None:
        """Update config with computed values."""
        df = to_polars(data)

        time_periods = np.sort(df[config.tname].unique().to_numpy())
        config.time_periods = time_periods
        config.n_periods = len(time_periods)
        config.n_units = df[config.idname].n_unique()

        cov_names: list[str] = []
        if config.xformla and config.xformla != "~1":
            formula_vars = extract_vars_from_formula(config.xformla)
            cov_names = [v for v in formula_vars if v != config.yname]
        config.covariate_names = cov_names


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
                ContDIDControlGroupFilter(),
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
                DIDInterTimeRanker(),
                WeightNormalizer(),
                SwitcherIdentifier(),
                ContinuousTreatmentProcessor(),
                FgVariationFilter(),
                ControlsTimeFilter(),
                DIDInterPanelBalancer(),
                TrendsLinTransformer(),
                DIDInterDataPreparer(),
                ContinuousTreatmentBinarizer(),
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

    @staticmethod
    def get_etwfe_pipeline() -> "DataTransformerPipeline":
        """Get ETWFE pipeline."""
        return DataTransformerPipeline(
            [
                ColumnSelector(),
                MissingDataHandler(),
                WeightNormalizer(),
                PanelBalancer(),
                DataSorter(),
            ]
        )

    @staticmethod
    def get_dyn_balancing_pipeline() -> "DataTransformerPipeline":
        """Get dynamic balancing pipeline."""
        return DataTransformerPipeline(
            [
                DynBalancingColumnSelector(),
                MissingDataHandler(),
                DynBalancingPooler(),
                DynBalancingPeriodFilter(),
                DynBalancingPanelBalancer(),
                DynBalancingFixedEffectDummifier(),
            ]
        )

    def transform(self, data, config, checks=None):
        """Run every step of the pipeline and update the config from the result.

        The structural checks run on the rows that the missing-data step
        keeps, before any later step drops units or balances the panel. They
        raise their errors at once.

        Parameters
        ----------
        data : DataFrame
            Data in long format.
        config : BasePreprocessConfig
            Configuration that the steps read and update.
        checks : CompositeValidator, optional
            Structural checks to run after the missing-data step.

        Returns
        -------
        pl.DataFrame
            The transformed data.
        """
        df = to_polars(data)
        for transformer in self.transformers:
            df = transformer.transform(df, config)
            if checks is not None and isinstance(transformer, MissingDataHandler):
                result = checks.validate(df, config)
                for message in result.warnings:
                    warnings.warn(message)
                result.raise_if_invalid()

        if isinstance(config, DynBalancingConfig):
            DynBalancingConfigUpdater.update(df, config)
        elif isinstance(config, DDDConfig):
            DDDConfigUpdater.update(df, config)
        elif isinstance(config, DIDInterConfig):
            DIDInterConfigUpdater.update(df, config)
        elif isinstance(config, EtwfeConfig):
            EtwfeConfigUpdater.update(df, config)
        else:
            ConfigUpdater.update(df, config)

        return df
