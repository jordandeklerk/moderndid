"""Builder pattern for constructing preprocessed data objects."""

import warnings
from typing import Any

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .config import BasePreprocessConfig, ContDIDConfig, DIDConfig, TwoPeriodDIDConfig
from .constants import WEIGHTS_COLUMN
from .models import ContDIDData, DIDData, TwoPeriodDIDData
from .tensors import TensorFactorySelector
from .transformers import DataTransformerPipeline
from .utils import extract_vars_from_formula
from .validators import CompositeValidator


class PreprocessDataBuilder:
    """Builder for constructing preprocessed data objects."""

    def __init__(self):
        """Initialize builder."""
        self._data: pl.DataFrame | None = None
        self._config: BasePreprocessConfig | None = None
        self._validator: CompositeValidator | None = None
        self._transformer: DataTransformerPipeline | None = None
        self._warnings: list[str] = []

    def with_data(self, data: DataFrame) -> "PreprocessDataBuilder":
        """Set the data.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame
            Input panel or cross-section data.

        Returns
        -------
        PreprocessDataBuilder
            Self for method chaining.
        """
        self._data = to_polars(data)
        return self

    def with_config(self, config):
        """Set the configuration.

        Parameters
        ----------
        config : BasePreprocessConfig
            Configuration object (DIDConfig, ContDIDConfig, or TwoPeriodDIDConfig).

        Returns
        -------
        PreprocessDataBuilder
            Self for method chaining.
        """
        self._config = config

        if isinstance(config, TwoPeriodDIDConfig):
            self._validator = CompositeValidator(config_type="two_period")
            self._transformer = DataTransformerPipeline.get_two_period_pipeline()
        elif isinstance(config, DIDConfig):
            self._validator = CompositeValidator(config_type="did")
            self._transformer = DataTransformerPipeline.get_did_pipeline()
        elif isinstance(config, ContDIDConfig):
            self._validator = CompositeValidator(config_type="cont_did")
            self._transformer = DataTransformerPipeline.get_cont_did_pipeline()
        else:
            self._validator = CompositeValidator(config_type="did")
            self._transformer = DataTransformerPipeline.get_did_pipeline()

        return self

    def with_config_dict(self, config_type="did", **kwargs):
        """Set configuration from keyword arguments.

        Parameters
        ----------
        config_type : str, default "did"
            Type of config to create ("did" or "cont_did").
        **kwargs
            Configuration parameters.

        Returns
        -------
        PreprocessDataBuilder
            Self for method chaining.
        """
        if config_type == "cont_did":
            self._config = ContDIDConfig(**kwargs)
        else:
            self._config = DIDConfig(**kwargs)

        return self.with_config(self._config)

    def validate(self) -> "PreprocessDataBuilder":
        """Validate data and configuration.

        Returns
        -------
        PreprocessDataBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If data or config not set, or if validation fails.
        """
        if self._data is None:
            raise ValueError("Data not set. Use with_data() first.")
        if self._config is None:
            raise ValueError("Configuration not set. Use with_config() first.")
        if self._validator is None:
            raise ValueError("Validator not initialized. Use with_config() first.")

        result = self._validator.validate(self._data, self._config)

        self._warnings.extend(result.warnings)

        if result.warnings:
            for warning in result.warnings:
                warnings.warn(warning)

        result.raise_if_invalid()

        return self

    def transform(self) -> "PreprocessDataBuilder":
        """Apply data transformations.

        Returns
        -------
        PreprocessDataBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If data, config, or transformer not set.
        """
        if self._data is None or self._config is None:
            raise ValueError("Must set data and config before transforming")
        if self._transformer is None:
            raise ValueError("Transformer not initialized. Use with_config() first.")

        self._data = self._transformer.transform(self._data, self._config)

        self._validate_transformed_data()

        return self

    def _validate_transformed_data(self) -> None:
        """Validate transformed data meets minimum requirements."""
        if self._data is None or self._config is None:
            return

        if isinstance(self._config, TwoPeriodDIDConfig):
            return

        glist = self._config.treated_groups
        if len(glist) == 0:
            raise ValueError(
                "No valid groups. The variable in 'gname' should be expressed as "
                "the time a unit is first treated (0 if never-treated)"
            )

        if self._config.panel:
            gsize = self._data.group_by(self._config.gname).len()
            gsize = gsize.with_columns((pl.col("len") / self._config.time_periods_count).alias("size"))
        else:
            gsize = self._data.group_by(self._config.gname).len().rename({"len": "size"})

        if self._config and self._config.xformla and self._config.xformla != "~1":
            formula_vars = extract_vars_from_formula(self._config.xformla)
            n_covs = len([v for v in formula_vars if v != self._config.yname])
        else:
            n_covs = 0
        reqsize = n_covs + 5

        small_groups = gsize.filter(pl.col("size") < reqsize)
        if len(small_groups) > 0:
            group_list = ", ".join([str(g) for g in small_groups[self._config.gname].to_list()])
            warning_msg = f"Be aware that there are some small groups in your dataset.\nCheck groups: {group_list}"
            warnings.warn(warning_msg)
            self._warnings.append(warning_msg)

            if isinstance(self._config, DIDConfig):
                from .constants import NEVER_TREATED_VALUE

                small_group_values = small_groups[self._config.gname].to_list()
                if NEVER_TREATED_VALUE in small_group_values and self._config.control_group.value == "nevertreated":
                    raise ValueError("Never treated group is too small, try setting control_group='notyettreated'")

    def build(self) -> DIDData | ContDIDData | TwoPeriodDIDData:
        """Build the final preprocessed data object.

        Returns
        -------
        DIDData | ContDIDData | TwoPeriodDIDData
            Preprocessed data container (type depends on config).

        Raises
        ------
        ValueError
            If data or config not set.
        """
        if self._data is None or self._config is None:
            raise ValueError("Must set data and config before building")

        if isinstance(self._config, TwoPeriodDIDConfig):
            return self._build_two_period_did_data()
        if isinstance(self._config, DIDConfig):
            return self._build_did_data()
        if isinstance(self._config, ContDIDConfig):
            return self._build_cont_did_data()
        raise ValueError(f"Unknown config type: {type(self._config)}")

    def _build_did_data(self) -> DIDData:
        """Build DIDData object with tensors."""
        tensor_factory = TensorFactorySelector()
        tensor_data = tensor_factory.create_tensors(self._data, self._config)

        did_data = DIDData(
            data=tensor_data["data"],
            time_invariant_data=tensor_data["time_invariant_data"],
            weights=tensor_data["weights"],
            cohort_counts=tensor_data["cohort_counts"],
            period_counts=tensor_data["period_counts"],
            crosstable_counts=tensor_data["crosstable_counts"],
            outcomes_tensor=tensor_data["outcomes_tensor"],
            covariates_matrix=tensor_data["covariates_matrix"],
            covariates_tensor=tensor_data["covariates_tensor"],
            cluster=tensor_data["cluster"],
            config=self._config,
        )

        return did_data

    def _build_two_period_did_data(self) -> TwoPeriodDIDData:
        """Build TwoPeriodDIDData object."""
        if not isinstance(self._config, TwoPeriodDIDConfig):
            raise ValueError("Config must be TwoPeriodDIDConfig")

        time_periods = sorted(self._data[self._config.tname].unique().to_list())
        if len(time_periods) != 2:
            raise ValueError("Data must have exactly 2 time periods")

        pre_period, post_period = time_periods

        covariate_columns = [
            col
            for col in self._data.columns
            if col.startswith("Intercept")
            or (
                col
                not in [
                    self._config.yname,
                    self._config.tname,
                    self._config.treat_col,
                    self._config.idname,
                    self._config.weightsname,
                    WEIGHTS_COLUMN,
                ]
            )
        ]

        if self._config.panel and self._config.idname:
            df_processed = self._data.sort([self._config.idname, self._config.tname])

            post_data = df_processed.filter(pl.col(self._config.tname) == post_period)
            pre_data = df_processed.filter(pl.col(self._config.tname) == pre_period)

            common_ids = (
                post_data.select(self._config.idname)
                .join(pre_data.select(self._config.idname), on=self._config.idname, how="inner")[self._config.idname]
                .to_list()
            )

            post_data = post_data.filter(pl.col(self._config.idname).is_in(common_ids)).sort(self._config.idname)
            pre_data = pre_data.filter(pl.col(self._config.idname).is_in(common_ids)).sort(self._config.idname)

            two_period_data = TwoPeriodDIDData(
                y1=post_data[self._config.yname].to_numpy(),
                y0=pre_data[self._config.yname].to_numpy(),
                D=post_data[self._config.treat_col].to_numpy(),
                covariates=post_data.select(covariate_columns).to_numpy(),
                weights=post_data[WEIGHTS_COLUMN].to_numpy(),
                covariate_names=covariate_columns,
                n_units=len(common_ids),
                config=self._config,
            )
        else:
            two_period_data = TwoPeriodDIDData(
                y=self._data[self._config.yname].to_numpy(),
                D=self._data[self._config.treat_col].to_numpy(),
                post=(self._data[self._config.tname] == post_period).cast(pl.Int64).to_numpy(),
                covariates=self._data.select(covariate_columns).to_numpy(),
                weights=self._data[WEIGHTS_COLUMN].to_numpy(),
                covariate_names=covariate_columns,
                n_obs=len(self._data),
                config=self._config,
            )

        return two_period_data

    def _build_cont_did_data(self) -> ContDIDData:
        """Build ContDIDData object."""
        time_invariant_data = self._create_time_invariant_data()
        summary_tables = self._create_summary_tables(time_invariant_data)

        cluster = self._extract_cluster_variable(time_invariant_data)
        weights = self._extract_weights(time_invariant_data)

        original_time_periods = self._config.time_periods.copy()

        cont_data = ContDIDData(
            data=self._data,
            time_invariant_data=time_invariant_data,
            weights=weights,
            cohort_counts=summary_tables["cohort_counts"],
            period_counts=summary_tables["period_counts"],
            crosstable_counts=summary_tables["crosstable_counts"],
            cluster=cluster,
            time_map=self._config.time_map or {},
            original_time_periods=original_time_periods,
            config=self._config,
        )

        return cont_data

    def _create_time_invariant_data(self) -> pl.DataFrame:
        """Extract time-invariant data."""
        time_invariant_cols = [self._config.idname, self._config.gname, WEIGHTS_COLUMN]

        if self._config.clustervars:
            time_invariant_cols.extend(self._config.clustervars)

        if self._config.xformla and self._config.xformla != "~1":
            formula_vars = extract_vars_from_formula(self._config.xformla)
            formula_vars = [v for v in formula_vars if v != self._config.yname]

            for var in formula_vars:
                if var in self._data.columns:
                    var_counts = self._data.group_by(self._config.idname).agg(pl.col(var).n_unique().alias("n_unique"))
                    if (var_counts["n_unique"] == 1).all():
                        time_invariant_cols.append(var)

        time_invariant_cols = list(dict.fromkeys(time_invariant_cols))
        cols_to_select = [col for col in time_invariant_cols if col in self._data.columns]
        return (
            self._data.group_by(self._config.idname, maintain_order=True)
            .first()
            .select([self._config.idname] + [c for c in cols_to_select if c != self._config.idname])
        )

    def _create_summary_tables(self, time_invariant_data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Create summary tables for cohorts and periods."""
        cohort_counts = (
            time_invariant_data.group_by(self._config.gname, maintain_order=True)
            .len()
            .rename({self._config.gname: "cohort", "len": "cohort_size"})
        )

        period_counts = (
            self._data.group_by(self._config.tname, maintain_order=True)
            .len()
            .rename({self._config.tname: "period", "len": "period_size"})
        )

        crosstable = (
            self._data.group_by([self._config.tname, self._config.gname], maintain_order=True)
            .len()
            .rename({self._config.tname: "period", self._config.gname: "cohort", "len": "count"})
        )
        crosstable_counts = crosstable.pivot(index="period", on="cohort", values="count").fill_null(0)

        return {
            "cohort_counts": cohort_counts,
            "period_counts": period_counts,
            "crosstable_counts": crosstable_counts,
        }

    def _extract_cluster_variable(self, time_invariant_data: pl.DataFrame) -> np.ndarray | None:
        """Extract cluster variable if specified."""
        if self._config.clustervars and len(self._config.clustervars) > 0:
            return time_invariant_data[self._config.clustervars[0]].to_numpy()
        return None

    @staticmethod
    def _extract_weights(time_invariant_data: pl.DataFrame) -> np.ndarray:
        """Extract normalized weights."""
        return time_invariant_data[WEIGHTS_COLUMN].to_numpy()

    def _get_did_summary(self, tensor_data: dict[str, Any]) -> str | None:
        """Get DiD preprocessing summary as a string."""
        if self._config is None or not isinstance(self._config, DIDConfig):
            return None

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("DiD Preprocessing Summary")
        lines.append("=" * 60)

        lines.append(f"Data Format: {self._config.data_format.value}")
        lines.append(f"Number of Units: {self._config.id_count}")
        lines.append(f"Number of Time Periods: {self._config.time_periods_count}")
        lines.append(f"Number of Treatment Cohorts: {self._config.treated_groups_count}")

        if self._config.treated_groups_count > 0:
            lines.append("\nTreatment Timing:")
            cohort_counts = tensor_data["cohort_counts"]
            for row in cohort_counts.iter_rows(named=True):
                cohort = row["cohort"]
                size = row["cohort_size"]
                if np.isfinite(cohort):
                    lines.append(f"  Period {int(cohort)}: {size} units")
                else:
                    lines.append(f"  Never Treated: {size} units")

        lines.append("\nSettings:")
        lines.append(f"  Control Group: {self._config.control_group.value}")
        lines.append(f"  Estimation Method: {self._config.est_method.value}")
        lines.append(f"  Anticipation Periods: {self._config.anticipation}")

        if self._warnings:
            lines.append(f"\nWarnings ({len(self._warnings)}):")
            for warning in self._warnings[:3]:
                lines.append(f"  - {warning}")
            if len(self._warnings) > 3:
                lines.append(f"  ... and {len(self._warnings) - 3} more")

        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def _get_cont_did_summary(self, summary_tables: dict[str, pl.DataFrame]) -> str | None:
        """Get continuous DiD preprocessing summary as a string."""
        if self._config is None or not isinstance(self._config, ContDIDConfig):
            return None

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("Continuous Treatment DiD Preprocessing Summary")
        lines.append("=" * 60)

        lines.append(f"Number of Units: {self._config.id_count}")
        lines.append(f"Number of Time Periods: {self._config.time_periods_count}")
        lines.append(f"Number of Treatment Cohorts: {self._config.treated_groups_count}")

        if self._config.has_dose:
            lines.append(f"\nDose Variable: {self._config.dname}")
            lines.append(f"  Spline Degree: {self._config.degree}")
            lines.append(f"  Number of Knots: {self._config.num_knots}")

        if self._config.treated_groups_count > 0:
            lines.append("\nTreatment Timing:")
            cohort_counts = summary_tables["cohort_counts"]
            for row in cohort_counts.head(10).iter_rows(named=True):
                cohort = row["cohort"]
                size = row["cohort_size"]
                if np.isfinite(cohort):
                    lines.append(f"  Period {int(cohort)}: {size} units")
                else:
                    lines.append(f"  Never Treated: {size} units")
            if len(cohort_counts) > 10:
                lines.append(f"  ... and {len(cohort_counts) - 10} more cohorts")

        lines.append("\nSettings:")
        lines.append(f"  Control Group: {self._config.control_group.value}")
        lines.append(f"  Anticipation Periods: {self._config.anticipation}")
        lines.append(f"  Required Pre-Periods: {self._config.required_pre_periods}")

        if self._warnings:
            lines.append(f"\nWarnings ({len(self._warnings)}):")
            for warning in self._warnings[:3]:
                lines.append(f"  - {warning}")
            if len(self._warnings) > 3:
                lines.append(f"  ... and {len(self._warnings) - 3} more")

        lines.append("=" * 60 + "\n")
        return "\n".join(lines)
