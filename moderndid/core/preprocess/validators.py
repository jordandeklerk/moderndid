"""Validation classes for preprocessing."""

from typing import Protocol

import polars as pl

from ..dataframe import DataFrame, to_polars
from .base import BaseValidator
from .config import BasePreprocessConfig, ContDIDConfig, DIDConfig, TwoPeriodDIDConfig
from .constants import BasePeriod, ControlGroup
from .models import ValidationResult


class DataValidator(Protocol):
    """Data validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""


class ColumnValidator(BaseValidator):
    """Column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []
        data_columns = df.columns

        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "gname": config.gname,
        }

        if config.panel and config.idname:
            required_cols["idname"] = config.idname

        for col_type, col_name in required_cols.items():
            if col_name not in data_columns:
                errors.append(f"{col_type} = '{col_name}' must be a column in the dataset")

        if config.weightsname and config.weightsname not in data_columns:
            errors.append(f"weightsname = '{config.weightsname}' must be a column in the dataset")

        if config.clustervars:
            for cluster_var in config.clustervars:
                if cluster_var not in data_columns:
                    errors.append(f"clustervars contains '{cluster_var}' which is not in the dataset")

        if isinstance(config, ContDIDConfig) and config.dname:
            if config.dname not in data_columns:
                errors.append(f"dname = '{config.dname}' must be a column in the dataset")

        if not errors:
            if config.tname in data_columns and not _is_numeric_dtype(df[config.tname]):
                errors.append(f"tname = '{config.tname}' is not numeric. Please convert it")

            if config.gname in data_columns and not _is_numeric_dtype(df[config.gname]):
                errors.append(f"gname = '{config.gname}' is not numeric. Please convert it")

            if config.idname and config.idname in data_columns:
                if not _is_numeric_dtype(df[config.idname]):
                    errors.append(f"idname = '{config.idname}' is not numeric. Please convert it")

            if isinstance(config, ContDIDConfig) and config.dname and config.dname in data_columns:
                if not _is_numeric_dtype(df[config.dname]):
                    errors.append(f"dname = '{config.dname}' is not numeric. Please convert it")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class TreatmentValidator(BaseValidator):
    """Treatment validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not config.panel or not config.idname:
            return self._create_result(errors, warnings)

        gname_by_id = df.group_by(config.idname).agg(pl.col(config.gname).n_unique().alias("n_unique"))
        if (gname_by_id["n_unique"] > 1).any():
            errors.append(
                "The value of gname (treatment variable) must be the same across all "
                "periods for each particular unit. The treatment must be irreversible."
            )

        first_period = df[config.tname].min()
        treated_first_mask = (pl.col(config.gname) > 0) & (pl.col(config.gname) <= first_period)
        treated_first_df = df.filter(treated_first_mask)

        if config.idname:
            n_first_period = treated_first_df[config.idname].n_unique()
        else:
            n_first_period = len(treated_first_df)

        if n_first_period > 0:
            warnings.append(f"{n_first_period} units were already treated in the first period and will be dropped")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class PanelStructureValidator(BaseValidator):
    """Panel structure validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not config.panel or not config.idname:
            return self._create_result(errors, warnings)

        if df.select([config.idname, config.tname]).is_duplicated().any():
            errors.append(
                "The value of idname must be unique (by tname). Some units are observed more than once in a period."
            )

        if not config.allow_unbalanced_panel:
            n_time_periods = df[config.tname].n_unique()
            unit_counts = df.group_by(config.idname).len()

            if not (unit_counts["len"] == n_time_periods).all():
                n_unbalanced = (unit_counts["len"] != n_time_periods).sum()
                warnings.append(f"{n_unbalanced} units have unbalanced observations and will be dropped")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class ClusterValidator(BaseValidator):
    """Cluster validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not config.clustervars:
            return self._create_result(errors, warnings)

        cluster_vars = [cv for cv in config.clustervars if cv != config.idname]

        if len(cluster_vars) > 1:
            errors.append("You can only provide 1 cluster variable additionally to the one provided in idname")
            return self._create_result(errors, warnings)

        if len(cluster_vars) > 0 and config.idname and config.panel:
            for clust_var in cluster_vars:
                clust_nunique = df.group_by(config.idname).agg(pl.col(clust_var).n_unique().alias("n_unique"))
                if (clust_nunique["n_unique"] > 1).any():
                    errors.append(
                        "DiD cannot handle time-varying cluster variables at the moment. "
                        "Please check your cluster variable."
                    )

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class ArgumentValidator(BaseValidator):
    """Argument validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        errors = []
        warnings = []

        if not isinstance(config.anticipation, int | float):
            errors.append("anticipation must be numeric")
        elif config.anticipation < 0:
            errors.append("anticipation must be positive")

        if not 0 < config.alp < 1:
            errors.append("alp must be between 0 and 1")

        if isinstance(config, DIDConfig):
            if config.control_group not in [ControlGroup.NEVER_TREATED, ControlGroup.NOT_YET_TREATED]:
                errors.append(
                    f"control_group must be either '{ControlGroup.NEVER_TREATED.value}' "
                    f"or '{ControlGroup.NOT_YET_TREATED.value}'"
                )

            if config.base_period not in [BasePeriod.UNIVERSAL, BasePeriod.VARYING]:
                errors.append(
                    f"base_period must be either '{BasePeriod.UNIVERSAL.value}' or '{BasePeriod.VARYING.value}'"
                )

        if isinstance(config, ContDIDConfig):
            if config.degree < 1:
                errors.append("degree must be at least 1")

            if config.num_knots < 0:
                errors.append("num_knots must be non-negative")

            if config.required_pre_periods < 0:
                errors.append("required_pre_periods must be non-negative")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DoseValidator(BaseValidator):
    """Dose validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not isinstance(config, ContDIDConfig) or not config.dname:
            return self._create_result(errors, warnings)

        if config.dname in df.columns:
            dose_values = df[config.dname]

            if (dose_values < 0).any():
                errors.append(f"dname = '{config.dname}' contains negative values")

            n_missing = dose_values.is_null().sum()
            if n_missing > 0:
                warnings.append(f"{n_missing} observations have missing dose values and will be handled")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class PrePostColumnValidator(BaseValidator):
    """Pre-post column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []
        data_columns = df.columns

        if not isinstance(config, TwoPeriodDIDConfig):
            return self._create_result(errors, warnings)

        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "treat_col": config.treat_col,
        }

        if config.panel and config.idname:
            required_cols["idname"] = config.idname

        for col_type, col_name in required_cols.items():
            if col_name not in data_columns:
                errors.append(f"{col_type} = '{col_name}' must be a column in the dataset")

        if config.weightsname and config.weightsname not in data_columns:
            errors.append(f"weightsname = '{config.weightsname}' must be a column in the dataset")

        if not errors:
            if config.tname in data_columns and not _is_numeric_dtype(df[config.tname]):
                errors.append(f"tname = '{config.tname}' is not numeric. Please convert it")

            if config.treat_col in data_columns and not _is_numeric_dtype(df[config.treat_col]):
                errors.append(f"treat_col = '{config.treat_col}' is not numeric. Please convert it")

            if config.idname and config.idname in data_columns:
                if not _is_numeric_dtype(df[config.idname]):
                    errors.append(f"idname = '{config.idname}' is not numeric. Please convert it")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class PrePostDataValidator(BaseValidator):
    """Pre-post data validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not isinstance(config, TwoPeriodDIDConfig):
            return self._create_result(errors, warnings)

        time_periods = sorted(df[config.tname].unique().to_list())
        if len(time_periods) != 2:
            errors.append("This package currently supports only two time periods (pre and post).")

        groups = sorted(df[config.treat_col].unique().to_list())
        if len(groups) != 2 or not all(g in [0, 1] for g in groups):
            errors.append("Treatment indicator column must contain only 0 (control) and 1 (treated).")

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class PrePostPanelValidator(BaseValidator):
    """Pre-post panel validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig | TwoPeriodDIDConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)
        errors = []
        warnings = []

        if not isinstance(config, TwoPeriodDIDConfig) or not config.panel or not config.idname:
            return self._create_result(errors, warnings)

        treat_counts = df.group_by(config.idname).agg(pl.col(config.treat_col).n_unique().alias("n_unique"))
        if (treat_counts["n_unique"] > 1).any():
            invalid_ids = treat_counts.filter(pl.col("n_unique") > 1)[config.idname].to_list()
            errors.append(
                f"Treatment indicator ('{config.treat_col}') must be unique for each ID ('{config.idname}'). "
                f"IDs with varying treatment: {invalid_ids}."
            )

        return self._create_result(errors, warnings)

    @staticmethod
    def _create_result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
        """Create result."""
        errors = errors or []
        warnings = warnings or []
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class CompositeValidator(BaseValidator):
    """Composite validator."""

    def __init__(self, validators: list[BaseValidator] | None = None, config_type: str = "did"):
        """Initialize composite validator."""
        if validators is not None:
            self.validators = validators
        else:
            self.validators = self._get_default_validators(config_type)

    @staticmethod
    def _get_default_validators(config_type: str = "did") -> list[BaseValidator]:
        """Get default validators."""
        if config_type == "two_period":
            return [
                PrePostColumnValidator(),
                PrePostDataValidator(),
                PrePostPanelValidator(),
            ]

        common_validators = [
            ArgumentValidator(),
            ColumnValidator(),
            TreatmentValidator(),
            PanelStructureValidator(),
            ClusterValidator(),
        ]

        if config_type == "cont_did":
            common_validators.append(DoseValidator())

        return common_validators

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        all_errors = []
        all_warnings = []

        for validator in self.validators:
            result = validator.validate(data, config)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(is_valid=len(all_errors) == 0, errors=all_errors, warnings=all_warnings)


def _is_numeric_dtype(series: pl.Series) -> bool:
    """Check if a polars series has a numeric dtype."""
    return series.dtype.is_numeric()
