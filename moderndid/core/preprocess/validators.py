"""Validation classes for preprocessing."""

from typing import Protocol

import numpy as np
import polars as pl

from ..dataframe import DataFrame, to_polars
from .base import BaseValidator
from .config import (
    BasePreprocessConfig,
    ContDIDConfig,
    DDDConfig,
    DIDInterConfig,
    DynBalancingConfig,
    TwoPeriodDIDConfig,
)
from .constants import ROW_ID_COLUMN, WEIGHTS_COLUMN
from .models import ValidationResult
from .utils import extract_vars_from_formula, get_formula_columns, nonfinite_to_null


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

        if isinstance(config, ContDIDConfig) and config.dname and config.dname not in data_columns:
            errors.append(f"dname = '{config.dname}' must be a column in the dataset")

        if not errors:
            if config.tname in data_columns and not _is_numeric_dtype(df[config.tname]):
                errors.append(f"tname = '{config.tname}' is not numeric. Please convert it")

            if config.gname in data_columns and not _is_numeric_dtype(df[config.gname]):
                errors.append(f"gname = '{config.gname}' is not numeric. Please convert it")

            if config.idname and config.idname in data_columns and not _is_numeric_dtype(df[config.idname]):
                errors.append(f"idname = '{config.idname}' is not numeric. Please convert it")

            if config.yname in data_columns and not _is_numeric_dtype(df[config.yname]):
                errors.append(f"yname = '{config.yname}' is not numeric. Please convert it")

            covariate_names = []
            if config.xformla and config.xformla != "~1":
                covariate_names = extract_vars_from_formula(config.xformla)
                for cov in covariate_names:
                    if cov not in data_columns:
                        errors.append(f"xformla contains '{cov}' which is not a column in the dataset")

            if (
                isinstance(config, ContDIDConfig)
                and config.dname
                and config.dname in data_columns
                and not _is_numeric_dtype(df[config.dname])
            ):
                errors.append(f"dname = '{config.dname}' is not numeric. Please convert it")

            named_columns = {
                "yname": config.yname,
                "tname": config.tname,
                "gname": config.gname,
                "idname": config.idname,
                "weightsname": config.weightsname,
                "clustervars": config.clustervars,
                "xformla": covariate_names,
                "dname": config.dname if isinstance(config, ContDIDConfig) else None,
            }
            errors.extend(_reserved_name_errors(named_columns, (WEIGHTS_COLUMN, ROW_ID_COLUMN)))

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

        n_first_period = treated_first_df[config.idname].n_unique() if config.idname else len(treated_first_df)

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

        mismatch_errors, mismatch_warnings = _check_panel_mismatch(df, config.idname, config.tname, config.panel)
        errors.extend(mismatch_errors)
        warnings.extend(mismatch_warnings)

        if config.panel and config.idname:
            duplicate_error = _duplicate_unit_period_error(df, config.idname, config.tname)
            if duplicate_error is not None:
                errors.append(duplicate_error)
            # Since a repeated row counts toward its unit's rows, the unbalanced count would misdescribe that unit.
            elif not config.allow_unbalanced_panel:
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

        if isinstance(config, ContDIDConfig) and config.required_pre_periods < 0:
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

            if config.idname and config.idname in data_columns and not _is_numeric_dtype(df[config.idname]):
                errors.append(f"idname = '{config.idname}' is not numeric. Please convert it")

            if config.yname in data_columns and not _is_numeric_dtype(df[config.yname]):
                errors.append(f"yname = '{config.yname}' is not numeric. Please convert it")

            # Since a transformed term such as I(x**2) still reads its columns, the check covers them too.
            formula_columns = []
            if config.xformla and config.xformla != "~1":
                formula_columns = get_formula_columns(config.xformla, data_columns)
            named_columns = {
                "yname": config.yname,
                "tname": config.tname,
                "treat_col": config.treat_col,
                "idname": config.idname,
                "weightsname": config.weightsname,
                "xformla": formula_columns,
            }
            errors.extend(_reserved_name_errors(named_columns, (WEIGHTS_COLUMN, "Intercept")))

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

        if not isinstance(config, TwoPeriodDIDConfig):
            return self._create_result(errors, warnings)

        mismatch_errors, mismatch_warnings = _check_panel_mismatch(df, config.idname, config.tname, config.panel)
        errors.extend(mismatch_errors)
        warnings.extend(mismatch_warnings)

        if config.panel and config.idname:
            duplicate_error = _duplicate_unit_period_error(df, config.idname, config.tname)
            if duplicate_error is not None:
                errors.append(duplicate_error)

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


class PrePostArgumentValidator(BaseValidator):
    """Argument validator for two-period DiD."""

    def validate(self, data: DataFrame, config) -> ValidationResult:
        """Validate two-period DiD arguments."""
        if not isinstance(config, TwoPeriodDIDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        return ValidationResult(is_valid=True, errors=[], warnings=[])


class DIDInterColumnValidator(BaseValidator):
    """DIDInter column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        if not isinstance(config, DIDInterConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []
        warnings = []
        data_columns = df.columns

        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "gname": config.gname,
            "dname": config.dname,
        }

        for col_type, col_name in required_cols.items():
            if col_name not in data_columns:
                errors.append(f"{col_type} = '{col_name}' must be a column in the dataset")

        if config.weightsname and config.weightsname not in data_columns:
            errors.append(f"weightsname = '{config.weightsname}' must be a column in the dataset")

        if config.cluster and config.cluster not in data_columns:
            errors.append(f"cluster = '{config.cluster}' must be a column in the dataset")

        covariate_names = []
        if config.xformla and config.xformla != "~1":
            covariate_names = extract_vars_from_formula(config.xformla)
            for ctrl in covariate_names:
                if ctrl not in data_columns:
                    errors.append(f"xformla contains '{ctrl}' which is not in the dataset")

        named_columns = {
            "yname": config.yname,
            "tname": config.tname,
            "gname": config.gname,
            "dname": config.dname,
            "weightsname": config.weightsname,
            "cluster": config.cluster,
            "xformla": covariate_names,
            "trends_nonparam": config.trends_nonparam,
            "predict_het": config.predict_het[0] if config.predict_het else None,
        }
        reserved = (
            WEIGHTS_COLUMN,
            "F_g",
            "d_sq",
            "d_sq_int",
            "d_fg",
            "S_g",
            "L_g",
            "T_g",
            "weight_gt",
            "first_obs_by_gp",
            "t_max_by_group",
        )
        errors.extend(_reserved_name_errors(named_columns, reserved))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DIDInterTreatmentValidator(BaseValidator):
    """DIDInter treatment validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        if not isinstance(config, DIDInterConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []
        warnings = []

        treatment_changes = df.group_by(config.gname).agg(pl.col(config.dname).n_unique().alias("n_unique"))
        n_switchers = int((treatment_changes["n_unique"] > 1).sum())

        if n_switchers == 0:
            errors.append("No units change treatment. Cannot estimate effects.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DIDInterArgumentValidator(BaseValidator):
    """DIDInter argument validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        if not isinstance(config, DIDInterConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        return ValidationResult(is_valid=True, errors=[], warnings=[])


class DIDInterPanelValidator(BaseValidator):
    """DIDInter panel validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        if not isinstance(config, DIDInterConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []
        warnings = []

        # The intertemporal config stores the caller's idname column in gname.
        duplicate_error = _duplicate_unit_period_error(df, config.gname, config.tname)
        if duplicate_error is not None:
            errors.append(duplicate_error)
        elif not config.allow_unbalanced_panel:
            n_time_periods = df[config.tname].n_unique()
            unit_counts = df.group_by(config.gname).len()

            if not (unit_counts["len"] == n_time_periods).all():
                n_unbalanced = int((unit_counts["len"] != n_time_periods).sum())
                warnings.append(f"{n_unbalanced} units have unbalanced observations and will be dropped")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DDDColumnValidator(BaseValidator):
    """DDD column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate DDD columns exist and have correct types."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []
        warnings = []
        data_columns = df.columns

        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "idname": config.idname,
            "gname": config.gname,
            "pname": config.pname,
        }

        for col_type, col_name in required_cols.items():
            if col_name not in data_columns:
                errors.append(f"{col_type}='{col_name}' not found in data.")

        if config.cluster is not None and config.cluster not in data_columns:
            errors.append(f"cluster='{config.cluster}' not found in data.")

        if config.weightsname is not None and config.weightsname not in data_columns:
            errors.append(f"weightsname='{config.weightsname}' not found in data.")

        covariate_vars = []
        if config.xformla != "~1":
            try:
                covariate_vars = extract_vars_from_formula(config.xformla)
            except ValueError as e:
                errors.append(f"Invalid formula: {e}")
                return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

            for var in covariate_vars:
                if var not in data_columns:
                    errors.append(f"Covariate '{var}' from formula not found in data.")

        if not errors:
            for col_type in ["yname", "tname", "idname", "gname"]:
                col_name = getattr(config, col_type)
                if col_name in data_columns and not _is_numeric_dtype(df[col_name]):
                    errors.append(f"{col_type}='{col_name}' is not numeric. Please convert it.")

        named_columns = {
            "yname": config.yname,
            "tname": config.tname,
            "idname": config.idname,
            "gname": config.gname,
            "pname": config.pname,
            "cluster": config.cluster,
            "weightsname": config.weightsname,
            "xformla": covariate_vars,
        }
        errors.extend(_reserved_name_errors(named_columns, (WEIGHTS_COLUMN, "_post", "_subgroup")))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DDDArgumentValidator(BaseValidator):
    """DDD argument validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate DDD arguments."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        return ValidationResult(is_valid=True, errors=[], warnings=[])


class DDDInvarianceValidator(BaseValidator):
    """DDD invariance validator for partition and treatment."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate partition and treatment are time-invariant."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []

        partition_per_id = df.group_by(config.idname).agg(pl.col(config.pname).n_unique().alias("n_unique"))
        if (partition_per_id["n_unique"] > 1).any():
            errors.append(f"The value of {config.pname} must be the same across all periods for each unit.")

        treat_per_id = df.group_by(config.idname).agg(pl.col(config.gname).n_unique().alias("n_unique"))
        if (treat_per_id["n_unique"] > 1).any():
            errors.append(f"The value of {config.gname} must be the same across all periods for each unit.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])


class DDDDataValidator(BaseValidator):
    """DDD data validator for time periods, treatment values, and the partition."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate exactly 2 time periods, 2 treatment values, and a 0/1 partition."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []

        tlist = np.sort(df[config.tname].unique().to_numpy())
        if len(tlist) != 2:
            errors.append(f"Data must have exactly 2 time periods, found {len(tlist)}.")

        glist = np.sort(df[config.gname].unique().to_numpy())
        if len(glist) != 2:
            errors.append(f"Treatment variable must have exactly 2 values (0 and treated group), found {len(glist)}.")
        elif glist[0] != 0:
            errors.append("Treatment variable must include 0 for never-treated units.")

        partition_error = _ddd_partition_error(df, config.pname)
        if partition_error is not None:
            errors.append(partition_error)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])


class DDDPanelStructureValidator(BaseValidator):
    """DDD panel structure validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate panel structure for DDD data."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        panel = getattr(config, "panel", True)
        errors, warnings = _check_panel_mismatch(df, config.idname, config.tname, panel)

        if panel and df.select([config.idname, config.tname]).is_duplicated().any():
            errors.append(
                "The value of idname must be unique (by tname). Some units are observed more than once in a period."
            )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DynBalancingColumnValidator(BaseValidator):
    """Dynamic balancing column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        if not isinstance(config, DynBalancingConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        errors = []
        warnings = []
        data_columns = df.columns

        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "idname": config.idname,
            "treatment_name": config.treatment_name,
        }

        for col_type, col_name in required_cols.items():
            if col_name not in data_columns:
                errors.append(f"{col_type}='{col_name}' not found in data.")

        if config.clustervars:
            for cv in config.clustervars:
                if cv not in data_columns:
                    errors.append(f"clustervars contains '{cv}' which is not in the dataset.")

        if config.fixed_effects:
            for fe in config.fixed_effects:
                if fe not in data_columns:
                    errors.append(f"fixed_effects contains '{fe}' which is not in the dataset.")

        if config.xformla and config.xformla != "~1":
            covariate_vars = extract_vars_from_formula(config.xformla)
            for var in covariate_vars:
                if var not in data_columns:
                    errors.append(f"xformla contains '{var}' which is not in the dataset.")

        if not errors:
            for col_type in ["yname", "tname", "idname"]:
                col_name = getattr(config, col_type)
                if col_name in data_columns and not _is_numeric_dtype(df[col_name]):
                    errors.append(f"{col_type}='{col_name}' is not numeric. Please convert it.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


class DynBalancingPanelValidator(BaseValidator):
    """Dynamic balancing panel validator."""

    def validate(self, data, config):
        """Check that each unit has at most one row per period."""
        if not isinstance(config, DynBalancingConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        duplicate_error = _duplicate_unit_period_error(to_polars(data), config.idname, config.tname)
        errors = [] if duplicate_error is None else [duplicate_error]
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])


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
                PrePostArgumentValidator(),
                PrePostDataValidator(),
                PrePostPanelValidator(),
            ]

        if config_type == "didinter":
            return [
                DIDInterColumnValidator(),
                DIDInterArgumentValidator(),
                DIDInterTreatmentValidator(),
                DIDInterPanelValidator(),
            ]

        if config_type == "etwfe":
            return [
                ColumnValidator(),
                PanelStructureValidator(),
            ]

        if config_type == "ddd":
            return [
                DDDColumnValidator(),
                DDDArgumentValidator(),
                DDDPanelStructureValidator(),
                DDDInvarianceValidator(),
                DDDDataValidator(),
            ]

        if config_type == "dyn_balancing":
            return [
                DynBalancingColumnValidator(),
                DynBalancingPanelValidator(),
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


def _check_panel_mismatch(df: pl.DataFrame, idname: str | None, tname: str, panel: bool) -> tuple[list[str], list[str]]:
    """Check for mismatches between the panel parameter and actual data structure."""
    errors = []
    warnings = []

    if not idname:
        return errors, warnings

    obs_per_unit = df.group_by(idname).len()
    max_obs = obs_per_unit["len"].max()
    n_time_periods = df[tname].n_unique()

    if panel and max_obs == 1:
        errors.append(
            "panel=True was specified, but no units appear in multiple time periods. "
            "Your data appears to be repeated cross-sections. "
            "Set panel=False to use the repeated cross-section estimator."
        )
    elif not panel and max_obs == n_time_periods and n_time_periods > 1:
        warnings.append(
            "panel=False was specified, but units appear across all time periods. "
            "Your data appears to be panel data. "
            "Consider setting panel=True to use the panel estimator."
        )

    return errors, warnings


def _duplicate_unit_period_error(df, idname, tname):
    """Describe the units that have more than one row in a period.

    The message names up to three of the repeated pairs. Rows that miss the
    unit or the period are left out, since the missing-data step drops them.

    Parameters
    ----------
    df : pl.DataFrame
        Data that holds the unit and period columns.
    idname : str
        Name of the unit identifier column.
    tname : str
        Name of the period column.

    Returns
    -------
    str or None
        The error message, or None when no unit has two rows in one period.
    """
    if idname not in df.columns or tname not in df.columns:
        return None

    # Since etwfe and dyn_balancing keep rows with an infinite unit or period, two such rows still repeat a pair.
    keys = nonfinite_to_null(df.select(idname, tname), keep_infinite=(idname, tname)).drop_nulls()
    repeated = keys.filter(keys.is_duplicated()).unique().sort(idname, tname)
    if repeated.height == 0:
        return None

    shown = [f"({unit}, {period})" for unit, period in repeated.head(3).iter_rows()]
    listed = " and ".join(shown) if len(shown) < 3 else f"{', '.join(shown[:-1])}, and {shown[-1]}"
    if repeated.height <= 3:
        noun = "pair" if repeated.height == 1 else "pairs"
        where = f"the ({idname}, {tname}) {noun} {listed}"
    else:
        where = f"{repeated.height} ({idname}, {tname}) pairs, such as {listed}"
    return (
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        f"Rows repeat for {where}."
    )


def _ddd_partition_error(df, pname):
    """Describe a partition that takes values other than 0 and 1.

    Since the missing-data step drops rows with null values, the check skips them.

    Parameters
    ----------
    df : pl.DataFrame
        Data that holds the partition column.
    pname : str
        Name of the partition column.

    Returns
    -------
    str or None
        The error message, or None when every partition value is 0 or 1.
    """
    if pname not in df.columns:
        return None

    values = df[pname].drop_nulls()
    if not (values.dtype.is_numeric() or values.dtype == pl.Boolean):
        return f"pname='{pname}' is not numeric. Code it 1 for eligible units and 0 for ineligible units."

    invalid = values.filter(~values.cast(pl.Float64).is_in([0.0, 1.0])).unique().sort()
    if len(invalid) == 0:
        return None
    return (
        f"pname='{pname}' must be 1 for eligible units and 0 for ineligible units, "
        f"but it also takes the values {invalid.head(5).to_list()}."
    )


def _ddd_subgroup_error(subgroup, gname, pname):
    """Describe the treatment and eligibility subgroups that have no units.

    Parameters
    ----------
    subgroup : ndarray
        Subgroup of each unit or observation, coded from 1 to 4.
    gname : str
        Name of the treatment group column.
    pname : str
        Name of the partition column.

    Returns
    -------
    str or None
        The error message, or None when all four subgroups have units.
    """
    labels = {
        4: "treated and eligible",
        3: "treated and ineligible",
        2: "untreated and eligible",
        1: "untreated and ineligible",
    }
    present = set(np.unique(subgroup).tolist())
    missing = [f"subgroup {sg} ({label})" for sg, label in labels.items() if sg not in present]
    if not missing:
        return None
    return (
        f"No units fall in {' or '.join(missing)}. "
        f"The triple difference needs units in every combination of {gname} and {pname}."
    )


def _weights_error(weights, weightsname):
    """Describe sampling weights that are negative or have a mean that is not positive.

    Since normalizing divides each weight by the mean weight, a mean of zero
    turns every weight into NaN. A negative weight flips the sign of its
    observation. The intertemporal estimator keeps rows with a missing weight
    and gives them zero weight. The check leaves those weights out.

    Parameters
    ----------
    weights : pl.Series
        Sampling weights of the rows that the missing-data step keeps.
    weightsname : str
        Name of the weights column.

    Returns
    -------
    str or None
        The error message, or None when the weights pass the check or are not
        numeric.
    """
    if not (weights.dtype.is_numeric() or weights.dtype == pl.Boolean):
        return None
    values = weights.drop_nulls().cast(pl.Float64)
    if (values < 0).any() or (len(values) > 0 and values.mean() <= 0):
        return f"The weights variable '{weightsname}' must be non-negative with a positive mean."
    return None


def _reserved_name_errors(named_columns, reserved):
    """Describe the columns of a call whose names an internal column also uses.

    Preprocessing adds its own columns next to the columns a call names. A
    user column with the same name would be overwritten or read in place of
    the internal one.

    Parameters
    ----------
    named_columns : dict
        Each argument, such as ``"yname"``, mapped to the column or list of
        columns it names. None names no column.
    reserved : tuple of str
        Names of the internal columns that the estimator adds.

    Returns
    -------
    list of str
        One error message for each named column whose name is reserved.
    """
    errors = []
    for argument, names in named_columns.items():
        names = [names] if isinstance(names, str) else names or []
        errors.extend(
            f"{argument} names the column '{name}'. Since moderndid uses that name for an internal column, "
            "rename the column."
            for name in dict.fromkeys(names)
            if name in reserved
        )
    return errors


def _is_numeric_dtype(series: pl.Series) -> bool:
    """Check if a polars series has a numeric dtype."""
    return series.dtype.is_numeric()
