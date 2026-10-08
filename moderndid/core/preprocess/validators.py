"""Validation classes for preprocessing."""

from difflib import SequenceMatcher
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
from .utils import extract_vars_from_formula, get_column_terms, get_formula_columns, nonfinite_to_null


class DataValidator(Protocol):
    """Data validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""


class ColumnValidator(BaseValidator):
    """Column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        df = to_polars(data)

        covariate_names = []
        if config.xformla and config.xformla != "~1":
            covariate_names = extract_vars_from_formula(config.xformla)
        dname = config.dname if isinstance(config, ContDIDConfig) else None
        named_columns = {
            "yname": config.yname,
            "tname": config.tname,
            "gname": config.gname,
            "idname": config.idname,
            "weightsname": config.weightsname,
            "clustervars": config.clustervars,
            "xformla": covariate_names,
            "dname": dname,
        }
        errors = _missing_column_errors(df.columns, named_columns)
        if errors:
            return self._create_result(errors)

        if not _is_numeric_dtype(df[config.tname]):
            errors.append(f"tname = '{config.tname}' is not numeric. Please convert it")

        if not _is_numeric_dtype(df[config.gname]):
            errors.append(f"gname = '{config.gname}' is not numeric. Please convert it")

        if config.idname and not _is_numeric_dtype(df[config.idname]):
            errors.append(f"idname = '{config.idname}' is not numeric. Please convert it")

        if not _is_numeric_dtype(df[config.yname]):
            errors.append(f"yname = '{config.yname}' is not numeric. Please convert it")

        if dname and not _is_numeric_dtype(df[dname]):
            errors.append(f"dname = '{dname}' is not numeric. Please convert it")

        errors.extend(_reserved_name_errors(named_columns, (WEIGHTS_COLUMN, ROW_ID_COLUMN)))

        return self._create_result(errors)

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

        if len([name for name in (config.clustervars or []) if name != config.idname]) > 1:
            errors.append("You can only provide 1 cluster variable additionally to the one provided in idname")

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

        if config.dname in df.columns and (df[config.dname] < 0).any():
            errors.append(f"dname = '{config.dname}' contains negative values")

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

        # Since the formula engine evaluates a transformed term such as I(x**2), only the plain terms are checked here.
        column_terms = []
        if config.xformla and config.xformla != "~1":
            column_terms = get_column_terms(config.xformla)
        required_cols = {
            "yname": config.yname,
            "tname": config.tname,
            "treat_col": config.treat_col,
            "idname": config.idname,
            "weightsname": config.weightsname,
            "xformla": column_terms,
        }
        errors.extend(_missing_column_errors(data_columns, required_cols))

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
        warnings = []

        covariate_names = []
        if config.xformla and config.xformla != "~1":
            covariate_names = extract_vars_from_formula(config.xformla)

        # The intertemporal config stores the caller's idname column in gname.
        named_columns = {
            "yname": config.yname,
            "tname": config.tname,
            "idname": config.gname,
            "dname": config.dname,
            "weightsname": config.weightsname,
            "cluster": config.cluster,
            "xformla": covariate_names,
            "trends_nonparam": config.trends_nonparam,
            "predict_het": config.predict_het[0] if config.predict_het else None,
        }
        errors = _missing_column_errors(df.columns, named_columns)
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
        # Every other column that preprocessing or the estimation adds starts with a dot.
        errors.extend(_reserved_name_errors(named_columns, reserved, prefix="."))

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

        # Since the intertemporal estimator keeps rows with a missing treatment, only the observed values count.
        treatment_changes = df.group_by(config.gname).agg(
            pl.col(config.dname).drop_nulls().n_unique().alias("n_unique")
        )
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

        # The intertemporal config stores the caller's idname column in gname.
        duplicate_error = _duplicate_unit_period_error(df, config.gname, config.tname)
        if duplicate_error is not None:
            errors.append(duplicate_error)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])


class DDDColumnValidator(BaseValidator):
    """DDD column validator."""

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate DDD columns exist and have correct types."""
        if not isinstance(config, DDDConfig):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        df = to_polars(data)
        warnings = []

        covariate_vars = []
        if config.xformla != "~1":
            try:
                covariate_vars = extract_vars_from_formula(config.xformla)
            except ValueError as e:
                return ValidationResult(is_valid=False, errors=[f"Invalid formula: {e}"], warnings=warnings)

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
        errors = _missing_column_errors(df.columns, named_columns)

        if not errors:
            for col_type in ["yname", "tname", "idname", "gname"]:
                col_name = getattr(config, col_type)
                if not _is_numeric_dtype(df[col_name]):
                    errors.append(f"{col_type}='{col_name}' is not numeric. Please convert it.")

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

        covariate_vars = []
        if config.xformla and config.xformla != "~1":
            covariate_vars = extract_vars_from_formula(config.xformla)
        named_columns = {
            "yname": config.yname,
            "tname": config.tname,
            "idname": config.idname,
            "treatment_name": config.treatment_name,
            "clustervars": config.clustervars,
            "fixed_effects": config.fixed_effects,
            "xformla": covariate_vars,
        }
        errors = _missing_column_errors(df.columns, named_columns)

        if not errors:
            for col_type in ["yname", "tname", "idname"]:
                col_name = getattr(config, col_type)
                if not _is_numeric_dtype(df[col_name]):
                    errors.append(f"{col_type}='{col_name}' is not numeric. Please convert it.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=[])


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
    """Run several validators and collect their errors and warnings.

    The default validators come in two phases. The ``"columns"`` phase checks
    the columns that the arguments name, their types, the reserved column
    names, and the argument values. Since these checks need no rows, they run
    on the data as given.

    The ``"structure"`` phase checks how the rows fit together, such as a
    cohort that changes within a unit or a unit observed twice in one period.
    It runs on the rows that the missing-data step keeps. A missing value in
    one row therefore never decides what these checks find.
    """

    def __init__(self, validators=None, config_type="did", phase="columns"):
        """Initialize composite validator."""
        if validators is not None:
            self.validators = validators
        else:
            self.validators = self._get_default_validators(config_type, phase)

    @staticmethod
    def _get_default_validators(config_type="did", phase="columns"):
        """Get the default validators of one phase."""
        if config_type == "two_period":
            columns = [PrePostColumnValidator(), PrePostArgumentValidator()]
            structure = [PrePostDataValidator(), PrePostPanelValidator()]
        elif config_type == "didinter":
            columns = [DIDInterColumnValidator(), DIDInterArgumentValidator()]
            structure = [DIDInterTreatmentValidator(), DIDInterPanelValidator()]
        elif config_type == "etwfe":
            columns = [ColumnValidator()]
            structure = [PanelStructureValidator()]
        elif config_type == "ddd":
            columns = [DDDColumnValidator(), DDDArgumentValidator()]
            structure = [DDDPanelStructureValidator(), DDDInvarianceValidator(), DDDDataValidator()]
        elif config_type == "dyn_balancing":
            columns = [DynBalancingColumnValidator()]
            structure = [DynBalancingPanelValidator()]
        else:
            columns = [ArgumentValidator(), ColumnValidator()]
            structure = [TreatmentValidator(), PanelStructureValidator(), ClusterValidator()]
            if config_type == "cont_did":
                structure.append(DoseValidator())

        return structure if phase == "structure" else columns

    def validate(self, data: DataFrame, config: BasePreprocessConfig) -> ValidationResult:
        """Validate data."""
        all_errors = []
        all_warnings = []

        for validator in self.validators:
            result = validator.validate(data, config)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return ValidationResult(is_valid=len(all_errors) == 0, errors=all_errors, warnings=all_warnings)


def check_columns(data, **arguments):
    """Check that the data hold every column that the arguments name.

    Each keyword is an argument of an estimator and gives the column it
    names, a list of columns, or None. The ``xformla`` keyword gives a
    formula whose plain terms name columns. A transformed term such as
    ``I(x**2)`` is left to the formula engine. A dict, such as the variance
    specification ``{"CRV1": "state"}``, names the columns in its values.
    Two-way clustering joins two of them with a plus sign.

    A single error lists every missing column. Each line names the argument
    and suggests the closest column names, as in "tname='yeer' is not a
    column in the data. Did you mean 'year'?".

    Parameters
    ----------
    data : DataFrame
        Data that the estimator reads.
    **arguments
        Each argument of the estimator that names columns, mapped to its
        value.
    """
    columns = to_polars(data).columns
    named_columns = {}
    for argument, value in arguments.items():
        if argument == "xformla":
            value = get_column_terms(value) if value else []
        elif isinstance(value, dict):
            specs = [spec for spec in value.values() if isinstance(spec, str)]
            value = [part.strip() for spec in specs for part in ([spec] if spec in columns else spec.split("+"))]
        named_columns[argument] = value

    errors = _missing_column_errors(columns, named_columns)
    if errors:
        raise ValueError("\n".join(errors))


def _missing_column_errors(columns, named_columns):
    """Describe each column that an argument names and the data lack.

    Parameters
    ----------
    columns : list of str
        Column names of the data.
    named_columns : dict
        Each argument, such as ``"tname"``, mapped to the column or list of
        columns it names. None names no column.

    Returns
    -------
    list of str
        One message for each missing column. It names the argument and
        suggests the closest column names.
    """
    available = set(columns)
    errors = []
    for argument, names in named_columns.items():
        listed = [names] if isinstance(names, str) else names or []
        for name in dict.fromkeys(listed):
            if name in available:
                continue
            where = f"{argument}='{name}'" if isinstance(names, str) else f"'{name}' in {argument}"
            message = f"{where} is not a column in the data."
            close = [f"'{column}'" for column in _closest_columns(name, columns)]
            if close:
                options = " or ".join(close) if len(close) < 3 else f"{', '.join(close[:-1])}, or {close[-1]}"
                message += f" Did you mean {options}?"
            errors.append(message)
    return errors


def _closest_columns(name, columns):
    """Return up to three column names that resemble a name."""
    lowered = str(name).lower()
    scores = {column: SequenceMatcher(None, lowered, column.lower()).ratio() for column in columns}
    close = [column for column in columns if scores[column] >= 0.6]
    return sorted(close, key=scores.get, reverse=True)[:3]


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


def _duplicate_unit_period_error(df, idname, tname, id_argument="idname", time_argument="tname"):
    """Describe the units that have more than one row in a period.

    The message names up to three of the repeated pairs. Since the missing-data
    step drops rows whose unit or period is null, NaN, or infinite, the check
    leaves them out.

    Parameters
    ----------
    df : pl.DataFrame
        Data that holds the unit and period columns.
    idname : str
        Name of the unit identifier column.
    tname : str
        Name of the period column.
    id_argument : str, default "idname"
        Name that the message gives the argument of the unit column.
    time_argument : str, default "tname"
        Name that the message gives the argument of the period column.

    Returns
    -------
    str or None
        The error message, or None when no unit has two rows in one period.
    """
    if idname not in df.columns or tname not in df.columns:
        return None

    keys = nonfinite_to_null(df.select(idname, tname)).drop_nulls()
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
        f"The value of {id_argument} must be unique (by {time_argument}). "
        "Some units are observed more than once in a period. "
        f"Rows repeat for {where}."
    )


def _ddd_partition_error(df, pname, argument="pname"):
    """Describe a partition that takes values other than 0 and 1.

    Since the missing-data step drops rows with null values, the check skips them.

    Parameters
    ----------
    df : pl.DataFrame
        Data that holds the partition column.
    pname : str
        Name of the partition column.
    argument : str, default "pname"
        Name that the message gives the argument of the partition column.

    Returns
    -------
    str or None
        The error message, or None when every partition value is 0 or 1.
    """
    if pname not in df.columns:
        return None

    values = df[pname].drop_nulls()
    if not (values.dtype.is_numeric() or values.dtype == pl.Boolean):
        return f"{argument}='{pname}' is not numeric. Code it 1 for eligible units and 0 for ineligible units."

    invalid = values.filter(~values.cast(pl.Float64).is_in([0.0, 1.0])).unique().sort()
    if len(invalid) == 0:
        return None
    return (
        f"{argument}='{pname}' must be 1 for eligible units and 0 for ineligible units, "
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


def _reserved_name_errors(named_columns, reserved, prefix=None):
    """Describe the columns of a call whose names an internal column also uses.

    Preprocessing adds its own columns next to the columns a call names. A
    user column with the same name would be overwritten or read in place of
    the internal one. When the other internal columns of an estimator share a
    prefix, every name that starts with it is reserved too.

    Parameters
    ----------
    named_columns : dict
        Each argument, such as ``"yname"``, mapped to the column or list of
        columns it names. None names no column.
    reserved : tuple of str
        Names of the internal columns that the estimator adds.
    prefix : str, optional
        Prefix that the estimator's other internal columns start with.

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
            if name in reserved or (prefix is not None and name.startswith(prefix))
        )
    return errors


def _is_numeric_dtype(series: pl.Series) -> bool:
    """Check if a polars series has a numeric dtype."""
    return series.dtype.is_numeric()
