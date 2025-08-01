"""Builder pattern for constructing DIDData objects."""

import warnings
from typing import Any

import numpy as np
import pandas as pd

from causaldid.utils import extract_vars_from_formula

from .models import DIDConfig, DIDData
from .tensors import TensorFactorySelector
from .transformers import DataTransformerPipeline
from .validators import CompositeValidator


class DIDDataBuilder:
    """Builder for constructing DIDData objects with fluent interface."""

    def __init__(self):
        """Initialize builder."""
        self._data: pd.DataFrame | None = None
        self._config: DIDConfig | None = None
        self._validator = CompositeValidator()
        self._transformer = DataTransformerPipeline()
        self._tensor_factory = TensorFactorySelector()
        self._warnings: list[str] = []

    def with_data(self, data: pd.DataFrame) -> "DIDDataBuilder":
        """Set the data."""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        self._data = data
        return self

    def with_config(self, config: DIDConfig) -> "DIDDataBuilder":
        """Set the configuration."""
        self._config = config
        return self

    def with_config_dict(self, **kwargs: Any) -> "DIDDataBuilder":
        """Create and set configuration from keyword arguments."""
        self._config = DIDConfig(**kwargs)
        return self

    def validate(self) -> "DIDDataBuilder":
        """Validate data and configuration."""
        if self._data is None:
            raise ValueError("Data not set. Use with_data() first.")
        if self._config is None:
            raise ValueError("Configuration not set. Use with_config() first.")

        result = self._validator.validate(self._data, self._config)

        self._warnings.extend(result.warnings)

        if self._config.print_details and result.warnings:
            for warning in result.warnings:
                warnings.warn(warning)

        result.raise_if_invalid()

        return self

    def transform(self) -> "DIDDataBuilder":
        """Apply data transformations."""
        if self._data is None or self._config is None:
            raise ValueError("Must set data and config before transforming")

        self._data = self._transformer.transform(self._data, self._config)

        self._validate_transformed_data()

        return self

    def _validate_transformed_data(self) -> None:
        """Validate transformed data meets minimum requirements."""
        if self._data is None or self._config is None:
            return

        glist = self._config.treated_groups
        if len(glist) == 0:
            raise ValueError(
                "No valid groups. The variable in 'gname' should be expressed as "
                "the time a unit is first treated (0 if never-treated)"
            )

        if self._config.panel:
            gsize = self._data.groupby(self._config.gname).size() / self._config.time_periods_count
        else:
            gsize = self._data.groupby(self._config.gname).size()

        if self._config and self._config.xformla and self._config.xformla != "~1":
            formula_vars = extract_vars_from_formula(self._config.xformla)
            n_covs = len([v for v in formula_vars if v != self._config.yname])
        else:
            n_covs = 0
        reqsize = n_covs + 5

        small_groups = gsize[gsize < reqsize]
        if len(small_groups) > 0:
            group_list = ", ".join([str(g) for g in small_groups.index])
            warning_msg = f"Be aware that there are some small groups in your dataset.\nCheck groups: {group_list}"
            warnings.warn(warning_msg)
            self._warnings.append(warning_msg)

            from .constants import NEVER_TREATED_VALUE

            if NEVER_TREATED_VALUE in small_groups.index and self._config.control_group.value == "nevertreated":
                raise ValueError("Never treated group is too small, try setting control_group='notyettreated'")

    def build(self) -> DIDData:
        """Build the final DIDData object."""
        if self._data is None or self._config is None:
            raise ValueError("Must set data and config before building")

        tensor_data = self._tensor_factory.create_tensors(self._data, self._config)

        if self._config.print_details:
            summary = self._get_summary(tensor_data)
            if summary:
                warnings.warn(summary, stacklevel=2)

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

    def _get_summary(self, tensor_data: dict[str, Any]) -> str | None:
        """Get preprocessing summary as a string."""
        if self._config is None:
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
            for _, row in cohort_counts.iterrows():
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
