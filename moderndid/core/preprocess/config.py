"""Configuration classes for preprocessing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_ANTICIPATION_PERIODS,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CORES,
    DEFAULT_NUM_KNOTS,
    DEFAULT_SPLINE_DEGREE,
    BasePeriod,
    BootstrapType,
    ControlGroup,
    DataFormat,
    EstimationMethod,
)


class ConfigMixin:
    """Mixin for config methods."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v.value if isinstance(v, Enum) else v for k, v in self.__dict__.items()}


@dataclass
class BasePreprocessConfig(ConfigMixin):
    """Base preprocess config."""

    yname: str
    tname: str
    gname: str

    idname: str | None = None
    xformla: str = "~1"
    panel: bool = True
    allow_unbalanced_panel: bool = True
    weightsname: str | None = None
    alp: float = DEFAULT_ALPHA
    bstrap: bool = False
    cband: bool = False
    biters: int = DEFAULT_BOOTSTRAP_ITERATIONS
    clustervars: list[str] = field(default_factory=list)
    anticipation: int = DEFAULT_ANTICIPATION_PERIODS
    faster_mode: bool = False
    pl: bool = False
    cores: int = DEFAULT_CORES

    true_repeated_cross_sections: bool = False
    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    treated_groups: np.ndarray = field(default_factory=lambda: np.array([]))
    treated_groups_count: int = 0
    id_count: int = 0
    data_format: DataFormat = DataFormat.PANEL


@dataclass
class DIDConfig(BasePreprocessConfig):
    """DID config."""

    control_group: ControlGroup = ControlGroup.NEVER_TREATED
    est_method: EstimationMethod = EstimationMethod.DOUBLY_ROBUST
    base_period: BasePeriod = BasePeriod.VARYING


@dataclass
class TwoPeriodDIDConfig(ConfigMixin):
    """Two-period DiD config."""

    yname: str
    tname: str
    treat_col: str
    idname: str | None = None
    xformla: str = "~1"
    panel: bool = True
    weightsname: str | None = None
    alp: float = DEFAULT_ALPHA
    bstrap: bool = False
    boot_type: BootstrapType = BootstrapType.WEIGHTED
    biters: int = DEFAULT_BOOTSTRAP_ITERATIONS
    est_method: str = "imp"
    trim_level: float = 0.995
    inf_func: bool = False
    normalized: bool = True

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    treated_groups: np.ndarray = field(default_factory=lambda: np.array([]))
    treated_groups_count: int = 0
    id_count: int = 0


@dataclass
class ContDIDConfig(BasePreprocessConfig):
    """ContDID config."""

    dname: str | None = None
    degree: int = DEFAULT_SPLINE_DEGREE
    num_knots: int = DEFAULT_NUM_KNOTS
    dvals: np.ndarray | None = None
    knots: np.ndarray | None = None
    boot_type: BootstrapType = BootstrapType.MULTIPLIER
    control_group: ControlGroup = ControlGroup.NOT_YET_TREATED
    base_period: BasePeriod = BasePeriod.VARYING
    required_pre_periods: int = 0
    gt_type: str = "att"
    ret_quantile: float | None = None
    target_parameter: str = "att"
    aggregation: str = "dose"
    treatment_type: str = "continuous"
    time_map: dict | None = None


@dataclass
class DDDConfig(ConfigMixin):
    """Triple Difference-in-Differences config."""

    yname: str = ""
    tname: str = ""
    idname: str = ""
    gname: str = ""
    pname: str = ""
    xformla: str = "~1"
    est_method: EstimationMethod = EstimationMethod.DOUBLY_ROBUST
    weightsname: str | None = None
    boot: bool = False
    boot_type: BootstrapType = BootstrapType.MULTIPLIER
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS
    cluster: str | None = None
    cband: bool = False
    alp: float = DEFAULT_ALPHA
    inf_func: bool = False

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    n_units: int = 0
