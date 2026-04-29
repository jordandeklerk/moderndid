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
    boot: bool = False
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
class DIDMLConfig(DIDConfig):
    """DIDML config."""

    nu_model: str = "rlearner"
    sigma_model: str = "rlearner"
    delta_model: str = "glm"

    k_folds: int = 10
    lambda_choice: str = "lambda.min"
    tune_penalty: bool = True
    t_func: bool = True

    use_gamma: bool = True
    zeta: float = 0.5

    compute_drdid_benchmark: bool = True

    random_state: int | None = None

    control_group: ControlGroup = ControlGroup.NOT_YET_TREATED

    def __post_init__(self) -> None:
        """Validate config fields."""
        if self.nu_model not in ("rlearner", "cf"):
            raise ValueError(f"nu_model must be 'rlearner' or 'cf', got {self.nu_model!r}.")
        if self.sigma_model not in ("rlearner", "cf"):
            raise ValueError(f"sigma_model must be 'rlearner' or 'cf', got {self.sigma_model!r}.")
        if self.delta_model not in ("glm", "stack"):
            raise ValueError(f"delta_model must be 'glm' or 'stack', got {self.delta_model!r}.")
        if self.lambda_choice not in ("lambda.min", "lambda.1se"):
            raise ValueError(f"lambda_choice must be 'lambda.min' or 'lambda.1se', got {self.lambda_choice!r}.")
        if not 0 < self.zeta < 1:
            raise ValueError(f"zeta must be in the open interval (0, 1), got {self.zeta!r}.")
        if self.k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {self.k_folds!r}.")


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
    boot: bool = False
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
    dose_est_method: str = "parametric"
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
    panel: bool = True
    allow_unbalanced_panel: bool = False

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    n_units: int = 0


@dataclass
class DIDInterConfig(ConfigMixin):
    """DIDInter config for heterogeneous/dynamic treatment effects."""

    yname: str = ""
    tname: str = ""
    gname: str = ""
    dname: str = ""

    cluster: str | None = None
    weightsname: str | None = None
    controls: list[str] | None = None
    xformla: str = "~1"
    trends_nonparam: list[str] | None = None

    effects: int = 1
    placebo: int = 0
    normalized: bool = False
    effects_equal: bool = False

    switchers: str = ""
    only_never_switchers: bool = False
    same_switchers: bool = False
    same_switchers_pl: bool = False

    ci_level: float = 95.0
    alp: float = DEFAULT_ALPHA
    less_conservative_se: bool = False

    trends_lin: bool = False
    continuous: int = 0

    keep_bidirectional_switchers: bool = False
    drop_missing_preswitch: bool = False

    predict_het: tuple[list[str], list[int]] | None = None
    predict_het_hc2bm: bool = False
    more_granular_demeaning: bool = False

    effects_equal_lb: int | None = None
    effects_equal_ub: int | None = None

    boot: bool = False
    biters: int = 1000
    random_state: int | None = None

    panel: bool = True
    allow_unbalanced_panel: bool = True

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    n_groups: int = 0
    id_count: int = 0
    max_effects_available: int = 0
    max_placebo_available: int = 0
    data_format: DataFormat = DataFormat.PANEL


@dataclass
class EtwfeConfig(ConfigMixin):
    """Extended Two-Way Fixed Effects config."""

    yname: str = ""
    tname: str = ""
    gname: str = ""
    idname: str | None = None
    xformla: str = "~1"
    xvar: str | None = None
    tref: int | float | None = None
    gref: int | float | None = None
    cgroup: str = "notyet"
    fe: str = "vs"
    family: str | None = None
    weightsname: str | None = None
    cluster: str | None = None
    alp: float = DEFAULT_ALPHA
    panel: bool = True

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    time_periods_count: int = 0
    treated_groups: np.ndarray = field(default_factory=lambda: np.array([]))
    treated_groups_count: int = 0
    n_units: int = 0
    n_obs: int = 0

    # Internal state set during prepare_etwfe_data / build_etwfe_formula
    _gref_min_flag: bool = False
    _ctrls: list = field(default_factory=list)
    _xvar_dm_cols: list = field(default_factory=list)
    _xvar_time_dummies: list = field(default_factory=list)
    _formula: str = ""


@dataclass
class DynBalancingConfig(ConfigMixin):
    """Dynamic covariate balancing config."""

    yname: str = ""
    tname: str = ""
    idname: str = ""
    treatment_name: str = ""
    ds1: list[int] = field(default_factory=list)
    ds2: list[int] = field(default_factory=list)
    xformla: str | None = None
    fixed_effects: list[str] | None = None
    pooled: bool = False
    clustervars: list[str] | None = None
    balancing: str = "dcb"
    method: str = "lasso_plain"
    alp: float = DEFAULT_ALPHA
    final_period: int | None = None
    initial_period: int | None = None
    adaptive_balancing: bool = True
    debias: bool = False
    continuous_treatment: bool = False
    lb: float = 0.0005
    ub: float = 2.0
    regularization: bool = True
    fast_adaptive: bool = False
    grid_length: int = 1000
    n_beta_nonsparse: float = 1e-4
    ratio_coefficients: float = 1 / 3
    nfolds: int = 10
    lags: int | None = None
    robust_quantile: bool = True
    demeaned_fe: bool = False

    time_periods: np.ndarray = field(default_factory=lambda: np.array([]))
    n_units: int = 0
    n_periods: int = 0
    covariate_names: list[str] = field(default_factory=list)
