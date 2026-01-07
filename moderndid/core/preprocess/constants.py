"""Constants and enums for preprocessing."""

from enum import Enum


class ControlGroup(str, Enum):
    """Control group."""

    NEVER_TREATED = "nevertreated"
    NOT_YET_TREATED = "notyettreated"


class EstimationMethod(str, Enum):
    """Estimation method."""

    DOUBLY_ROBUST = "dr"
    IPW = "ipw"
    REGRESSION = "reg"


class BasePeriod(str, Enum):
    """Base period."""

    UNIVERSAL = "universal"
    VARYING = "varying"


class BootstrapType(str, Enum):
    """Bootstrap type."""

    WEIGHTED = "weighted"
    MULTIPLIER = "multiplier"
    EMPIRICAL = "empirical"


class DataFormat(str, Enum):
    """Data format."""

    PANEL = "panel"
    REPEATED_CROSS_SECTION = "repeated_cross_section"
    UNBALANCED_PANEL = "unbalanced_panel"


DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_ANTICIPATION_PERIODS = 0
DEFAULT_TRIM_LEVEL = 0.995
DEFAULT_CORES = 1

WEIGHTS_COLUMN = "weights"
ROW_ID_COLUMN = ".rowid"

NEVER_TREATED_VALUE = float("inf")

MIN_GROUP_SIZE_BASE = 5
MIN_GROUP_SIZE_PER_COVARIATE = 1

DEFAULT_SPLINE_DEGREE = 2
DEFAULT_NUM_KNOTS = 3
