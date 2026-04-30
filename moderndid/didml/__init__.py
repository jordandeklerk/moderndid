"""Machine learning models for Difference-in-Differences."""

try:
    import cvxpy
    import econml
    import sklearn
    import xgboost
except ImportError as e:
    raise ImportError(
        "The 'didml' module requires additional dependencies. Install them with: uv pip install 'moderndid[didml]'"
    ) from e

from moderndid.core.preprocess.config import DIDMLConfig

from .container import (
    BLPResult,
    CLANResult,
    DIDMLAggResult,
    DIDMLResult,
    blp_result,
    clan_result,
    didml_agg,
    didml_result,
    summary_didml,
    summary_didml_agg,
)
from .weights import solve_minimax_weights

__all__ = [
    "BLPResult",
    "CLANResult",
    "DIDMLAggResult",
    "DIDMLConfig",
    "DIDMLResult",
    "blp_result",
    "clan_result",
    "didml_agg",
    "didml_result",
    "solve_minimax_weights",
    "summary_didml",
    "summary_didml_agg",
]
