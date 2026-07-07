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
from .aggte import aggte_didml, dynamic_cates
from .didml import didml
from .heterogeneity import blp_eventtimes, clan_glhtest, clan_ttest, het_prep
from .lnw import lnw_did
from .weights import amle_weights
from . import format as _format

__all__ = [
    "BLPResult",
    "CLANResult",
    "DIDMLAggResult",
    "DIDMLConfig",
    "DIDMLResult",
    "aggte_didml",
    "amle_weights",
    "blp_eventtimes",
    "blp_result",
    "clan_glhtest",
    "clan_result",
    "clan_ttest",
    "didml",
    "didml_agg",
    "didml_result",
    "dynamic_cates",
    "het_prep",
    "lnw_did",
    "summary_didml",
    "summary_didml_agg",
]
