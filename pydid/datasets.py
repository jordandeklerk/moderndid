"""Dataset functions for pyDiD examples and testing."""

import gzip
import pickle
from pathlib import Path

import pandas as pd

__all__ = ["load_nsw"]


def load_nsw() -> pd.DataFrame:
    """Load the NSW (National Supported Work) demonstration dataset.

    This dataset is from the National Supported Work (NSW) Demonstration,
    a randomized employment training program operated in the mid-1970s.
    It has been widely used in the causal inference literature, particularly
    for demonstrating difference-in-differences methods.

    The dataset is in long format with observations for 1975 (pre-treatment)
    and 1978 (post-treatment) periods.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:

        - *id*: Individual identifier
        - *year*: Year (1975 or 1978)
        - *experimental*: Treatment indicator (1 if treated, 0 if control)
        - *re*: Real earnings (outcome variable)
        - *age*: Age in years
        - *educ*: Years of education
        - *black*: Indicator for Black race
        - *married*: Indicator for married status
        - *nodegree*: Indicator for no high school degree
        - *hisp*: Indicator for Hispanic ethnicity
        - *re74*: Real earnings in 1974

    Notes
    -----
    This dataset was used in Lalonde (1986) and has been extensively analyzed
    in the treatment effects literature. The version included here is formatted
    for panel data difference-in-differences analysis.

    References
    ----------
    .. [1] Lalonde, R. (1986). Evaluating the econometric evaluations of
        training programs with experimental data. American Economic Review,
        76(4), 604-620.
    """
    module_path = Path(__file__).parent
    data_path = module_path / "data" / "nsw_long.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"NSW data file not found at {data_path}. "
            "Please ensure the data file is included in the pyDiD installation."
        )

    with gzip.open(data_path, "rb") as f:
        nsw_data = pickle.load(f)

    return nsw_data
