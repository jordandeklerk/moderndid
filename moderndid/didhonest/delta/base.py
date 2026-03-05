"""Base classes and protocols for delta methods."""

from typing import NamedTuple, Protocol


class DeltaResult(NamedTuple):
    """Container for delta method computation results.

    Attributes
    ----------
    id_lb : float
        Lower bound of the identified set.
    id_ub : float
        Upper bound of the identified set.
    """

    #: Lower bound of the identified set.
    id_lb: float
    #: Upper bound of the identified set.
    id_ub: float


class ConditionalCSResult(NamedTuple):
    """Container for conditional confidence set computation results.

    Attributes
    ----------
    grid : ndarray
        Grid of parameter values tested.
    accept : ndarray
        Boolean array indicating acceptance at each grid point.
    """

    #: Grid of parameter values tested.
    grid: any
    #: Boolean array indicating acceptance at each grid point.
    accept: any


class DeltaMethodProtocol(Protocol):
    """Protocol for delta method functions."""

    def __call__(
        self,
        betahat,
        sigma,
        num_pre_periods,
        num_post_periods,
        l_vec=None,
        alpha=0.05,
        hybrid_flag="FLCI",
        grid_points=1000,
        grid_lb=None,
        grid_ub=None,
        **kwargs,
    ) -> dict:
        """Compute conditional confidence set under delta restrictions."""
