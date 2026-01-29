"""Difference-in-Differences for Intertemporal Treatment Effects."""

from .results import ATEResult, DIDInterResult, EffectsResult, PlacebosResult

__all__ = [
    "DIDInterResult",
    "EffectsResult",
    "PlacebosResult",
    "ATEResult",
]
