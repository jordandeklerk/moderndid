"""Preprocessing for multi-period DiD estimators."""

from .builders import DIDDataBuilder
from .constants import (
    BasePeriod,
    ControlGroup,
    DataFormat,
    EstimationMethod,
)
from .models import DIDConfig, DIDData, ValidationResult
from .tensors import TensorFactorySelector
from .transformers import DataTransformerPipeline
from .validators import CompositeValidator

__all__ = [
    # Builders
    "DIDDataBuilder",
    # Constants
    "BasePeriod",
    "ControlGroup",
    "DataFormat",
    "EstimationMethod",
    # Models
    "DIDConfig",
    "DIDData",
    "ValidationResult",
    # Factories and Pipelines
    "TensorFactorySelector",
    "DataTransformerPipeline",
    "CompositeValidator",
]
