"""Abstract base classes for preprocessing."""

from abc import ABC, abstractmethod
from typing import Protocol

from ..dataframe import DataFrame


class BaseConfig(ABC):
    """Base config."""


class BasePreprocessedData(ABC):
    """Base preprocessed data."""


class DataTransformer(Protocol):
    """Data transformer."""

    def transform(self, data: DataFrame, config: BaseConfig):
        """Transform data."""


class BaseTransformer(ABC):
    """Base transformer."""

    @abstractmethod
    def transform(self, data: DataFrame, config: BaseConfig):
        """Transform data."""


class DataValidator(Protocol):
    """Data validator."""

    def validate(self, data: DataFrame, config: BaseConfig):
        """Validate data."""


class BaseValidator(ABC):
    """Base validator."""

    @abstractmethod
    def validate(self, data: DataFrame, config: BaseConfig):
        """Validate data."""


class TensorFactory(Protocol):
    """Tensor factory."""

    def create_tensors(self, data: DataFrame, config: BaseConfig):
        """Create tensors."""


class BaseTensorFactory(ABC):
    """Base tensor factory."""

    @abstractmethod
    def create_outcomes_tensor(self, data: DataFrame, config: BaseConfig):
        """Create outcomes tensor."""

    @abstractmethod
    def create_covariates_tensor(self, data: DataFrame, config: BaseConfig):
        """Create covariates tensor."""
