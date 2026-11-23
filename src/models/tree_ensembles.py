"""Model contracts for ML3 tree ensembles.

Estimator construction is intentionally deferred until model libraries,
hyperparameters and imbalance handling have been selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TreeModelSpec:
    """Serializable model request; it does not instantiate an estimator."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")


@runtime_checkable
class BinaryProbabilisticEstimator(Protocol):
    """Minimum estimator surface needed by the future ML3 trainer."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "BinaryProbabilisticEstimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TreeEstimatorFactory(Protocol):
    """Factory injected by the training configuration at a later stage."""

    def create(self, spec: TreeModelSpec, *, random_seed: int) -> BinaryProbabilisticEstimator: ...
