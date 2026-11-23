"""Configuration contracts for the future ML3 training runner.

This module deliberately contains no estimator selection, class-imbalance
policy, fitting loop or threshold policy yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from src.models.tree_ensembles import BinaryProbabilisticEstimator, TreeModelSpec

ProtocolName = Literal["within", "loso"]


@dataclass(frozen=True)
class FoldDefinition:
    """Indices and identity of one repository-defined evaluation fold."""

    protocol: ProtocolName
    subject_id: str
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if np.intersect1d(self.train_indices, self.test_indices).size:
            raise ValueError("train and test indices must be disjoint")


@dataclass(frozen=True)
class ML3TrainingPlan:
    """A dry, serializable plan that can be finalized before training."""

    feature_path: Path
    output_dir: Path
    model_specs: tuple[TreeModelSpec, ...] = ()
    protocols: tuple[ProtocolName, ...] = ("within", "loso")
    random_seed: int = 42
    checkpoint_each_fold: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.model_specs)


@runtime_checkable
class FoldTrainer(Protocol):
    """Interface for the fitting policy that will be selected later."""

    def fit_fold(
        self,
        estimator: BinaryProbabilisticEstimator,
        fold: FoldDefinition,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Mapping[str, object]: ...


def require_configured_plan(plan: ML3TrainingPlan) -> None:
    """Prevent scaffold entrypoints from accidentally producing fake results."""

    if not plan.is_configured:
        raise ValueError(
            "ML3 training is not configured: choose model specifications and "
            "training policy before running folds"
        )
