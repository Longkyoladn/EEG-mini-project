from pathlib import Path

import numpy as np
import pytest

from src.training.ml3_training import FoldDefinition, ML3TrainingPlan, require_configured_plan


def test_fold_contract_rejects_train_test_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        FoldDefinition("loso", "sub-01", 0, np.array([0, 1]), np.array([1, 2]))


def test_unconfigured_plan_cannot_start_training() -> None:
    plan = ML3TrainingPlan(Path("features.npz"), Path("results/ml3"))
    with pytest.raises(ValueError, match="not configured"):
        require_configured_plan(plan)
