from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.handover_validation import PER_FOLD_COLUMNS, validate_handover_directory


def _write_valid_pair(directory) -> None:
    pd.DataFrame(
        [["sub-01", 0, 0.5, 0.7, 0.3, 0.4, 0.6, 0.4]],
        columns=PER_FOLD_COLUMNS,
    ).to_csv(directory / "demo_loso_per_fold.csv", index=False)
    np.savez_compressed(
        directory / "demo_loso_preds.npz",
        subject_id=np.array(["sub-01"]),
        fold=np.array([0]),
        trial_idx=np.array([0]),
        trial_uid=np.array(["trial-0"]),
        y_true=np.array([1]),
        y_prob=np.array([0.8]),
        y_pred=np.array([1]),
    )


def test_valid_handover_pair_passes(tmp_path) -> None:
    _write_valid_pair(tmp_path)
    report = validate_handover_directory(tmp_path, require_prediction_pair=True)
    assert report.is_valid
    assert report.files_checked == 2


def test_metric_outside_unit_interval_fails(tmp_path) -> None:
    _write_valid_pair(tmp_path)
    path = tmp_path / "demo_loso_per_fold.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "roc_auc"] = 1.5
    frame.to_csv(path, index=False)
    report = validate_handover_directory(tmp_path)
    assert not report.is_valid
    assert any("outside [0, 1]" in issue.message for issue in report.issues)
