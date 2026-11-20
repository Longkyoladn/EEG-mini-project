"""Tải dữ liệu chuẩn Data Contract và chuẩn hóa Zero-Leakage cho nhánh DL2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

VALID_DATA_VARIANTS = ("clean", "no_ica")
VALID_PROTOCOLS = ("within_subject", "cross_subject_5fold", "cross_subject_loso")


def natural_sort_key(value: str) -> List[Union[int, str]]:
    """Khóa sắp xếp tự nhiên theo số học (vd: sub-02 trước sub-10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def compute_trial_uid_sha256(trial_uids: Sequence[str]) -> str:
    """Tính mã băm SHA-256 của danh sách `trial_uid` theo đúng chuẩn Data Contract."""
    payload = "\n".join(str(uid) for uid in trial_uids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class EEGDatasetBundle:
    """Gói dữ liệu đầu vào đã được xác thực hợp đồng dữ liệu."""

    data_dir: Path
    data_variant: str
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    metadata: pd.DataFrame
    splits: Dict[str, Any]
    trial_uid_sha256: str


class ChannelStandardScaler:
    """Bộ chuẩn hóa Z-score theo từng kênh EEG với cơ chế chống sập đơn vị Volts.

    Dữ liệu EEG thô lưu theo chuẩn MNE ở đơn vị Volts (độ lệch chuẩn ~1e-5 V).
    Nếu dùng trực tiếp `eps=1e-5` ở đơn vị Volts, mẫu số `(std + eps)` sẽ bóp méo
    toàn bộ biên độ tín hiệu. Lớp này quy đổi nội bộ sang microvolts (`* 1e6`) trên
    độ chính xác kép `float64` với `eps=1e-8`, sau đó trả về tensor `float32`.
    """

    def __init__(self, eps: float = 1e-8, scale_factor: float = 1e6) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps phải > 0, nhận được {eps}.")
        if scale_factor <= 0.0:
            raise ValueError(f"scale_factor phải > 0, nhận được {scale_factor}.")
        self.eps = float(eps)
        self.scale_factor = float(scale_factor)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None and self.n_channels_ is not None

    def fit(self, X_train: np.ndarray) -> "ChannelStandardScaler":
        """Chỉ ước lượng mean và std theo kênh từ tập huấn luyện `X_train` `(N, C, T)`."""
        arr = np.asarray(X_train)
        if arr.ndim != 3:
            raise ValueError(f"Kỳ vọng tensor 3D (N, C, T), nhận được shape {arr.shape}.")
        if arr.shape[0] == 0 or arr.shape[2] == 0:
            raise ValueError(f"Tập X_train rỗng không thể fit scaler: shape={arr.shape}.")

        x_uv = arr.astype(np.float64) * self.scale_factor
        mean = np.mean(x_uv, axis=(0, 2), keepdims=True)
        std = np.std(x_uv, axis=(0, 2), keepdims=True)

        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError("Phát hiện giá trị NaN/Inf trong X_train khi fit ChannelStandardScaler.")

        self.mean_ = mean
        self.std_ = std
        self.n_channels_ = int(arr.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa `X` bằng đúng thống kê đã fit trên train (tuyệt đối không refit)."""
        if not self.is_fitted:
            raise RuntimeError("ChannelStandardScaler chưa được fit trước khi gọi transform().")

        arr = np.asarray(X)
        if arr.ndim != 3:
            raise ValueError(f"Kỳ vọng tensor 3D (N, C, T), nhận được shape {arr.shape}.")
        if arr.shape[1] != self.n_channels_:
            raise ValueError(
                f"Số kênh không khớp scaler đã fit ({self.n_channels_}), nhận được {arr.shape[1]}."
            )

        assert self.mean_ is not None and self.std_ is not None
        x_uv = arr.astype(np.float64) * self.scale_factor
        normed = (x_uv - self.mean_) / (self.std_ + self.eps)
        return normed.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit trên `X_train` và trả về `X_train` đã chuẩn hóa `float32`."""
        return self.fit(X_train).transform(X_train)

    def state_dict(self) -> Dict[str, Any]:
        """Xuất trạng thái scaler tương thích `torch.save(..., weights_only=True)`."""
        if not self.is_fitted:
            raise RuntimeError("Không thể xuất state_dict() khi scaler chưa được fit.")
        assert self.mean_ is not None and self.std_ is not None and self.n_channels_ is not None
        return {
            "mean": torch.from_numpy(self.mean_.copy()).to(dtype=torch.float64),
            "std": torch.from_numpy(self.std_.copy()).to(dtype=torch.float64),
            "eps": float(self.eps),
            "scale_factor": float(self.scale_factor),
            "n_channels": int(self.n_channels_),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ChannelStandardScaler":
        """Nạp lại trạng thái scaler từ checkpoint `.pt`."""
        for key in ("mean", "std", "eps", "scale_factor"):
            if key not in state:
                raise KeyError(f"state_dict thiếu khóa bắt buộc '{key}'.")

        mean_raw = state["mean"]
        std_raw = state["std"]
        mean_arr = (
            mean_raw.detach().cpu().numpy().astype(np.float64)
            if isinstance(mean_raw, torch.Tensor)
            else np.asarray(mean_raw, dtype=np.float64)
        )
        std_arr = (
            std_raw.detach().cpu().numpy().astype(np.float64)
            if isinstance(std_raw, torch.Tensor)
            else np.asarray(std_raw, dtype=np.float64)
        )

        if mean_arr.ndim != 3 or std_arr.ndim != 3 or mean_arr.shape != std_arr.shape:
            raise ValueError(
                f"Shape mean/std không hợp lệ trong state_dict: {mean_arr.shape} vs {std_arr.shape}."
            )

        self.eps = float(state["eps"])
        self.scale_factor = float(state["scale_factor"])
        self.mean_ = mean_arr
        self.std_ = std_arr
        self.n_channels_ = int(state.get("n_channels", mean_arr.shape[1]))
        return self

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ChannelStandardScaler":
        """Khởi tạo mới một `ChannelStandardScaler` từ `state_dict` đã lưu."""
        scaler = cls(
            eps=float(state["eps"]),
            scale_factor=float(state["scale_factor"]),
        )
        return scaler.load_state_dict(state)


class EEGTrialDataset(Dataset):
    """Dataset PyTorch cho single-trial EEG `(127, 251)` và nhãn nhị phân `(1,)`."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_arr.ndim != 3:
            raise ValueError(f"X phải là mảng 3D (N, C, T), nhận được shape {x_arr.shape}.")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Số lượng mẫu không khớp giữa X ({x_arr.shape[0]}) và y ({y_arr.shape[0]})."
            )
        self.X = torch.from_numpy(np.ascontiguousarray(x_arr, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y_arr, dtype=np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _build_within_splits_for_arrays(
    subjects: np.ndarray,
    y: np.ndarray,
    unique_subjects: List[str],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Sinh lại chia tập `within_subject` theo cùng seed khi dùng nhánh `no_ica`."""
    within_splits: Dict[str, List[Dict[str, Any]]] = {}
    for sub_id in unique_subjects:
        sub_indices = np.where(subjects == sub_id)[0]
        sub_y = y[sub_indices]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sub_folds: List[Dict[str, Any]] = []
        for fold_idx, (tr_val_pos, te_pos) in enumerate(skf.split(sub_indices, sub_y)):
            train_val_idx = sub_indices[tr_val_pos]
            test_idx = sub_indices[te_pos]
            train_val_y = y[train_val_idx]

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=seed + fold_idx
            )
            tr_pos, val_pos = next(sss.split(train_val_idx, train_val_y))
            sub_folds.append(
                {
                    "fold": int(fold_idx),
                    "train_idx": np.sort(train_val_idx[tr_pos]).astype(int).tolist(),
                    "val_idx": np.sort(train_val_idx[val_pos]).astype(int).tolist(),
                    "test_idx": np.sort(test_idx).astype(int).tolist(),
                }
            )
        within_splits[sub_id] = sub_folds
    return within_splits


def _map_subject_folds_to_indices(
    clean_folds: Sequence[Dict[str, Any]],
    subjects_target: np.ndarray,
) -> List[Dict[str, Any]]:
    """Ánh xạ danh sách subject theo từng fold từ `splits.json` sang chỉ số mẫu thực tế."""
    mapped_folds: List[Dict[str, Any]] = []
    available_subjects = set(np.unique(subjects_target).tolist())

    for fold in clean_folds:
        train_subjects = [str(s) for s in fold["train_subjects"]]
        val_subjects = [str(s) for s in fold["val_subjects"]]
        test_subjects = [str(s) for s in fold["test_subjects"]]

        expected_subjects = set(train_subjects) | set(val_subjects) | set(test_subjects)
        missing = expected_subjects - available_subjects
        if missing:
            raise ValueError(
                f"Nhánh no_ica thiếu các subject có trong splits.json: {sorted(missing)}"
            )

        train_idx = np.where(np.isin(subjects_target, train_subjects))[0]
        val_idx = np.where(np.isin(subjects_target, val_subjects))[0]
        test_idx = np.where(np.isin(subjects_target, test_subjects))[0]

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Fold {fold.get('fold')} của no_ica có tập rỗng "
                f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})."
            )

        mapped_folds.append(
            {
                "fold": int(fold["fold"]),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "train_idx": np.sort(train_idx).astype(int).tolist(),
                "val_idx": np.sort(val_idx).astype(int).tolist(),
                "test_idx": np.sort(test_idx).astype(int).tolist(),
            }
        )
    return mapped_folds


def _build_no_ica_splits(
    clean_splits: Dict[str, Any],
    subjects_no_ica: np.ndarray,
    y_no_ica: np.ndarray,
    no_ica_sha256: str,
) -> Dict[str, Any]:
    """Tạo bộ splits đồng bộ subject với `splits.json` cho biến thể `no_ica`."""
    seed = int(clean_splits.get("seed", 42))
    unique_subjects = sorted(np.unique(subjects_no_ica).tolist(), key=natural_sort_key)

    return {
        "schema_version": clean_splits.get("schema_version", "2.5.0"),
        "seed": seed,
        "n_samples": int(len(subjects_no_ica)),
        "n_subjects": int(len(unique_subjects)),
        "subjects": unique_subjects,
        "trial_uid_sha256": no_ica_sha256,
        "clean_trial_uid_sha256": clean_splits.get("trial_uid_sha256"),
        "within_subject": _build_within_splits_for_arrays(
            subjects_no_ica, y_no_ica, unique_subjects, seed
        ),
        "cross_subject_5fold": _map_subject_folds_to_indices(
            clean_splits.get("cross_subject_5fold", []), subjects_no_ica
        ),
        "cross_subject_loso": _map_subject_folds_to_indices(
            clean_splits.get("cross_subject_loso", []), subjects_no_ica
        ),
    }


def _validate_loaded_arrays(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    metadata: pd.DataFrame,
    variant: str,
) -> None:
    """Xác thực shape, kiểu dữ liệu và tính nhất quán giữa mảng numpy và metadata."""
    if X.ndim != 3 or X.shape[1:] != (127, 251):
        raise ValueError(
            f"[{variant}] X phải có shape (N, 127, 251), nhận được {X.shape}."
        )
    n_samples = X.shape[0]
    if y.shape != (n_samples,):
        raise ValueError(f"[{variant}] y có shape {y.shape}, kỳ vọng ({n_samples},).")
    if subject_ids.shape != (n_samples,):
        raise ValueError(
            f"[{variant}] subject_ids có shape {subject_ids.shape}, kỳ vọng ({n_samples},)."
        )
    if len(metadata) != n_samples:
        raise ValueError(
            f"[{variant}] metadata có {len(metadata)} dòng, không khớp X ({n_samples})."
        )
    for required_col in ("trial_uid", "subject_id", "label"):
        if required_col not in metadata.columns:
            raise ValueError(f"[{variant}] metadata thiếu cột bắt buộc '{required_col}'.")
    if not np.isfinite(X).all():
        raise ValueError(f"[{variant}] Dữ liệu X chứa NaN hoặc Inf.")
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.int64), y):
        raise ValueError(f"[{variant}] Cột 'label' trong metadata không khớp y.npy.")
    if not np.array_equal(metadata["subject_id"].astype(str).to_numpy(), subject_ids):
        raise ValueError(
            f"[{variant}] Cột 'subject_id' trong metadata không khớp subject_ids.npy."
        )


def load_eeg_dataset(
    data_dir: Union[str, Path],
    data_variant: str = "clean",
) -> EEGDatasetBundle:
    """Tải tập dữ liệu EEG đã tiền xử lý theo chuẩn Data Contract.

    Args:
        data_dir: Thư mục chứa tệp `.npy`, `metadata*.csv` và `splits.json`.
        data_variant: Biến thể dữ liệu (`"clean"` hoặc `"no_ica"`).

    Returns:
        `EEGDatasetBundle` chứa dữ liệu, metadata, splits và hash `trial_uid_sha256`.
    """
    if data_variant not in VALID_DATA_VARIANTS:
        raise ValueError(
            f"data_variant phải thuộc {VALID_DATA_VARIANTS}, nhận được {data_variant!r}."
        )

    root = Path(data_dir)
    splits_path = root / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Không tìm thấy splits.json tại: {splits_path.resolve()}")

    with open(splits_path, "r", encoding="utf-8") as f:
        clean_splits = json.load(f)

    if data_variant == "clean":
        x_path = root / "X_clean.npy"
        y_path = root / "y.npy"
        sub_path = root / "subject_ids.npy"
        meta_path = root / "metadata.csv"

        for p in (x_path, y_path, sub_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Thiếu tệp dữ liệu clean bắt buộc: {p.resolve()}")

        X = np.load(x_path).astype(np.float32, copy=False)
        y = np.load(y_path).astype(np.int64, copy=False)
        subject_ids = np.load(sub_path).astype(str)
        metadata = pd.read_csv(meta_path)

        _validate_loaded_arrays(X, y, subject_ids, metadata, variant="clean")

        actual_sha256 = compute_trial_uid_sha256(metadata["trial_uid"].astype(str).tolist())
        expected_sha256 = str(clean_splits.get("trial_uid_sha256", ""))
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "SHA-256 của metadata['trial_uid'] không khớp trial_uid_sha256 trong splits.json: "
                f"actual={actual_sha256} != expected={expected_sha256}"
            )

        return EEGDatasetBundle(
            data_dir=root,
            data_variant="clean",
            X=X,
            y=y,
            subject_ids=subject_ids,
            metadata=metadata,
            splits=clean_splits,
            trial_uid_sha256=actual_sha256,
        )

    # Biến thể đối chứng no_ica (giữ nguyên toàn bộ trial không ICA, không epoch rejection)
    x_path = root / "X_no_ica_no_reject.npy"
    y_path = root / "y_no_ica_no_reject.npy"
    sub_path = root / "subject_ids_no_ica_no_reject.npy"
    meta_path = root / "metadata_no_ica.csv"

    for p in (x_path, y_path, sub_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Thiếu tệp dữ liệu no_ica bắt buộc: {p.resolve()}")

    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int64, copy=False)
    subject_ids = np.load(sub_path).astype(str)
    metadata = pd.read_csv(meta_path)

    _validate_loaded_arrays(X, y, subject_ids, metadata, variant="no_ica")

    actual_sha256 = compute_trial_uid_sha256(metadata["trial_uid"].astype(str).tolist())
    no_ica_splits = _build_no_ica_splits(clean_splits, subject_ids, y, actual_sha256)

    return EEGDatasetBundle(
        data_dir=root,
        data_variant="no_ica",
        X=X,
        y=y,
        subject_ids=subject_ids,
        metadata=metadata,
        splits=no_ica_splits,
        trial_uid_sha256=actual_sha256,
    )


def get_protocol_folds(
    splits: Dict[str, Any],
    protocol: str,
    subjects: Optional[Sequence[str]] = None,
    folds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Trích xuất danh sách fold chuẩn hóa cho một giao thức đánh giá.

    Mỗi phần tử trả về có khóa:
        `fold_key`, `protocol`, `fold`, `subject_id`, `train_idx`, `val_idx`, `test_idx`,
        và tùy chọn `train_subjects`, `val_subjects`, `test_subjects`.
    """
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"protocol phải thuộc {VALID_PROTOCOLS}, nhận được {protocol!r}.")

    subject_filter = set(str(s) for s in subjects) if subjects else None
    fold_filter = set(int(f) for f in folds) if folds is not None else None
    specs: List[Dict[str, Any]] = []

    if protocol == "within_subject":
        within_dict = splits.get("within_subject", {})
        ordered_subs = sorted(within_dict.keys(), key=natural_sort_key)
        for sub_id in ordered_subs:
            if subject_filter is not None and sub_id not in subject_filter:
                continue
            for item in within_dict[sub_id]:
                fold_idx = int(item["fold"])
                if fold_filter is not None and fold_idx not in fold_filter:
                    continue
                specs.append(
                    {
                        "fold_key": f"{sub_id}_fold{fold_idx}",
                        "protocol": protocol,
                        "fold": fold_idx,
                        "subject_id": sub_id,
                        "train_idx": item["train_idx"],
                        "val_idx": item["val_idx"],
                        "test_idx": item["test_idx"],
                    }
                )
        return specs

    raw_folds = splits.get(protocol, [])
    for item in raw_folds:
        fold_idx = int(item["fold"])
        if fold_filter is not None and fold_idx not in fold_filter:
            continue
        test_subs = [str(s) for s in item.get("test_subjects", [])]
        if subject_filter is not None and not (set(test_subs) & subject_filter):
            continue

        if protocol == "cross_subject_loso" and len(test_subs) == 1:
            sub_label = test_subs[0]
            fold_key = f"{sub_label}_fold{fold_idx}"
        else:
            sub_label = ",".join(test_subs) if test_subs else "all"
            fold_key = f"fold{fold_idx}"

        specs.append(
            {
                "fold_key": fold_key,
                "protocol": protocol,
                "fold": fold_idx,
                "subject_id": sub_label,
                "train_subjects": [str(s) for s in item.get("train_subjects", [])],
                "val_subjects": [str(s) for s in item.get("val_subjects", [])],
                "test_subjects": test_subs,
                "train_idx": item["train_idx"],
                "val_idx": item["val_idx"],
                "test_idx": item["test_idx"],
            }
        )
    return specs
