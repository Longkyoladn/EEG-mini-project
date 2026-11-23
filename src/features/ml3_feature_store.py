"""Processed-data adapter and resumable per-subject ML3 feature storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.ml3_features import (
    FeatureExtractionConfig,
    FeatureExtractor,
    SubjectEpochs,
    extract_subject_features,
)


REQUIRED_PROCESSED_FILES = (
    "X_clean.npy",
    "y.npy",
    "subject_ids.npy",
    "times.npy",
    "channel_names.json",
    "metadata.csv",
)


@dataclass(frozen=True)
class ProcessedFeatureSource:
    input_dir: Path
    epochs: np.ndarray
    targets: np.ndarray
    subject_ids: np.ndarray
    trial_uids: np.ndarray
    times_s: np.ndarray
    channel_names: tuple[str, ...]

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted(np.unique(self.subject_ids.astype(str)).tolist()))

    def subject_data(self, subject_id: str) -> SubjectEpochs:
        indices = np.flatnonzero(self.subject_ids.astype(str) == subject_id)
        if not len(indices):
            raise KeyError(f"unknown subject: {subject_id}")
        # The preprocessing pipeline writes subjects contiguously. Preserve a
        # memmap view in that common case; fall back to indexed loading safely.
        if np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            selector: slice | np.ndarray = slice(int(indices[0]), int(indices[-1]) + 1)
        else:
            selector = indices
        return SubjectEpochs(
            subject_id=subject_id,
            epochs=self.epochs[selector],
            epoch_ids=self.trial_uids[selector],
            targets=self.targets[selector],
        )


def load_processed_feature_source(input_dir: Path) -> ProcessedFeatureSource:
    """Open the repository's processed contract without loading EEG into RAM."""

    root = Path(input_dir)
    missing = [name for name in REQUIRED_PROCESSED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"processed directory is missing: {missing}")

    epochs = np.load(root / "X_clean.npy", mmap_mode="r")
    targets = np.load(root / "y.npy", mmap_mode="r")
    subject_ids = np.load(root / "subject_ids.npy", mmap_mode="r")
    times_s = np.load(root / "times.npy")
    channel_names = tuple(
        str(name)
        for name in json.loads((root / "channel_names.json").read_text(encoding="utf-8"))
    )
    metadata = pd.read_csv(root / "metadata.csv", usecols=["trial_uid", "subject_id", "label"])
    trial_uids = metadata["trial_uid"].astype(str).to_numpy()

    n_epochs = epochs.shape[0]
    if epochs.ndim != 3:
        raise ValueError("X_clean.npy must have shape (n_epochs, n_channels, n_times)")
    if epochs.shape[1:] != (len(channel_names), len(times_s)):
        raise ValueError("X_clean shape does not match channel_names.json and times.npy")
    if any(len(array) != n_epochs for array in (targets, subject_ids, trial_uids, metadata)):
        raise ValueError("processed arrays and metadata do not contain the same number of epochs")
    if not metadata["trial_uid"].is_unique:
        raise ValueError("metadata trial_uid values must be unique")
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.int64), targets):
        raise ValueError("metadata labels do not match y.npy")
    if not np.array_equal(metadata["subject_id"].astype(str).to_numpy(), subject_ids.astype(str)):
        raise ValueError("metadata subject_id values do not match subject_ids.npy")
    if not set(np.unique(targets).tolist()).issubset({0, 1}):
        raise ValueError("y.npy must contain binary values in {0, 1}")

    return ProcessedFeatureSource(
        input_dir=root,
        epochs=epochs,
        targets=targets,
        subject_ids=subject_ids,
        trial_uids=trial_uids,
        times_s=np.asarray(times_s, dtype=np.float64),
        channel_names=channel_names,
    )


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _configuration_payload(
    extractor: FeatureExtractor,
    runtime: FeatureExtractionConfig,
    source: ProcessedFeatureSource,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "extractor": extractor.configuration(),
        "runtime": asdict(runtime),
        "source": {
            "n_channels": len(source.channel_names),
            "n_times": len(source.times_s),
            "time_start_s": float(source.times_s[0]),
            "time_stop_s": float(source.times_s[-1]),
            "channel_names": list(source.channel_names),
        },
    }


def _configuration_hash(payload: Mapping[str, object]) -> str:
    # Batch size is operational only: changing it must not invalidate already
    # extracted values, which are explicitly tested for batch invariance.
    semantic_payload = dict(payload)
    runtime = dict(semantic_payload.get("runtime", {}))
    runtime.pop("batch_size", None)
    semantic_payload["runtime"] = runtime
    return hashlib.sha256(_stable_json(semantic_payload).encode("utf-8")).hexdigest()


def _validate_existing_output(
    path: Path,
    *,
    subject: SubjectEpochs,
    expected_hash: str,
) -> dict[str, object]:
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {
                "features",
                "feature_names",
                "subject_id",
                "trial_uid",
                "target",
                "config_sha256",
                "config_json",
            }
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"missing arrays: {sorted(missing)}")
            if str(saved["subject_id"].item()) != subject.subject_id:
                raise ValueError("subject_id does not match")
            if str(saved["config_sha256"].item()) != expected_hash:
                raise ValueError("extractor configuration changed")
            if not np.array_equal(saved["trial_uid"].astype(str), subject.epoch_ids.astype(str)):
                raise ValueError("trial_uid order changed")
            if not np.array_equal(saved["target"].astype(np.int64), subject.targets.astype(np.int64)):
                raise ValueError("targets changed")
            shape = tuple(int(value) for value in saved["features"].shape)
            if shape[0] != len(subject.targets):
                raise ValueError("feature row count does not match subject epochs")
            if shape[1] != len(saved["feature_names"]):
                raise ValueError("feature names do not match feature columns")
    except Exception as exc:
        raise ValueError(
            f"existing output {path} is incompatible ({exc}); use --overwrite"
        ) from exc
    return {"path": str(path), "shape": list(shape), "status": "reused"}


def save_subject_feature_table(
    path: Path,
    *,
    table,
    feature_group: str,
    configuration: Mapping[str, object],
    configuration_hash: str,
) -> None:
    """Write one self-describing feature file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=table.values,
        feature_names=np.asarray(table.feature_names, dtype=str),
        feature_group=np.asarray(feature_group),
        subject_id=np.asarray(str(table.subject_ids[0])),
        trial_uid=np.asarray(table.epoch_ids, dtype=str),
        target=np.asarray(table.targets, dtype=np.int64),
        config_sha256=np.asarray(configuration_hash),
        config_json=np.asarray(_stable_json(configuration)),
    )
    temporary.replace(path)


def extract_feature_groups(
    source: ProcessedFeatureSource,
    output_dir: Path,
    extractors: Sequence[FeatureExtractor],
    runtime: FeatureExtractionConfig,
    *,
    subjects: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract requested groups per subject with resume-safe output files."""

    extractor_by_group = {extractor.feature_group: extractor for extractor in extractors}
    if not extractor_by_group:
        raise ValueError("at least one extractor is required")
    if len(extractor_by_group) != len(extractors):
        raise ValueError("feature_group names must be unique")

    selected_subjects = tuple(subjects) if subjects is not None else source.subjects
    unknown = sorted(set(selected_subjects) - set(source.subjects))
    if unknown:
        raise ValueError(f"unknown subjects requested: {unknown}")

    root = Path(output_dir)
    records: list[dict[str, object]] = []
    group_configurations = {
        group: _configuration_payload(extractor, runtime, source)
        for group, extractor in extractor_by_group.items()
    }
    group_hashes = {
        group: _configuration_hash(configuration)
        for group, configuration in group_configurations.items()
    }

    for subject_id in selected_subjects:
        subject = source.subject_data(subject_id)
        for group, extractor in extractor_by_group.items():
            output_path = root / group / f"{subject_id}.npz"
            if output_path.exists() and not overwrite:
                record = _validate_existing_output(
                    output_path,
                    subject=subject,
                    expected_hash=group_hashes[group],
                )
            else:
                table = extract_subject_features(subject, extractor, runtime)
                save_subject_feature_table(
                    output_path,
                    table=table,
                    feature_group=group,
                    configuration=group_configurations[group],
                    configuration_hash=group_hashes[group],
                )
                record = {
                    "path": str(output_path),
                    "shape": list(table.values.shape),
                    "status": "written",
                }
            records.append({"subject_id": subject_id, "feature_group": group, **record})

    manifest: dict[str, object] = {
        "schema_version": 1,
        "input_dir": str(source.input_dir),
        "output_dir": str(root),
        "subjects": list(selected_subjects),
        "feature_groups": list(extractor_by_group),
        "configurations": group_configurations,
        "configuration_sha256": group_hashes,
        "files": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "feature_manifest.json"
    temporary_manifest = root / "feature_manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest
