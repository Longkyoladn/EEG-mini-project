"""Chạy MS-CNN-BiGRU theo splits đã khóa; lưu kết quả từng fold để tiếp tục sau gián đoạn."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.training.dataset_dl1 import get_protocol_folds, load_eeg_dataset
from src.training.trainer_dl1 import train_single_fold
from src.evaluation.metrics import METRIC_NAMES, aggregate_metrics, save_metrics_json, save_predictions_csv

DL1_SOURCES = (
    "src/models/ms_cnn_bigru.py",
    "src/training/dataset_dl1.py",
    "src/training/trainer_dl1.py",
)


def _save_experiment(results_dir, exp, bundle, protocol, epochs, metrics, predictions):
    directory = results_dir / "metrics" / "dl1"
    agg = aggregate_metrics(metrics, protocol)
    config = dict(exp_name=exp, model="ms_cnn_bigru", model_kwargs={}, protocol=protocol,
                  data_dir=str(bundle.data_dir), data_variant=bundle.data_variant,
                  epochs=epochs, patience=15, seed=42, lr=1e-3, weight_decay=1e-2,
                  batch_size=32 if protocol == "within_subject" else 64,
                  early_stop_metric="val_roc_auc", threshold_metric="balanced_accuracy")
    save_predictions_csv(predictions, directory / f"predictions_{exp}.csv")
    save_metrics_json(directory / f"metrics_{exp}.json", config, bundle.trial_uid_sha256,
                      agg["summary"], metrics, agg["per_subject"])


def run_dl1_protocol(bundle, protocol, reference_file, label, mode, results_dir):
    """Lưu từng fold đã chấm test, rồi ghép các dự đoán theo thứ tự metadata."""
    ref = json.loads((REPO_DIR / "splits" / reference_file).read_text(encoding="utf-8"))
    if ref["trial_uid_sha256"] != bundle.trial_uid_sha256 or ref[protocol] != bundle.splits[protocol]:
        raise ValueError(f"{label}: dữ liệu hoặc fold khác chuẩn của nhóm")
    specs = get_protocol_folds(bundle.splits, protocol)
    expected_count = 195 if label == "within" else 39
    if len(specs) != expected_count:
        raise ValueError(f"{label}: cần {expected_count} fold, tìm được {len(specs)}")
    epochs = 1 if mode == "pilot" else 80
    if mode == "pilot":
        specs = specs[:1]

    prefix = "pilot" if mode == "pilot" else "dl1"
    short_protocol = "ws" if label == "within" else "loso"
    exp = f"{prefix}_{short_protocol}_ms_cnn_bigru"
    root = results_dir / "checkpoints" / "dl1" / exp
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    legacy = results_dir / "dl1" / mode / label
    code_hash = hashlib.sha256()
    for rel in DL1_SOURCES:
        code_hash.update((REPO_DIR / rel).read_bytes())
    run_id = f"{bundle.trial_uid_sha256}:{code_hash.hexdigest()}:{epochs}:seed42"
    metrics, predictions = [], []
    for number, fold in enumerate(specs, 1):
        key = fold["fold_key"]
        record = cache / f"{key}_metrics.json"
        pred_file = cache / f"{key}_pred.csv"
        ckpt_dir = root if label == "loso" else None
        weights = root / f"{key}.pt"
        saved = None
        write_cache = True
        if record.is_file() and pred_file.is_file() and (ckpt_dir is None or weights.is_file()):
            fold_metrics = json.loads(record.read_text(encoding="utf-8"))
            saved = dict(run_id=fold_metrics.pop("run_id"), metrics=fold_metrics,
                         predictions=pd.read_csv(pred_file).to_dict("records"))
            write_cache = False
        elif (legacy / f"{key}.json").is_file():
            old_weights = legacy / "checkpoints" / weights.name
            if ckpt_dir is None or old_weights.is_file():
                saved = json.loads((legacy / f"{key}.json").read_text(encoding="utf-8"))
                if ckpt_dir is not None and saved["run_id"] == run_id:
                    tmp_weights = weights.with_suffix(".pt.tmp")
                    shutil.copyfile(old_weights, tmp_weights)
                    tmp_weights.replace(weights)
        if saved is not None and saved["run_id"] != run_id:
            print(f"[{number}/{len(specs)}] run_id thay đổi, huấn luyện lại {label} {key}...", flush=True)
            saved = None
            write_cache = True
        if saved is None:
            print(f"[{number}/{len(specs)}] Train {label} {key}", flush=True)
            result = train_single_fold(bundle, fold, epochs=epochs, patience=15, seed=42,
                                       checkpoint_dir=ckpt_dir)
            saved = dict(run_id=run_id, metrics=result["fold_metrics"],
                         best_epoch=result["best_epoch"], threshold=result["best_threshold"],
                         predictions=result["predictions_df"].to_dict("records"))
            print(f"  ROC-AUC={saved['metrics']['roc_auc']:.4f}, "
                  f"BA={saved['metrics']['balanced_accuracy']:.4f}, "
                  f"{result['train_time_sec']:.1f} giây", flush=True)
        else:
            print(f"[{number}/{len(specs)}] Dùng lại {label} {key}")
        if write_cache:
            saved["metrics"].update(best_epoch=saved["best_epoch"], threshold=saved["threshold"])
            tmp_pred = pred_file.with_suffix(".csv.tmp")
            save_predictions_csv(pd.DataFrame(saved["predictions"]), tmp_pred)
            shutil.move(str(tmp_pred), str(pred_file))
            tmp = record.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({**saved["metrics"], "run_id": run_id}), encoding="utf-8")
            shutil.move(str(tmp), str(record))
        metrics.append(saved["metrics"])
        predictions.extend(saved["predictions"])

    frame = pd.DataFrame(predictions)
    _save_experiment(results_dir, exp, bundle, protocol, epochs, metrics, frame)
    if mode == "pilot":
        return
    scores = pd.DataFrame(metrics).rename(columns={"balanced_accuracy": "bal_acc"})
    columns = ["subject_id", "fold", "bal_acc", "roc_auc", "pr_auc", "f1_target", "sensitivity", "specificity"]
    pred = frame.set_index("trial_uid", verify_integrity=True)
    ids = bundle.metadata["trial_uid"].astype(str)
    if not ids.is_unique or len(pred) != len(ids) or not ids.isin(pred.index).all():
        raise ValueError(f"{label}: dự đoán thiếu hoặc trùng trial")
    pred = pred.loc[ids]
    if not np.array_equal(pred["y_true"], bundle.y) or not np.array_equal(pred["subject_id"], bundle.subject_ids):
        raise ValueError(f"{label}: dự đoán lệch metadata")
    if not pred["y_prob"].between(0, 1).all():
        raise ValueError(f"{label}: xác suất không hợp lệ")
    save_predictions_csv(pred.reset_index(), results_dir / "metrics" / "dl1" / f"handover_predictions_dl1_{label}.csv")
    handover = results_dir / "handover_standard"
    handover.mkdir(parents=True, exist_ok=True)
    prefix = f"ms_cnn_bigru_{label}"
    with tempfile.TemporaryDirectory(dir=handover) as folder:
        stage = Path(folder)
        scores[columns].to_csv(stage / f"{prefix}_per_fold.csv", index=False)
        np.savez_compressed(stage / f"{prefix}_preds.npz",
            subject_id=pred["subject_id"].to_numpy(dtype=str), fold=pred["fold"].to_numpy(dtype=np.int64),
            trial_idx=np.arange(len(ids), dtype=np.int64), trial_uid=ids.to_numpy(dtype=str),
            y_true=pred["y_true"].to_numpy(dtype=np.int64), y_prob=pred["y_prob"].to_numpy(dtype=np.float64),
            y_pred=pred["y_pred"].to_numpy(dtype=np.int64))
        for path in stage.iterdir():
            shutil.move(str(path), str(handover / path.name))
    print(f"Đã xuất {len(scores)} fold và {len(pred)} dự đoán: {handover / prefix}")


def _save_subject_summary(results_dir, data_dir):
    directory = results_dir / "metrics" / "dl1"
    tables = []
    for protocol in ("ws", "loso"):
        report = directory / f"metrics_dl1_{protocol}_ms_cnn_bigru.json"
        records = json.loads(report.read_text(encoding="utf-8"))["per_subject"]
        table = pd.DataFrame.from_dict(records, orient="index")[list(METRIC_NAMES)]
        tables.append(table.add_prefix(f"{protocol}_"))
    summary = pd.concat(tables, axis=1).rename_axis("subject_id")
    qc_file = data_dir / "preprocessing_qc_summary.csv"
    if qc_file.is_file():
        qc = pd.read_csv(qc_file).set_index("subject_id")
        summary = summary.join(qc[["n_clean_epochs_kept", "n_frequent", "n_target", "drop_rate_pct"]])
    audit = REPO_DIR / "reports" / "audit_per_subject.csv"
    if audit.is_file():
        summary = summary.join(pd.read_csv(audit).set_index("subject_id")[["hit_rate_pct"]])
    summary.to_csv(directory / "handover_per_subject_summary_dl1.csv")


def run_cs5_pilot(data_dir, results_dir):
    """Đánh giá fold 0 của cả hai biến thể với phân nhóm subject của nhóm."""
    for variant in ("clean", "no_ica"):
        bundle = load_eeg_dataset(data_dir, data_variant=variant, cs5_path=REPO_DIR / "splits/cross_5fold.json")
        fold, = get_protocol_folds(bundle.splits, "cross_subject_5fold", folds=[0])
        exp = f"pilot_cs5_ms_cnn_bigru_{variant}"
        out = results_dir / "checkpoints" / "dl1" / exp
        result = train_single_fold(bundle, fold, epochs=1, checkpoint_dir=out)
        fm = dict(result["fold_metrics"], best_epoch=result["best_epoch"], threshold=result["best_threshold"])
        cache = out / "cache"
        save_predictions_csv(result["predictions_df"], cache / "fold0_pred.csv")
        (cache / "fold0_metrics.json").write_text(json.dumps(fm), encoding="utf-8")
        _save_experiment(results_dir, exp, bundle, fold["protocol"], 1, [fm], result["predictions_df"])
        print(variant, result["fold_metrics"], result["checkpoint_path"], flush=True)
        del bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Thư mục EEG đã xử lý")
    parser.add_argument("--results-dir", type=Path, required=True, help="Thư mục lưu kết quả")
    parser.add_argument("--mode", choices=("pilot", "full", "cs5-pilot"), default="pilot")
    args = parser.parse_args(argv)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 60, flush=True)
    print(f">>> KHỞI ĐỘNG TRAIN DL1 (MS-CNN-BiGRU) | Mode: {args.mode.upper()}", flush=True)
    print(f">>> Thiết bị huấn luyện: {device_name}", flush=True)
    print(f">>> Data directory: {args.data_dir}", flush=True)
    print(f">>> Results directory: {args.results_dir}", flush=True)
    print("=" * 60, flush=True)
    if args.mode == "cs5-pilot":
        run_cs5_pilot(args.data_dir, args.results_dir)
        return
    print(f">>> [1/3] Đang tải và kiểm tra dữ liệu từ {args.data_dir}...", flush=True)
    bundle = load_eeg_dataset(args.data_dir, data_variant="clean")
    print(f"-> Nạp thành công: {len(bundle.metadata)} trials, {len(set(bundle.subject_ids))} subjects.", flush=True)
    for protocol, reference, label in [
        ("within_subject", "within_5fold.json", "within"),
        ("cross_subject_loso", "loso.json", "loso"),
    ]:
        print(f"\n>>> [2/3] Bắt đầu giao thức: {label.upper()} ({protocol})", flush=True)
        run_dl1_protocol(bundle, protocol, reference, label, args.mode, args.results_dir)
    if args.mode == "full":
        print("\n>>> [3/3] Đang tổng hợp báo cáo kết quả theo từng subject...", flush=True)
        _save_subject_summary(args.results_dir, args.data_dir)
    print(f"\n{'='*60}\n>>> HOÀN TẤT TOÀN BỘ {args.mode.upper()}: {args.results_dir / 'metrics' / 'dl1'}\n{'='*60}", flush=True)


if __name__ == "__main__":
    main()
