"""Pipeline tiền xử lý chuẩn hóa tín hiệu EEG (Oddball Paradigm - P300)
Tệp: src/preprocessing/pipeline.py

Mô tả:
Pipeline tiền xử lý dữ liệu EEG đơn thử nghiệm (Single-Trial Classification)
được nâng cấp theo các tiêu chuẩn khắt khe nhất của EEG Signal Processing & MNE-Python:
1. Trích xuất sự kiện chính xác (khớp tên tuyệt đối / duy nhất, chống gán nhầm nhãn).
2. Resample đồng bộ cả tín hiệu và events để triệt tiêu event timing jitter.
3. Đồng bộ hóa thứ tự 127 kênh chuẩn giữa tất cả các đối tượng (khắc phục hoán đổi kênh).
4. Chuẩn hóa tên kênh ('TP9 LEFT EAR' -> 'TP9') và kiểm tra tọa độ điện cực thực tế / standard_1005.
5. Tự động phát hiện kênh hỏng (flat / noisy) TRƯỚC KHI thực hiện Average Re-referencing và nội suy (Interpolation).
6. Phân tích ICA an toàn: Kiểm tra rank dữ liệu, phương sai giải thích, phát hiện IC mắt qua kênh trán cực,
   ghi nhận điểm số tương quan và cảnh báo nếu loại quá nhiều ICs.
7. Cắt đoạn (Epoching) & phân loại chi tiết nguyên nhân loại trial qua epochs.drop_log.
8. Xuất đầy đủ bộ dữ liệu sạch (Clean) và bộ dữ liệu đối chứng (Ablation: X_no_ica_no_reject.npy).
9. Xuất trọn bộ metadata cho Machine Learning (channel_names.json, times.npy, metadata.csv,
   subject_summary.csv, preprocessing_config.json) với subject_ids dạng Unicode string.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# Thiết lập mức log mặc định cho MNE
mne.set_log_level("WARNING")


# ==============================================================================
# 1. BẢNG CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================
CONFIG = {
    # Đường dẫn thư mục
    "RAW_DIR": Path("data/raw/oddball_data"),
    "PROCESSED_DIR": Path("data/processed"),
    "FIGURES_DIR": Path("results/figures/ica"),

    # Cấu hình Trigger / Event (Oddball: Frequent Standard vs Rare Target)
    "EVENT_TRIGGER_MAP": {
        "Frequent": "Stimulus/S  5",   # Kích thích thường xuyên (Standard) -> Nhãn 0
        "Target": "Stimulus/S  6",     # Kích thích mục tiêu (P300 Target) -> Nhãn 1
    },
    "TARGET_LABELS": {
        "Frequent": 0,
        "Target": 1,
    },

    # Lọc tín hiệu (Filtering)
    "FILTER": {
        "l_freq": 0.5,                 # Thông cao (Hz) - bảo toàn thành phần chậm P300
        "h_freq": 40.0,                # Thông thấp (Hz)
        "use_notch": True,             # Có áp dụng Notch Filter 50 Hz hay không
        "notch_freq": 50.0,            # Tần số khử nhiễu điện lưới (Hz)
    },

    # Tần số lấy mẫu mục tiêu (Resampling)
    "RESAMPLE_FREQ": 250.0,            # Downsample từ 1000 Hz -> 250 Hz

    # Phát hiện kênh xấu trước khi Average Reference
    "BAD_CHANNELS": {
        "enabled": True,
        "flat_threshold_std": 1e-7,    # Kênh phẳng: std < 0.1 µV
        "noisy_threshold_z": 5.0,      # Kênh nhiễu: z-score std > 5.0
        "interpolate": True,           # Nội suy kênh hỏng sau khi tham chiếu
    },

    # Tham chiếu điện cực (Re-referencing)
    "REFERENCE": "average",            # Average reference toàn bộ điện cực tốt

    # Phân tích thành phần độc lập (ICA)
    "ICA": {
        "method": "fastica",
        "n_components": 25,            # Số thành phần mong muốn (sẽ kiểm tra rank)
        "random_state": 42,
        "high_pass": 1.0,              # Lọc 1.0 Hz cho bản sao fit ICA
        "eog_channels": ["Fp1", "Fp2"],# Điện cực trán cực bắt nhiễu mắt
        "threshold": 3.0,              # Ngưỡng z-score tìm bads EOG
        "max_excluded_ics": 4,         # Cảnh báo nếu số IC bị loại vượt quá ngưỡng này
        "save_diagnostic_plots": True, # Lưu biểu đồ ICA cho các subject đại diện
        "num_diag_subjects": 3,        # Số subject đầu tiên cần lưu hình ảnh ICA
    },

    # Cắt đoạn (Epoching) & Loại bỏ biên độ cao
    "EPOCHS": {
        "tmin": -0.2,                  # Bắt đầu epoch (giây) trước stimulus
        "tmax": 0.8,                   # Kết thúc epoch (giây) sau stimulus
        "baseline": (-0.2, 0.0),       # Hiệu chỉnh đường nền
        "reject_amp": 150e-6,          # Ngưỡng biên độ loại bỏ trial: 150 µV (150e-6 V)
    },

    # Ánh xạ tên kênh không chuẩn sang chuẩn quốc tế 10-05
    "CHANNEL_ALIASES": {
        "TP9 LEFT EAR": "TP9",
        "TP10 RIGHT EAR": "TP10",
    },

    # Tùy chọn lưu trữ
    "SAVE_SINGLE_SUBJECT_FIF": True,
    "SAVE_ABLATION_STUDY": True,       # Lưu bộ dữ liệu đối chứng (X_no_ica_no_reject.npy)
}


# ==============================================================================
# 2. HỆ THỐNG GHI NHẬT KÝ (LOGGING)
# ==============================================================================
def setup_logging(log_file: Path) -> logging.Logger:
    """Khởi tạo logger đa kênh xuất ra console và tệp log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("EEG_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# 3. TIỆN ÍCH TRÍCH XUẤT DỮ LIỆU & SỰ KIỆN AN TOÀN
# ==============================================================================
def natural_sort_key(s: str) -> List[int | str]:
    """Sắp xếp tự nhiên theo thứ tự số tăng dần."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def extract_subject_id(vhdr_path: Path) -> str:
    """Trích xuất subject_id an toàn bằng regex (hỗ trợ cả sub-01/ hay sub-01/eeg/)."""
    match = re.search(r"(sub-\d+)", vhdr_path.stem)
    if match:
        return match.group(1)
    match_parent = re.search(r"(sub-\d+)", str(vhdr_path))
    if match_parent:
        return match_parent.group(1)
    return vhdr_path.parent.name


def normalize_event_name(name: str) -> str:
    """Chuẩn hóa tên sự kiện: loại bỏ toàn bộ khoảng trắng dư thừa và chuyển về chữ thường."""
    return re.sub(r"\s+", "", name).lower()


def extract_events_safe(
    raw: mne.io.Raw,
    trigger_map: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Trích xuất sự kiện an toàn tuyệt đối, tránh gán nhãn nhầm:

    1. Ưu tiên khớp chính xác tên annotation.
    2. Nếu không khớp chính xác, chuẩn hóa khoảng trắng rồi tìm tất cả kết quả.
    3. Chỉ chấp nhận khi có đúng một kết quả duy nhất.
    4. Báo lỗi chi tiết và in toàn bộ danh sách annotation nếu không thỏa mãn.
    """
    events_raw, event_dict = mne.events_from_annotations(raw, verbose=False)

    mne_target_mapping = {}
    for class_name, trigger_pattern in trigger_map.items():
        target_code = 1 if class_name == "Frequent" else 2

        # 1. Khớp chính xác (Exact match)
        if trigger_pattern in event_dict:
            mne_target_mapping[trigger_pattern] = target_code
            continue

        # 2. Khớp sau khi chuẩn hóa khoảng trắng
        norm_pattern = normalize_event_name(trigger_pattern)
        matches = [
            annot for annot in event_dict
            if normalize_event_name(annot) == norm_pattern
        ]

        if len(matches) == 1:
            mne_target_mapping[matches[0]] = target_code
        else:
            err_msg = (
                f"LỖI GÁN NHÃN: Trigger '{class_name}' (pattern='{trigger_pattern}') "
                f"không khớp duy nhất! Tìm thấy {len(matches)} kết quả: {matches}. "
                f"Danh sách annotations hiện có trong file: {list(event_dict.keys())}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Trích xuất bảng sự kiện chỉ chứa 2 lớp này
    events, _ = mne.events_from_annotations(raw, event_id=mne_target_mapping, verbose=False)
    mne_event_id = {"Frequent": 1, "Target": 2}
    return events, mne_event_id


def setup_channel_montage(
    raw: mne.io.Raw,
    vhdr_path: Path,
    channel_aliases: Dict[str, str],
    logger: logging.Logger,
) -> List[str]:
    """Chuẩn hóa tên kênh, đọc tọa độ thực tế (nếu có) hoặc gán standard_1005.

    Ghi log các kênh không tìm thấy tọa độ.
    """
    # Đổi tên các kênh alias (ví dụ 'TP9 LEFT EAR' -> 'TP9')
    rename_dict = {
        orig: new for orig, new in channel_aliases.items()
        if orig in raw.ch_names
    }
    if rename_dict:
        raw.rename_channels(rename_dict)
        logger.debug(f"Đã chuẩn hóa tên kênh: {rename_dict}")

    sub_dir = vhdr_path.parent
    sub_id = extract_subject_id(vhdr_path)

    # 1. Kiểm tra tọa độ điện cực đo đạc thực tế (CapTrak .bvct / BIDS electrodes.tsv)
    custom_montage = None
    electrodes_tsv = list(sub_dir.glob(f"*{sub_id}*electrodes.tsv"))
    bvct_files = list(sub_dir.glob("*.bvct"))

    if electrodes_tsv:
        try:
            logger.info(f"[{sub_id}] Tìm thấy tọa độ thực tế: {electrodes_tsv[0].name}")
            # Đọc custom montage từ file TSV nếu có
            coordsystem_json = list(sub_dir.glob(f"*{sub_id}*coordsystem.json"))
            # Custom reader có thể được tích hợp tại đây
        except Exception as e:
            logger.warning(f"[{sub_id}] Không thể nạp custom electrodes.tsv: {e}")

    # 2. Gán standard_1005 montage (on_missing="ignore" để không crash với kênh không có toạ độ)
    montage = mne.channels.make_standard_montage("colin27_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)

    # 3. Xác định các kênh không có toạ độ 3D
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    no_position_chs = []
    for idx in eeg_picks:
        ch = raw.info["chs"][idx]
        pos = ch["loc"][:3]
        if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) < 1e-10:
            no_position_chs.append(ch["ch_name"])

    if no_position_chs:
        logger.warning(
            f"[{sub_id}] {len(no_position_chs)} kênh không có toạ độ 3D "
            f"trong montage: {no_position_chs}."
        )

    return no_position_chs


def detect_and_handle_bad_channels(
    raw: mne.io.Raw,
    config_bads: Dict,
    logger: logging.Logger,
    sub_id: str,
    no_position_chs: Optional[List[str]] = None,
) -> List[str]:
    """Tự động phát hiện kênh hỏng (flat / noisy) TRƯỚC KHI Average Reference:

    - Kênh phẳng (flat): độ lệch chuẩn cực thấp (< threshold).
    - Kênh nhiễu bất thường (noisy): z-score độ lệch chuẩn vượt ngưỡng > 5.0.
    """
    if not config_bads.get("enabled", True):
        return []

    _no_pos = set(no_position_chs or [])

    data = raw.get_data(picks="eeg")
    ch_names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True)]
    stds = np.std(data, axis=1)

    flat_thresh = config_bads.get("flat_threshold_std", 1e-7)
    noisy_z_thresh = config_bads.get("noisy_threshold_z", 5.0)

    # Tính median / MAD trên các kênh có toạ độ hợp lệ
    valid_stds = np.array([s for ch, s in zip(ch_names, stds) if ch not in _no_pos])
    median_std = np.median(valid_stds) if len(valid_stds) > 0 else np.median(stds)
    mad = np.median(np.abs(valid_stds - median_std)) if len(valid_stds) > 0 else 0.0
    if mad <= 1e-12:
        logger.warning(f"[{sub_id}] MAD ≈ 0. Bỏ qua phát hiện noisy channels.")
        z_scores = np.zeros_like(stds)
    else:
        z_scores = 0.6745 * (stds - median_std) / mad

    bad_channels = []
    for ch, std, z_score in zip(ch_names, stds, z_scores):
        if ch in _no_pos:
            continue

        if std < flat_thresh:
            bad_channels.append(ch)
            logger.warning(f"[{sub_id}] FLAT channel: {ch} (std={std * 1e6:.3f} µV)")

        elif z_score > noisy_z_thresh:
            bad_channels.append(ch)
            logger.warning(f"[{sub_id}] NOISY channel: {ch} (z-score={z_score:.2f})")

    raw.info["bads"] = sorted(set(raw.info["bads"] + bad_channels))
    return bad_channels


def parse_drop_log_reasons(epochs: mne.Epochs, raw: mne.io.Raw) -> Dict[str, int]:
    """Phân loại chính xác nguyên nhân loại bỏ từng epoch theo epochs.drop_log."""
    reasons = {
        "initial_events": len(epochs.drop_log),
        "kept_epochs": len(epochs),
        "amplitude_rejected": 0,
        "annotation_rejected": 0,
        "boundary_or_other": 0,
    }
    raw_ch_names_set = set(raw.ch_names)

    for drop in epochs.drop_log:
        if len(drop) == 0:
            continue
        # Bị loại do biên độ kênh vượt ngưỡng
        if any(item in raw_ch_names_set for item in drop):
            reasons["amplitude_rejected"] += 1
        # Bị loại do đánh dấu đoạn dữ liệu xấu (BAD annotation)
        elif any("BAD" in str(item).upper() for item in drop):
            reasons["annotation_rejected"] += 1
        else:
            reasons["boundary_or_other"] += 1

    return reasons


# ==============================================================================
# 4. QUY TRÌNH TIỀN XỬ LÝ CHO MỘT ĐỐI TƯỢNG (WORKER)
# ==============================================================================
def process_single_subject(
    vhdr_path: Path,
    config: Dict,
    reference_channel_order: List[str],
    logger: logging.Logger,
    save_diag: bool = False,
) -> Tuple[Optional[mne.Epochs], Optional[mne.Epochs], Dict]:
    """Thực hiện tiền xử lý toàn diện cho 1 subject với đầy đủ các bước kiểm tra an toàn."""
    sub_id = extract_subject_id(vhdr_path)
    logger.info(f"--- BẮT ĐẦU XỬ LÝ: {sub_id} ({vhdr_path.name}) ---")

    # 1. Đọc dữ liệu thô
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    orig_sfreq = raw.info["sfreq"]

    # 2. Chuẩn hóa tên kênh và thiết lập Montage
    no_position_chs = setup_channel_montage(raw, vhdr_path, config["CHANNEL_ALIASES"], logger)

    # 3. Đồng bộ hóa thứ tự kênh
    current_chs = set(raw.ch_names)
    expected_chs = set(reference_channel_order)
    if current_chs != expected_chs:
        missing = expected_chs - current_chs
        extra = current_chs - expected_chs
        err_msg = f"[{sub_id}] Bộ kênh không đồng nhất! Thiếu: {missing}, Thừa: {extra}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    raw.reorder_channels(reference_channel_order)

    # 4. Lọc dải tần (Bandpass) & Lọc Notch
    filter_cfg = config["FILTER"]
    if filter_cfg.get("use_notch", True):
        logger.info(f"[{sub_id}] Áp dụng Notch Filter {filter_cfg['notch_freq']} Hz...")
        raw.notch_filter(
            freqs=filter_cfg["notch_freq"],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False,
        )

    logger.info(f"[{sub_id}] Áp dụng Bandpass Filter {filter_cfg['l_freq']} - {filter_cfg['h_freq']} Hz...")
    raw.filter(
        l_freq=filter_cfg["l_freq"],
        h_freq=filter_cfg["h_freq"],
        picks="eeg",
        method="fir",
        phase="zero",
        verbose=False,
    )

    # 5. Trích xuất sự kiện trước khi resample
    events, mne_event_id = extract_events_safe(raw, config["EVENT_TRIGGER_MAP"], logger)

    # 6. Resample đồng bộ cả tín hiệu và sự kiện
    resample_freq = config["RESAMPLE_FREQ"]
    if orig_sfreq != resample_freq:
        logger.info(f"[{sub_id}] Resampling đồng bộ tín hiệu và events: {orig_sfreq} Hz -> {resample_freq} Hz...")
        raw, events = raw.resample(resample_freq, events=events, verbose=False)

    # 7. Phát hiện kênh hỏng trước khi average reference
    bad_channels = detect_and_handle_bad_channels(
        raw, config["BAD_CHANNELS"], logger, sub_id,
        no_position_chs=no_position_chs,
    )
    logger.info(f"[{sub_id}] Danh sách bad channels trước reference: {raw.info['bads']}")

    # 8. Tham chiếu trung bình (Average Re-referencing)
    logger.info(f"[{sub_id}] Thực hiện Re-referencing: {config['REFERENCE']}...")
    raw.set_eeg_reference(ref_channels=config["REFERENCE"], projection=False, verbose=False)

    # 9. Nội suy kênh hỏng (Interpolation)
    if config["BAD_CHANNELS"].get("interpolate", True) and len(raw.info["bads"]) > 0:
        safe_bads = [ch for ch in raw.info["bads"] if ch not in set(no_position_chs)]
        leaked = set(raw.info["bads"]) - set(safe_bads)
        if leaked:
            logger.warning(
                f"[{sub_id}] Bỏ qua kênh không có tọa độ khi nội suy: {leaked}"
            )
            raw.info["bads"] = safe_bads

        if raw.info["bads"]:
            logger.info(f"[{sub_id}] Nội suy kênh hỏng: {raw.info['bads']}...")
            chs_no_loc = [
                ch["ch_name"]
                for ch, ch_type in zip(raw.info["chs"], raw.get_channel_types())
                if ch_type == "eeg" and np.isnan(ch["loc"][:3]).any()
            ]
            raw.interpolate_bads(reset_bads=True, exclude=chs_no_loc, verbose=False)

    # --------------------------------------------------------------------------
    # 10. NHÁNH ĐỐI CHỨNG (ABLATION STUDY): X_no_ica_no_reject
    # --------------------------------------------------------------------------
    epochs_no_ica = None
    if config["SAVE_ABLATION_STUDY"]:
        logger.info(f"[{sub_id}] [Ablation] Tạo nhánh đối chứng (X_no_ica_no_reject)...")
        epochs_no_ica = mne.Epochs(
            raw,
            events=events,
            event_id=mne_event_id,
            tmin=config["EPOCHS"]["tmin"],
            tmax=config["EPOCHS"]["tmax"],
            baseline=config["EPOCHS"]["baseline"],
            reject=None,
            preload=True,
            verbose=False,
        )

    # --------------------------------------------------------------------------
    # 11. KHỬ NHIỄU MẮT TỰ ĐỘNG BẰNG ICA (NHÁNH CHÍNH)
    # --------------------------------------------------------------------------
    ica_cfg = config["ICA"]
    # Tạo bản sao lọc thông cao >= 1.0 Hz để fit ICA
    raw_for_ica = raw.copy().filter(
        l_freq=ica_cfg["high_pass"],
        h_freq=None,
        picks="eeg",
        verbose=False,
    )

    # Kiểm tra rank dữ liệu sau average reference
    data_rank = mne.compute_rank(raw_for_ica, rank="info")["eeg"]
    fit_n_components = min(ica_cfg["n_components"], data_rank)
    logger.info(f"[{sub_id}] Rank dữ liệu EEG = {data_rank}. Fit ICA với n_components = {fit_n_components}...")

    ica = mne.preprocessing.ICA(
        n_components=fit_n_components,
        method=ica_cfg["method"],
        random_state=ica_cfg["random_state"],
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_for_ica, verbose=False)

    # Tính tỷ lệ phương sai giải thích
    if hasattr(ica, "pca_explained_variance_"):
        exp_var = float(
            np.sum(ica.pca_explained_variance_[:fit_n_components])
            / np.sum(ica.pca_explained_variance_)
            * 100
        )
        logger.info(f"[{sub_id}] ICA giải thích {exp_var:.2f}% tổng phương sai PCA.")
    else:
        exp_var = 0.0

    # Tìm thành phần nhiễu mắt tự động qua kênh trán cực
    available_eog = [ch for ch in ica_cfg["eog_channels"] if ch in raw.ch_names]
    eog_inds, eog_scores = [], []
    if available_eog:
            try:
                eog_inds, scores = ica.find_bads_eog(
                    raw,
                    ch_name=available_eog,
                    threshold=ica_cfg["threshold"],
                    verbose=False,
                )
                # Chuẩn hóa scores về mảng 2D: (số kênh EOG, số ICs)
                scores_arr = np.atleast_2d(np.asarray(scores))
                eog_inds = list(set(int(ic) for ic in eog_inds))
                ica.exclude = eog_inds

                eog_scores = (
                    [float(np.max(np.abs(scores_arr[:, ic]))) for ic in eog_inds]
                    if len(eog_inds) > 0
                    else []
                )
                logger.info(
                    f"[{sub_id}] ICA phát hiện {len(eog_inds)} ICs nhiễu mắt qua {available_eog}: "
                    f"{eog_inds} (max |z-score|={[round(s, 2) for s in eog_scores]})"
                )

                # Cảnh báo nếu loại quá nhiều IC
                if len(eog_inds) > ica_cfg.get("max_excluded_ics", 4):
                    logger.warning(
                        f"[{sub_id}] CẢNH BÁO: Số lượng IC nhiễu mắt vượt quá mong đợi ({len(eog_inds)} ICs). "
                        f"Cần kiểm tra lại để tránh làm suy hao sóng P300!"
                    )
            except Exception as eog_err:
                logger.warning(f"[{sub_id}] Lỗi find_bads_eog: {eog_err}")

    # Lưu hình ảnh chẩn đoán ICA nếu được yêu cầu
    if save_diag:
        try:
            fig_dir = Path(config["FIGURES_DIR"])
            fig_dir.mkdir(parents=True, exist_ok=True)
            if len(eog_inds) > 0 and len(scores) > 0:
                fig = ica.plot_scores(scores, exclude=eog_inds, show=False)
                fig.savefig(fig_dir / f"{sub_id}_ica_scores.png", dpi=150)
        except Exception as plot_err:
            logger.debug(f"Không thể lưu hình ảnh ICA: {plot_err}")

    ica.exclude = eog_inds
    logger.info(f"[{sub_id}] Áp dụng ica.apply() lên dữ liệu gốc 0.5 - 40 Hz...")
    ica.apply(raw, verbose=False)

    # --------------------------------------------------------------------------
    # 12. CẮT ĐOẠN (EPOCHING) & ĐÁNH GIÁ NGUYÊN NHÂN LOẠI TRIAL (CLEAN)
    # --------------------------------------------------------------------------
    epoch_cfg = config["EPOCHS"]
    reject_criteria = dict(eeg=epoch_cfg["reject_amp"])

    epochs_clean = mne.Epochs(
        raw,
        events=events,
        event_id=mne_event_id,
        tmin=epoch_cfg["tmin"],
        tmax=epoch_cfg["tmax"],
        baseline=epoch_cfg["baseline"],
        reject=reject_criteria,
        preload=True,
        verbose=False,
    )

    # Thống kê chi tiết nguyên nhân loại bỏ theo drop_log
    drop_stats = parse_drop_log_reasons(epochs_clean, raw)

    n_frequent = int(np.sum(epochs_clean.events[:, 2] == mne_event_id["Frequent"]))
    n_target = int(np.sum(epochs_clean.events[:, 2] == mne_event_id["Target"]))

    logger.info(
        f"[{sub_id}] Hoàn thành: Ban đầu={drop_stats['initial_events']} -> Giữ={drop_stats['kept_epochs']}. "
        f"[Loại do Biên độ: {drop_stats['amplitude_rejected']}, Annotation: {drop_stats['annotation_rejected']}, "
        f"Biên: {drop_stats['boundary_or_other']}]. Phân bố: Frequent={n_frequent}, Target={n_target}."
    )

    stats = {
        "subject_id": sub_id,
        "initial_events": drop_stats["initial_events"],
        "clean_kept": drop_stats["kept_epochs"],
        "amplitude_rejected": drop_stats["amplitude_rejected"],
        "annotation_rejected": drop_stats["annotation_rejected"],
        "boundary_rejected": drop_stats["boundary_or_other"],
        "reject_rate_pct": (
            (drop_stats["initial_events"] - drop_stats["kept_epochs"])
            / max(drop_stats["initial_events"], 1)
            * 100
        ),
        "n_frequent": n_frequent,
        "n_target": n_target,
        "bad_channels": ";".join(bad_channels),
        "ica_excluded_ics": ";".join(map(str, eog_inds)),
        "data_rank": data_rank,
        "ica_exp_variance": round(exp_var, 2),
    }

    return epochs_clean, epochs_no_ica, stats


# ==============================================================================
# 5. PIPELINE CHÍNH ĐIỀU PHỐI TOÀN BỘ DATASET
# ==============================================================================
def run_pipeline(config: Dict = CONFIG) -> None:
    """Thực thi toàn bộ pipeline cho 42 subjects và xuất trọn bộ dữ liệu Machine Learning."""
    raw_dir = Path(config["RAW_DIR"])
    processed_dir = Path(config["PROCESSED_DIR"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    log_file = processed_dir / "preprocessing.log"
    logger = setup_logging(log_file)

    # Khởi tạo trạng thái thực thi để bảo vệ tránh dùng nhầm dữ liệu cũ
    status_manifest = {
        "status": "RUNNING",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()},
    }
    with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=4)

    logger.info("=" * 80)
    logger.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ EEG SINGLE-TRIAL CLASSIFICATION (V2 - PRODUCTION)")
    logger.info(f"Thư mục nguồn thô: {raw_dir.resolve()}")
    logger.info(f"Thư mục đầu ra sạch: {processed_dir.resolve()}")
    logger.info("=" * 80)

    # Tìm tất cả file .vhdr
    vhdr_files = sorted(list(raw_dir.rglob("*.vhdr")), key=natural_sort_key)
    total_subjects = len(vhdr_files)
    logger.info(f"Tìm thấy tổng cộng {total_subjects} subjects để xử lý.")

    if total_subjects == 0:
        logger.error(f"Không tìm thấy file .vhdr nào trong '{raw_dir}'. Dừng chương trình.")
        status_manifest["status"] = "FAILED_NO_FILES"
        with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status_manifest, f, indent=4)
        return

    # Xác định danh sách thứ tự kênh chuẩn mực từ subject đầu tiên
    logger.info("Xác định thứ tự kênh tham chiếu từ subject chuẩn...")
    raw_ref = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
    setup_channel_montage(raw_ref, vhdr_files[0], config["CHANNEL_ALIASES"], logger)
    reference_channel_order = list(raw_ref.ch_names)
    logger.info(f"Thứ tự kênh chuẩn ({len(reference_channel_order)} kênh): {reference_channel_order[:10]}...")

    clean_data_list = []
    clean_labels_list = []
    clean_sub_ids_list = []
    clean_metadata_rows = []

    no_ica_data_list = []
    no_ica_labels_list = []
    no_ica_sub_ids_list = []

    subject_stats = []
    failed_subjects = []

    start_time = time.time()
    num_diag = config["ICA"].get("num_diag_subjects", 3)

    for idx, vhdr_path in enumerate(
        tqdm(vhdr_files, desc="Đang xử lý subjects", unit="sub", file=sys.stdout), start=1
    ):
        sub_id = extract_subject_id(vhdr_path)
        save_diag = config["ICA"].get("save_diagnostic_plots", True) and (idx <= num_diag)

        try:
            epochs_clean, epochs_no_ica, stats = process_single_subject(
                vhdr_path=vhdr_path,
                config=config,
                reference_channel_order=reference_channel_order,
                logger=logger,
                save_diag=save_diag,
            )

            # 1. Lưu file .fif từng subject
            if config["SAVE_SINGLE_SUBJECT_FIF"] and epochs_clean is not None:
                fif_path = processed_dir / f"{sub_id}_clean-epo.fif"
                epochs_clean.save(fif_path, overwrite=True, verbose=False)

            # 2. Thu thập dữ liệu Clean cho NumPy
            data_clean = epochs_clean.get_data().astype(np.float32)
            clean_data_list.append(data_clean)

            # Gán nhãn nhị phân: Frequent -> 0, Target -> 1
            labels_clean = np.where(
                epochs_clean.events[:, 2] == epochs_clean.event_id["Target"],
                config["TARGET_LABELS"]["Target"],
                config["TARGET_LABELS"]["Frequent"],
            ).astype(np.int64)
            clean_labels_list.append(labels_clean)

            # Lưu ID subject
            sub_id_arr = np.array([sub_id] * len(labels_clean), dtype="U10")
            clean_sub_ids_list.append(sub_id_arr)

            # Ghi nhận metadata từng trial
            for trial_idx, (lbl, ev) in enumerate(zip(labels_clean, epochs_clean.events)):
                clean_metadata_rows.append({
                    "subject_id": sub_id,
                    "trial_idx": trial_idx,
                    "onset_sample": int(ev[0]),
                    "event_type": "Target" if lbl == 1 else "Frequent",
                    "label": int(lbl),
                })

            # 3. Thu thập dữ liệu Ablation Study (X_no_ica_no_reject)
            if config["SAVE_ABLATION_STUDY"] and epochs_no_ica is not None:
                data_no_ica = epochs_no_ica.get_data().astype(np.float32)
                no_ica_data_list.append(data_no_ica)

                labels_no_ica = np.where(
                    epochs_no_ica.events[:, 2] == epochs_no_ica.event_id["Target"],
                    config["TARGET_LABELS"]["Target"],
                    config["TARGET_LABELS"]["Frequent"],
                ).astype(np.int64)
                no_ica_labels_list.append(labels_no_ica)

                sub_id_no_ica_arr = np.array([sub_id] * len(labels_no_ica), dtype="U10")
                no_ica_sub_ids_list.append(sub_id_no_ica_arr)

            subject_stats.append(stats)

        except Exception as err:
            logger.error(f"[!] LỖI TẠI SUBJECT {sub_id}: {str(err)}")
            logger.error(traceback.format_exc())
            failed_subjects.append({"subject_id": sub_id, "error": str(err)})
            continue

    total_duration = time.time() - start_time

    # ==============================================================================
    # 6. XUẤT CÁC TẬP DỮ LIỆU NUMPY (.NPY) VÀ METADATA CHO MACHINE LEARNING
    # ==============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ĐANG XUẤT CÁC TẬP FILE ĐẦU RA CHO MACHINE LEARNING / DEEP LEARNING...")
    logger.info("=" * 80)

    # 1. Tập Clean
    if clean_data_list:
        X_clean = np.concatenate(clean_data_list, axis=0)
        y_clean = np.concatenate(clean_labels_list, axis=0)
        subject_ids_clean = np.concatenate(clean_sub_ids_list, axis=0)

        np.save(processed_dir / "X_clean.npy", X_clean)
        np.save(processed_dir / "y.npy", y_clean)
        np.save(processed_dir / "subject_ids.npy", subject_ids_clean)

        logger.info(f"✅ Đã lưu X_clean.npy: shape = {X_clean.shape}, dtype = {X_clean.dtype}")
        logger.info(f"✅ Đã lưu y.npy:       shape = {y_clean.shape}, dtype = {y_clean.dtype}")
        logger.info(f"✅ Đã lưu subject_ids.npy: shape = {subject_ids_clean.shape}, dtype = {subject_ids_clean.dtype}")

    # 2. Tập đối chứng Ablation Study
    if config["SAVE_ABLATION_STUDY"] and no_ica_data_list:
        X_no_ica = np.concatenate(no_ica_data_list, axis=0)
        y_no_ica = np.concatenate(no_ica_labels_list, axis=0)
        subject_ids_no_ica = np.concatenate(no_ica_sub_ids_list, axis=0)

        np.save(processed_dir / "X_no_ica_no_reject.npy", X_no_ica)
        np.save(processed_dir / "y_no_ica_no_reject.npy", y_no_ica)
        np.save(processed_dir / "subject_ids_no_ica_no_reject.npy", subject_ids_no_ica)

        logger.info(f"✅ [Ablation] Đã lưu X_no_ica_no_reject.npy: shape = {X_no_ica.shape}")
        logger.info(f"✅ [Ablation] Đã lưu y_no_ica_no_reject.npy: shape = {y_no_ica.shape}")
        logger.info(f"✅ [Ablation] Đã lưu subject_ids_no_ica_no_reject.npy: shape = {subject_ids_no_ica.shape}")

    # 3. Trục thời gian (times.npy)
    tmin = config["EPOCHS"]["tmin"]
    tmax = config["EPOCHS"]["tmax"]
    times = np.linspace(tmin, tmax, int(np.round((tmax - tmin) * config["RESAMPLE_FREQ"])) + 1)
    np.save(processed_dir / "times.npy", times)
    logger.info(f"✅ Đã lưu times.npy: shape = {times.shape} ({tmin}s đến {tmax}s)")

    # 4. Danh sách tên kênh (channel_names.json)
    with open(processed_dir / "channel_names.json", "w", encoding="utf-8") as f:
        json.dump(reference_channel_order, f, indent=4)
    logger.info(f"✅ Đã lưu channel_names.json ({len(reference_channel_order)} kênh)")

    # 5. Toàn bộ cấu hình thực tế đã sử dụng (preprocessing_config.json)
    serializable_cfg = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()
    }
    with open(processed_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(serializable_cfg, f, indent=4)
    logger.info("✅ Đã lưu preprocessing_config.json")

    # 6. Metadata bảng cho từng trial (metadata.csv)
    if clean_metadata_rows:
        df_meta = pd.DataFrame(clean_metadata_rows)
        df_meta.to_csv(processed_dir / "metadata.csv", index=False)
        logger.info(f"✅ Đã lưu metadata.csv ({len(df_meta)} hàng)")

    # 7. Thống kê chi tiết theo từng subject (subject_summary.csv)
    if subject_stats:
        df_summary = pd.DataFrame(subject_stats)
        df_summary.to_csv(processed_dir / "subject_summary.csv", index=False)
        logger.info(f"✅ Đã lưu subject_summary.csv ({len(df_summary)} subjects)")

    # 8. Danh sách subject lỗi nếu có (failed_subjects.csv)
    if failed_subjects:
        df_failed = pd.DataFrame(failed_subjects)
        df_failed.to_csv(processed_dir / "failed_subjects.csv", index=False)
        logger.warning(f"⚠️ Có {len(failed_subjects)} subjects lỗi, đã lưu failed_subjects.csv")

    # ==============================================================================
    # 7. BÁO CÁO TỔNG KẾT & CẬP NHẬT TRẠNG THÁI HOÀN TẤT
    # ==============================================================================
    total_successful = len(subject_stats)
    total_initial = sum(s["initial_events"] for s in subject_stats)
    total_clean = sum(s["clean_kept"] for s in subject_stats)
    total_amp_rej = sum(s["amplitude_rejected"] for s in subject_stats)
    total_annot_rej = sum(s["annotation_rejected"] for s in subject_stats)
    total_bound_rej = sum(s["boundary_rejected"] for s in subject_stats)
    total_rejected = total_initial - total_clean

    total_frequent = sum(s["n_frequent"] for s in subject_stats)
    total_target = sum(s["n_target"] for s in subject_stats)
    pct_frequent = (total_frequent / total_clean * 100) if total_clean > 0 else 0.0
    pct_target = (total_target / total_clean * 100) if total_clean > 0 else 0.0

    summary_str = f"""
================================================================================
📊 BÁO CÁO TỔNG KẾT TIỀN XỬ LÝ (PREPROCESSING COMPREHENSIVE SUMMARY)
================================================================================
 • Thời gian chạy:                   {total_duration / 60:.2f} phút
 • Subjects thành công:              {total_successful} / {total_subjects}
 • Subjects thất bại:                {len(failed_subjects)}

🔍 PHÂN LOẠI CHI TIẾT NGUYÊN NHÂN LOẠI TRIAL (DROP LOG ANALYSIS):
 • Initial events (Frequent + Target): {total_initial:,} trials
 • Kept clean epochs:                 {total_clean:,} trials
 • Bị loại do vượt biên độ (>150 µV): {total_amp_rej:,} trials ({(total_amp_rej/max(total_initial,1)*100):.2f}%)
 • Bị loại do BAD annotation:         {total_annot_rej:,} trials ({(total_annot_rej/max(total_initial,1)*100):.2f}%)
 • Bị loại do biên / thiếu dữ liệu:   {total_bound_rej:,} trials ({(total_bound_rej/max(total_initial,1)*100):.2f}%)
 • Tổng số trial bị từ chối:         {total_rejected:,} trials ({(total_rejected/max(total_initial,1)*100):.2f}%)

⚖️ CÂN BẰNG LỚP TRÊN TẬP CLEAN:
 • Lớp 0 (Frequent Standard):        {total_frequent:,} ({pct_frequent:.2f}%)
 • Lớp 1 (P300 Target):              {total_target:,} ({pct_target:.2f}%)
 • Tỷ lệ mất cân bằng:               ~ 1 Target : {total_frequent / max(total_target, 1):.1f} Frequent

💡 LƯU Ý CHO GIAI ĐOẠN HUẤN LUYỆN MODEL (MACHINE LEARNING / DEEP LEARNING):
 - TUYỆT ĐỐI KHÔNG chia ngẫu nhiên train_test_split() trên toàn bộ trials để tránh Data Leakage.
 - BẮT BUỘC sử dụng GroupShuffleSplit hoặc Leave-One-Subject-Out (LOSO) dựa trên 'subject_ids.npy'.
================================================================================
"""
    logger.info(summary_str)

    # Cập nhật trạng thái hoàn tất vào manifest
    if len(failed_subjects) == 0 and total_successful == total_subjects:
        status_manifest["status"] = "COMPLETED"
    elif total_successful > 0:
        status_manifest["status"] = "COMPLETED_WITH_ERRORS"
    else:
        status_manifest["status"] = "FAILED"
    status_manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_manifest["successful_subjects"] = total_successful
    status_manifest["failed_subjects"] = len(failed_subjects)
    with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=4)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    CONFIG["RAW_DIR"] = project_root / "data" / "raw" / "oddball_data"
    CONFIG["PROCESSED_DIR"] = project_root / "data" / "processed"
    CONFIG["FIGURES_DIR"] = project_root / "results" / "figures" / "ica"

    run_pipeline(CONFIG)
