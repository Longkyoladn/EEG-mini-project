"""Pipeline tiền xử lý chuẩn hóa tín hiệu EEG (Oddball Paradigm - P300)  -  V3
Tệp: src/preprocessing/pipeline.py

Mô tả:
Pipeline tiền xử lý dữ liệu EEG đơn thử nghiệm (Single-Trial Classification)
theo Data Contract đã kiểm toán tại `reports/DATA_AUDIT_REPORT.md`:
1. Cohort Policy: Mặc định loại 3 subjects sai luật hành vi (`sub-02, sub-14, sub-17`),
   giữ `sub-38` (775 stimuli) tạo tập chuẩn 39 subjects sạch.
2. Cắt gọn `raw.crop()` về cửa sổ task `[first_stim - 5s, last_stim + 5s]` TRƯỚC mọi
   bước lọc, phát hiện kênh xấu, và ICA để loại bỏ bão hòa cuối file (`sub-08`, `sub-35`).
3. Hệ 127 kênh chuẩn & Montage `colin27_1005` (`on_missing='raise'`): áp dụng đủ 4 aliases
   `{'TP9 LEFT EAR': 'TP9', 'TP10 RIGHT EAR': 'TP10', 'O9': 'I1', 'O10': 'I2'}`,
   đồng bộ thứ tự kênh theo `CANONICAL_127_CHANNELS`, đọc `_channels.tsv` với `encoding='latin1'`.
4. Phát hiện và nội suy kênh xấu TRƯỚC `set_eeg_reference('average')` để kênh hỏng
   (`I2` của `sub-16`, `sub-41`) không làm ô nhiễm Average Reference.
5. Lọc hành vi chính xác (`S  5` Frequent CR vs `S  6` Target Hit với `100 <= RT <= 1500 ms`),
   sinh `trial_uid = f"{sub_id}_{onset_sample_1000hz}"`, `rt_ms`, `rt_group`.
6. Đồng bộ trục mẫu qua `crop()` + `resample(250.0, events=...)`, đánh dấu `BAD boundary`
   với `reject_by_annotation=True`, lọc Notch 50 Hz + Bandpass 0.5-40 Hz, ICA 250 Hz,
   xuất tensor `(N, 127, 251)` `float32` cho nhánh `clean` và nhánh đối chứng `no_ica`.
7. Hỗ trợ CLI (`--subjects`, `--all`, `--out-dir`, `--resume`) và cache từng subject.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import freqz
from tqdm import tqdm

mne.set_log_level("WARNING")


# ==============================================================================
# 1. HẰNG SỐ DATA CONTRACT & BẢNG CẤU HÌNH HỆ THỐNG (CONFIG)
# ==============================================================================
EXCLUDE_SUBJECTS = {"sub-02", "sub-14", "sub-17"}

CHANNEL_ALIASES: Dict[str, str] = {
    "TP9 LEFT EAR": "TP9",
    "TP10 RIGHT EAR": "TP10",
    "O9": "I1",
    "O10": "I2",
}

# Thứ tự 127 kênh chuẩn của sub-01 sau khi chuẩn hóa 4 aliases
CANONICAL_127_CHANNELS: List[str] = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9",
    "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
    "TP10", "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4",
    "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1",
    "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8",
    "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6",
    "AF8", "AF4", "F2", "F9", "AFF1h", "FFC1h", "FFC5h", "FTT7h", "FCC3h", "CCP1h",
    "CCP5h", "TPP7h", "P9", "PPO9h", "PO9", "I1", "OI1h", "PPO1h", "CPP3h", "CPP4h",
    "PPO2h", "OI2h", "I2", "PO10", "PPO10h", "P10", "TPP8h", "CCP6h", "CCP2h", "FCC4h",
    "FTT8h", "FFC6h", "FFC2h", "AFF2h", "F10", "AFp1", "AFF5h", "FFT9h", "FFT7h", "FFC3h",
    "FCC1h", "FCC5h", "FTT9h", "TTP7h", "CCP3h", "CPP1h", "CPP5h", "TPP9h", "POO9h", "PPO5h",
    "POO1", "POO2", "PPO6h", "POO10h", "TPP10h", "CPP6h", "CPP2h", "CCP4h", "TTP8h", "FTT10h",
    "FCC6h", "FCC2h", "FFC4h", "FFT8h", "FFT10h", "AFF6h", "AFp2",
]

METADATA_COLUMNS: List[str] = [
    "trial_uid",
    "subject_id",
    "trial_idx",
    "onset_sample",
    "onset_sec",
    "event_type",
    "label",
    "rt_ms",
    "rt_group",
]

CONFIG: Dict[str, Any] = {
    # Đường dẫn thư mục
    "RAW_DIR": Path("data/raw/oddball_data"),
    "PROCESSED_DIR": Path("data/processed"),
    "FIGURES_DIR": Path("results/figures/ica"),

    # Chính sách chọn lọc subject (Cohort Policy)
    "EXCLUDE_SUBJECTS": sorted(EXCLUDE_SUBJECTS),

    # Cắt đoạn bao quanh task trước khi lọc / ICA (cứu sub-08, sub-35)
    "CROP_PAD_SEC": 5.0,

    # Cấu hình Trigger / Event (Oddball: Frequent Standard vs Rare Target)
    "EVENT_TRIGGER_MAP": {
        "Frequent": "Stimulus/S  5",
        "Target": "Stimulus/S  6",

    },
    "BEHAVIOR": {
        "rt_min_sec": 0.100,
        "rt_max_sec": 1.500,
        "rt_fast_ms": 450.0,
        "rt_slow_ms": 550.0,
    },

    "RESPONSE_TRIGGER": "Stimulus/S  1",
    "RT_WINDOW_MS": (150.0, 1500.0),

    "TARGET_LABELS": {"Frequent": 0, "Target": 1},

    "FILTER": {
        "l_freq": 0.5,
        "h_freq": 40.0,
        "notch_mode": "auto",        # "auto" | "always" | "never"
        "notch_freq": 50.0,
        "notch_residual_db": 3.0,    # auto: chỉ notch nếu đỉnh 50 Hz còn nổi > 3 dB SAU bandpass
        "line_noise_max_sec": 120.0, # đo nhiễu lưới trên tối đa 120 s đầu
    },

    "RESAMPLE_FREQ": 250.0,

    "BAD_CHANNELS": {
        "enabled": True,
        "flat_threshold_std": 1e-7,    # Kênh phẳng: std < 0.1 µV
        "noisy_threshold_z": 5.0,      # Kênh nhiễu: z-score MAD std > 5.0
        "interpolate": True,           # Nội suy kênh hỏng TRƯỚC khi average reference
    },

    # Montage & Tham chiếu điện cực (Re-referencing)
    "MONTAGE": "colin27_1005",
    "REFERENCE": "average",            # Average reference toàn bộ điện cực chuẩn

    "ICA": {
        "method": "picard",          # "picard" (tự fallback "fastica") | "fastica" | "infomax"
        "n_components": 25,
        "random_state": 42,
        "high_pass": 1.0,
        "fit_epoch_sec": 1.0,        # ICA được fit trên epoch cố định 1 s
        "fit_reject_ptp": 500e-6,    # bỏ epoch có ptp > 500 µV (giữ lại chớp mắt, bỏ nhiễu cơ/chuyển động)
        "min_fit_epochs": 60,
        "eog_proxy": {
            "veog": ["Fp1", "Fp2"],              # trung bình -> proxy chớp mắt dọc
            "heog": [["F7", "AF7"], ["F8", "AF8"]],  # (trái - phải) -> proxy liếc ngang (dùng cặp đầu tiên có đủ)
        },
        "frontal_channels": ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8", "F7", "F8"],
        "threshold": 3.0,            # z-score của |r| giữa các IC
        "corr_min": 0.4,             # |r| tối thiểu với proxy
        "min_frontal_ratio": 1.5,    # tải trung bình vùng trán / tải trung bình toàn bộ >= 1.5
        "max_excluded_ics": 4,       # GIỚI HẠN CỨNG: chỉ loại tối đa N IC (theo |r| giảm dần)
        "save_diagnostic_plots": True,
        "p300_channels": ["Pz", "CPz", "Cz"],
        "p300_window": (0.3, 0.5),
        "p300_change_warn_pct": 30.0,
    },

    "EPOCHS": {
        "tmin": -0.2,
        "tmax": 0.8,
        "baseline": (-0.2, 0.0),
        "reject_amp": 150e-6,
    },

    # Ánh xạ tên kênh không chuẩn sang chuẩn quốc tế 10-05 / colin27_1005
    "CHANNEL_ALIASES": dict(CHANNEL_ALIASES),

    "EXCLUDE_SUBJECTS": ["sub-02", "sub-14", "sub-17"],

    

    "SAVE_SINGLE_SUBJECT_FIF": True,
    "SAVE_ABLATION_STUDY": True,    # lưu A (no ICA/no reject), B (ICA/no reject), keep_mask
}


# ==============================================================================
# 2. HỆ THỐNG GHI NHẬT KÝ (LOGGING) & TIỆN ÍCH CHUNG
# ==============================================================================
def default_project_root() -> Path:
    """SỬA [6]: không crash khi chạy trong notebook (không có __file__)."""
    env = os.getenv("EEG_ROOT")
    if env:
        return Path(env)
    try:
        return Path(__file__).resolve().parents[2]
    except (NameError, IndexError):
        return Path.cwd()


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    root = default_project_root()
    if cfg.get("RAW_DIR") is None:
        cfg["RAW_DIR"] = root / "data" / "raw" / "oddball_data"
    if cfg.get("PROCESSED_DIR") is None:
        cfg["PROCESSED_DIR"] = root / "data" / "processed"
    if cfg.get("FIGURES_DIR") is None:
        cfg["FIGURES_DIR"] = root / "results" / "figures" / "ica"
    for k in ("RAW_DIR", "PROCESSED_DIR", "FIGURES_DIR", "STAGE_LOCAL_DIR"):
        if cfg.get(k) is not None:
            cfg[k] = Path(cfg[k])
    return cfg


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("EEG_Pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")  # 'a' để giữ log khi RESUME
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _to_json_serializable(obj: Any) -> Any:
    """Chuyển đổi Path, set, numpy scalar sang kiểu JSON hợp lệ."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def natural_sort_key(s: str) -> List[int | str]:
    """Sắp xếp tự nhiên theo thứ tự số tăng dần."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def extract_subject_id(vhdr_path: Path) -> str:
    m = re.search(r"(sub-\d+)", vhdr_path.stem)
    if m:
        return m.group(1)
    m = re.search(r"(sub-\d+)", str(vhdr_path))
    if m:
        return m.group(1)
    return vhdr_path.parent.name


def normalize_event_name(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


# ==============================================================================
# 3. TIỆN ÍCH KÊNH CHUẨN, MONTAGE, SỰ KIỆN HÀNH VI & KÊNH XẤU
# ==============================================================================
def get_canonical_127_channels(
    raw_dir: Path,
    channel_aliases: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Lấy danh sách thứ tự 127 kênh chuẩn từ sub-01 (sau alias), fallback về hằng số."""
    aliases = channel_aliases or CHANNEL_ALIASES
    sub01_candidates = sorted(raw_dir.rglob("sub-01*_eeg.vhdr"), key=natural_sort_key)
    if sub01_candidates:
        raw_ref = mne.io.read_raw_brainvision(sub01_candidates[0], preload=False, verbose=False)
        rename_dict = {orig: new for orig, new in aliases.items() if orig in raw_ref.ch_names}
        if rename_dict:
            raw_ref.rename_channels(rename_dict)
        ch_order = list(raw_ref.ch_names)
        if set(ch_order) == set(CANONICAL_127_CHANNELS) and len(ch_order) == 127:
            return ch_order
        if logger is not None:
            logger.warning(
                "Bộ kênh của sub-01 không khớp CANONICAL_127_CHANNELS; dùng hằng số mặc định."
            )
    return list(CANONICAL_127_CHANNELS)


def validate_channels_tsv(
    vhdr_path: Path,
    raw_ch_names: Sequence[str],
    logger: logging.Logger,
) -> None:
    """Đọc tệp _channels.tsv với encoding='latin1' để xác thực danh sách 127 kênh."""
    sub_id = extract_subject_id(vhdr_path)
    ch_tsv_candidates = sorted(vhdr_path.parent.glob("*_channels.tsv"))
    if not ch_tsv_candidates:
        return

    df_ch = pd.read_csv(ch_tsv_candidates[0], sep="\t", encoding="latin1")
    if "name" not in df_ch.columns:
        logger.warning(f"[{sub_id}] {ch_tsv_candidates[0].name} thiếu cột 'name'.")
        return

    tsv_names = df_ch["name"].astype(str).tolist()
    if len(tsv_names) != len(raw_ch_names) or tsv_names != list(raw_ch_names):
        logger.warning(
            f"[{sub_id}] Danh sách kênh trong {ch_tsv_candidates[0].name} "
            f"({len(tsv_names)}) khác với .vhdr ({len(raw_ch_names)})."
        )


def normalize_boundary_annotations(
    raw: mne.io.Raw,
    logger: logging.Logger,
    sub_id: str,
) -> List[float]:
    """Đổi annotation 'boundary'/'New Segment' (onset > 0.01s) thành 'BAD boundary'."""
    if len(raw.annotations) == 0:
        return []

    onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
    durations = np.asarray(raw.annotations.duration, dtype=np.float64)
    orig_descs = [str(d) for d in raw.annotations.description]

    new_descs: List[str] = []
    boundary_onsets: List[float] = []
    for onset, desc in zip(onsets, orig_descs):
        desc_lower = desc.lower()
        if ("boundary" in desc_lower or "new segment" in desc_lower) and onset > 0.01:
            new_descs.append("BAD boundary")
            boundary_onsets.append(float(onset))
        else:
            new_descs.append(desc)

    if boundary_onsets:
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=new_descs,
                orig_time=raw.annotations.orig_time,
            ),
            verbose=False,
        )
        unique_boundaries = sorted(set(round(x, 6) for x in boundary_onsets))
        logger.info(
            f"[{sub_id}] Đánh dấu {len(unique_boundaries)} mốc ghép nối BAD boundary: "
            f"{unique_boundaries}"
        )
        return unique_boundaries

    return []


def extract_behaviorally_clean_events(
    events_tsv_path: Path,
    sub_id: str,
    vmrk_boundary_sec: Optional[List[float]] = None,
    sfreq: float = 1000.0,
    rt_min: float = 0.100,
    rt_max: float = 1.500,
    rt_fast_ms: float = 450.0,
    rt_slow_ms: float = 550.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Lọc hành vi chuẩn Oddball theo Mục 6.2.C của DATA_AUDIT_REPORT.md.

    - Giữ `S  5` (Frequent CR, label=0) khi KHÔNG có `S  1` trong [0, 1.500] s trước
      stimulus / break (`S 10`, `S 12`, boundary) tiếp theo (`rt_ms = np.nan`, `rt_group = 'na'`).
    - Giữ `S  6` (Target Hit, label=1) khi có `S  1` đầu tiên thỏa `0.100 <= rt <= 1.500` s
      trước stimulus / break kế tiếp (`rt_ms = rt * 1000.0`).
    - Loại bỏ hoàn toàn Distractor (`S  7`), Target Miss, Anticipation (< 100 ms), Frequent FA.
    - Trả về mảng MNE events (N, 3) dùng sample 0-based tại `sfreq` (mã 1=Frequent, 2=Target)
      và DataFrame metadata chứa `trial_uid = f"{sub_id}_{onset_sample_1000hz}"`.
    """
    df = pd.read_csv(events_tsv_path, sep="\t")
    stim_codes = {"S  5", "S  6", "S  7"}
    break_codes = {"S 10", "S 12", "__BOUNDARY__"}

    # Đọc timeline hợp nhất giữa _events.tsv và các mốc boundary từ .vmrk
    timeline: List[Tuple[float, int, str]] = []
    for _, row in df.iterrows():
        onset_val = float(row["onset"])
        code_val = str(row["event_type"])
        priority = 0 if code_val in break_codes else 1
        timeline.append((onset_val, priority, code_val))

    for b_sec in vmrk_boundary_sec or []:
        timeline.append((float(b_sec), 0, "__BOUNDARY__"))

    timeline.sort(key=lambda item: (item[0], item[1]))
    records = [(t, c) for t, _, c in timeline]
    stim_idx = [i for i, (_, c) in enumerate(records) if c in stim_codes]

    clean_events: List[List[int]] = []
    meta_rows: List[Dict[str, Any]] = []
    seen_uids = set()

    for k, i in enumerate(stim_idx):
        t_stim, code = records[i]
        j_end = stim_idx[k + 1] if k + 1 < len(stim_idx) else len(records)
        resps: List[float] = []
        for j in range(i + 1, j_end):
            t_j, c_j = records[j]
            if c_j in break_codes:
                break
            if c_j == "S  1":
                rt_sec = round(t_j - t_stim, 6)
                if rt_sec >= 0.0:
                    resps.append(rt_sec)

        # BIDS _events.tsv lưu onset = (sample_1based) / 1000.0 -> trừ 1 để khớp mẫu 0-based của MNE
        sample_0based = max(0, int(round(t_stim * sfreq)) - 1)
        onset_sec = float(sample_0based / sfreq)
        trial_uid = f"{sub_id}_{sample_0based}"

        if code == "S  5" and not any(r <= rt_max for r in resps):
            if trial_uid in seen_uids:
                raise ValueError(f"[{sub_id}] Trùng trial_uid: {trial_uid}")
            seen_uids.add(trial_uid)
            clean_events.append([sample_0based, 0, 1])  # 1 -> Frequent (label 0)
            meta_rows.append(
                {
                    "trial_uid": trial_uid,
                    "subject_id": sub_id,
                    "trial_idx": -1,
                    "onset_sample": int(sample_0based),
                    "onset_sec": onset_sec,
                    "event_type": "Frequent",
                    "label": 0,
                    "rt_ms": np.nan,
                    "rt_group": "na",
                }
            )
        elif code == "S  6" and resps and (rt_min <= resps[0] <= rt_max):
            if trial_uid in seen_uids:
                raise ValueError(f"[{sub_id}] Trùng trial_uid: {trial_uid}")
            seen_uids.add(trial_uid)
            rt_ms = round(float(resps[0] * 1000.0), 3)
            if rt_ms < rt_fast_ms:
                rt_group = "fast"
            elif rt_ms <= rt_slow_ms:
                rt_group = "mid"
            else:
                rt_group = "slow"

            clean_events.append([sample_0based, 0, 2])  # 2 -> Target (label 1)
            meta_rows.append(
                {
                    "trial_uid": trial_uid,
                    "subject_id": sub_id,
                    "trial_idx": -1,
                    "onset_sample": int(sample_0based),
                    "onset_sec": onset_sec,
                    "event_type": "Target",
                    "label": 1,
                    "rt_ms": rt_ms,
                    "rt_group": rt_group,
                }
            )

    if not clean_events:
        raise ValueError(f"[{sub_id}] Không có trial hợp lệ nào sau lọc hành vi!")

    events_arr = np.asarray(clean_events, dtype=np.int64)
    df_meta = pd.DataFrame(meta_rows, columns=METADATA_COLUMNS)
    return events_arr, df_meta


def extract_events_safe(
    raw,
    trigger_map: Dict[str, str],
    logger: logging.Logger,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Giữ lại helper cũ để bảo đảm tương thích ngược nếu module khác gọi trực tiếp."""
    _, event_dict = mne.events_from_annotations(raw, verbose=False)

    mne_target_mapping = {}
    for class_name, trigger_pattern in trigger_map.items():
        target_code = 1 if class_name == "Frequent" else 2

        if trigger_pattern in event_dict:
            mne_target_mapping[trigger_pattern] = target_code
            continue

        norm_pattern = normalize_event_name(trigger_pattern)
        matches = [
            annot for annot in event_dict
            if normalize_event_name(annot) == norm_pattern
        ]

        if len(matches) == 1:
            mapping[matches[0]] = code
        else:
            err_msg = (
                f"LỖI GÁN NHÃN: Trigger '{class_name}' (pattern='{trigger_pattern}') "
                f"không khớp duy nhất! Tìm thấy {len(matches)} kết quả: {matches}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    events, _ = mne.events_from_annotations(raw, event_id=mne_target_mapping, verbose=False)
    mne_event_id = {"Frequent": 1, "Target": 2}
    return events, mne_event_id


def setup_channel_montage(
    raw: mne.io.Raw,
    vhdr_path: Path,
    channel_aliases: Dict[str, str],
    logger: logging.Logger,
    montage_name: str = "colin27_1005",
) -> List[str]:
    """Chuẩn hóa đủ 4 tên kênh alias và gán montage colin27_1005 (on_missing='raise')."""
    rename_dict = {
        orig: new for orig, new in channel_aliases.items()
        if orig in raw.ch_names
    }
    if rename_dict:
        raw.rename_channels(rename_dict)
        logger.debug(f"Đã chuẩn hóa tên kênh: {rename_dict}")

    sub_id = extract_subject_id(vhdr_path)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="raise", verbose=False)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    no_position_chs: List[str] = []
    for idx in eeg_picks:
        ch = raw.info["chs"][idx]
        pos = ch["loc"][:3]
        if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) < 1e-10:
            no_position_chs.append(ch["ch_name"])

    if no_position_chs:
        raise ValueError(
            f"[{sub_id}] Phát hiện {len(no_position_chs)} kênh thiếu tọa độ 3D "
            f"sau gán montage '{montage_name}': {no_position_chs}"
        )

def template_montage():
    """Montage 10-05 mẫu. MNE >= 1.13 đã deprecate 'standard_1005' (thay bằng 'colin27_1005')."""
    available = mne.channels.get_builtin_montages()
    name = "colin27_1005" if "colin27_1005" in available else "standard_1005"
    return mne.channels.make_standard_montage(name), name


def detect_and_handle_bad_channels(
    raw: mne.io.Raw,
    config_bads: Dict[str, Any],
    logger: logging.Logger,
    sub_id: str,
    no_position_chs: Optional[List[str]] = None,
) -> List[str]:
    """Tự động phát hiện kênh hỏng (flat / noisy) TRƯỚC KHI Average Reference."""
    if not config_bads.get("enabled", True):
        return []

    _no_pos = set(no_position_chs or [])
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[p] for p in eeg_picks]
    stds = np.std(data, axis=1)

    flat_thresh = float(config_bads.get("flat_threshold_std", 1e-7))
    noisy_z_thresh = float(config_bads.get("noisy_threshold_z", 5.0))

    valid_stds = np.array([s for ch, s in zip(ch_names, stds) if ch not in _no_pos])
    median_std = float(np.median(valid_stds)) if len(valid_stds) > 0 else float(np.median(stds))
    mad = float(np.median(np.abs(valid_stds - median_std))) if len(valid_stds) > 0 else 0.0
    if mad <= 1e-12:
        logger.warning(f"[{sub_id}] MAD ≈ 0. Bỏ qua phát hiện noisy channels.")
        z = np.zeros_like(stds)
    else:
        z_scores = 0.6745 * (stds - median_std) / mad

    bad_channels: List[str] = []
    for ch, std, z_score in zip(ch_names, stds, z_scores):
        if ch in _no_pos:
            continue

        if std < flat_thresh:
            bad_channels.append(ch)
            logger.warning(f"[{sub_id}] FLAT channel: {ch} (std={std * 1e6:.3f} µV)")
        elif z_score > noisy_z_thresh:
            bad_channels.append(ch)
            logger.warning(
                f"[{sub_id}] NOISY channel: {ch} (std={std * 1e6:.1f} µV, z-score={z_score:.2f})"
            )

    raw.info["bads"] = sorted(set(raw.info["bads"] + bad_channels))
    return sorted(set(bad_channels))


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
        if any(item in raw_ch_names_set for item in drop):
            reasons["amplitude_rejected"] += 1
        elif any("BAD" in str(item).upper() or "BOUNDARY" in str(item).upper() for item in drop):
            reasons["annotation_rejected"] += 1
        else:
            out["boundary_or_other"] += 1
    return out


# ==============================================================================
# 6. ICA (SỬA [3], [5])
# ==============================================================================
def build_eog_proxies(raw, proxy_cfg: Dict) -> Dict[str, np.ndarray]:
    names = set(raw.ch_names)
    proxies: Dict[str, np.ndarray] = {}
    v = [c for c in proxy_cfg["veog"] if c in names]
    if v:
        proxies["VEOG"] = raw.get_data(picks=v).mean(axis=0)
    lefts, rights = proxy_cfg["heog"]
    for left, right in zip(lefts, rights):
        if left in names and right in names:
            proxies["HEOG"] = raw.get_data(picks=[left])[0] - raw.get_data(picks=[right])[0]
            break
    return proxies


def select_eog_components(ica, raw, ica_cfg: Dict, logger, sub_id: str) -> Tuple[List[int], List[Dict[str, float]]]:
    """Chọn IC mắt: |r| với proxy VEOG/HEOG, z-score giữa các IC, và tải vùng trán."""
    proxies = build_eog_proxies(raw, ica_cfg["eog_proxy"])
    if not proxies:
        logger.warning(f"[{sub_id}] Không có kênh trán để tạo proxy EOG -> không loại IC nào.")
        return [], []
    S = ica.get_sources(raw).get_data()
    S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-20)

    A = np.abs(ica.get_components())
    front = [i for i, c in enumerate(ica.ch_names) if c in set(ica_cfg["frontal_channels"])]
    ratio = (A[front].mean(axis=0) / (A.mean(axis=0) + 1e-20)) if front else np.zeros(A.shape[1])

    best_r = np.zeros(S.shape[0])
    for _, p in proxies.items():
        p = (p - p.mean()) / (p.std() + 1e-20)
        r = np.abs(S @ p) / S.shape[1]
        best_r = np.maximum(best_r, r)
    z = (best_r - best_r.mean()) / (best_r.std() + 1e-20)

    cand = [i for i in range(len(best_r))
            if best_r[i] >= ica_cfg["corr_min"] and z[i] >= ica_cfg["threshold"]
            and ratio[i] >= ica_cfg["min_frontal_ratio"]]
    cand = sorted(cand, key=lambda i: -best_r[i])
    if len(cand) > ica_cfg["max_excluded_ics"]:
        logger.warning(f"[{sub_id}] {len(cand)} IC thỏa tiêu chí mắt > giới hạn {ica_cfg['max_excluded_ics']}; "
                       f"chỉ giữ top theo |r|. Nên kiểm tra thủ công subject này.")
        cand = cand[: ica_cfg["max_excluded_ics"]]
    details = [{"ic": i, "abs_r": float(best_r[i]), "z": float(z[i]), "frontal_ratio": float(ratio[i])} for i in range(len(best_r))]
    return sorted(cand), details


def save_ica_diag(details: List[Dict], excluded: List[int], fig_dir: Path, sub_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    idx = [d["ic"] for d in details]
    r = [d["abs_r"] for d in details]
    fr = [d["frontal_ratio"] for d in details]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    cols = ["#d62728" if i in excluded else "#7f7f7f" for i in idx]
    ax[0].bar(idx, r, color=cols); ax[0].set_title(f"{sub_id}: |r| với proxy EOG"); ax[0].set_xlabel("IC")
    ax[1].bar(idx, fr, color=cols); ax[1].set_title("Tỷ lệ tải vùng trán"); ax[1].set_xlabel("IC")
    fig.tight_layout(); fig.savefig(fig_dir / f"{sub_id}_ica_diag.png", dpi=120); plt.close(fig)


def p300_effect(epochs, channels: List[str], window: Tuple[float, float]) -> Optional[float]:
    ch = next((c for c in channels if c in epochs.ch_names), None)
    if ch is None or len(epochs["Target"]) == 0 or len(epochs["Frequent"]) == 0:
        return None
    t = epochs.times
    sel = (t >= window[0]) & (t <= window[1])
    tgt = epochs["Target"].get_data(picks=[ch])[:, 0, :].mean(axis=0)
    frq = epochs["Frequent"].get_data(picks=[ch])[:, 0, :].mean(axis=0)
    return float((tgt - frq)[sel].mean())


# ==============================================================================
# 4. QUY TRÌNH TIỀN XỬ LÝ CHO MỘT ĐỐI TƯỢNG (WORKER) + CACHING
# ==============================================================================
def _locate_events_tsv(vhdr_path: Path, sub_id: str) -> Path:
    """Tìm tệp _events.tsv tương ứng của subject."""
    preferred = vhdr_path.parent / f"{sub_id}_task-oddball_events.tsv"
    if preferred.exists():
        return preferred
    candidates = sorted(vhdr_path.parent.glob("*_events.tsv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"[{sub_id}] Không tìm thấy tệp _events.tsv trong {vhdr_path.parent}")


def _save_subject_cache(
    cache_path: Path,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_no_ica: np.ndarray,
    y_no_ica: np.ndarray,
    meta_clean: pd.DataFrame,
    meta_no_ica: pd.DataFrame,
    stats: Dict[str, Any],
    channel_names: List[str],
) -> None:
    """Lưu cache từng subject theo cách atomic vào file .npz."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.npz")
    payload = {
        "X_clean": X_clean.astype(np.float32),
        "y_clean": y_clean.astype(np.int64),
        "X_no_ica": X_no_ica.astype(np.float32),
        "y_no_ica": y_no_ica.astype(np.int64),
        "meta_clean_json": np.array(meta_clean.to_json(orient="records")),
        "meta_no_ica_json": np.array(meta_no_ica.to_json(orient="records")),
        "stats_json": np.array(json.dumps(_to_json_serializable(stats))),
        "channel_names": np.array(channel_names),
    }
    with open(tmp_path, "wb") as f:
        np.savez(f, **payload)
    tmp_path.replace(cache_path)


def _load_subject_cache(
    cache_path: Path,
    expected_channels: List[str],
    expected_n_times: int,
) -> Optional[Dict[str, Any]]:
    """Nạp cache subject nếu tồn tại và khớp schema (127 kênh, 251 mẫu)."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_channels = data["channel_names"].tolist()
            if cached_channels != expected_channels:
                return None
            X_clean = np.asarray(data["X_clean"], dtype=np.float32)
            y_clean = np.asarray(data["y_clean"], dtype=np.int64)
            X_no_ica = np.asarray(data["X_no_ica"], dtype=np.float32)
            y_no_ica = np.asarray(data["y_no_ica"], dtype=np.int64)
            if X_clean.ndim != 3 or X_clean.shape[1:] != (len(expected_channels), expected_n_times):
                return None
            meta_clean_records = json.loads(str(data["meta_clean_json"].item()))
            meta_no_ica_records = json.loads(str(data["meta_no_ica_json"].item()))
            stats = json.loads(str(data["stats_json"].item()))

        meta_clean = pd.DataFrame(meta_clean_records, columns=METADATA_COLUMNS)
        meta_no_ica = pd.DataFrame(
            meta_no_ica_records, columns=METADATA_COLUMNS + ["kept_in_clean"]
        )
        meta_no_ica["kept_in_clean"] = meta_no_ica["kept_in_clean"].astype(bool)
        return {
            "X_clean": X_clean,
            "y_clean": y_clean,
            "X_no_ica": X_no_ica,
            "y_no_ica": y_no_ica,
            "meta_clean": meta_clean,
            "meta_no_ica": meta_no_ica,
            "stats": stats,
        }
    except Exception:
        return None


def process_single_subject(
    vhdr_path: Path,
    config: Dict[str, Any],
    reference_channel_order: List[str],
    logger: logging.Logger,
    save_diag: bool = False,
) -> Tuple[mne.Epochs, Optional[mne.Epochs], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Thực hiện tiền xử lý chuẩn Data Contract cho 1 subject."""
    sub_id = extract_subject_id(vhdr_path)
    events_tsv_path = _locate_events_tsv(vhdr_path, sub_id)
    logger.info(f"--- BẮT ĐẦU XỬ LÝ: {sub_id} ({vhdr_path.name}) ---")
    n_jobs = config.get("N_JOBS", 1)
    fcfg, ica_cfg, ecfg = config["FILTER"], config["ICA"], config["EPOCHS"]
    files = _subject_files(out_dir / "subjects", sub_id)
    files["stats"].parent.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu thô & kiểm tra _channels.tsv (Latin-1)
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    orig_sfreq = float(raw.info["sfreq"])
    validate_channels_tsv(vhdr_path, raw.ch_names, logger)

    # 2. Đánh dấu các mốc boundary / New Segment giữa bản ghi thành 'BAD boundary'
    boundary_onsets_sec = normalize_boundary_annotations(raw, logger, sub_id)

    # 3. BẮT BUỘC: raw.crop() về cửa sổ task TRƯỚC lọc, tìm kênh xấu & ICA (cứu sub-08, sub-35)
    ev_df = pd.read_csv(events_tsv_path, sep="\t")
    stim_mask = ev_df["event_type"].astype(str).isin(["S  5", "S  6", "S  7"])
    stim_onsets = ev_df.loc[stim_mask, "onset"].astype(float)
    if stim_onsets.empty:
        raise ValueError(f"[{sub_id}] Không tìm thấy kích thích S  5 / S  6 / S  7 trong {events_tsv_path.name}")

    pad_sec = float(config.get("CROP_PAD_SEC", 5.0))
    crop_tmin = max(0.0, float(stim_onsets.min()) - pad_sec)
    crop_tmax = min(float(raw.times[-1]), float(stim_onsets.max()) + pad_sec)
    raw.crop(tmin=crop_tmin, tmax=crop_tmax)
    logger.info(
        f"[{sub_id}] Đã raw.crop([{crop_tmin:.2f}s, {crop_tmax:.2f}s]) "
        f"(first_samp={raw.first_samp}, n_times={raw.n_times})."
    )

    # 4. Đổi tên đủ 4 aliases & gán montage colin27_1005 (on_missing='raise') + đồng bộ 127 kênh
    no_position_chs = setup_channel_montage(
        raw,
        vhdr_path,
        config["CHANNEL_ALIASES"],
        logger,
        montage_name=config.get("MONTAGE", "colin27_1005"),
    )

    current_chs = set(raw.ch_names)
    expected_chs = set(reference_channel_order)
    if current_chs != expected_chs:
        missing = expected_chs - current_chs
        extra = current_chs - expected_chs
        raise ValueError(f"[{sub_id}] Bộ kênh không đồng nhất! Thiếu: {missing}, Thừa: {extra}")

    raw.reorder_channels(reference_channel_order)

    # 5. Phát hiện và nội suy kênh xấu SAU đổi tên/montage, TRƯỚC filter/resample và Average Reference
    bad_channels = detect_and_handle_bad_channels(
        raw,
        config["BAD_CHANNELS"],
        logger,
        sub_id,
        no_position_chs=no_position_chs,
    )
    if config["BAD_CHANNELS"].get("interpolate", True) and len(raw.info["bads"]) > 0:
        logger.info(f"[{sub_id}] Nội suy {len(raw.info['bads'])} kênh xấu trước Average Reference: {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        logger.info(f"[{sub_id}] Không phát hiện kênh xấu cần nội suy.")

    # 6. Lọc hành vi chính xác ở 1000 Hz & tạo trial_uid = f"{sub_id}_{onset_sample_1000hz}"
    beh_cfg = config.get("BEHAVIOR", {})
    events_1000hz, df_events_meta = extract_behaviorally_clean_events(
        events_tsv_path=events_tsv_path,
        sub_id=sub_id,
        vmrk_boundary_sec=boundary_onsets_sec,
        sfreq=orig_sfreq,
        rt_min=float(beh_cfg.get("rt_min_sec", 0.100)),
        rt_max=float(beh_cfg.get("rt_max_sec", 1.500)),
        rt_fast_ms=float(beh_cfg.get("rt_fast_ms", 450.0)),
        rt_slow_ms=float(beh_cfg.get("rt_slow_ms", 550.0)),
    )
    n_behavior_clean = int(len(df_events_meta))

    # Kiểm chứng chắc chắn toàn bộ events nằm trong [raw.first_samp, raw.last_samp]
    if events_1000hz[:, 0].min() < raw.first_samp or events_1000hz[:, 0].max() > raw.last_samp:
        raise RuntimeError(
            f"[{sub_id}] Lỗi căn chỉnh mẫu sau crop: events trong "
            f"[{events_1000hz[:, 0].min()}, {events_1000hz[:, 0].max()}] ngoài "
            f"[{raw.first_samp}, {raw.last_samp}]"
        )

    # 7. Resample đồng bộ về 250 Hz (với anti-aliasing chuẩn MNE) TRƯỚC bước Filter & ICA
    resample_freq = float(config["RESAMPLE_FREQ"])
    if orig_sfreq != resample_freq:
        logger.info(f"[{sub_id}] Resampling đồng bộ tín hiệu & events: {orig_sfreq} Hz -> {resample_freq} Hz...")
        raw, events_res = raw.resample(resample_freq, events=events_1000hz, verbose=False)
    else:
        events_res = events_1000hz.copy()

    # 8. Lọc Notch 50 Hz & Bandpass 0.5 - 40 Hz ở 250 Hz
    filter_cfg = config["FILTER"]
    if filter_cfg.get("use_notch", True):
        raw.notch_filter(
            freqs=filter_cfg["notch_freq"],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=False,
        )

    raw.filter(
        l_freq=filter_cfg["l_freq"],
        h_freq=filter_cfg["h_freq"],
        picks="eeg",
        method="fir",
        phase="zero",
        verbose=False,
    )

    # 9. Đặt Average Reference & khóa lại thứ tự 127 kênh chuẩn
    raw.set_eeg_reference(ref_channels=config["REFERENCE"], projection=False, verbose=False)
    raw.reorder_channels(reference_channel_order)

    post_filter_std_uv = float(np.median(np.std(raw.get_data(picks="eeg"), axis=1)) * 1e6)
    logger.info(f"[{sub_id}] Median channel std sau lọc & avg ref: {post_filter_std_uv:.2f} µV")

    mne_event_id = {"Frequent": 1, "Target": 2}
    epoch_cfg = config["EPOCHS"]

    # --------------------------------------------------------------------------
    # 10. NHÁNH ĐỐI CHỨNG (ABLATION STUDY): X_no_ica_no_reject
    # --------------------------------------------------------------------------
    epochs_no_ica: Optional[mne.Epochs] = None
    if config.get("SAVE_ABLATION_STUDY", True):
        logger.info(f"[{sub_id}] [Ablation] Tạo nhánh đối chứng (X_no_ica_no_reject)...")
        epochs_no_ica = mne.Epochs(
            raw,
            events=events_res,
            event_id=mne_event_id,
            tmin=epoch_cfg["tmin"],
            tmax=epoch_cfg["tmax"],
            baseline=epoch_cfg["baseline"],
            reject=None,
            reject_by_annotation=True,
            metadata=df_events_meta.copy(),
            preload=True,
            verbose=False,
        )

    # --------------------------------------------------------------------------
    # 11. KHỬ NHIỄU MẮT TỰ ĐỘNG BẰNG ICA TRÊN TÍN HIỆU 250 Hz
    # --------------------------------------------------------------------------
    ica_cfg = config["ICA"]
    raw_for_ica = raw.copy().filter(
        l_freq=ica_cfg["high_pass"],
        h_freq=None,
        picks="eeg",
        verbose=False,
    )

    data_rank = int(mne.compute_rank(raw_for_ica, rank="info")["eeg"])
    fit_n_components = min(int(ica_cfg["n_components"]), data_rank)
    logger.info(f"[{sub_id}] Rank EEG = {data_rank}. Fit ICA (n_components={fit_n_components}) ở {resample_freq} Hz...")

    ica = mne.preprocessing.ICA(
        n_components=fit_n_components,
        method=ica_cfg["method"],
        random_state=ica_cfg["random_state"],
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_for_ica, reject_by_annotation=True, verbose=False)

    if hasattr(ica, "pca_explained_variance_") and np.sum(ica.pca_explained_variance_) > 0:
        exp_var = float(
            np.sum(ica.pca_explained_variance_[:fit_n_components])
            / np.sum(ica.pca_explained_variance_)
            * 100.0
        )
    else:
        exp_var = 0.0

    available_eog = [ch for ch in ica_cfg["eog_channels"] if ch in raw.ch_names]
    eog_inds: List[int] = []
    scores = []
    if available_eog:
        try:
            found_inds, scores = ica.find_bads_eog(
                raw,
                ch_name=available_eog,
                threshold=ica_cfg["threshold"],
                verbose=False,
            )
            eog_inds = sorted(set(int(ic) for ic in found_inds))
            scores_arr = np.atleast_2d(np.asarray(scores))
            eog_scores = (
                [float(np.max(np.abs(scores_arr[:, ic]))) for ic in eog_inds]
                if len(eog_inds) > 0
                else []
            )
            logger.info(
                f"[{sub_id}] ICA loại {len(eog_inds)} ICs mắt qua {available_eog}: "
                f"{eog_inds} (max |z|={[round(s, 2) for s in eog_scores]})"
            )
            if len(eog_inds) > int(ica_cfg.get("max_excluded_ics", 4)):
                logger.warning(
                    f"[{sub_id}] Số IC nhiễu mắt ({len(eog_inds)}) vượt ngưỡng cảnh báo."
                )
        except Exception as eog_err:
            logger.warning(f"[{sub_id}] Lỗi find_bads_eog: {eog_err}")

    if save_diag:
        try:
            fig_dir = Path(config["FIGURES_DIR"])
            fig_dir.mkdir(parents=True, exist_ok=True)
            if len(eog_inds) > 0 and len(scores) > 0:
                fig = ica.plot_scores(scores, exclude=eog_inds, show=False)
                fig.savefig(fig_dir / f"{sub_id}_ica_scores.png", dpi=150)
        except Exception as plot_err:
            logger.debug(f"[{sub_id}] Không thể lưu hình ICA: {plot_err}")

    ica.exclude = eog_inds
    ica.apply(raw, verbose=False)
    raw.reorder_channels(reference_channel_order)

    # --------------------------------------------------------------------------
    # 12. CẮT ĐOẠN SẠCH (EPOCHING) VỚI reject_by_annotation=True
    # --------------------------------------------------------------------------
    reject_criteria = dict(eeg=float(epoch_cfg["reject_amp"]))
    epochs_clean = mne.Epochs(
        raw,
        events=events_res,
        event_id=mne_event_id,
        tmin=epoch_cfg["tmin"],
        tmax=epoch_cfg["tmax"],
        baseline=epoch_cfg["baseline"],
        reject=reject_criteria,
        reject_by_annotation=True,
        metadata=df_events_meta.copy(),
        preload=True,
        verbose=False,
    )

    if list(epochs_clean.ch_names) != reference_channel_order:
        raise RuntimeError(f"[{sub_id}] Thứ tự kênh epochs_clean không khớp canonical order!")

    drop_stats = parse_drop_log_reasons(epochs_clean, raw)

    meta_clean = epochs_clean.metadata.reset_index(drop=True).copy()
    meta_clean["trial_idx"] = np.arange(len(meta_clean), dtype=np.int64)
    meta_clean = meta_clean[METADATA_COLUMNS]

    meta_no_ica: Optional[pd.DataFrame] = None
    if epochs_no_ica is not None:
        meta_no_ica = epochs_no_ica.metadata.reset_index(drop=True).copy()
        meta_no_ica["trial_idx"] = np.arange(len(meta_no_ica), dtype=np.int64)
        kept_uid_set = set(meta_clean["trial_uid"].astype(str).tolist())
        meta_no_ica["kept_in_clean"] = meta_no_ica["trial_uid"].astype(str).isin(kept_uid_set)
        meta_no_ica = meta_no_ica[METADATA_COLUMNS + ["kept_in_clean"]]

    n_frequent = int((meta_clean["label"] == 0).sum())
    n_target = int((meta_clean["label"] == 1).sum())
    clean_epoch_std_uv = (
        float(np.median(np.std(epochs_clean.get_data(), axis=(0, 2))) * 1e6)
        if len(epochs_clean) > 0
        else 0.0
    )
    drop_rate_pct = (
        (n_behavior_clean - len(epochs_clean)) / max(n_behavior_clean, 1) * 100.0
    )

    logger.info(
        f"[{sub_id}] Hoàn thành: Hành vi={n_behavior_clean} -> Giữ={len(epochs_clean)} "
        f"(Drop={drop_rate_pct:.2f}% | Amp={drop_stats['amplitude_rejected']}, "
        f"Annot={drop_stats['annotation_rejected']}, Biên={drop_stats['boundary_or_other']}). "
        f"Frequent={n_frequent}, Target={n_target}."
    )

    stats = {
        "subject_id": sub_id,
        "n_interpolated_channels": len(bad_channels),
        "interpolated_channels": ";".join(bad_channels),
        "n_ica_excluded": len(eog_inds),
        "ica_excluded_ics": ";".join(map(str, eog_inds)),
        "n_behavior_clean_epochs": n_behavior_clean,
        "n_clean_epochs_kept": int(len(epochs_clean)),
        "drop_rate_pct": round(float(drop_rate_pct), 2),
        "post_filter_std_uv": round(post_filter_std_uv, 2),
        "clean_epoch_std_uv": round(clean_epoch_std_uv, 2),
        "n_frequent": n_frequent,
        "n_target": n_target,
        "amplitude_rejected": int(drop_stats["amplitude_rejected"]),
        "annotation_rejected": int(drop_stats["annotation_rejected"]),
        "boundary_rejected": int(drop_stats["boundary_or_other"]),
        "data_rank": data_rank,
        "ica_exp_variance": round(exp_var, 2),
    }
    files["stats"].write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")
    logger.info(f"[{sub_id}] Xong: {initial} -> {stats['clean_kept']} epoch "
                f"(biên độ loại {amp_rej}, biên {stats['boundary_rejected']}); {stats['seconds']} s.")
    del raw, epochs_b, XA, XB, ica
    gc.collect()
    return stats

    return epochs_clean, epochs_no_ica, meta_clean, meta_no_ica, stats


# ==============================================================================
# 9. PIPELINE CHÍNH
# ==============================================================================
def run_pipeline(
    config: Dict[str, Any] = CONFIG,
    subjects: Optional[Sequence[str]] = None,
    resume: bool = True,
    include_excluded: bool = False,
) -> None:
    """Thực thi pipeline tiền xử lý theo Cohort Policy & hỗ trợ caching theo subject."""
    raw_dir = Path(config["RAW_DIR"])
    processed_dir = Path(config["PROCESSED_DIR"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = processed_dir / "per_subject"

    log_file = processed_dir / "preprocessing.log"
    logger = setup_logging(log_file)

    status_manifest = {
        "status": "RUNNING",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _to_json_serializable(config),
    }
    with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=4)

    logger.info("=" * 80)
    logger.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ EEG SINGLE-TRIAL (V2.5 - DATA CONTRACT)")
    logger.info(f"Thư mục nguồn thô: {raw_dir.resolve()}")
    logger.info(f"Thư mục đầu ra sạch: {processed_dir.resolve()}")
    logger.info("=" * 80)

    all_vhdr_files = sorted(list(raw_dir.rglob("*.vhdr")), key=natural_sort_key)
    if not all_vhdr_files:
        logger.error(f"Không tìm thấy file .vhdr nào trong '{raw_dir}'. Dừng chương trình.")
        status_manifest["status"] = "FAILED_NO_FILES"
        with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status_manifest, f, indent=4)
        return

    excluded_set = set() if include_excluded else set(config.get("EXCLUDE_SUBJECTS", EXCLUDE_SUBJECTS))
    requested_set = set(subjects) if subjects else None

    vhdr_files: List[Path] = []
    for p in all_vhdr_files:
        sid = extract_subject_id(p)
        if sid in excluded_set:
            logger.info(f"Bỏ qua subject bị loại theo Cohort Policy: {sid}")
            continue
        if requested_set is not None and sid not in requested_set:
            continue
        vhdr_files.append(p)

    if requested_set is not None:
        found_ids = {extract_subject_id(p) for p in vhdr_files}
        missing_req = sorted(requested_set - found_ids, key=natural_sort_key)
        if missing_req:
            logger.warning(f"Các subject yêu cầu không khả dụng hoặc bị loại: {missing_req}")

    total_subjects = len(vhdr_files)
    logger.info(f"Tổng số subjects sẽ xử lý: {total_subjects} (exclude={sorted(excluded_set)})")
    if total_subjects == 0:
        logger.error("Không còn subject nào để xử lý!")
        status_manifest["status"] = "FAILED_NO_SUBJECTS"
        with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status_manifest, f, indent=4)
        return

    # Khóa thứ tự 127 kênh chuẩn theo sub-01 sau khi chuẩn hóa 4 aliases
    reference_channel_order = get_canonical_127_channels(
        raw_dir, channel_aliases=config["CHANNEL_ALIASES"], logger=logger
    )
    logger.info(
        f"Thứ tự 127 kênh chuẩn ({len(reference_channel_order)} kênh): "
        f"{reference_channel_order[:8]} ... {reference_channel_order[-4:]}"
    )

    tmin = float(config["EPOCHS"]["tmin"])
    tmax = float(config["EPOCHS"]["tmax"])
    expected_n_times = int(np.round((tmax - tmin) * float(config["RESAMPLE_FREQ"]))) + 1

    clean_data_list: List[np.ndarray] = []
    clean_labels_list: List[np.ndarray] = []
    clean_sub_ids_list: List[np.ndarray] = []
    clean_meta_dfs: List[pd.DataFrame] = []

    no_ica_data_list: List[np.ndarray] = []
    no_ica_labels_list: List[np.ndarray] = []
    no_ica_sub_ids_list: List[np.ndarray] = []
    no_ica_meta_dfs: List[pd.DataFrame] = []

    subject_stats: List[Dict[str, Any]] = []
    failed_subjects: List[Dict[str, str]] = []

    start_time = time.time()
    num_diag = int(config["ICA"].get("num_diag_subjects", 3))

    for idx, vhdr_path in enumerate(
        tqdm(vhdr_files, desc="Đang xử lý subjects", unit="sub", file=sys.stdout), start=1
    ):
        sub_id = extract_subject_id(vhdr_path)
        cache_path = cache_dir / f"{sub_id}_cache.npz"
        save_diag = bool(config["ICA"].get("save_diagnostic_plots", True)) and (idx <= num_diag)

        try:
            cached = (
                _load_subject_cache(cache_path, reference_channel_order, expected_n_times)
                if resume
                else None
            )

            if cached is not None:
                logger.info(f"[{sub_id}] Nạp từ cache: {cache_path.name}")
                data_clean = cached["X_clean"]
                labels_clean = cached["y_clean"]
                meta_clean = cached["meta_clean"]
                data_no_ica = cached["X_no_ica"]
                labels_no_ica = cached["y_no_ica"]
                meta_no_ica = cached["meta_no_ica"]
                stats = cached["stats"]
            else:
                epochs_clean, epochs_no_ica, meta_clean, meta_no_ica, stats = (
                    process_single_subject(
                        vhdr_path=vhdr_path,
                        config=config,
                        reference_channel_order=reference_channel_order,
                        logger=logger,
                        save_diag=save_diag,
                    )
                )

                if config.get("SAVE_SINGLE_SUBJECT_FIF", True):
                    fif_path = processed_dir / f"{sub_id}_clean-epo.fif"
                    epochs_clean.save(fif_path, overwrite=True, verbose=False)

                data_clean = epochs_clean.get_data().astype(np.float32)
                labels_clean = meta_clean["label"].to_numpy(dtype=np.int64)

                if config.get("SAVE_ABLATION_STUDY", True) and epochs_no_ica is not None and meta_no_ica is not None:
                    data_no_ica = epochs_no_ica.get_data().astype(np.float32)
                    labels_no_ica = meta_no_ica["label"].to_numpy(dtype=np.int64)
                else:
                    data_no_ica = np.empty((0, len(reference_channel_order), expected_n_times), dtype=np.float32)
                    labels_no_ica = np.empty((0,), dtype=np.int64)
                    meta_no_ica = pd.DataFrame(columns=METADATA_COLUMNS + ["kept_in_clean"])

                _save_subject_cache(
                    cache_path=cache_path,
                    X_clean=data_clean,
                    y_clean=labels_clean,
                    X_no_ica=data_no_ica,
                    y_no_ica=labels_no_ica,
                    meta_clean=meta_clean,
                    meta_no_ica=meta_no_ica,
                    stats=stats,
                    channel_names=reference_channel_order,
                )

            clean_data_list.append(data_clean)
            clean_labels_list.append(labels_clean)
            clean_sub_ids_list.append(np.array([sub_id] * len(labels_clean), dtype="<U10"))
            clean_meta_dfs.append(meta_clean)

            if config.get("SAVE_ABLATION_STUDY", True) and len(labels_no_ica) > 0:
                no_ica_data_list.append(data_no_ica)
                no_ica_labels_list.append(labels_no_ica)
                no_ica_sub_ids_list.append(np.array([sub_id] * len(labels_no_ica), dtype="<U10"))
                no_ica_meta_dfs.append(meta_no_ica)

            subject_stats.append(stats)

        except Exception as err:
            logger.error(f"[!] LỖI TẠI {sub_id}: {err}")
            logger.error(traceback.format_exc())
            failed_subjects.append({"subject_id": sub_id, "error": str(err)})
            continue

    total_duration = time.time() - start_time

    # ==============================================================================
    # 6. XUẤT CÁC TẬP DỮ LIỆU NUMPY (.NPY) VÀ METADATA CHUẨN
    # ==============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ĐANG XUẤT CÁC TẬP FILE ĐẦU RA THEO DATA CONTRACT...")
    logger.info("=" * 80)

    if clean_data_list:
        X_clean = np.concatenate(clean_data_list, axis=0).astype(np.float32)
        y_clean = np.concatenate(clean_labels_list, axis=0).astype(np.int64)
        subject_ids_clean = np.concatenate(clean_sub_ids_list, axis=0).astype("<U10")
        df_meta = pd.concat(clean_meta_dfs, ignore_index=True)[METADATA_COLUMNS]

        np.save(processed_dir / "X_clean.npy", X_clean)
        np.save(processed_dir / "y.npy", y_clean)
        np.save(processed_dir / "subject_ids.npy", subject_ids_clean)
        df_meta.to_csv(processed_dir / "metadata.csv", index=False)

        logger.info(f"Đã lưu X_clean.npy: shape={X_clean.shape}, dtype={X_clean.dtype}")
        logger.info(f"Đã lưu y.npy: shape={y_clean.shape}, dtype={y_clean.dtype}")
        logger.info(f"Đã lưu subject_ids.npy: shape={subject_ids_clean.shape}, dtype={subject_ids_clean.dtype}")
        logger.info(f"Đã lưu metadata.csv: {len(df_meta)} hàng")

    if config.get("SAVE_ABLATION_STUDY", True) and no_ica_data_list:
        X_no_ica = np.concatenate(no_ica_data_list, axis=0).astype(np.float32)
        y_no_ica = np.concatenate(no_ica_labels_list, axis=0).astype(np.int64)
        subject_ids_no_ica = np.concatenate(no_ica_sub_ids_list, axis=0).astype("<U10")
        df_meta_no_ica = pd.concat(no_ica_meta_dfs, ignore_index=True)[
            METADATA_COLUMNS + ["kept_in_clean"]
        ]

        np.save(processed_dir / "X_no_ica_no_reject.npy", X_no_ica)
        np.save(processed_dir / "y_no_ica_no_reject.npy", y_no_ica)
        np.save(processed_dir / "subject_ids_no_ica_no_reject.npy", subject_ids_no_ica)
        df_meta_no_ica.to_csv(processed_dir / "metadata_no_ica.csv", index=False)

        logger.info(f"[Ablation] Đã lưu X_no_ica_no_reject.npy: shape={X_no_ica.shape}")
        logger.info(f"[Ablation] Đã lưu metadata_no_ica.csv: {len(df_meta_no_ica)} hàng")

    times = np.linspace(tmin, tmax, expected_n_times, dtype=np.float64)
    np.save(processed_dir / "times.npy", times)
    logger.info(f"Đã lưu times.npy: shape={times.shape}, dtype={times.dtype}")

    with open(processed_dir / "channel_names.json", "w", encoding="utf-8") as f:
        json.dump(reference_channel_order, f, indent=4)
    logger.info(f"Đã lưu channel_names.json ({len(reference_channel_order)} kênh)")

    with open(processed_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(config), f, indent=4)

    if subject_stats:
        df_summary = pd.DataFrame(subject_stats)
        df_summary.to_csv(processed_dir / "preprocessing_qc_summary.csv", index=False)
        df_summary.to_csv(processed_dir / "subject_summary.csv", index=False)
        logger.info(f"Đã lưu preprocessing_qc_summary.csv ({len(df_summary)} subjects)")

    if failed_subjects:
        df_failed = pd.DataFrame(failed_subjects)
        df_failed.to_csv(processed_dir / "failed_subjects.csv", index=False)
        logger.warning(f"Có {len(failed_subjects)} subjects lỗi -> failed_subjects.csv")

    total_successful = len(subject_stats)
    total_behavior = sum(s["n_behavior_clean_epochs"] for s in subject_stats)
    total_clean = sum(s["n_clean_epochs_kept"] for s in subject_stats)
    total_frequent = sum(s["n_frequent"] for s in subject_stats)
    total_target = sum(s["n_target"] for s in subject_stats)

    logger.info(
        f"Hoàn tất {total_successful}/{total_subjects} subjects trong {total_duration:.2f}s "
        f"({total_duration / 60:.2f} phút). Clean epochs={total_clean}/{total_behavior} "
        f"(Frequent={total_frequent}, Target={total_target})."
    )

    if len(failed_subjects) == 0 and total_successful == total_subjects:
        status_manifest["status"] = "COMPLETED"
    elif total_successful > 0:
        status_manifest["status"] = "COMPLETED_WITH_ERRORS"
    else:
        status_manifest["status"] = "FAILED"
    status_manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_manifest["duration_sec"] = round(total_duration, 2)
    status_manifest["successful_subjects"] = total_successful
    status_manifest["failed_subjects"] = len(failed_subjects)
    with open(processed_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
        json.dump(status_manifest, f, indent=4)


# ==============================================================================
# 6. CLI ENTRYPOINT
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline tiền xử lý dữ liệu EEG Oddball P300 theo Data Contract"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Danh sách subject cần xử lý (vd: --subjects sub-03 sub-05). Mặc định chạy toàn bộ 39 subjects sạch.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy toàn bộ cohort 39 subjects sạch (mặc định nếu không truyền --subjects).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Thư mục dữ liệu thô BIDS (mặc định: data/raw/oddball_data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Thư mục lưu đầu ra (mặc định: data/processed).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tái sử dụng cache {out_dir}/per_subject/{sub_id}_cache.npz (mặc định: bật).",
    )
    parser.add_argument(
        "--no-save-fif",
        action="store_true",
        help="Không xuất file lẻ *_clean-epo.fif để tiết kiệm dung lượng đĩa.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Không tự động loại EXCLUDE_SUBJECTS (chỉ dùng để debug).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    cfg = dict(CONFIG)
    cfg["RAW_DIR"] = args.raw_dir if args.raw_dir is not None else (project_root / "data" / "raw" / "oddball_data")
    cfg["PROCESSED_DIR"] = args.out_dir if args.out_dir is not None else (project_root / "data" / "processed")
    cfg["FIGURES_DIR"] = project_root / "results" / "figures" / "ica"
    if args.no_save_fif:
        cfg["SAVE_SINGLE_SUBJECT_FIF"] = False

    selected_subjects = None if args.all else args.subjects
    run_pipeline(
        config=cfg,
        subjects=selected_subjects,
        resume=bool(args.resume),
        include_excluded=bool(args.include_excluded),
    )


if __name__ == "__main__":
    main()
