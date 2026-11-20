"""Pipeline tiền xử lý chuẩn hóa tín hiệu EEG (Oddball Paradigm - P300)  -  V3
Tệp: src/preprocessing/pipeline.py

Các thay đổi so với V2 (xem mục "SỬA" trong từng hàm):
  [1] Montage: ĐỌC tọa độ điện cực đo thực tế (BIDS electrodes.tsv / .bvct), kiểm tra tính hợp lý,
      fallback montage mẫu 10-05 (colin27_1005/standard_1005 tùy phiên bản MNE). Ghi nguồn montage vào subject_summary.csv.
  [2] Bộ nhớ: mỗi subject ghi ra đĩa rồi giải phóng RAM; ghép cuối bằng np.memmap (không np.concatenate).
      Có RESUME: subject đã xong sẽ được bỏ qua khi chạy lại.
  [3] Tốc độ: resample 1000 -> 250 Hz NGAY SAU KHI ĐỌC (trước khi lọc), ICA fit trên các epoch 1 s đã
      loại đoạn nhiễu lớn, ưu tiên Picard (tự fallback FastICA).
  [4] Notch 50 Hz: tự quyết định dựa trên đo đạc (độ nổi 50 Hz trừ suy giảm của bộ bandpass).
  [5] IC mắt: dùng proxy VEOG/HEOG từ kênh trán + tương quan + hệ số tải vùng trán + giới hạn số IC loại;
      so sánh biên độ P300 trước/sau ICA để phát hiện mất tín hiệu hữu ích.
  [6] Đường dẫn: không phụ thuộc __file__; nhận từ tham số dòng lệnh / biến môi trường EEG_ROOT / Colab.
  [7] Ablation: 3 nhánh KHỚP TỪNG TRIAL: A (no ICA, no reject), B (ICA, no reject), C (ICA + reject = B[keep_mask]).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import freqz
from tqdm import tqdm

mne.set_log_level("WARNING")


# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
CONFIG: Dict[str, Any] = {
    # Đường dẫn: để None -> tự suy ra từ EEG_ROOT / vị trí file / thư mục hiện tại (xem resolve_paths)
    "RAW_DIR": None,
    "PROCESSED_DIR": None,
    "FIGURES_DIR": None,
    # Colab: đặt "/content/tmp_raw" để copy từng subject từ Drive về đĩa local trước khi đọc (nhanh hơn nhiều)
    "STAGE_LOCAL_DIR": None,
    "RESUME": True,          # bỏ qua subject đã xử lý xong
    "N_JOBS": 2,             # số luồng cho filter/resample (Colab thường có 2 core)

    "EVENT_TRIGGER_MAP": {
        "Frequent": "Stimulus/S  5",
        "Target": "Stimulus/S  6",

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
        "flat_threshold_std": 1e-7,
        "noisy_threshold_z": 5.0,
        "interpolate": True,
        "max_bad_fraction": 0.10,    # cảnh báo nếu > 10% kênh hỏng
        # Kênh trán có std lớn hơn một cách tự nhiên do chớp mắt -> nới ngưỡng noisy để không mất proxy EOG
        "frontal_channels": ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8", "AFz", "F7", "F8"],
        "frontal_z_multiplier": 2.0,
    },

    "REFERENCE": "average",

    "MONTAGE": {
        "min_measured_coverage": 0.95,  # dùng tọa độ đo thực tế nếu phủ >= 95% số kênh EEG
        "radius_range_m": (0.05, 0.15),  # bán kính trung vị hợp lệ của điện cực trên da đầu (m)
    },

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

    "CHANNEL_ALIASES": {
        "TP9 LEFT EAR": "TP9",
        "TP10 RIGHT EAR": "TP10",
        "O9": "I1",
        "O10" : "I2",
    },

    "EXCLUDE_SUBJECTS": ["sub-02", "sub-14", "sub-17"],

    

    "SAVE_SINGLE_SUBJECT_FIF": True,
    "SAVE_ABLATION_STUDY": True,    # lưu A (no ICA/no reject), B (ICA/no reject), keep_mask
}


# ==============================================================================
# 2. ĐƯỜNG DẪN & LOGGING
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


# ==============================================================================
# 3. TIỆN ÍCH SỰ KIỆN
# ==============================================================================
def natural_sort_key(s: str) -> List[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


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


def extract_events_safe(
    raw,
    trigger_map: Dict[str, str],
    logger,
    response_trigger: Optional[str] = None,
    rt_window_ms: Tuple[float, float] = (150.0, 1500.0),
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """Khớp trigger, lọc hành vi (Target Hit & Frequent CR) và tính rt_ms."""
    _, event_dict = mne.events_from_annotations(raw, verbose=False)
    mapping = {}
    for class_name, pattern in trigger_map.items():
        code = 1 if class_name == "Frequent" else 2
        if pattern in event_dict:
            mapping[pattern] = code
            continue
        norm = normalize_event_name(pattern)
        matches = [a for a in event_dict if normalize_event_name(a) == norm]
        if len(matches) == 1:
            mapping[matches[0]] = code
        else:
            msg = (f"LỖI GÁN NHÃN: Trigger '{class_name}' (pattern='{pattern}') không khớp duy nhất! "
                   f"Tìm thấy {len(matches)}: {matches}. Annotations trong file: {list(event_dict.keys())}")
            logger.error(msg)
            raise ValueError(msg)

    # Nếu không truyền response_trigger thì giữ nguyên hành vi cũ
    if not response_trigger:
        events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
        return events, {"Frequent": 1, "Target": 2}, np.full(len(events), np.nan, dtype=np.float32)

    # Tìm marker phản hồi (Response)
    if response_trigger in event_dict:
        mapping[response_trigger] = 99
    else:
        norm_r = normalize_event_name(response_trigger)
        matches_r = [a for a in event_dict if normalize_event_name(a) == norm_r]
        if len(matches_r) == 1:
            mapping[matches_r[0]] = 99
        else:
            msg = (f"LỖI GÁN NHÃN: Response trigger '{response_trigger}' không khớp duy nhất! "
                   f"Tìm thấy {len(matches_r)}: {matches_r}. Annotations: {list(event_dict.keys())}")
            logger.error(msg)
            raise ValueError(msg)

    all_events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose=False)
    sfreq = raw.info["sfreq"]
    rt_min_samp = int(round(rt_window_ms[0] * sfreq / 1000.0))
    rt_max_samp = int(round(rt_window_ms[1] * sfreq / 1000.0))

    stim_events = all_events[np.isin(all_events[:, 2], [1, 2])]
    resp_samples = all_events[all_events[:, 2] == 99, 0]

    kept_events = []
    kept_rt_ms = []
    for ev in stim_events:
        s_stim, code = ev[0], ev[2]
        # Tìm nút bấm đầu tiên nằm trong cửa sổ [rt_min, rt_max]
        idx = np.searchsorted(resp_samples, s_stim + rt_min_samp, side="left")
        has_resp = idx < len(resp_samples) and (resp_samples[idx] - s_stim) <= rt_max_samp

        if code == 2 and has_resp:  # Target Hit -> giữ và tính rt_ms
            kept_events.append(ev)
            kept_rt_ms.append((resp_samples[idx] - s_stim) * 1000.0 / sfreq)
        elif code == 1 and not has_resp:  # Frequent CR -> giữ, rt_ms = NaN
            kept_events.append(ev)
            kept_rt_ms.append(np.nan)

    events = np.array(kept_events, dtype=np.int64) if kept_events else np.empty((0, 3), dtype=np.int64)
    rt_ms = np.array(kept_rt_ms, dtype=np.float32)
    return events, {"Frequent": 1, "Target": 2}, rt_ms


# ==============================================================================
# 4. MONTAGE (SỬA [1])
# ==============================================================================
def _read_bids_electrodes(tsv: Path, coordsystem: Optional[Path], aliases: Dict[str, str]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(tsv, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    if not {"name", "x", "y", "z"}.issubset(cols):
        raise ValueError(f"electrodes.tsv thiếu cột name/x/y/z (có: {list(df.columns)})")
    scale = 1.0
    if coordsystem is not None and coordsystem.exists():
        meta = json.loads(coordsystem.read_text(encoding="utf-8"))
        unit = str(meta.get("EEGCoordinateUnits", meta.get("iEEGCoordinateUnits", "m"))).lower()
        scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}.get(unit, 1.0)
    out = {}
    for _, r in df.iterrows():
        xyz = pd.to_numeric(pd.Series([r[cols["x"]], r[cols["y"]], r[cols["z"]]]), errors="coerce").to_numpy(float)
        if np.all(np.isfinite(xyz)):
            name = aliases.get(str(r[cols["name"]]), str(r[cols["name"]]))
            out[name] = xyz * scale
    return out


def _measured_positions(sub_dir: Path, sub_id: str, aliases: Dict[str, str], logger) -> Tuple[Dict[str, np.ndarray], str]:
    tsvs = sorted(sub_dir.glob(f"*{sub_id}*electrodes.tsv")) or sorted(sub_dir.glob("*electrodes.tsv"))
    if tsvs:
        cs = sorted(sub_dir.glob(f"*{sub_id}*coordsystem.json")) or sorted(sub_dir.glob("*coordsystem.json"))
        try:
            pos = _read_bids_electrodes(tsvs[0], cs[0] if cs else None, aliases)
            return pos, f"measured:{tsvs[0].name}"
        except Exception as e:
            logger.warning(f"[{sub_id}] Không đọc được {tsvs[0].name}: {e}")
    bvct = sorted(sub_dir.glob("*.bvct"))
    if bvct:
        try:
            m = mne.channels.read_dig_captrak(bvct[0])
            pos = {aliases.get(k, k): np.asarray(v) for k, v in m.get_positions()["ch_pos"].items()}
            return pos, f"measured:{bvct[0].name}"
        except Exception as e:
            logger.warning(f"[{sub_id}] Không đọc được {bvct[0].name}: {e}")
    return {}, ""


def template_montage():
    """Montage 10-05 mẫu. MNE >= 1.13 đã deprecate 'standard_1005' (thay bằng 'colin27_1005')."""
    available = mne.channels.get_builtin_montages()
    name = "colin27_1005" if "colin27_1005" in available else "standard_1005"
    return mne.channels.make_standard_montage(name), name


def setup_channel_montage(raw, vhdr_path: Path, channel_aliases: Dict[str, str], logger,
                          montage_cfg: Optional[Dict] = None) -> Tuple[List[str], str]:
    """Đổi tên kênh alias, gán tọa độ (đo thực tế nếu hợp lệ, ngược lại standard_1005).

    Trả về (danh sách kênh không có tọa độ, nguồn montage).
    """
    montage_cfg = montage_cfg or CONFIG["MONTAGE"]
    rename = {o: n for o, n in channel_aliases.items() if o in raw.ch_names}
    if rename:
        raw.rename_channels(rename)

    sub_id = extract_subject_id(vhdr_path)
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]

    source = "template"
    montage = None
    measured, tag = _measured_positions(vhdr_path.parent, sub_id, channel_aliases, logger)
    if measured:
        covered = [c for c in eeg_names if c in measured]
        coverage = len(covered) / max(len(eeg_names), 1)
        radii = np.linalg.norm(np.array([measured[c] for c in covered]), axis=1) if covered else np.array([0.0])
        r_lo, r_hi = montage_cfg["radius_range_m"]
        if coverage >= montage_cfg["min_measured_coverage"] and r_lo <= float(np.median(radii)) <= r_hi:
            montage = mne.channels.make_dig_montage(ch_pos={c: measured[c] for c in covered}, coord_frame="head")
            source = tag
        else:
            logger.warning(
                f"[{sub_id}] Tọa độ đo thực tế bị từ chối (phủ {coverage:.0%}, bán kính trung vị "
                f"{np.median(radii):.4f} m, kỳ vọng {r_lo}-{r_hi} m; kiểm tra đơn vị/hệ tọa độ). Dùng montage mẫu 10-05."
            )
    if montage is None:
        montage, tname = template_montage()
        source = f"template:{tname}"

    raw.set_montage(montage, on_missing="ignore", verbose=False)

    no_pos = []
    for idx in mne.pick_types(raw.info, eeg=True, exclude=[]):
        pos = raw.info["chs"][idx]["loc"][:3]
        if not np.all(np.isfinite(pos)) or np.linalg.norm(pos) < 1e-10:
            no_pos.append(raw.info["chs"][idx]["ch_name"])
    if no_pos:
        logger.warning(f"[{sub_id}] {len(no_pos)} kênh không có tọa độ ({source}): {no_pos}")
    return no_pos, source


# ==============================================================================
# 5. NHIỄU LƯỚI, KÊNH HỎNG
# ==============================================================================
def measure_line_noise_db(raw, line_freq: float, max_sec: float) -> Optional[float]:
    """Độ nổi (dB) của đỉnh nhiễu lưới so với phổ lân cận, trên dữ liệu CHƯA lọc."""
    sfreq = raw.info["sfreq"]
    if line_freq + 8 >= sfreq / 2:
        return None
    tmax = min(max_sec, raw.times[-1])
    n_fft = int(min(sfreq * 4, tmax * sfreq))
    psd = raw.compute_psd(method="welch", fmin=line_freq - 8, fmax=line_freq + 8, tmax=tmax,
                          picks="eeg", n_fft=n_fft, verbose=False)
    data, freqs = psd.get_data(return_freqs=True)
    db = (10 * np.log10(np.maximum(data, 1e-30))).mean(axis=0)
    d = np.abs(freqs - line_freq)
    on, off = d <= 0.5, (d >= 2.0) & (d <= 6.0)
    if not on.any() or not off.any():
        return None
    return float(db[on].max() - np.median(db[off]))


def fir_attenuation_db(sfreq: float, l_freq: float, h_freq: float, at_freq: float) -> float:
    h = mne.filter.create_filter(None, sfreq, l_freq, h_freq, method="fir", phase="zero",
                                 fir_design="firwin", verbose=False)
    _, resp = freqz(h, worN=[2 * np.pi * at_freq / sfreq])
    return float(20 * np.log10(max(abs(resp[0]), 1e-12)))


def decide_notch(raw, fcfg: Dict, logger, sub_id: str) -> Tuple[bool, Optional[float], Optional[float]]:
    mode = fcfg.get("notch_mode", "auto")
    if mode == "never":
        return False, None, None
    prom = measure_line_noise_db(raw, fcfg["notch_freq"], fcfg["line_noise_max_sec"])
    if mode == "always":
        return True, prom, None
    if prom is None:
        return False, None, None
    atten = fir_attenuation_db(raw.info["sfreq"], fcfg["l_freq"], fcfg["h_freq"], fcfg["notch_freq"])
    residual = prom + atten
    apply = residual > fcfg["notch_residual_db"]
    logger.info(f"[{sub_id}] Nhiễu {fcfg['notch_freq']:.0f} Hz: nổi {prom:.1f} dB, bandpass suy giảm {atten:.1f} dB "
                f"-> còn {residual:.1f} dB => {'CẦN' if apply else 'KHÔNG cần'} notch.")
    return apply, prom, residual


def detect_and_handle_bad_channels(raw, cfg: Dict, logger, sub_id: str) -> List[str]:
    """SỬA [1]: KHÔNG còn miễn trừ kênh thiếu tọa độ khỏi việc phát hiện kênh hỏng."""
    if not cfg.get("enabled", True):
        return []
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)
    # Phân vị 25% của std theo cửa sổ 2 s: bền với chớp mắt/nhiễu thoáng qua (std toàn bản ghi sẽ khiến Fp1/Fp2
    # bị đánh dấu "noisy" chỉ vì chớp mắt, làm mất kênh proxy EOG cho bước ICA).
    win = int(2 * raw.info["sfreq"])
    n_win = data.shape[1] // win
    if n_win >= 5:
        stds = np.percentile(data[:, : n_win * win].reshape(len(picks), n_win, win).std(axis=2), 25, axis=1)
    else:
        stds = data.std(axis=1)
    del data
    med = np.median(stds)
    mad = np.median(np.abs(stds - med))
    if mad <= 1e-12:
        logger.warning(f"[{sub_id}] MAD ≈ 0. Bỏ qua phát hiện noisy channels.")
        z = np.zeros_like(stds)
    else:
        z = 0.6745 * (stds - med) / mad
    bads = []
    front_set = set(cfg.get("frontal_channels", []))
    for ch, s, zz in zip(names, stds, z):
        if s < cfg["flat_threshold_std"]:
            bads.append(ch)
            logger.warning(f"[{sub_id}] FLAT channel: {ch} (std={s * 1e6:.3f} µV)")
        elif zz > cfg["noisy_threshold_z"] * (cfg.get("frontal_z_multiplier", 1.0) if ch in front_set else 1.0):
            bads.append(ch)
            logger.warning(f"[{sub_id}] NOISY channel: {ch} (z={zz:.2f})")
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bads))
    if len(bads) / max(len(names), 1) > cfg.get("max_bad_fraction", 0.1):
        logger.warning(f"[{sub_id}] CẢNH BÁO: {len(bads)}/{len(names)} kênh hỏng (> {cfg['max_bad_fraction']:.0%}).")
    return bads


def parse_drop_log_reasons(epochs) -> Dict[str, int]:
    """Phân loại nguyên nhân loại epoch (epochs không dùng reject biên độ)."""
    out = {"annotation_rejected": 0, "boundary_or_other": 0}
    for drop in epochs.drop_log:
        if len(drop) == 0:
            continue
        if any("BAD" in str(i).upper() for i in drop):
            out["annotation_rejected"] += 1
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
# 7. XỬ LÝ 1 SUBJECT (ghi thẳng ra đĩa, trả về stats nhỏ)
# ==============================================================================
def _subject_files(sub_dir: Path, sub_id: str) -> Dict[str, Path]:
    return {
        "X_clean": sub_dir / f"{sub_id}_X_clean.npy",
        "y_clean": sub_dir / f"{sub_id}_y_clean.npy",
        "X_a": sub_dir / f"{sub_id}_X_no_ica.npy",
        "X_b": sub_dir / f"{sub_id}_X_ica_all.npy",
        "y_all": sub_dir / f"{sub_id}_y_all.npy",
        "keep": sub_dir / f"{sub_id}_keep.npy",
        "meta": sub_dir / f"{sub_id}_meta.csv",
        "stats": sub_dir / f"{sub_id}_stats.json",
    }


def process_single_subject(vhdr_path: Path, config: Dict, reference_channel_order: List[str],
                           logger, out_dir: Path) -> Dict[str, Any]:
    t0 = time.time()
    sub_id = extract_subject_id(vhdr_path)
    logger.info(f"--- BẮT ĐẦU XỬ LÝ: {sub_id} ({vhdr_path.name}) ---")
    n_jobs = config.get("N_JOBS", 1)
    fcfg, ica_cfg, ecfg = config["FILTER"], config["ICA"], config["EPOCHS"]
    files = _subject_files(out_dir / "subjects", sub_id)
    files["stats"].parent.mkdir(parents=True, exist_ok=True)

    # 1. Đọc, chuẩn hóa tên kênh, montage
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    orig_sfreq = raw.info["sfreq"]
    no_pos, montage_source = setup_channel_montage(raw, vhdr_path, config["CHANNEL_ALIASES"], logger, config["MONTAGE"])

    # 2. Đồng bộ thứ tự kênh
    if set(raw.ch_names) != set(reference_channel_order):
        missing = set(reference_channel_order) - set(raw.ch_names)
        extra = set(raw.ch_names) - set(reference_channel_order)
        msg = f"[{sub_id}] Bộ kênh không đồng nhất! Thiếu: {missing}, Thừa: {extra}"
        logger.error(msg)
        raise ValueError(msg)
    raw.reorder_channels(reference_channel_order)

    # 3. Lấy events gốc -> CROP trước khi resample & filter
    events_raw, _, _ = extract_events_safe(raw, config["EVENT_TRIGGER_MAP"], logger)
    
    # Cắt bỏ phần đầu/cuối thừa (tránh đoạn rút mũ bão hòa ở sub-08, sub-35)
    t_first = events_raw[0, 0] / orig_sfreq
    t_last = events_raw[-1, 0] / orig_sfreq
    tmin_crop = max(0.0, t_first - 5.0)
    tmax_crop = min(raw.times[-1], t_last + 5.0)
    logger.info(f"[{sub_id}] Crop tín hiệu về [{tmin_crop:.2f}s, {tmax_crop:.2f}s] (gốc: {raw.times[-1]:.2f}s)")
    raw.crop(tmin=tmin_crop, tmax=tmax_crop)
    
    # Lấy lại events sau khi crop (kèm lọc hành vi Target Hit / Frequent CR và tính rt_ms)
    events, event_id, rt_ms_all = extract_events_safe(
        raw,
        config["EVENT_TRIGGER_MAP"],
        logger,
        response_trigger=config.get("RESPONSE_TRIGGER"),
        rt_window_ms=config.get("RT_WINDOW_MS", (150.0, 1500.0)),
    )
    if orig_sfreq != config["RESAMPLE_FREQ"]:
        logger.info(f"[{sub_id}] Resample {orig_sfreq:g} -> {config['RESAMPLE_FREQ']:g} Hz (cùng events)...")
        raw, events = raw.resample(config["RESAMPLE_FREQ"], events=events, npad="auto", n_jobs=n_jobs, verbose=False)

    # 4. Nhiễu lưới -> bandpass -> (notch nếu cần)  (SỬA [4])
    use_notch, line_prom, line_resid = decide_notch(raw, fcfg, logger, sub_id)
    raw.filter(fcfg["l_freq"], fcfg["h_freq"], picks="eeg", method="fir", phase="zero", n_jobs=n_jobs, verbose=False)
    if use_notch:
        raw.notch_filter(fcfg["notch_freq"], picks="eeg", method="fir", phase="zero", n_jobs=n_jobs, verbose=False)

    # 5. Kênh hỏng -> average reference (bỏ kênh hỏng) -> nội suy
    bad_channels = detect_and_handle_bad_channels(raw, config["BAD_CHANNELS"], logger, sub_id)
    logger.info(f"[{sub_id}] Bad channels trước reference: {raw.info['bads']}")
    raw.set_eeg_reference(ref_channels=config["REFERENCE"], projection=False, verbose=False)

    unrecoverable: List[str] = []
    n_interp = 0
    if raw.info["bads"]:
        if config["BAD_CHANNELS"].get("interpolate", True):
            unrecoverable = [c for c in raw.info["bads"] if c in set(no_pos)]
            interp = [c for c in raw.info["bads"] if c not in set(no_pos)]
            if interp:
                raw.info["bads"] = interp
                raw.interpolate_bads(reset_bads=True, verbose=False)
                n_interp = len(interp)
            if unrecoverable:
                # Không có tọa độ -> không nội suy được: đặt về 0 (kênh bị loại, shape giữ nguyên 127)
                logger.warning(f"[{sub_id}] Kênh hỏng không có tọa độ, đặt về 0: {unrecoverable}")
                raw._data[[raw.ch_names.index(c) for c in unrecoverable], :] = 0.0
            raw.info["bads"] = []
        else:
            raw.info["bads"] = []

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    good_for_rank = len(eeg_picks) - 1 - n_interp - len(unrecoverable)  # -1: average reference

    # 6. NHÁNH A: epoch trước ICA, không reject
    def make_epochs(r):
        return mne.Epochs(r, events=events, event_id=event_id, tmin=ecfg["tmin"], tmax=ecfg["tmax"],
                          baseline=ecfg["baseline"], picks=eeg_picks, reject=None, preload=True, verbose=False)

    epochs_a = make_epochs(raw)
    p300_a = p300_effect(epochs_a, ica_cfg["p300_channels"], ica_cfg["p300_window"])
    XA = epochs_a.get_data().astype(np.float32)
    sel_a = epochs_a.selection.copy()
    del epochs_a

    # 7. ICA: fit trên epoch 1 s đã loại đoạn nhiễu lớn
    var_before = float(np.var(raw.get_data(picks=eeg_picks), axis=1).sum())
    raw_fit = raw.copy().filter(ica_cfg["high_pass"], None, picks="eeg", n_jobs=n_jobs, verbose=False)
    fit_ep = mne.make_fixed_length_epochs(raw_fit, duration=ica_cfg["fit_epoch_sec"], preload=True,
                                          reject_by_annotation=False, verbose=False)
    n_before = len(fit_ep)
    fit_ep.drop_bad(reject=dict(eeg=ica_cfg["fit_reject_ptp"]), verbose=False)
    n_fit_epochs = len(fit_ep)
    if n_fit_epochs < ica_cfg["min_fit_epochs"]:
        logger.warning(f"[{sub_id}] Chỉ còn {n_fit_epochs}/{n_before} epoch 1 s để fit ICA (ít) -> kết quả ICA kém tin cậy.")

    n_comp = int(min(ica_cfg["n_components"], good_for_rank))
    method, fit_params = ica_cfg["method"], None
    if method == "picard":
        try:
            import picard  # noqa: F401
            fit_params = dict(ortho=False, extended=True)
        except ImportError: 
            logger.warning(f"[{sub_id}] Chưa cài python-picard -> dùng fastica (pip install python-picard).")
            method = "fastica"
    logger.info(f"[{sub_id}] ICA: rank≈{good_for_rank}, n_components={n_comp}, method={method}, fit trên {n_fit_epochs} epoch.")
    ica = mne.preprocessing.ICA(n_components=n_comp, method=method, fit_params=fit_params,
                                random_state=ica_cfg["random_state"], max_iter="auto", verbose=False)
    ica.fit(fit_ep, verbose=False)
    del raw_fit, fit_ep
    gc.collect()

    eog_inds, ic_details = select_eog_components(ica, raw, ica_cfg, logger, sub_id)
    logger.info(f"[{sub_id}] IC bị loại: {eog_inds} "
                f"(|r|={[round(ic_details[i]['abs_r'], 2) for i in eog_inds]})")
    if ica_cfg.get("save_diagnostic_plots", True) and ic_details:
        try:
            save_ica_diag(ic_details, eog_inds, Path(config["FIGURES_DIR"]), sub_id)
        except Exception as e:  # không để lỗi vẽ hình làm hỏng subject
            logger.debug(f"Không lưu được hình ICA: {e}")

    ica.exclude = eog_inds
    ica.apply(raw, verbose=False)
    var_after = float(np.var(raw.get_data(picks=eeg_picks), axis=1).sum())
    removed_var_pct = 100.0 * (1.0 - var_after / max(var_before, 1e-30))

    # 8. NHÁNH B (ICA, no reject) và C (ICA + reject) KHỚP TỪNG TRIAL  (SỬA [7])
    epochs_b = make_epochs(raw)
    if not np.array_equal(sel_a, epochs_b.selection):
        raise RuntimeError(f"[{sub_id}] Nhánh A và B không cùng tập epoch!")
    p300_b = p300_effect(epochs_b, ica_cfg["p300_channels"], ica_cfg["p300_window"])
    p300_change = None
    if p300_a is not None and p300_b is not None and abs(p300_a) > 1e-9:
        p300_change = 100.0 * (p300_b - p300_a) / abs(p300_a)
        if abs(p300_change) > ica_cfg["p300_change_warn_pct"]:
            logger.warning(f"[{sub_id}] CẢNH BÁO: biên độ P300 (Target-Frequent) đổi {p300_change:+.1f}% sau ICA. "
                           f"Kiểm tra IC bị loại {eog_inds}.")

    probe = epochs_b.copy().drop_bad(reject=dict(eeg=ecfg["reject_amp"]), verbose=False)
    keep = np.isin(epochs_b.selection, probe.selection)
    del probe

    XB = epochs_b.get_data().astype(np.float32)
    y_all = np.where(epochs_b.events[:, 2] == event_id["Target"],
                     config["TARGET_LABELS"]["Target"], config["TARGET_LABELS"]["Frequent"]).astype(np.int64)
    onset = epochs_b.events[:, 0].astype(np.int64)
    drop_b = parse_drop_log_reasons(epochs_b)

    # 9. Ghi ra đĩa
    np.save(files["X_clean"], XB[keep])
    np.save(files["y_clean"], y_all[keep])
    if config["SAVE_ABLATION_STUDY"]:
        np.save(files["X_a"], XA)
        np.save(files["X_b"], XB)
        np.save(files["y_all"], y_all)
        np.save(files["keep"], keep)
    pd.DataFrame({
        "subject_id": sub_id, "trial_idx": np.arange(len(y_all)), "onset_sample": onset,
        "event_type": np.where(y_all == 1, "Target", "Frequent"), "label": y_all,"rt_ms": rt_ms_all[epochs_b.selection],
         "keep": keep,
    }).to_csv(files["meta"], index=False)
    if config["SAVE_SINGLE_SUBJECT_FIF"] and keep.any():
        epochs_b[keep].save(out_dir / f"{sub_id}_clean-epo.fif", overwrite=True, verbose=False)

    initial = len(events)
    amp_rej = int((~keep).sum())
    stats = {
        "subject_id": sub_id,
        "initial_events": initial,
        "ica_no_reject_kept": len(epochs_b),
        "clean_kept": int(keep.sum()),
        "amplitude_rejected": amp_rej,
        "annotation_rejected": drop_b["annotation_rejected"],
        "boundary_rejected": initial - len(epochs_b),
        "reject_rate_pct": 100.0 * (initial - int(keep.sum())) / max(initial, 1),
        "n_frequent": int(((y_all == 0) & keep).sum()),
        "n_target": int(((y_all == 1) & keep).sum()),
        "montage_source": montage_source,
        "n_no_position": len(no_pos),
        "bad_channels": ";".join(bad_channels),
        "n_interpolated": n_interp,
        "unrecoverable_bads": ";".join(unrecoverable),
        "line_prominence_db": line_prom,
        "line_residual_db": line_resid,
        "notch_applied": bool(use_notch),
        "data_rank": int(good_for_rank),
        "ica_method": method,
        "ica_n_components": n_comp,
        "ica_n_fit_epochs": n_fit_epochs,
        "ica_excluded_ics": ";".join(map(str, eog_inds)),
        "ica_removed_var_pct": round(removed_var_pct, 3),
        "p300_diff_before_uV": None if p300_a is None else round(p300_a * 1e6, 3),
        "p300_diff_after_uV": None if p300_b is None else round(p300_b * 1e6, 3),
        "p300_change_pct": None if p300_change is None else round(p300_change, 2),
        "seconds": round(time.time() - t0, 1),
    }
    files["stats"].write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")
    logger.info(f"[{sub_id}] Xong: {initial} -> {stats['clean_kept']} epoch "
                f"(biên độ loại {amp_rej}, biên {stats['boundary_rejected']}); {stats['seconds']} s.")
    del raw, epochs_b, XA, XB, ica
    gc.collect()
    return stats


# ==============================================================================
# 8. GHÉP KẾT QUẢ BẰNG MEMMAP (SỬA [2])
# ==============================================================================
def merge_npy(paths: List[Path], out_path: Path, mask_paths: Optional[List[Path]] = None) -> Tuple[int, ...]:
    first = np.load(paths[0], mmap_mode="r")
    counts = []
    for i, p in enumerate(paths):
        n = np.load(p, mmap_mode="r").shape[0]
        counts.append(int(np.load(mask_paths[i]).sum()) if mask_paths else n)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=first.dtype, shape=(sum(counts), *first.shape[1:]))
    i = 0
    for k, p in enumerate(paths):
        a = np.load(p, mmap_mode="r")
        a = a[np.load(mask_paths[k])] if mask_paths else np.asarray(a)
        out[i:i + len(a)] = a
        i += len(a)
    out.flush()
    shape = out.shape
    del out
    return shape


def stage_subject_files(vhdr_path: Path, stage_dir: Path, sub_id: str) -> Path:
    """Copy file của 1 subject từ Drive về đĩa local (Colab), trả về đường dẫn .vhdr local."""
    dst = stage_dir / sub_id
    dst.mkdir(parents=True, exist_ok=True)
    for f in vhdr_path.parent.iterdir():
        if not f.is_file():
            continue
        same_recording = f.stem == vhdr_path.stem
        coord = f.name.endswith(("electrodes.tsv", "coordsystem.json")) and sub_id in f.name or f.suffix == ".bvct"
        if same_recording or coord:
            shutil.copy2(f, dst / f.name)
    return dst / vhdr_path.name


# ==============================================================================
# 9. PIPELINE CHÍNH
# ==============================================================================
def run_pipeline(config: Dict = CONFIG) -> None:

    config = resolve_paths(config)
    raw_dir, processed_dir = Path(config["RAW_DIR"]), Path(config["PROCESSED_DIR"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(processed_dir / "preprocessing.log")

    status = {"status": "RUNNING", "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "config": _jsonable(config)}
    (processed_dir / "pipeline_status.json").write_text(json.dumps(status, indent=4), encoding="utf-8")

    logger.info("=" * 80)
    logger.info("PIPELINE TIỀN XỬ LÝ EEG P300 (V3)")
    logger.info(f"Nguồn thô:  {raw_dir.resolve()}")
    logger.info(f"Đầu ra:     {processed_dir.resolve()}")
    logger.info("=" * 80)

    exclude_set = set(config.get("EXCLUDE_SUBJECTS", []))
    vhdr_files = [
        f for f in sorted(raw_dir.rglob("*.vhdr"), key=natural_sort_key)
        if extract_subject_id(f) not in exclude_set
    ]
    total = len(vhdr_files)
    logger.info(f"Tìm thấy {total} subjects.")
    if total == 0:
        status["status"] = "FAILED_NO_FILES"
        (processed_dir / "pipeline_status.json").write_text(json.dumps(status, indent=4), encoding="utf-8")
        return

    raw_ref = mne.io.read_raw_brainvision(vhdr_files[0], preload=False, verbose=False)
    setup_channel_montage(raw_ref, vhdr_files[0], config["CHANNEL_ALIASES"], logger, config["MONTAGE"])
    reference_channel_order = list(raw_ref.ch_names)
    del raw_ref
    logger.info(f"Thứ tự kênh chuẩn ({len(reference_channel_order)} kênh): {reference_channel_order[:10]}...")

    sub_dir = processed_dir / "subjects"
    sub_dir.mkdir(exist_ok=True)
    stage_dir = Path(config["STAGE_LOCAL_DIR"]) if config.get("STAGE_LOCAL_DIR") else None

    stats_list: List[Dict] = []
    failed: List[Dict] = []
    t_start = time.time()

    for vhdr in tqdm(vhdr_files, desc="Subjects", unit="sub", file=sys.stdout):
        sub_id = extract_subject_id(vhdr)
        files = _subject_files(sub_dir, sub_id)
        need = [files["X_clean"], files["y_clean"], files["meta"], files["stats"]]
        if config["SAVE_ABLATION_STUDY"]:
            need += [files["X_a"], files["X_b"], files["y_all"], files["keep"]]
        if config.get("RESUME", True) and all(p.exists() for p in need):
            logger.info(f"[{sub_id}] Đã có kết quả -> bỏ qua (RESUME).")
            stats_list.append(json.loads(files["stats"].read_text(encoding="utf-8")))
            continue
        staged = None
        try:
            path = vhdr
            if stage_dir is not None:
                path = staged = stage_subject_files(vhdr, stage_dir, sub_id)
            stats_list.append(process_single_subject(path, config, reference_channel_order, logger, processed_dir))
        except Exception as err:
            logger.error(f"[!] LỖI TẠI {sub_id}: {err}")
            logger.error(traceback.format_exc())
            failed.append({"subject_id": sub_id, "error": str(err)})
        finally:
            if staged is not None:
                shutil.rmtree(staged.parent, ignore_errors=True)
            gc.collect()

    duration = time.time() - t_start

    # ---- Ghép đầu ra (memmap, không tốn RAM) ----
    ok_ids = [s["subject_id"] for s in stats_list]
    F = {sid: _subject_files(sub_dir, sid) for sid in ok_ids}
    logger.info("=" * 80 + "\nĐANG GHÉP DỮ LIỆU ĐẦU RA (memmap)...")
    if ok_ids:
        shape = merge_npy([F[s]["X_clean"] for s in ok_ids], processed_dir / "X_clean.npy")
        y_shape = merge_npy([F[s]["y_clean"] for s in ok_ids], processed_dir / "y.npy")
        ids_arr = np.concatenate([np.array([s] * int(np.load(F[s]["y_clean"], mmap_mode="r").shape[0]), dtype="U10") for s in ok_ids])
        np.save(processed_dir / "subject_ids.npy", ids_arr)
        logger.info(f"✅ X_clean.npy {shape} | y.npy {y_shape} | subject_ids.npy {ids_arr.shape}")

        if config["SAVE_ABLATION_STUDY"]:
            sa = merge_npy([F[s]["X_a"] for s in ok_ids], processed_dir / "X_no_ica_no_reject.npy")
            sb = merge_npy([F[s]["X_b"] for s in ok_ids], processed_dir / "X_ica_no_reject.npy")
            merge_npy([F[s]["y_all"] for s in ok_ids], processed_dir / "y_all.npy")
            merge_npy([F[s]["keep"] for s in ok_ids], processed_dir / "keep_mask.npy")
            ids_all = np.concatenate([np.array([s] * int(np.load(F[s]["y_all"], mmap_mode="r").shape[0]), dtype="U10") for s in ok_ids])
            np.save(processed_dir / "subject_ids_all.npy", ids_all)
            logger.info(f"✅ [Ablation] A={sa} | B={sb} | keep_mask/y_all/subject_ids_all cùng độ dài. "
                        f"C = B[keep_mask] = X_clean.")

        meta = pd.concat([pd.read_csv(F[s]["meta"]) for s in ok_ids], ignore_index=True)
        meta.to_csv(processed_dir / "metadata_all.csv", index=False)
        meta[meta["keep"]].drop(columns="keep").reset_index(drop=True).to_csv(processed_dir / "metadata.csv", index=False)
        pd.DataFrame(stats_list).to_csv(processed_dir / "subject_summary.csv", index=False)

    tmin, tmax = config["EPOCHS"]["tmin"], config["EPOCHS"]["tmax"]
    n_t = int(np.round((tmax - tmin) * config["RESAMPLE_FREQ"])) + 1
    np.save(processed_dir / "times.npy", np.linspace(tmin, tmax, n_t))
    (processed_dir / "channel_names.json").write_text(json.dumps(reference_channel_order, indent=4), encoding="utf-8")
    (processed_dir / "preprocessing_config.json").write_text(json.dumps(_jsonable(config), indent=4), encoding="utf-8")
    if failed:
        pd.DataFrame(failed).to_csv(processed_dir / "failed_subjects.csv", index=False)
        logger.warning(f"⚠️ {len(failed)} subjects lỗi (failed_subjects.csv). Chạy lại để RESUME các subject còn thiếu.")

    # ---- Tổng kết ----
    df = pd.DataFrame(stats_list)
    if len(df):
        ini, clean = int(df["initial_events"].sum()), int(df["clean_kept"].sum())
        nf, nt = int(df["n_frequent"].sum()), int(df["n_target"].sum())
        flagged = df[df["p300_change_pct"].abs() > config["ICA"]["p300_change_warn_pct"]]["subject_id"].tolist() \
            if "p300_change_pct" in df else []
        logger.info(f"""
================================================================================
BÁO CÁO TỔNG KẾT
================================================================================
 • Thời gian (lần chạy này):  {duration / 60:.1f} phút
 • Subjects thành công/lỗi:   {len(df)} / {len(failed)} (tổng {total})
 • Initial events:            {ini:,}
 • Epoch giữ lại (clean):     {clean:,}
 • Loại do biên độ:           {int(df['amplitude_rejected'].sum()):,}
 • Loại do biên/thiếu dữ liệu:{int(df['boundary_rejected'].sum()):,}
 • Lớp 0 Frequent / lớp 1 Target: {nf:,} / {nt:,} (~1 Target : {nf / max(nt, 1):.1f} Frequent)
 • Montage đo thực tế:        {int(df['montage_source'].astype(str).str.startswith('measured').sum())}/{len(df)} subjects
 • Notch được áp dụng:        {int(df['notch_applied'].sum())}/{len(df)} subjects
 • Subject P300 đổi > {config['ICA']['p300_change_warn_pct']:.0f}% sau ICA: {flagged}
LƯU Ý: dùng GroupKFold / LOSO theo subject_ids.npy, KHÔNG train_test_split ngẫu nhiên.
================================================================================""")

    if not failed and len(df) == total:
        status["status"] = "COMPLETED"
    elif len(df) > 0:
        status["status"] = "COMPLETED_WITH_ERRORS"
    else:
        status["status"] = "FAILED"
    status.update(end_time=time.strftime("%Y-%m-%d %H:%M:%S"), successful_subjects=len(df), failed_subjects=len(failed))
    (processed_dir / "pipeline_status.json").write_text(json.dumps(status, indent=4), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir"); p.add_argument("--out-dir"); p.add_argument("--figures-dir")
    p.add_argument("--stage-local-dir", help="Colab: thư mục tạm trên đĩa local, vd /content/tmp_raw")
    p.add_argument("--no-resume", action="store_true")
    args, _ = p.parse_known_args(argv)  # parse_known_args: bỏ qua tham số '-f' của Jupyter
    cfg = dict(CONFIG)
    if args.raw_dir: cfg["RAW_DIR"] = args.raw_dir
    if args.out_dir: cfg["PROCESSED_DIR"] = args.out_dir
    if args.figures_dir: cfg["FIGURES_DIR"] = args.figures_dir
    if args.stage_local_dir: cfg["STAGE_LOCAL_DIR"] = args.stage_local_dir
    if args.no_resume: cfg["RESUME"] = False
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
