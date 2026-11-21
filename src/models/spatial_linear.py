"""Standard ML Nhánh 1: Các Pipeline Phân loại EEG Tuyến tính & Không gian.

Module này ánh xạ (re-export) cấu hình pipeline từ `scripts/spatial_linear.py`
để đảm bảo khả năng tương thích ngược hoàn toàn với `src.models`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc của project có trong sys.path để import scripts.spatial_linear
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.spatial_linear import (
    MemoryEfficientXdawn,
    TimeDecimator,
    get_spatial_linear_models,
)

__all__ = [
    "MemoryEfficientXdawn",
    "TimeDecimator",
    "get_spatial_linear_models",
]
