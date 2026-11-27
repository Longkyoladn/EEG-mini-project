"""Interactive VS Code Colab runner (`# %%` cells) for ML3 P300 XGBoost training.

Cách mở file `.py` này thành môi trường Colab Kernel ngay trong VS Code:
1. Cài 2 extension chính chủ trên VS Code: **Google Colab** (`googlecolab.colab`) và **Jupyter** (`ms-toolsai.jupyter`).
2. Mở file này (`scripts/training/colab_ml3_runner.py`), bấm nút **Run Cell** phía trên dòng `# %%` đầu tiên -> cửa sổ **Interactive Window** sẽ bật lên.
3. Ở góc trên bên phải của **Interactive Window**, bấm chọn Kernel -> **Select Another Kernel...** -> **Colab** -> **New Colab Server** (chọn GPU T4/L4) và đăng nhập Google.
4. Nhấn `Ctrl+Shift+P` -> gõ **`Colab: Mount Google Drive to Server...`** để gắn `/content/drive` vào máy ảo Colab.
5. Chạy lần lượt các cell `# %%` bên dưới (`Shift+Enter`).
"""

# %% [markdown]
# ### Cell 1: Kiểm tra GPU máy ảo Colab, đồng bộ repo vào `/content` & cài `xgboost`
# Lưu ý: Code chạy trên máy ảo Colab từ xa (`/content`), nên cần clone/pull repo GitHub (hoặc copy code từ Drive) vào `/content/EEG-mini-project`.

# %%
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Chỉnh lại URL repo hoặc đường dẫn repo trên Drive nếu cần
REPO_GIT_URL = "https://github.com/Longkyoladn/EEG-mini-project.git"
REPO_BRANCH = "main"
COLAB_REPO_DIR = Path("/content/EEG-mini-project")
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/EEG-mini-project")
LOCAL_FEATURES_DIR = Path("/content/features")

subprocess.run(["nvidia-smi"], check=False)

if not COLAB_REPO_DIR.exists():
    if (DRIVE_PROJECT_DIR / "src").is_dir() and (DRIVE_PROJECT_DIR / "splits").is_dir():
        print(f"Copying repo code from Drive {DRIVE_PROJECT_DIR} -> {COLAB_REPO_DIR} ...")
        shutil.copytree(
            DRIVE_PROJECT_DIR,
            COLAB_REPO_DIR,
            ignore=shutil.ignore_patterns("features", "results", "data", "oddball_data", ".git"),
        )
    else:
        subprocess.run(
            ["git", "clone", "-b", REPO_BRANCH, REPO_GIT_URL, str(COLAB_REPO_DIR)],
            check=True,
        )
else:
    if (COLAB_REPO_DIR / ".git").is_dir():
        subprocess.run(["git", "-C", str(COLAB_REPO_DIR), "pull"], check=False)

os.chdir(COLAB_REPO_DIR)
if str(COLAB_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_REPO_DIR))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost>=2.0.0"], check=True)
print(f"Working directory ready: {Path.cwd()}")

# %% [markdown]
# ### Cell 2: Kiểm tra Google Drive & Copy `features` từ MyDrive sang `/content/features` (SSD cục bộ)
# - Nếu chưa mount Drive trong VS Code: nhấn `Ctrl+Shift+P` -> **`Colab: Mount Google Drive to Server...`**.
# - Hàm dưới đây chỉ copy các file `.npz` chưa có trong `/content/features` (chạy lại mất < 0.1s), giúp quá trình train đọc từ ổ NVMe `/content` cực nhanh còn kết quả vẫn lưu an toàn về Google Drive!

# %%
from scripts.training.train_ml3 import main as train_ml3_main, setup_colab_drive_paths

paths = setup_colab_drive_paths(
    drive_project_dir=DRIVE_PROJECT_DIR,
    stage_to_content=True,
    local_features_dir=LOCAL_FEATURES_DIR,
)
for k, v in paths.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ### Cell 3: Kiểm tra nhanh cấu hình & số lượng feature (`--plan-only`)

# %%
train_ml3_main(
    [
        "--features-dir", str(paths["features_dir"]),   # /content/features (tốc độ cao)
        "--splits-dir", str(COLAB_REPO_DIR / "splits"),
        "--protocol", "cs5",
        "--output-dir", str(paths["output_dir"]),       # lưu thẳng về Google Drive
        "--device", "cuda",
        "--plan-only",
    ]
)

# %% [markdown]
# ### Cell 4: Chạy Cross-Subject 5-Fold (5 kịch bản chính)
# Đọc feature từ `/content/features`, lưu checkpoint từng fold + biểu đồ về `/content/drive/MyDrive/EEG-mini-project/results/ml3/cs5`.

# %%
train_ml3_main(
    [
        "--features-dir", str(paths["features_dir"]),
        "--splits-dir", str(COLAB_REPO_DIR / "splits"),
        "--protocol", "cs5",
        "--experiments", "temporal", "wavelet_w5", "wavelet_w8", "combined_w5", "combined_w8",
        "--output-dir", str(paths["output_dir"]),
        "--device", "cuda",
        "--resume",
    ]
)

# %% [markdown]
# ### Cell 5: Chạy LOSO (3 kịch bản chính: `temporal`, `combined_w5`, `combined_w8`)

# %%
train_ml3_main(
    [
        "--features-dir", str(paths["features_dir"]),
        "--splits-dir", str(COLAB_REPO_DIR / "splits"),
        "--protocol", "loso",
        "--experiments", "temporal", "combined_w5", "combined_w8",
        "--output-dir", str(paths["output_dir"]),
        "--device", "cuda",
        "--resume",
    ]
)

# %% [markdown]
# ### Cell 6 (Tùy chọn): Kiểm tra bổ sung Gamma trên CS5
# Đổi `"combined_w5_gamma"` thành `"combined_w8_gamma"` nếu `W8` là cấu hình cần đối chứng.

# %%
# Bỏ comment khi muốn chạy thí nghiệm bổ sung Gamma:
# train_ml3_main(
#     [
#         "--features-dir", str(paths["features_dir"]),
#         "--splits-dir", str(COLAB_REPO_DIR / "splits"),
#         "--protocol", "cs5",
#         "--experiments", "combined_w5_gamma",
#         "--output-dir", str(paths["output_dir"]),
#         "--device", "cuda",
#         "--resume",
#     ]
# )
