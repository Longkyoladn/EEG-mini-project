import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

# Tự động lưu ngay tại thư mục chứa script này (oddball_data)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Toàn bộ 41 subjects hợp lệ của bài test Oddball (sub-01 đến sub-42, trừ sub-38 bị khuyết)
SUBJECTS = [i for i in range(1, 43) if i != 38]

# 4 file bắt buộc của task-oddball
REQUIRED_SUFFIXES = (
    "task-oddball_eeg.vhdr",
    "task-oddball_eeg.vmrk",
    "task-oddball_eeg.eeg",
    "task-oddball_events.tsv",
)

BUCKET = "openneuro.org"
PREFIX = "ds004621/"
MAX_WORKERS = 4  # Tải song song 4 luồng cùng lúc


def download_single_file(task):
    sub_str, suffix = task
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    sub_dir = os.path.join(DATA_DIR, sub_str)
    os.makedirs(sub_dir, exist_ok=True)

    fname = f"{sub_str}_{suffix}"
    s3_key = f"{PREFIX}{sub_str}/eeg/{fname}"
    local_path = os.path.join(sub_dir, fname)

    try:
        meta = s3.head_object(Bucket=BUCKET, Key=s3_key)
        fsize = meta["ContentLength"]

        # Nếu file đã tải đủ dung lượng từ trước (ví dụ sub-01 -> sub-05) thì bỏ qua ngay
        if os.path.exists(local_path) and os.path.getsize(local_path) == fsize:
            return f"[Đã có] {fname}"

        s3.download_file(BUCKET, s3_key, local_path)
        return f"[Hoàn tất] {fname} ({fsize / (1024*1024):.1f} MB)"
    except Exception as e:
        return f"[Lỗi] {fname}: {e}"


def download_all():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    
    # 1. Tải participants.tsv
    part_path = os.path.join(DATA_DIR, "participants.tsv")
    if not os.path.exists(part_path):
        s3.download_file(BUCKET, f"{PREFIX}participants.tsv", part_path)
        print("[OK] Đã tải participants.tsv")

    # 2. Tạo danh sách toàn bộ 164 files cần tải (41 subjects x 4 files)
    tasks = [
        (f"sub-{sub_id:02d}", suffix)
        for sub_id in SUBJECTS
        for suffix in REQUIRED_SUFFIXES
    ]

    print(f"Bắt đầu tải toàn bộ {len(SUBJECTS)} subjects ({len(tasks)} files) với {MAX_WORKERS} luồng song song...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_file, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Tiến độ tổng"):
            tqdm.write(future.result())


if __name__ == "__main__":
    download_all()
    print(f"\nĐã tải xong toàn bộ bộ dữ liệu tại: {DATA_DIR}")