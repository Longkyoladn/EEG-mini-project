import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

# Lưu trực tiếp tại thư mục chứa script này
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Toàn bộ 41 subjects hợp lệ của bài test Oddball (trừ sub-38 bị lỗi trong dataset gốc)
SUBJECTS = [i for i in range(1, 43) if i != 38]

BUCKET = "openneuro.org"
PREFIX = "ds004621/"
MAX_WORKERS = 6  # Số luồng tải song song


def get_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def collect_download_Manifest(s3):
    """
    Quét S3 để lập danh sách toàn bộ các file cần tải (Gồm cả Metadata + EEG Oddball).
    Trả về list các tuple: (s3_key, local_path, file_size)
    """
    manifest = []

    # 1. Quét các file metadata ở thư mục gốc (README, participants.tsv, task-oddball_events.json...)
    root_resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/")
    for obj in root_resp.get("Contents", []):
        key = obj["Key"]
        fname = os.path.basename(key)
        # Lấy các file chung của dataset hoặc file cấu hình riêng của task-oddball
        if fname and not fname.startswith(".") and ("task-" not in fname or "task-oddball" in fname):
            local_path = os.path.join(DATA_DIR, fname)
            manifest.append((key, local_path, obj["Size"]))

    # 2. Quét từng Subject để lấy dữ liệu EEG task-oddball + Metadata điện cực/kênh
    for sub_id in SUBJECTS:
        sub_str = f"sub-{sub_id:02d}"
        sub_prefix = f"{PREFIX}{sub_str}/eeg/"
        sub_dir = os.path.join(DATA_DIR, sub_str)

        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=sub_prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            fname = os.path.basename(key)

            # Chỉ lọc lấy:
            # - Toàn bộ file của task-oddball (.eeg, .vhdr, .vmrk, _events.tsv, _channels.tsv, _eeg.json)
            # - Các file tọa độ điện cực dùng chung của subject (_electrodes.tsv, _coordsystem.json)
            if (
                "task-oddball" in fname
                or "electrodes.tsv" in fname
                or "coordsystem.json" in fname
            ):
                local_path = os.path.join(sub_dir, fname)
                manifest.append((key, local_path, obj["Size"]))

    return manifest


def download_worker(item):
    s3_key, local_path, expected_size = item
    fname = os.path.basename(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Kiểm tra nếu file đã tồn tại và khớp chính xác dung lượng byte trên S3 thì bỏ qua
    if os.path.exists(local_path) and os.path.getsize(local_path) == expected_size:
        return ("SKIPPED", fname, expected_size)

    s3 = get_s3_client()
    # Tải vào file tạm .part trước, hoàn tất mới đổi tên để tránh file lỗi khi đứt mạng/dừng ngang
    temp_path = local_path + ".part"
    try:
        s3.download_file(BUCKET, s3_key, temp_path)
        os.replace(temp_path, local_path)
        return ("DOWNLOADED", fname, expected_size)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return ("ERROR", f"{fname}: {e}", 0)


def main():
    print(">>> Đang kết nối S3 và lập danh sách toàn bộ file (EEG + Metadata)...")
    s3 = get_s3_client()
    manifest = collect_download_Manifest(s3)

    total_files = len(manifest)
    total_gb = sum(item[2] for item in manifest) / (1024**3)
    print(f">>> Tìm thấy {total_files} files (Tổng dung lượng: {total_gb:.2f} GB).")

    stats = {"DOWNLOADED": 0, "SKIPPED": 0, "ERROR": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_worker, item): item for item in manifest}
        with tqdm(total=total_files, desc="Tiến độ đồng bộ", unit="file") as pbar:
            for future in as_completed(futures):
                status, msg, _ = future.result()
                stats[status] += 1
                if status == "ERROR":
                    tqdm.write(f"[LỖI] {msg}")
                pbar.update(1)

    print("\n" + "=" * 55)
    print(f"HOÀN TẤT ĐỒNG BỘ DỮ LIỆU TẠI: {DATA_DIR}")
    print(f" - Tải mới thành công : {stats['DOWNLOADED']} files")
    print(f" - Đã có sẵn (Bỏ qua) : {stats['SKIPPED']} files")
    print(f" - Lỗi                : {stats['ERROR']} files")
    print("=" * 55)


if __name__ == "__main__":
    main()