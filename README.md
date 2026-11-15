# EEG Single-Trial Classification (Oddball Paradigm - P300)

Dự án nghiên cứu và phân loại tín hiệu điện não đồ (EEG) theo từng thử nghiệm đơn lẻ (single-trial classification) trên bài toán **Oddball Paradigm (P300)** thuộc học phần **Tính toán thần kinh**.

---

## 📌 1. Giới thiệu bài toán & Dữ liệu

- **Bài toán**: Nhận diện và phân loại đáp ứng điện thế liên quan sự kiện (Event-Related Potential - ERP / sóng **P300**) đối với kích thích mục tiêu (Target) và kích thích chuẩn (Non-target/Standard).
- **Dataset**: [OpenNeuro ds004621](https://openneuro.org/datasets/ds004621).
  - Tác vụ: `task-oddball`
  - Đối tượng: 41 đối tượng đo hợp lệ (`sub-01` đến `sub-42`, loại trừ `sub-38` do lỗi dữ liệu gốc).
  - Tín hiệu: Dữ liệu EEG đa kênh kèm tọa độ điện cực và nhãn sự kiện.

---

## 📂 2. Cấu trúc dự án

```text
EEG-Single-Trial-Classification/
│
├── data/
│   ├── raw/
│   │   ├── download_exact_oddball.py   # Script tự động đồng bộ dữ liệu S3
│   │   └── oddball_data/               # Thư mục chứa 41 subjects và metadata (ignored by git)
│   └── processed/                      # Dữ liệu sau tiền xử lý (epochs, features, .fif, .npy)
│
├── notebooks/
│   ├── 01_data_exploration/            # Khám phá cấu trúc dữ liệu, kênh đo, tín hiệu thô
│   ├── 02_preprocessing/               # Lọc dải tần (Bandpass), loại bỏ nhiễu/ICA, Epoching
│   ├── 03_erp_analysis/                # Tính trung bình ERP, trực quan hóa sóng P300
│   ├── 04_standard_ml/                 # Phân loại bằng ML truyền thống (CSP, SVM, LDA, RF)
│   └── 05_deep_learning/               # Mô hình học sâu (EEGNet, ShallowFBCSPNet, CNN-LSTM)
│
├── src/
│   ├── preprocessing/                  # Module tiền xử lý tín hiệu
│   ├── models/                         # Kiến trúc mô hình ML & DL
│   ├── training/                       # Pipeline huấn luyện, tối ưu siêu tham số
│   └── evaluation/                     # Đánh giá (ROC-AUC, Confusion Matrix, Cross-Validation)
│
├── results/
│   ├── figures/                        # Biểu đồ trực quan hóa, sóng ERP, phổ công suất
│   ├── metrics/                        # Báo cáo kết quả phân loại (accuracy, f1-score, auc)
│   └── checkpoints/                    # Trọng số mô hình đã huấn luyện (.pt, .pth, .pkl)
│
├── reports/                            # Báo cáo đồ án / thuyết trình
├── requirements.txt                    # Thư viện phụ thuộc của dự án
├── README.md
└── .gitignore
```

---

## 🚀 3. Hướng dẫn cài đặt & Khởi tạo

### 3.1. Thiết lập môi trường

Khuyến nghị sử dụng Python 3.9 - 3.11 trong môi trường ảo:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

*(Các thư viện cốt lõi thường dùng: `mne`, `boto3`, `tqdm`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch` / `braindecode`)*.

---

## 📥 4. Tải dữ liệu thô (Raw Data)

Dữ liệu được tải tự động đa luồng trực tiếp từ S3 Bucket của OpenNeuro qua script:

```powershell
python data/raw/download_exact_oddball.py
```

- Toàn bộ dữ liệu 41 subjects và metadata đi kèm sẽ được tải và lưu tập trung tại thư mục `data/raw/oddball_data/`.
- Script hỗ trợ tiếp tục tải (skip file đã hoàn tất) và kiểm tra toàn vẹn dung lượng file.
- Thư mục dữ liệu `oddball_data/` đã được cấu hình trong `.gitignore`, không lo bị vô tình commit lên Git.

---

## 🔄 5. Quy trình thực hiện (Pipeline)

1. **Khám phá dữ liệu (`notebooks/01_data_exploration/`)**:
   - Kiểm tra thông tin kênh đo, tỷ lệ lấy mẫu, sự kiện (Events/Triggers: Target vs. Standard).
2. **Tiền xử lý (`notebooks/02_preprocessing/` & `src/preprocessing/`)**:
   - Lọc dải tần (Band-pass filter, e.g., 0.1 - 30 Hz), lọc Notching 50Hz.
   - Loại bỏ artifacts bằng ICA hoặc ngưỡng biên độ.
   - Cắt epoch (-200ms đến 800ms quanh kích thích) và hiệu chỉnh baseline.
3. **Phân tích ERP (`notebooks/03_erp_analysis/`)**:
   - Trực quan hóa hình thái sóng P300 (đỉnh dương xuất hiện khoảng 300 - 500ms sau kích thích mục tiêu, rõ nhất tại các điện cực đỉnh/trung tâm như Pz, Cz).
4. **Phân loại Single-Trial**:
   - **Machine Learning (`notebooks/04_standard_ml/`)**: Trích xuất đặc trưng thời gian - tần số, CSP (Common Spatial Pattern) kết hợp SVM / LDA / Random Forest.
   - **Deep Learning (`notebooks/05_deep_learning/`)**: Huấn luyện các kiến trúc chuyên biệt cho EEG như EEGNet, ShallowFBCSPNet.
5. **Đánh giá (`results/` & `reports/`)**:
   - Cross-Validation (Within-subject và Cross-subject).
   - Đánh giá qua các chỉ số: Balanced Accuracy, F1-Score, ROC-AUC.
