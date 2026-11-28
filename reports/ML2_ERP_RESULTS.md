# BÁO CÁO NHÁNH ML2 — TRÍCH XUẤT ĐẶC TRƯNG ERP & PHÂN LOẠI SINGLE-TRIAL

- **Nguồn dữ liệu chuẩn (Single Source of Truth)**: `data/processed/X_clean.npy` (`39` subjects sạch, `20,886` epochs: `18,036` Frequent / `2,850` Target, tỷ lệ lệch lớp `6.33 : 1`, `250 Hz`, `[-0.2, 0.8] s`, `127` kênh).
- **Mã băm định danh mẫu (`trial_uid_sha256`)**: `7c5ce364eec45fc9c7d2ac3a2fc5d5723fd961b5953d9653cf403acda19eb524`.
- **Splits cố định (`seed = 42`)**: `splits/within_5fold.json` (`195` folds) và `splits/loso.json` (`39` folds).
- **Nguyên tắc chống rò rỉ (Zero Data Leakage)**: Toàn bộ phép trích xuất đặc trưng ERP tính độc lập trên từng lượt thử (`µV`, `float64`); `StandardScaler` chỉ `.fit()` trên `train_idx`; siêu tham số, Platt scaling (`LogisticRegression` 1D trên `decision_function` với SVM) và ngưỡng quyết định (`find_optimal_threshold` trong `[0.05, 0.95]`) chỉ học trên `val_idx`; tập `test_idx` chỉ chấm điểm đúng 1 lần.
- **Xử lý mất cân bằng lớp (`6.33 : 1`)**: `SVM` (`LinearSVC` / `SVC(kernel='rbf')`) bật `class_weight='balanced'` kết hợp Platt scaling và tối ưu ngưỡng `balanced_accuracy` trên `val_idx`; `KNN` (`weights='distance'`) và `GaussianNB` (không hỗ trợ tham số `class_weight` trong `scikit-learn`) bù lệch lớp thông qua tối ưu ngưỡng `balanced_accuracy` trên `val_idx`.
- **Môi trường thực thi**: Huấn luyện song song trên **CPU `16` luồng** (`joblib.Parallel(n_jobs=-1)`); suy luận single-trial (`batch_size = 1`) đo đơn luồng trên CPU.

---

## 1. Kiến trúc Đặc trưng Miền Thời gian ERP (`src/features/erp_time_features.py`)

Dùng `14` kênh đường giữa & đỉnh–trung tâm (`core_14`: `Fz, FC1, FC2, Cz, CPz, Pz, POz, Oz, CP1, CP2, P1, P2, P3, P4`, tránh cụm vận động `C3/C4`), chuyển từ Volts sang `µV` (`float64`), với 3 mức đặc trưng:

1. **`erp_peaks_only` (`126` chiều)**: Biên độ đỉnh và độ trễ (trên tín hiệu làm mượt cục bộ 5 mẫu = `20 ms` trong từng trial) tại `N100 [80–150 ms]`, `P200 [150–250 ms]`, `N200 [200–300 ms]`, `P300 [300–600 ms]` + hiệu đỉnh-đáy `P300 − N200`.
2. **`erp_windows_and_peaks` (`224` chiều)**: Bổ sung trung bình biên độ trên `6` cửa sổ sinh lý (`N100`, `P200`, `N200`, `P300_early [300–450 ms]`, `P300_late [450–600 ms]`, `Late_SW [600–800 ms]`) + tích phân có dấu `auc_P300_uv_s` (`[300–600 ms]`).
3. **`erp_full_binned` (`352` chiều — Bộ chính cho ML2)**: Bổ sung `16` lát thời gian đều `50 ms` (`[0, 50) → [750, 800] ms`) trên `8` kênh lõi đường giữa (`Fz, Cz, CPz, Pz, POz, Oz, CP1, CP2`).

### Bảng 1 — Chẩn đoán Sức mạnh Phân biệt Đơn biến & Latency Jitter trên `20,886` Epochs

- **Top 5 đặc trưng đơn biến mạnh nhất (`erp_windows_and_peaks`)**:
  1. `P1__P300_late_mean_uv`: `Cohen's d = +1.2601`, `AUC = 0.8037` (`Target = +6.99 ± 5.80` vs `Freq = +0.71 ± 4.85 µV`)
  2. `Pz__P300_late_mean_uv`: `Cohen's d = +1.2505`, `AUC = 0.8041` (`Target = +7.41 ± 6.14` vs `Freq = +0.72 ± 5.22 µV`)
  3. `CPz__P300_late_mean_uv`: `Cohen's d = +1.1994`, `AUC = 0.7967`
  4. `CP1__P300_late_mean_uv`: `Cohen's d = +1.1807`, `AUC = 0.7859`
  5. `CP2__P300_late_mean_uv`: `Cohen's d = +1.1070`, `AUC = 0.7786`

| Đặc trưng tại điện cực `Pz` | Cohen's `d` | Single-Feature ROC-AUC | Target (`Mean ± SD`) | Frequent (`Mean ± SD`) |
| :--- | ---: | ---: | :---: | :---: |
| `Pz__P300_peak_lat_ms` *(độ trễ đỉnh)* | `+0.3580` | `0.6080` | `463.8 ± 86.9 ms` | `429.2 ± 98.0 ms` |
| `Pz__P300_early_mean_uv` (`300–450 ms`) | `+0.6530` | `0.6745` | `+5.63 ± 6.03 µV` | `+2.07 ± 5.36 µV` |
| `Pz__P300_peak_amp_uv` *(bắt đỉnh thủ công)* | `+0.6637` | `0.6918` | `+14.16 ± 6.81 µV` | `+9.86 ± 6.42 µV` |
| `Pz__auc_P300_uv_s` *(tích phân `300–600 ms`)* | `+1.0292` | `0.7630` | `+1.96 ± 1.69 µV·s` | `+0.42 ± 1.47 µV·s` |
| **`Pz__P300_late_mean_uv` (`450–600 ms`)** | **`+1.2505`** | **`0.8041`** | **`+7.41 ± 6.14 µV`** | **`+0.72 ± 5.22 µV`** |

- **Bằng chứng Latency Jitter & Nhiễu Single-Trial**:
  - Trên Grand Average (`Target − Frequent`), đỉnh P300 tại `Pz` rất sắc nét ở `+6.91 µV @ 488.0 ms`.
  - Tuy nhiên ở từng trial riêng lẻ, độ trễ đỉnh P300 Target phân tán rất rộng (`463.8 ± 86.9 ms`, `median = 464.0 ms`, `IQR = [396.0, 536.0] ms`) và tương quan thuận có ý nghĩa thống kê với Reaction Time hành vi (`540.6 ± 119.9 ms`, `N = 2,850`: `Spearman r = +0.2363`, `p = 1.82e-37`; `Pearson r = +0.1783`, `p = 8.55e-22`).
  - Do nhiễu nền single-trial thường tạo cực đại giả (`Frequent` cũng có đỉnh giả trung bình `+9.86 ± 6.42 µV`), phép bắt đỉnh đơn điểm (`Pz__P300_peak_amp_uv`) chỉ đạt `d = +0.6637` (`AUC = 0.6918`), kém gần gấp đôi so với lấy trung bình cửa sổ `P300_late` (`d = +1.2505`, `AUC = 0.8041`).

---

## 2. Kết quả Thực nghiệm Phân loại Single-Trial (`Mean ± SD` qua 39 Subjects)

### Bảng 2 — So sánh 3 Họ Mô hình ML2 trên `erp_full_binned` (`352` chiều, chạy song song 16 luồng CPU)

| Giao thức | Mô hình | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Sensitivity | Specificity | Train Time (16T CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| Within (5-Fold, 195 folds) | **SVM (`ml2_erp_svm`)** | **0.9466 ± 0.0608** | **0.8353 ± 0.1389** | **0.8724 ± 0.0693** | **0.7186 ± 0.1385** | **0.8315 ± 0.0933** | **0.9134 ± 0.0568** | 12.72 s |
| Within (5-Fold, 195 folds) | KNN (`ml2_erp_knn`) | 0.8919 ± 0.0799 | 0.7062 ± 0.1620 | 0.8078 ± 0.0840 | 0.5882 ± 0.1466 | 0.7651 ± 0.1146 | 0.8506 ± 0.0824 | 2.00 s |
| Within (5-Fold, 195 folds) | GaussianNB (`ml2_erp_gnb`) | 0.8693 ± 0.0760 | 0.5353 ± 0.1138 | 0.8029 ± 0.0767 | 0.5678 ± 0.1164 | 0.7527 ± 0.1232 | 0.8532 ± 0.0422 | 1.20 s |
| LOSO (39-Fold, 39 subs) | **SVM (`ml2_erp_svm`)** | **0.9330 ± 0.0551** | **0.7654 ± 0.1493** | **0.8515 ± 0.0772** | **0.6277 ± 0.1204** | **0.8410 ± 0.1608** | **0.8620 ± 0.0773** | 3235.58 s |
| LOSO (39-Fold, 39 subs) | KNN (`ml2_erp_knn`) | 0.8628 ± 0.0889 | 0.6161 ± 0.1742 | 0.7803 ± 0.0846 | 0.5098 ± 0.1228 | 0.7566 ± 0.1946 | 0.8040 ± 0.0984 | 229.19 s |
| LOSO (39-Fold, 39 subs) | GaussianNB (`ml2_erp_gnb`) | 0.8215 ± 0.0871 | 0.4301 ± 0.1500 | 0.7367 ± 0.0913 | 0.4597 ± 0.1368 | 0.6740 ± 0.2317 | 0.7993 ± 0.1239 | 7.94 s |

- **Phân bố kernel được chọn trên Validation (`ml2_erp_svm`)**: Within: `120` LinearSVC / `75` RBF-SVC (`195` folds); LOSO: `17` LinearSVC / `22` RBF-SVC (`39` folds).

### Bảng 3 — Ablation 3 Bộ Đặc trưng ERP trên SVM (Định lượng nhược điểm bắt đỉnh thủ công)

| Giao thức | Bộ đặc trưng | Dims | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | ΔROC vs `peaks_only` |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | ---: |
| Within-Subject | `erp_peaks_only` | 126 | 0.8605 ± 0.1045 | 0.6273 ± 0.1932 | 0.7785 ± 0.0976 | 0.5351 ± 0.1497 | — |
| Within-Subject | `erp_windows_and_peaks` | 224 | 0.9368 ± 0.0604 | 0.8041 ± 0.1424 | 0.8622 ± 0.0675 | 0.6880 ± 0.1325 | **+0.0763** |
| Within-Subject | **`erp_full_binned`** | **352** | **0.9466 ± 0.0608** | **0.8353 ± 0.1389** | **0.8724 ± 0.0693** | **0.7186 ± 0.1385** | **+0.0861** |
| Cross-Subject LOSO | `erp_peaks_only` | 126 | 0.8409 ± 0.0998 | 0.5529 ± 0.1859 | 0.7549 ± 0.0885 | 0.4676 ± 0.1241 | — |
| Cross-Subject LOSO | `erp_windows_and_peaks` | 224 | 0.9286 ± 0.0569 | 0.7500 ± 0.1579 | 0.8458 ± 0.0776 | 0.6128 ± 0.1141 | **+0.0877** |
| Cross-Subject LOSO | **`erp_full_binned`** | **352** | **0.9330 ± 0.0551** | **0.7654 ± 0.1493** | **0.8515 ± 0.0772** | **0.6277 ± 0.1204** | **+0.0921** |

### Bảng 4 — Sensitivity Analysis (`ml2_erp_svm`: 39 vs 37 vs 31 Subjects)

| Giao thức | Cohort | N | ROC-AUC | PR-AUC | Balanced Acc | F1-Target |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: |
| Within-Subject | Full (chuẩn) | 39 | 0.9466 ± 0.0608 | 0.8353 ± 0.1389 | 0.8724 ± 0.0693 | 0.7186 ± 0.1385 |
| Within-Subject | Loại `drop > 30%` (`sub-07, 10`) | 37 | 0.9546 ± 0.0420 | 0.8516 ± 0.1080 | 0.8810 ± 0.0532 | 0.7329 ± 0.1182 |
| Within-Subject | Loại `Hit < 90%` | 31 | 0.9597 ± 0.0359 | 0.8597 ± 0.1061 | 0.8882 ± 0.0500 | 0.7469 ± 0.1076 |
| Cross-Subject LOSO | Full (chuẩn) | 39 | 0.9330 ± 0.0551 | 0.7654 ± 0.1493 | 0.8515 ± 0.0772 | 0.6277 ± 0.1204 |
| Cross-Subject LOSO | Loại `drop > 30%` (`sub-07, 10`) | 37 | 0.9361 ± 0.0502 | 0.7725 ± 0.1375 | 0.8549 ± 0.0740 | 0.6320 ± 0.1165 |
| Cross-Subject LOSO | Loại `Hit < 90%` | 31 | 0.9358 ± 0.0499 | 0.7746 ± 0.1306 | 0.8538 ± 0.0762 | 0.6443 ± 0.1172 |

### Bảng 5 — Top 5 & Bottom 5 Subjects (`ml2_erp_svm`)

| Giao thức | Nhóm | Top/Bottom Subjects (`ROC-AUC`) |
| :--- | :--- | :--- |
| Within-Subject | Top 5 | `sub-12` (0.9953), `sub-37` (0.9940), `sub-31` (0.9936), `sub-28` (0.9925), `sub-35` (0.9914) |
| Within-Subject | Bottom 5 | `sub-07` (0.6749), `sub-01` (0.8159), `sub-04` (0.8610), `sub-41` (0.8759), `sub-32` (0.8869) |
| Cross-Subject LOSO | Top 5 | `sub-39` (0.9938), `sub-31` (0.9911), `sub-08` (0.9849), `sub-35` (0.9817), `sub-12` (0.9791) |
| Cross-Subject LOSO | Bottom 5 | `sub-32` (0.7787), `sub-07` (0.7822), `sub-11` (0.8139), `sub-01` (0.8304), `sub-41` (0.8446) |

---

## 3. Chi phí Tính toán Single-Trial (`batch_size = 1`, CPU) & So sánh Nhanh với DL2

Trích xuất đặc trưng `ERPFeatureExtractor` (`erp_full_binned`, `352` chiều): `0.473 ± 0.121 ms/epoch` (`median = 0.426 ms`).

| Mô hình | Fit fold 0 (`N_tr`) | E2E Latency (`Mean ± SD ms/ep`) | Median (ms) | p95 (ms) | Throughput (ep/s) |
| :--- | :--- | :---: | ---: | ---: | ---: |
| `ml2_erp_svm` (Linear `C=0.01`) | `within` (`344`) | **0.707 ± 0.081** | 0.668 | 0.882 | **1,415.4** |
| `ml2_erp_svm` (RBF `C=1.0, scale`) | `within` (`344`) | 0.919 ± 0.151 | 0.859 | 1.127 | 1,087.7 |
| `ml2_erp_knn` (`k=35, manhattan`) | `within` (`344`) | 4.800 ± 0.260 | 4.763 | 5.242 | 208.3 |
| `ml2_erp_gnb` (`var_smooth=0.1`) | `within` (`344`) | 0.892 ± 0.153 | 0.867 | 1.123 | 1,121.4 |
| `ml2_erp_svm` (Linear `C=0.01`) | `loso` (`18,338`) | **0.834 ± 0.089** | 0.864 | 0.919 | **1,198.3** |
| `ml2_erp_svm` (RBF `C=1.0, scale`) | `loso` (`18,338`) | 2.636 ± 0.318 | 2.540 | 3.284 | 379.3 |
| `ml2_erp_knn` (`k=35, manhattan`) | `loso` (`18,338`) | 2.573 ± 1.018 | 2.495 | 2.951 | 388.7 |
| `ml2_erp_gnb` (`var_smooth=0.1`) | `loso` (`18,338`) | 0.940 ± 0.225 | 0.870 | 1.404 | 1,064.0 |
| **Tham chiếu DL2 (`P300ATCNet`, CPU)** | — | `2.775 ± 0.583` | `2.993` | `3.443` | `360.3` |

### Nhận xét phục vụ RQ2 (ML2 vs DL2 — kết hợp `fig2_single_trial_and_rt`)
1. **Sample-Efficiency (Within-Subject ít dữ liệu):** Với chỉ `~344` mẫu train/fold, `ml2_erp_svm` đạt **`0.9466`** ROC-AUC (bám sát `P300-ATCNet` `0.9622`, chỉ kém `-0.0156`), cho thấy đặc trưng ERP thiết kế theo tri thức miền rất hiệu quả khi dữ liệu từng người ít và chỉ tốn `12.72 s` wall-clock trên CPU 16 luồng (`n_jobs=-1`) cho toàn bộ `195` folds.
2. **Model Capacity (Cross-Subject LOSO giàu dữ liệu):** Khi chuyển sang `LOSO` (`18,338` mẫu train/fold), `ml2_erp_svm` giảm nhẹ từ `0.9466 → 0.9330` (do khác biệt hình thái & độ trễ P300 giữa 39 người mà bộ đặc trưng tĩnh không tự căn chỉnh pha được), trong khi `P300-ATCNet` tăng lên `0.9778` (`+0.0448` ROC-AUC, `+0.0632` Balanced Acc).
3. **Lý do bắt đỉnh ERP thủ công yếu ở single-trial**: Kết hợp với phân bố RT ở `fig2_single_trial_and_rt` (`540.6 ± 119.9 ms`) và `erp_feature_diagnostics.json`, độ trễ P300 biến thiên mạnh theo từng lượt thử và nhiễu cao tần khiến bắt đỉnh (`erp_peaks_only`) tụt `-0.0861` đến `-0.0921` ROC-AUC so với tích hợp lát thời gian (`erp_full_binned`).

---

## 4. Danh mục Sản phẩm Bàn giao của Hào

- **Code & Scripts**:
  - `tests/smoke_test_data.py` — Lớp 2 Data Smoke Test (kiểm tra `20,886` clean + `22,477` no_ica, `sub-08`, `trial_uid_sha256`, 3 file `splits/*.json`).
  - `src/features/erp_time_features.py` — `ERPFeatureExtractor` (`erp_peaks_only`, `erp_windows_and_peaks`, `erp_full_binned`).
  - `src/ml2/ml2_erp.py`, `scripts/extract_erp_features.py`, `scripts/train_ml2.py`, `scripts/benchmark_latency_ml2.py`, `scripts/export_ml2_handover.py`.
- **Unit Tests**: `tests/test_erp_features.py`, `tests/test_ml2_erp.py`, `tests/test_ml2_handover.py`.
- **4 File Bàn giao Chuẩn Mục 3.5 (`results/handover_standard/`)**:
  - `ml2_erp_svm_within_per_fold.csv` (`195` dòng) & `ml2_erp_svm_within_preds.npz` (`20,886` mẫu, `allow_pickle=False`)
  - `ml2_erp_svm_loso_per_fold.csv` (`39` dòng) & `ml2_erp_svm_loso_preds.npz` (`20,886` mẫu, `allow_pickle=False`)
- **Log chi tiết cục bộ (`results/metrics/ml2/`, ignored bởi `.gitignore`)**: `erp_feature_diagnostics.json`, `erp_feature_stats.csv`, `ml2_comparison_summary.{csv,json}`, `ml2_ablation_summary.{csv,json}`, `inference_latency_ml2.json`, cùng 10 cặp `metrics_ml2_*.json` & `predictions_ml2_*.csv`.
