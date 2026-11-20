# BƯỚC 5 — BÁO CÁO KẾT QUẢ FULL TRAINING (39 SUBJECTS)

- **Tập dữ liệu**: `data/processed` (39 subjects sạch, 20,886 epochs)
- **Phần cứng thực thi**: `NVIDIA GeForce RTX 5060 Ti`
- **Mô hình chủ lực DL2**: `P300ATCNet(readout='attn_pool', n_windows=2, shared_branch=False, tcn_filters=32, temp_kernel_1=125, spatial_max_norm=2.0, dropout=0.3)`
- **Hai bản vá kỹ thuật áp dụng trước Full Run**:
  1. Kẹp ngưỡng quyết định `find_optimal_threshold()` gọn trong `[0.05, 0.95]` (ưu tiên gần `0.5` khi hòa).
  2. Tối ưu GPU-Resident (`ChannelStandardScaler` `float64` trên GPU + mini-batch GPU `torch.randperm` + `spatial_conv` cuBLAS batched GEMM trong `trainer_dl2.py`).

### Bảng 1 — Tốc độ huấn luyện sau tối ưu GPU-Resident trên RTX 5060 Ti

| Thực nghiệm | Số folds | Tổng epochs | Giây/epoch | Giây/fold | Tổng thời gian |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `dl2_ws_p300_atcnet` (Within, 39 subs) | 195 | 6,351 | 0.167s | 5.44s | 1061.8s (17.70m) |
| `dl2_loso_p300_atcnet` (LOSO, 39 subs) | 39 | 1,325 | 4.091s | 138.99s | 5420.7s (90.35m) |
| `dl2_cs5_p300_atcnet_clean` (CS5 clean) | 5 | 209 | 3.106s | 129.82s | 649.1s (10.82m) |
| `dl2_cs5_p300_atcnet_no_ica` (CS5 no_ica) | 5 | 190 | 3.365s | 127.89s | 639.4s (10.66m) |

> Tổng thời gian huấn luyện thực tế của các thực nghiệm Bước 5: **`7771.0s` (`129.52 phút`)**.

### Bảng 2 — Hiệu năng Full 39 Subjects & So sánh đối chứng (`Mean ± SD`)

| Giao thức | Mô hình / Biến thể | Folds / Unit | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Sensitivity | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Within-Subject | P300-ATCNet (`attn_pool`) | 195 / 39 subs | 0.9622 ± 0.0446 | 0.8804 ± 0.1192 | 0.9036 ± 0.0621 | 0.7884 ± 0.1285 | 0.8671 ± 0.0854 | 0.9402 ± 0.0501 |
| Cross-Subject LOSO | P300-ATCNet (`attn_pool`) | 39 / 39 subs | 0.9778 ± 0.0295 | 0.9167 ± 0.0887 | 0.9147 ± 0.0633 | 0.7692 ± 0.1162 | 0.9066 ± 0.1268 | 0.9229 ± 0.0631 |
| Cross-Subject 5-Fold | P300-ATCNet (`clean`) | 5 / fold | 0.9680 ± 0.0165 | 0.8834 ± 0.0559 | 0.9121 ± 0.0256 | 0.7780 ± 0.0710 | 0.8878 ± 0.0339 | 0.9363 ± 0.0268 |
| Cross-Subject 5-Fold | P300-ATCNet (`no_ica`) | 5 / fold | 0.9577 ± 0.0199 | 0.8615 ± 0.0608 | 0.8975 ± 0.0286 | 0.7248 ± 0.0878 | 0.8849 ± 0.0242 | 0.9102 ± 0.0465 |

### Bảng 3 — Sensitivity Analysis (39 subjects vs. 37 subjects loại `drop > 30%` vs. 31 subjects loại `Hit Rate < 90%`)

- **37 subjects**: loại `sub-07, sub-10` (`drop_rate_pct > 30%`).
- **31 subjects**: loại `sub-01, sub-07, sub-08, sub-09, sub-10, sub-25, sub-29, sub-30` (`Hit Rate < 90%`).

| Giao thức & Mô hình | Cohort | N | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Sensitivity | Specificity |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Within-Subject (`P300-ATCNet`) | Full (chuẩn) | 39 | 0.9622 ± 0.0446 | 0.8804 ± 0.1192 | 0.9036 ± 0.0621 | 0.7884 ± 0.1285 | 0.8671 ± 0.0854 | 0.9402 ± 0.0501 |
| Within-Subject (`P300-ATCNet`) | Loại drop > 30% | 37 | 0.9672 ± 0.0365 | 0.8934 ± 0.1001 | 0.9101 ± 0.0557 | 0.8005 ± 0.1175 | 0.8750 ± 0.0801 | 0.9452 ± 0.0424 |
| Within-Subject (`P300-ATCNet`) | Loại Hit < 90% | 31 | 0.9674 ± 0.0376 | 0.8960 ± 0.1015 | 0.9101 ± 0.0557 | 0.8036 ± 0.1161 | 0.8752 ± 0.0797 | 0.9451 ± 0.0437 |
| Within-Subject (`P300-ATCNet`) | **Δ (37 − 39)** | — | +0.0050 | +0.0129 | +0.0064 | +0.0121 | +0.0079 | +0.0050 |
| Within-Subject (`P300-ATCNet`) | **Δ (31 − 39)** | — | +0.0053 | +0.0156 | +0.0065 | +0.0152 | +0.0081 | +0.0049 |
| Cross-Subject LOSO (`P300-ATCNet`) | Full (chuẩn) | 39 | 0.9778 ± 0.0295 | 0.9167 ± 0.0887 | 0.9147 ± 0.0633 | 0.7692 ± 0.1162 | 0.9066 ± 0.1268 | 0.9229 ± 0.0631 |
| Cross-Subject LOSO (`P300-ATCNet`) | Loại drop > 30% | 37 | 0.9802 ± 0.0247 | 0.9238 ± 0.0740 | 0.9182 ± 0.0591 | 0.7763 ± 0.1053 | 0.9091 ± 0.1284 | 0.9272 ± 0.0564 |
| Cross-Subject LOSO (`P300-ATCNet`) | Loại Hit < 90% | 31 | 0.9809 ± 0.0221 | 0.9249 ± 0.0676 | 0.9161 ± 0.0602 | 0.7748 ± 0.1008 | 0.9064 ± 0.1350 | 0.9257 ± 0.0579 |
| Cross-Subject LOSO (`P300-ATCNet`) | **Δ (37 − 39)** | — | +0.0024 | +0.0071 | +0.0034 | +0.0071 | +0.0025 | +0.0044 |
| Cross-Subject LOSO (`P300-ATCNet`) | **Δ (31 − 39)** | — | +0.0031 | +0.0082 | +0.0013 | +0.0055 | -0.0002 | +0.0029 |

### Bảng 4 — Top 5 Subjects Hiệu năng Cao nhất và Thấp nhất (`P300-ATCNet attn_pool`)

| Giao thức | Nhóm xếp hạng | Subject | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | Clean Epochs (Target) | Drop Rate (%) | Hit Rate (%) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| Within-Subject | Top 5 Cao nhất #1 | `sub-22` | 0.9965 | 0.9851 | 0.9472 | 0.9029 | 579 (81) | 0.00% | 100.00% |
| Within-Subject | Top 5 Cao nhất #2 | `sub-42` | 0.9965 | 0.9734 | 0.9702 | 0.9180 | 578 (80) | 0.00% | 98.77% |
| Within-Subject | Top 5 Cao nhất #3 | `sub-31` | 0.9949 | 0.9745 | 0.9820 | 0.9584 | 571 (78) | 0.87% | 97.53% |
| Within-Subject | Top 5 Cao nhất #4 | `sub-08` | 0.9940 | 0.9662 | 0.9607 | 0.8733 | 545 (68) | 3.54% | 85.19% |
| Within-Subject | Top 5 Cao nhất #5 | `sub-27` | 0.9925 | 0.9674 | 0.9287 | 0.8812 | 573 (81) | 1.04% | 100.00% |
| Within-Subject | Bottom 5 Thấp nhất #1 | `sub-07` | 0.8000 | 0.4697 | 0.7346 | 0.4531 | 273 (36) | 49.63% | 60.49% |
| Within-Subject | Bottom 5 Thấp nhất #2 | `sub-41` | 0.8458 | 0.5809 | 0.7538 | 0.5167 | 572 (78) | 0.87% | 97.53% |
| Within-Subject | Bottom 5 Thấp nhất #3 | `sub-32` | 0.8532 | 0.5763 | 0.7632 | 0.4907 | 575 (79) | 0.35% | 97.53% |
| Within-Subject | Bottom 5 Thấp nhất #4 | `sub-01` | 0.9019 | 0.6898 | 0.7940 | 0.5346 | 539 (61) | 1.28% | 76.54% |
| Within-Subject | Bottom 5 Thấp nhất #5 | `sub-06` | 0.9057 | 0.7919 | 0.8460 | 0.6136 | 541 (73) | 5.91% | 96.30% |
| Cross-Subject LOSO | Top 5 Cao nhất #1 | `sub-31` | 0.9993 | 0.9961 | 0.9784 | 0.9059 | 571 (78) | 0.87% | 97.53% |
| Cross-Subject LOSO | Top 5 Cao nhất #2 | `sub-33` | 0.9991 | 0.9949 | 0.9816 | 0.9554 | 542 (77) | 6.23% | 100.00% |
| Cross-Subject LOSO | Top 5 Cao nhất #3 | `sub-30` | 0.9982 | 0.9874 | 0.9711 | 0.8250 | 551 (66) | 2.65% | 83.95% |
| Cross-Subject LOSO | Top 5 Cao nhất #4 | `sub-27` | 0.9981 | 0.9876 | 0.9085 | 0.6429 | 573 (81) | 1.04% | 100.00% |
| Cross-Subject LOSO | Top 5 Cao nhất #5 | `sub-08` | 0.9976 | 0.9860 | 0.9611 | 0.9474 | 545 (68) | 3.54% | 85.19% |
| Cross-Subject LOSO | Bottom 5 Thấp nhất #1 | `sub-07` | 0.8744 | 0.6047 | 0.7560 | 0.4409 | 273 (36) | 49.63% | 60.49% |
| Cross-Subject LOSO | Bottom 5 Thấp nhất #2 | `sub-01` | 0.8998 | 0.7023 | 0.8218 | 0.5960 | 539 (61) | 1.28% | 76.54% |
| Cross-Subject LOSO | Bottom 5 Thấp nhất #3 | `sub-41` | 0.9155 | 0.7782 | 0.8576 | 0.6598 | 572 (78) | 0.87% | 97.53% |
| Cross-Subject LOSO | Bottom 5 Thấp nhất #4 | `sub-32` | 0.9205 | 0.7555 | 0.7504 | 0.6357 | 575 (79) | 0.35% | 97.53% |
| Cross-Subject LOSO | Bottom 5 Thấp nhất #5 | `sub-21` | 0.9310 | 0.7551 | 0.7615 | 0.6462 | 568 (77) | 1.56% | 97.53% |
