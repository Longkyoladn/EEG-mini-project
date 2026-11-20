# BƯỚC 6 — BÁO CÁO EXPLAINABLE AI (RQ3 — 4 LỚP KIỂM CHỨNG) & PHÂN TÍCH THẦN KINH HỌC

- **Mô hình giải thích**: `P300ATCNet(readout='attn_pool', n_windows=2, shared_branch=False)` trên toàn bộ **39 folds LOSO** (`dl2_loso_p300_atcnet`, **Zero Leakage**).
- **Phương pháp XAI**: Captum `IntegratedGradients(model)` (`target=0`, `n_steps=64`, `internal_batch_size=32`, baseline = trung bình lớp Frequent tập train đã scale, đối chiếu với `baseline = 0` và baseline Frequent tập test).
- **Định lý Completeness**: trên toàn bộ `2,850` trials Target, `max |delta| = 9.8433e-02`, `mean |delta| = 2.2804e-03`.
- **Montage không gian**: `colin27_1005` (`127/127` kênh khớp tọa độ 3D từ `channel_names.json`, `on_missing='raise'`).

---

## Lớp 1 — Chiều Thời gian & Sanity Check (Baseline vs P300)

- **Đỉnh Temporal Attribution toàn tập (`|IG|`)**: **`484.0 ms`** (per-subject peak: `489.2 ± 47.8 ms`, median `500.0 ms`).
- **Tỷ số cường độ P300 `[300, 600] ms` / Baseline `[-200, 0) ms`**: **`11.13×`** (`± 2.83`, median `10.67×`, Wilcoxon `p = 3.64e-12`).
- **Đối chiếu Baseline (`Frequent mean` vs `0`)**: Temporal `r = 0.9990` (`ρ = 0.9991`); Topomap `300–500 ms`: `r = 0.9926` (`ρ = 0.9906`).
- **Đối chiếu Baseline (`Train Frequent mean` vs `Test Frequent mean`)**: Temporal `r = 0.9994`; Topomap `300–500 ms`: `r = 0.9990`.

### Bảng 1 — Phân bố Năng lượng & Cường độ Attribution qua 4 Cửa sổ Sinh lý (39 subjects)

| Cửa sổ thời gian | Số mẫu | Tỷ lệ thời lượng (%) | Tỷ trọng năng lượng `|IG|` (%) (Mean ± SD) | 95% CI (%) | Cường độ `mean |IG|` (Mean ± SD) |
| :--- | ---: | ---: | :---: | :---: | :---: |
| [-200, 0) ms | 50 | 19.9% | 3.17 ± 0.68% | ±0.22% | 0.00097 ± 0.00029 |
| [0, 300) ms | 75 | 29.9% | 21.58 ± 2.99% | ±0.97% | 0.00442 ± 0.00109 |
| [300, 600] ms | 76 | 30.3% | 50.95 ± 2.42% | ±0.78% | 0.01035 ± 0.00240 |
| (600, 800] ms | 50 | 19.9% | 24.30 ± 3.28% | ±1.06% | 0.00744 ± 0.00171 |

- **Top 7 kênh có trọng số Depthwise Spatial Conv (`|W_spatial|`) cao nhất qua 39 folds LOSO**: `PPO10h` (0.1121), `C1` (0.1048), `AFz` (0.1046), `FCC3h` (0.1004), `PPO2h` (0.0974), `TPP10h` (0.0955), `P2` (0.0940).

---

## Lớp 2 — Kiểm chứng Nhiễu Chớp mắt EOG (`clean` vs `no_ica`, CS 5-Fold)

- **Cụm `EOG_ROI` (8/127 kênh = `6.30%` nếu phân bố đều)**: `Fp1, Fp2, AF7, AF3, AF8, AF4, AFp1, AFp2`.
- **Hiệu năng phân loại CS5**: `clean` ROC-AUC = **`0.9680 ± 0.0165`** (F1 = `0.7780`) vs `no_ica` ROC-AUC = **`0.9577 ± 0.0199`** (F1 = `0.7248`).

### Bảng 2 — Tỷ trọng Attribution rơi vào cụm trán `EOG_ROI` Trước vs. Sau ICA (39 subjects)

| Chỉ số `EOG_ROI` | `clean` (N=2,850 Target) | `no_ica` khớp 1-1 (N=2,850) | `no_ica` toàn tập (N=3,002) | Wilcoxon p (khớp 1-1) |
| :--- | :---: | :---: | :---: | ---: |
| Tỷ trọng toàn epoch (%) | 6.41 ± 1.45% | 4.98 ± 0.68% | 4.96 ± 0.68% | 2.78e-06 |
| Tỷ trọng cửa sổ `300–600 ms` (%) | 6.63 ± 1.70% | 5.82 ± 0.89% | 5.79 ± 0.89% | 0.0102 |
| Mật độ TB mỗi kênh `EOG_ROI` | 0.00704 ± 0.00184 | 0.00487 ± 0.00120 | 0.00487 ± 0.00122 | 1.36e-08 |
| Tỷ số mật độ `EOG / Non-EOG` | 1.022 ± 0.249 | 0.780 ± 0.111 | — | 2.78e-06 |

#### Kiểm tra riêng 2 ca nội suy đồng thời `Fp1/Fp2` (`sub-18`, `sub-40`)

| Nhóm subject | `clean` EOG share (%) | `no_ica` khớp 1-1 EOG share (%) | `no_ica` toàn tập (%) | Chênh lệch (clean − no_ica) |
| :--- | ---: | ---: | ---: | ---: |
| `sub-18` (nội suy Fp1/Fp2) | 7.60% | 4.64% | 4.80% | +2.96% |
| `sub-40` (nội suy Fp1/Fp2) | 4.91% | 4.80% | 4.89% | +0.11% |
| 37 subjects còn lại (Mean ± SD) | 6.41 ± 1.45% | 4.99 ± 0.69% | — | +1.42% (p=5.03e-06) |

---

## Lớp 3 — Bóc tách Nhiễu Vận động Bấm tay (Motor Confound — Trọng tâm RQ3)

### Bảng 3A — Tỷ số ROI Không gian & Chỉ số Lệch Bán cầu Vận động (39 subjects LOSO)

| Cửa sổ | `P300_ROI` (7 chs) | `LEFT_MOTOR_ROI` (6 chs) | `RIGHT_MOTOR_ROI` (6 chs) | Tỷ số `P300 / Left_Motor` | `Motor Laterality (L−R)/(L+R)` | Wilcoxon `P300 vs Left` | Wilcoxon `Left vs Right` |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| `300–600 ms` | 0.01398 ± 0.00436 | 0.01580 ± 0.00445 | 0.01008 ± 0.00292 | **0.90 ± 0.21×** | **+0.2183 ± 0.1035** | 0.0003 | 3.64e-12 |
| `300–500 ms` | 0.01377 ± 0.00484 | 0.01594 ± 0.00479 | 0.01004 ± 0.00313 | 0.86× | +0.2230 ± 0.1114 | — | 1.82e-11 |

### Bảng 3B — Phân tầng Reaction Time (`Fast < 450 ms`, `Mid 450–550 ms`, `Slow > 550 ms`)

| Nhóm RT | Số trials | RT (ms) (Mean ± SD) | Peak XAI (ms) | `P300 / Left_Motor` (`300–600`) | Laterality (`300–600`) | `P300 / Left_Motor` (`450–650`) | Laterality (`450–650`) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| Fast (`< 450 ms`) | 641 | 405.9 ± 30.7 | **452.0** | 0.81× | +0.2862 | 0.79× | +0.2528 |
| Mid (`450–550 ms`) | 1027 | 500.1 ± 28.5 | **484.0** | 0.90× | +0.2197 | 1.00× | +0.1643 |
| Slow (`> 550 ms`) | 1182 | 648.9 ± 102.2 | **532.0** | 0.94× | +0.1474 | 1.04× | +0.1577 |

- **Tương quan `subject_mean_rt_ms` vs hiệu năng LOSO (39 subjects)**:
  - vs `LOSO ROC-AUC`: Spearman `ρ = -0.3893` (`p = 0.0143`), Pearson `r = -0.5312` (`p = 0.0005`).
  - vs `LOSO PR-AUC`: Spearman `ρ = -0.3609` (`p = 0.0240`).
  - vs `LOSO Balanced Acc`: Spearman `ρ = -0.2834` (`p = 0.0804`).

### Bảng 3C — Kiểm chứng Cửa sổ Cắt Sớm (`Early-Window Cutoff <= 450 ms`, Pre-Median-RT)

| Giao thức / Mô hình | Cửa sổ tín hiệu | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | ΔROC-AUC vs Gốc |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| LOSO 39 folds (Gốc đủ cửa sổ) | `[-200, 800] ms` | 0.9778 ± 0.0295 | 0.9167 ± 0.0887 | 0.9147 ± 0.0633 | 0.7692 ± 0.1162 | — |
| LOSO 39 folds (Inference-only mask `> 450 ms`) | `[-200, 450] ms` | 0.8241 ± 0.0993 | 0.5475 ± 0.2032 | 0.7317 ± 0.1008 | 0.4686 ± 0.1357 | -0.1537 (p=3.64e-12) |
| CS 5-Fold (Gốc đủ cửa sổ) | `[-200, 800] ms` | 0.9680 ± 0.0165 | 0.8834 ± 0.0559 | 0.9121 ± 0.0256 | 0.7780 ± 0.0710 | — |
| CS 5-Fold (Inference-only mask `> 450 ms`) | `[-200, 450] ms` | 0.8002 ± 0.0531 | 0.5016 ± 0.1310 | 0.7186 ± 0.0434 | 0.4851 ± 0.0774 | -0.1678 |
| **CS 5-Fold (Retrain-from-scratch `n_times=163`)** | **`[-200, 450] ms`** | **0.9044 ± 0.0293** | **0.7161 ± 0.0722** | **0.8217 ± 0.0371** | **0.5867 ± 0.0741** | **-0.0636** |

---

## Lớp 4 — Chiều Tần số, Đối chiếu ERP & Error Analysis

### Bảng 4A — Frequency Occlusion trên toàn bộ 39 folds LOSO (Butterworth SOS bậc 4)

| Dải tần bị triệt tiêu | ROC-AUC sau occlusion | `ΔROC-AUC` (Mean ± SD) | Wilcoxon p (ROC) | PR-AUC sau occlusion | `ΔPR-AUC` (Mean ± SD) | Wilcoxon p (PR) |
| :--- | :---: | :---: | ---: | :---: | :---: | ---: |
| Sub-Delta (<1 Hz) | 0.9090 ± 0.0657 | **-0.0688 ± 0.0443** | 7.28e-12 | 0.7183 ± 0.1634 | **-0.1984 ± 0.1024** | 3.64e-12 |
| Delta (1–4 Hz) | 0.8150 ± 0.0767 | **-0.1628 ± 0.0649** | 3.64e-12 | 0.4650 ± 0.1402 | **-0.4517 ± 0.1188** | 3.64e-12 |
| Theta (4–8 Hz) | 0.9735 ± 0.0327 | **-0.0043 ± 0.0057** | 2.32e-07 | 0.9042 ± 0.0950 | **-0.0125 ± 0.0154** | 6.25e-07 |
| Alpha (8–13 Hz) | 0.9718 ± 0.0359 | **-0.0060 ± 0.0087** | 7.53e-10 | 0.8969 ± 0.1060 | **-0.0197 ± 0.0248** | 1.63e-09 |
| Beta (13–30 Hz) | 0.9766 ± 0.0297 | **-0.0012 ± 0.0019** | 7.35e-05 | 0.9119 ± 0.0890 | **-0.0047 ± 0.0087** | 2.57e-05 |
| Gamma (30–40 Hz) | 0.9779 ± 0.0296 | **+0.0001 ± 0.0003** | 0.9152 | 0.9168 ± 0.0887 | **+0.0001 ± 0.0010** | 0.8382 |
| Delta+Theta (1–8 Hz) | 0.7847 ± 0.0770 | **-0.1931 ± 0.0674** | 3.64e-12 | 0.4145 ± 0.1262 | **-0.5021 ± 0.1115** | 3.64e-12 |

### Bảng 4B — Tương quan Không gian 127 Kênh (XAI `IG` vs Grand Average ERP `Target − Frequent`)

| Cửa sổ | So sánh cặp bản đồ 127 kênh | Pearson `r` (p-value) | Spearman `ρ` (p-value) | Cosine Sim |
| :--- | :--- | :---: | :---: | ---: |
| `300–500 ms` | `|IG|` vs `ERP (Target − Frequent)` | +0.1105 (0.2161) | +0.0362 (0.6865) | +0.0460 |
| `300–500 ms` | `|IG|` vs `|ERP (Target − Frequent)|` | **+0.2177** (0.0139) | **+0.2338** (0.0082) | +0.7648 |
| `300–500 ms` | `Signed IG` vs `ERP (Target − Frequent)` | +0.3273 (0.0002) | +0.0382 (0.6695) | +0.3136 |
| `300–600 ms` | `|IG|` vs `ERP (Target − Frequent)` | +0.1230 (0.1682) | +0.0663 (0.4586) | +0.0504 |
| `300–600 ms` | `|IG|` vs `|ERP (Target − Frequent)|` | +0.2666 (0.0024) | +0.2658 (0.0025) | +0.8021 |
| `300–600 ms` | `Signed IG` vs `ERP (Target − Frequent)` | +0.3017 (0.0006) | +0.0913 (0.3073) | +0.2885 |

- **Tương quan không gian trung bình từng subject (`300–500 ms`, `|IG|` vs `|ERP diff|`, 39 subjects)**: `r = +0.1817 ± 0.1440`, `ρ = +0.1723 ± 0.1405`.

### Bảng 4C — Error Analysis (`True Positive` vs `False Negative` trên 2,850 Target trials LOSO)

- **True Positive (TP)**: **`2,582`** trials (`90.60%`) | **False Negative (FN)**: **`268`** trials (`9.40%`).
- **Biên độ ERP `Pz` trung bình trong `[300, 600] ms`**: TP = **`+6.86 ± 5.43 µV`** vs FN = **`+3.25 ± 6.27 µV`** (Mann-Whitney `p = 2.33e-25`).
- **Phân bố Reaction Time (`rt_ms`)**: TP = **`533.4 ± 108.6 ms`** (median `521.0 ms`) vs FN = **`610.2 ± 184.6 ms`** (median `571.0 ms`, Mann-Whitney `p = 2.22e-11`).
- **Cường độ `|IG|` tại `P300_ROI` (`300–600 ms`)**: TP = `0.01475` vs FN = `0.00774` (Mann-Whitney `p = 5.37e-64`).

#### Phân tích cụm Bottom-5 LOSO (`sub-07, sub-01, sub-41, sub-32, sub-21`)

| Subject | Target (TP / FN) | FN Rate (%) | Mean RT (ms) | `% RT > 550ms` | `Pz` P300 Mean (µV) | LOSO ROC-AUC | LOSO PR-AUC | `P300/Left_Motor` | Laterality |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sub-07` | 36 (28 / 8) | 22.2% | 671.3 | 63.9% | +3.01 | 0.8744 | 0.6047 | 0.59× | +0.2942 |
| `sub-01` | 61 (45 / 16) | 26.2% | 573.0 | 50.8% | +2.13 | 0.8998 | 0.7023 | 0.65× | +0.1611 |
| `sub-41` | 78 (64 / 14) | 17.9% | 666.1 | 84.6% | +4.96 | 0.9155 | 0.7782 | 0.97× | +0.1870 |
| `sub-32` | 79 (41 / 38) | 48.1% | 650.6 | 77.2% | +2.24 | 0.9205 | 0.7555 | 0.81× | +0.1282 |
| `sub-21` | 77 (42 / 35) | 45.5% | 604.8 | 59.7% | +0.67 | 0.9310 | 0.7551 | 0.71× | +0.1711 |

---

## Phụ lục Bước 6.5 — Kiểm chứng Bổ sung Phản biện (Addendum RQ3)

### Phụ lục A — Hiệu năng & XAI theo phân tầng RT của mô hình Retrain Early-Window (`[-200, 450] ms`, `n_times=163`)

| Phân tầng Target | Số Target (N) | Early450 Fold ROC-AUC (Mean ± SD) | Early450 Fold PR-AUC (Mean ± SD) | Early450 Pooled ROC / PR | Early450 Sens (%) | Full `[-200, 800]` Fold ROC-AUC | Full Fold PR-AUC |
| :--- | ---: | :---: | :---: | :---: | ---: | :---: | :---: |
| All Targets (N=2,850) | 2,850 | 0.9044 ± 0.0293 | 0.7161 ± 0.0722 | 0.8924 / 0.6852 | 79.05% | 0.9680 ± 0.0165 | 0.8834 ± 0.0559 |
| Fast RT (< 450 ms) | 641 | 0.9150 ± 0.0422 | 0.5232 ± 0.1086 | 0.9049 / 0.5172 | 82.37% | 0.9669 ± 0.0174 | 0.7360 ± 0.1113 |
| Mid RT (450–550 ms) | 1,027 | 0.9319 ± 0.0223 | 0.6246 ± 0.0997 | 0.9224 / 0.5707 | 85.30% | 0.9797 ± 0.0157 | 0.8465 ± 0.0824 |
| Slow RT (> 550 ms) | 1,182 | 0.8791 ± 0.0280 | 0.4941 ± 0.1036 | 0.8595 / 0.4391 | 71.83% | 0.9608 ± 0.0190 | 0.7498 ± 0.0974 |
| Very Slow RT (> 600 ms) | 711 | 0.8638 ± 0.0303 | 0.3815 ± 0.1259 | 0.8417 / 0.3270 | 69.62% | 0.9539 ± 0.0230 | 0.6179 ± 0.1305 |

#### Tỷ số ROI `[300, 450] ms` trên mô hình Retrain `[-200, 450] ms` (Captum IG, `n_steps=64`, baseline = `Frequent mean` test fold)

| Mô hình & Phân nhóm Target | Đơn vị / N | `P300_ROI` (7 chs) | `LEFT_MOTOR_ROI` (6 chs) | `RIGHT_MOTOR_ROI` (6 chs) | `P300 / Left_Motor` | `Motor Laterality (L−R)/(L+R)` | Wilcoxon `P300 vs Left` | Wilcoxon `Left vs Right` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| Full `[-200, 800]` CS5 (All Target, `[300, 450]`) | 39 subs | 0.01385 | 0.01674 | 0.01062 | **0.84×** | **+0.2092** | 7.89e-08 | 4.98e-10 |
| **Early450 `[-200, 450]` (All Target, subject-level)** | **39 subs** | 0.01218 | 0.01075 | 0.00608 | **1.21 ± 0.31×** | **+0.2907 ± 0.1029** | 0.0065 | 3.64e-12 |
| Early450 (`All Targets`) | 2,850 trials | 0.01211 | 0.01065 | 0.00604 | **1.14×** | **+0.2772** | 1.08e-57 | 0.00e+00 |
| Early450 (`Fast RT < 450 ms`) | 641 trials | 0.01402 | 0.01115 | 0.00602 | **1.26×** | **+0.3103** | 1.52e-33 | 6.80e-94 |
| Early450 (`Mid RT 450–550 ms`) | 1,027 trials | 0.01244 | 0.01090 | 0.00619 | **1.14×** | **+0.2797** | 3.28e-23 | 4.08e-145 |
| Early450 (`Slow RT > 550 ms`) | 1,182 trials | 0.01080 | 0.01017 | 0.00591 | **1.06×** | **+0.2570** | 2.00e-10 | 2.42e-158 |
| Early450 (`Very Slow RT > 600 ms`) | 711 trials | 0.01060 | 0.01021 | 0.00593 | **1.04×** | **+0.2512** | 0.0003 | 3.53e-94 |

### Phụ lục B — Sai số Tương đối Completeness của Captum IG trên toàn bộ `2,850` Target trials LOSO

- **`|delta|` (tuyệt đối)**: Median = `8.6737e-04`, P95 = `7.6741e-03`, P99 = `1.9958e-02`, Max = `9.8433e-02` (Mean = `2.2804e-03`).
- **`rel_delta = |delta| / (|F(x) − F(x')| + 1e-8)`**: Median = **`0.0075%`**, P95 = **`0.0474%`**, P99 = **`0.1199%`**, Max = **`0.6771%`** (Mean = `0.0155%`).

| Hạng | `trial_uid` | Subject | `rt_ms` | `F(x)` | `F(x')` | `F(x) − F(x')` | `|delta|` | `rel_delta (%)` |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| #1 | `sub-36_567579` | `sub-36` | 477.0 | +8.6992 | -7.7639 | +16.4630 | 0.09843 | 0.5979% |
| #2 | `sub-36_1079575` | `sub-36` | 598.0 | +6.7675 | -7.7639 | +14.5314 | 0.09839 | 0.6771% |
| #3 | `sub-33_1160958` | `sub-33` | 462.0 | +11.2542 | -9.2793 | +20.5334 | 0.09080 | 0.4422% |
| #4 | `sub-05_900734` | `sub-05` | 458.0 | +8.6632 | -7.5100 | +16.1732 | 0.06638 | 0.4104% |
| #5 | `sub-05_144325` | `sub-05` | 444.0 | +10.3587 | -7.5100 | +17.8686 | 0.06291 | 0.3521% |

### Phụ lục C — Chẩn đoán Threshold Miscalibration trên cụm Bottom-5 LOSO *(Diagnostic Only — Không thay thế điểm test chính thức)*

| Subject | `thr_loso` | Gốc: Sens / Spec / F1 / FN (FN%) | `thr_oracle (bAcc)` | Oracle bAcc: Sens / Spec / F1 / FN (FN%) | FN khôi phục | `thr_oracle (F1)` (F1) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: |
| `sub-07` | 0.050 | 77.8% / 73.4% / 0.441 / 8 (22.2%) | 0.144 | 77.8% / 84.0% / 0.549 / 8 (22.2%) | **0/8 (0.0%)** | 0.895 (0.580) |
| `sub-01` | 0.075 | 73.8% / 90.6% / 0.596 / 16 (26.2%) | 0.056 | 78.7% / 89.7% / 0.608 / 13 (21.3%) | **3/16 (18.8%)** | 0.320 (0.667) |
| `sub-41` | 0.160 | 82.1% / 89.5% / 0.660 / 14 (17.9%) | 0.220 | 82.1% / 90.7% / 0.681 / 14 (17.9%) | **0/14 (0.0%)** | 0.770 (0.732) |
| `sub-32` | 0.950 | 51.9% / 98.2% / 0.636 / 38 (48.1%) | 0.085 | 79.7% / 91.7% / 0.689 / 16 (20.3%) | **22/38 (57.9%)** | 0.555 (0.725) |
| `sub-21` | 0.440 | 54.5% / 97.8% / 0.646 / 35 (45.5%) | 0.050 | 83.1% / 87.2% / 0.627 / 13 (16.9%) | **22/35 (62.9%)** | 0.190 (0.671) |
