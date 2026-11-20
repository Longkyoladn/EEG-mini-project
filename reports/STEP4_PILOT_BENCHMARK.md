# BƯỚC 4 — BÁO CÁO GO/NO-GO PILOT BENCHMARK (EEGNet-v4 vs P300-ATCNet)

- **Tập dữ liệu**: `data/processed` (39 subjects sạch, 20,886 epochs)
- **Phần cứng thực thi**: `NVIDIA GeForce RTX 5060 Ti`
- **Siêu tham số chung**: `epochs=80`, `patience=15`, `lr=1e-3`, `weight_decay=1e-2`, `seed=42`, `early_stop_metric=val_roc_auc`, `threshold_metric=balanced_accuracy`

### Bảng 1 — Cross-Subject 5-Fold (39 subjects, `batch_size=64`)

| Thực nghiệm | Mô hình / Readout | Params | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) | F1-Target (Mean ± SD) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | Mean Best Epoch | Thời gian chạy |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| `pilot_cs5_eegnet` | EEGNet-v4 | 3,721 | 0.9593 ± 0.0202 | 0.8441 ± 0.0675 | 0.8947 ± 0.0294 | 0.7030 ± 0.1040 | 0.9001 ± 0.0281 | 0.8894 ± 0.0580 | 29.2 | 1630.0s (27.17m) |
| `pilot_cs5_atcnet_avg` | P300-ATCNet (avg_pool) | 49,650 | 0.9597 ± 0.0219 | 0.8636 ± 0.0550 | 0.8946 ± 0.0318 | 0.7300 ± 0.1083 | 0.8784 ± 0.0111 | 0.9107 ± 0.0551 | 26.2 | 1861.9s (31.03m) |
| `pilot_cs5_atcnet_attn` | P300-ATCNet (attn_pool) | 49,716 | 0.9703 ± 0.0183 | 0.8896 ± 0.0600 | 0.9110 ± 0.0279 | 0.7835 ± 0.0797 | 0.8825 ± 0.0295 | 0.9396 ± 0.0313 | 16.6 | 1450.7s (24.18m) |
| `pilot_cs5_atcnet_last` | P300-ATCNet (last_step) | 49,650 | 0.9717 ± 0.0158 | 0.8933 ± 0.0591 | 0.9122 ± 0.0250 | 0.7613 ± 0.0952 | 0.9023 ± 0.0164 | 0.9222 ± 0.0451 | 29.8 | 2027.0s (33.78m) |

### Bảng 2 — Within-Subject 5-Fold Tổng hợp (4 subjects: `sub-03`, `sub-05`, `sub-07`, `sub-10`, `batch_size=32`)

| Thực nghiệm | Mô hình / Cấu hình | Params | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) | F1-Target (Mean ± SD) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | Mean Best Epoch | Thời gian chạy |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| `pilot_ws_eegnet` | EEGNet-v4 | 3,721 | 0.9237 ± 0.0968 | 0.7748 ± 0.2367 | 0.8482 ± 0.1264 | 0.6854 ± 0.2244 | 0.8381 ± 0.1005 | 0.8583 ± 0.1542 | 15.8 | 93.6s (1.56m) |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | 49,650 | 0.9147 ± 0.1013 | 0.7595 ± 0.2535 | 0.8262 ± 0.1303 | 0.6472 ± 0.2378 | 0.7744 ± 0.1601 | 0.8780 ± 0.1041 | 18.1 | 140.6s (2.34m) |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | 49,716 | 0.8937 ± 0.1365 | 0.7709 ± 0.2633 | 0.8320 ± 0.1382 | 0.6851 ± 0.2674 | 0.7641 ± 0.1508 | 0.8999 ± 0.1309 | 18.1 | 140.8s (2.35m) |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | 28,722 | 0.8976 ± 0.1080 | 0.7270 ± 0.2373 | 0.8205 ± 0.0850 | 0.6208 ± 0.1656 | 0.7948 ± 0.0648 | 0.8463 ± 0.1339 | 18.7 | 134.9s (2.25m) |

### Bảng 3 — Phân rã Within-Subject theo từng cá nhân (`sub-03`, `sub-05`, `sub-07`, `sub-10`)

| Thực nghiệm | Mô hình | Subject / Nhóm | Epochs (Target) | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `pilot_ws_eegnet` | EEGNet-v4 | sub-03 (Sạch) | 558 (81) | 0.9780 ± 0.0232 | 0.8725 ± 0.1353 | 0.9309 ± 0.0488 |
| `pilot_ws_eegnet` | EEGNet-v4 | sub-05 (Sạch) | 561 (76) | 0.9877 ± 0.0112 | 0.9514 ± 0.0451 | 0.9355 ± 0.0507 |
| `pilot_ws_eegnet` | EEGNet-v4 | sub-07 (Ít Target) | 273 (36) | 0.7807 ± 0.1552 | 0.4259 ± 0.1912 | 0.6656 ± 0.1404 |
| `pilot_ws_eegnet` | EEGNet-v4 | sub-10 (Ít Target) | 251 (35) | 0.9482 ± 0.0428 | 0.8495 ± 0.1150 | 0.8609 ± 0.0923 |
| `pilot_ws_eegnet` | EEGNet-v4 | **TB Nhóm sạch (03, 05)** | — | **0.9829** | **0.9120** | **0.9332** |
| `pilot_ws_eegnet` | EEGNet-v4 | **TB Nhóm ít Target (07, 10)** | — | **0.8645** | **0.6377** | **0.7633** |
| `pilot_ws_eegnet` | EEGNet-v4 | **Δ (Sạch − Ít Target)** | — | **+0.1184** | **+0.2743** | **+0.1699** |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | sub-03 (Sạch) | 558 (81) | 0.9884 ± 0.0093 | 0.9385 ± 0.0474 | 0.9333 ± 0.0652 |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | sub-05 (Sạch) | 561 (76) | 0.9765 ± 0.0181 | 0.9106 ± 0.0615 | 0.9111 ± 0.0673 |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | sub-07 (Ít Target) | 273 (36) | 0.7683 ± 0.0590 | 0.3903 ± 0.0795 | 0.6471 ± 0.0482 |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | sub-10 (Ít Target) | 251 (35) | 0.9257 ± 0.0404 | 0.7985 ± 0.0769 | 0.8134 ± 0.0728 |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | **TB Nhóm sạch (03, 05)** | — | **0.9825** | **0.9246** | **0.9222** |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | **TB Nhóm ít Target (07, 10)** | — | **0.8470** | **0.5944** | **0.7303** |
| `pilot_ws_atcnet_avg` | P300-ATCNet (avg_pool) | **Δ (Sạch − Ít Target)** | — | **+0.1354** | **+0.3302** | **+0.1919** |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | sub-03 (Sạch) | 558 (81) | 0.9920 ± 0.0069 | 0.9588 ± 0.0373 | 0.9545 ± 0.0351 |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | sub-05 (Sạch) | 561 (76) | 0.9807 ± 0.0149 | 0.9350 ± 0.0524 | 0.8956 ± 0.0979 |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | sub-07 (Ít Target) | 273 (36) | 0.6974 ± 0.1320 | 0.3902 ± 0.2243 | 0.6366 ± 0.1229 |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | sub-10 (Ít Target) | 251 (35) | 0.9045 ± 0.0525 | 0.7997 ± 0.1010 | 0.8412 ± 0.0460 |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | **TB Nhóm sạch (03, 05)** | — | **0.9864** | **0.9469** | **0.9250** |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | **TB Nhóm ít Target (07, 10)** | — | **0.8009** | **0.5949** | **0.7389** |
| `pilot_ws_atcnet_attn` | P300-ATCNet (attn_pool) | **Δ (Sạch − Ít Target)** | — | **+0.1855** | **+0.3519** | **+0.1861** |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | sub-03 (Sạch) | 558 (81) | 0.9828 ± 0.0057 | 0.9286 ± 0.0223 | 0.8784 ± 0.0578 |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | sub-05 (Sạch) | 561 (76) | 0.9471 ± 0.0361 | 0.8285 ± 0.1090 | 0.8829 ± 0.0560 |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | sub-07 (Ít Target) | 273 (36) | 0.7402 ± 0.1219 | 0.3857 ± 0.2065 | 0.7004 ± 0.1316 |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | sub-10 (Ít Target) | 251 (35) | 0.9203 ± 0.0481 | 0.7652 ± 0.1445 | 0.8204 ± 0.1026 |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | **TB Nhóm sạch (03, 05)** | — | **0.9650** | **0.8785** | **0.8806** |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | **TB Nhóm ít Target (07, 10)** | — | **0.8303** | **0.5755** | **0.7604** |
| `pilot_ws_atcnet_shared` | P300-ATCNet (avg_pool, shared) | **Δ (Sạch − Ít Target)** | — | **+0.1347** | **+0.3031** | **+0.1203** |

### Bảng 4 — Kiểm tra chênh lệch Validation vs Test & Ước lượng thời gian chạy Full Bước 5

#### Bảng 4A — Kiểm tra chênh lệch `best_val_metric` (Val ROC-AUC) vs `test_roc_auc` & An toàn hội tụ

| Thực nghiệm | Protocol | Val ROC-AUC (Mean ± SD) | Test ROC-AUC (Mean ± SD) | Δ (Val − Test) | best_epoch (Mean [Min..Max]) | Folds ep < 5 | NaN (Fold / Pred) |
| :--- | :--- | :---: | :---: | ---: | :---: | ---: | :---: |
| `pilot_cs5_eegnet` | `cross_subject_5fold` | 0.9683 ± 0.0147 | 0.9593 ± 0.0202 | +0.0090 | 29.2 [18..50] | 0/5 | 0 / 0 |
| `pilot_cs5_atcnet_avg` | `cross_subject_5fold` | 0.9727 ± 0.0110 | 0.9597 ± 0.0219 | +0.0130 | 26.2 [16..37] | 0/5 | 0 / 0 |
| `pilot_cs5_atcnet_attn` | `cross_subject_5fold` | 0.9800 ± 0.0099 | 0.9703 ± 0.0183 | +0.0098 | 16.6 [12..22] | 0/5 | 0 / 0 |
| `pilot_cs5_atcnet_last` | `cross_subject_5fold` | 0.9809 ± 0.0095 | 0.9717 ± 0.0158 | +0.0092 | 29.8 [15..47] | 0/5 | 0 / 0 |
| `pilot_ws_eegnet` | `within_subject` | 0.9638 ± 0.0604 | 0.9237 ± 0.1140 | +0.0401 | 15.8 [1..44] | 5/20 | 0 / 0 |
| `pilot_ws_atcnet_avg` | `within_subject` | 0.9571 ± 0.0559 | 0.9147 ± 0.0963 | +0.0424 | 18.1 [5..59] | 0/20 | 0 / 0 |
| `pilot_ws_atcnet_attn` | `within_subject` | 0.9529 ± 0.0685 | 0.8937 ± 0.1379 | +0.0592 | 18.1 [5..44] | 0/20 | 0 / 0 |
| `pilot_ws_atcnet_shared` | `within_subject` | 0.9456 ± 0.0645 | 0.8976 ± 0.1145 | +0.0480 | 18.7 [2..41] | 2/20 | 0 / 0 |

#### Bảng 4B — Ước lượng thời gian chạy Full Bước 5 trên GPU RTX 5060 Ti (195 folds Within-Subject + 39 folds LOSO)

- Hệ số hiệu chỉnh quy mô tập train từ `splits.json`: Within-Subject `342.4 / 262.4 = 1.3047x` (195 folds vs 20 pilot folds); LOSO `18,195.6 / 13,487.2 = 1.3491x` (39 folds vs 5 CS5 folds).

| Kiến trúc / Cấu hình | WS Pilot (20f) | WS-195 Trực tiếp | WS-195 Hiệu chỉnh | CS5 Pilot (5f) | LOSO-39 Trực tiếp | LOSO-39 Hiệu chỉnh | Tổng Bước 5 (Hiệu chỉnh) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EEGNet-v4 | 93.6s (4.68s/f) | 15.21 phút (913s) | 19.85 phút (1191s) | 1630.0s (326.01s/f) | 211.91 phút (12714s) | 285.88 phút (17153s) | **305.73 phút (18344s)** |
| P300-ATCNet (avg_pool) | 140.6s (7.03s/f) | 22.85 phút (1371s) | 29.81 phút (1789s) | 1861.9s (372.39s/f) | 242.05 phút (14523s) | 326.55 phút (19593s) | **356.36 phút (21382s)** |
| P300-ATCNet (attn_pool) | 140.8s (7.04s/f) | 22.88 phút (1373s) | 29.85 phút (1791s) | 1450.7s (290.14s/f) | 188.59 phút (11316s) | 254.43 phút (15266s) | **284.28 phút (17057s)** |
| P300-ATCNet (last_step) | — | — | — | 2027.0s (405.39s/f) | 263.51 phút (15810s) | 355.50 phút (21330s) | 355.50 phút (21330s) (chỉ LOSO) |
| P300-ATCNet (avg_pool, shared) | 134.9s (6.75s/f) | 21.93 phút (1316s) | 28.61 phút (1717s) | — | — | — | 28.61 phút (1717s) (chỉ WS) |

> **Tổng thời gian thực chạy 8 thực nghiệm Pilot**: `7479.6s (124.66m)`.
> **Dự kiến tổng thời gian chạy cặp mô hình chính Bước 5 (`EEGNet-v4` + `P300-ATCNet avg_pool`, 195 WS + 39 LOSO mỗi mô hình)**: `662.09 phút (39726s)` (hiệu chỉnh theo quy mô tập train).
