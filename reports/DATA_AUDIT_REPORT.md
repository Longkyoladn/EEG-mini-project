# BÁO CÁO KIỂM TOÁN DỮ LIỆU EEG TOÀN DIỆN (DATA AUDIT REPORT)

- **Dataset**: `The Nencki-Symfonia EEG/ERP dataset` (`doi:10.18112/openneuro.ds004621.v1.0.4`)
- **Tác vụ**: Visual 3-Stimulus Oddball (`task-oddball`, sóng P300, phân loại Single-trial `Frequent` vs `Target`)
- **Thư mục nguồn (READ-ONLY)**: `D:\EEG-mini-project\data\raw\oddball_data`
- **Thời điểm chạy audit**: `2026-11-20 02:24:49`
- **Tổng dung lượng trên đĩa**: `27.19 GB` (`42/42` subjects, `210` files task-oddball + `7` files metadata gốc)

> [!CAUTION]
> **TÓM TẮT CÁC BẪY KỸ THUẬT LỚN NHẤT (RED FLAGS) CẦN XỬ LÝ NGAY TRƯỚC KHI PREPROCESSING:**
> 1. **3 subjects hiểu sai luật hành vi (`sub-02, sub-14, sub-17`) + 8 subjects Hit Rate thấp (`< 90%`)**: `sub-02, sub-14, sub-17` bấm nút cho cả Distractor (`S  7`) với tỷ lệ False Alarm là `98.77%`, `97.53%`, `76.54%` (trùng bài báo Dzianok et al., 2022 - Table 5) -> **bắt buộc loại**. Ngoài ra, có `8` subjects khác có `Hit Rate < 90%` (thấp nhất là `sub-07`: `60.49%`, `sub-01`: `76.54%`, `sub-25`: `77.78%`, `sub-10`: `82.72%`).
> 2. **Bẫy 127 vs 128 kênh & Lỗi nhiễm Average Reference từ `O9/O10`**: File raw chỉ có **127 kênh EEG** (`FCz` là tham chiếu phần cứng). Trong 127 kênh có **4 kênh không khớp `standard_1005`/`colin27_1005`**: `TP9 LEFT EAR`, `TP10 RIGHT EAR`, `O9`, `O10`. Trong chuẩn `1005` của MNE, `O9` và `O10` là **`I1`** và **`I2`** (sai khác tọa độ `0.000 m`). Vì `src/preprocessing/pipeline.py` chưa map `O9/O10 -> I1/I2`, các kênh đang nhiễu nặng như `O10` của `sub-16` (`7,212 µV`) và `sub-41` (`7,961 µV`) bị bỏ qua khỏi khâu phát hiện/nội suy kênh xấu và **lọt thẳng vào Average Reference làm ô nhiễm toàn bộ 127 kênh**!
> 3. **Bẫy rút mũ / bão hòa cuối file trên `sub-08` (và jump cuối file ở `sub-35`)**: `sub-08` kết thúc task ở `1278.6s`, nhưng máy vẫn ghi tiếp tới `1694.9s`; tại `~1446.2s` tín hiệu nhảy lên bão hòa **`820,810.9 µV`** làm `median_std` toàn file tăng vọt lên **`291,895.27 µV`** (gấp >1000x bình thường!). Nếu không `crop()` về cửa sổ task trước khi chạy `filter(0.5, 40)` và `ICA`, toàn bộ dữ liệu `sub-08` bị phá hỏng (`filtered median_std = 3,080.82 µV`); sau `crop()` về `[2.1s, 1283.6s]`, tín hiệu sạch chuẩn `9.82 µV`!
> 4. **Đảo thứ tự kênh giữa các subjects**: Có **3/42 subjects** (`sub-13, sub-41, sub-42`) bị hoán đổi thứ tự kênh trong `.vhdr` so với `sub-01`. Nối trực tiếp mảng NumPy mà không gọi `raw.reorder_channels()` sẽ làm sai lệch không gian điện cực.
> 5. **Bẫy mã hóa `_channels.tsv` (`Latin-1`) & thiếu cột `status`**: Toàn bộ 42 file `_channels.tsv` dùng byte `0xb5` (`µV`) chuẩn Latin-1 (đọc bằng UTF-8 sẽ văng `UnicodeDecodeError`), và không có cột `status` đánh dấu kênh `bad` sẵn.
> 6. **Marker `boundary` ẩn trong `_events.tsv` & giải mã `sub-38`**: **5 subjects (`sub-15, sub-23, sub-24, sub-37, sub-38`)** bị ngắt/khởi động lại trong lúc ghi (`New Segment` + `boundary` trong `.vmrk` bị ẩn khỏi `_events.tsv`). Riêng **`sub-38` KHÔNG hề thiếu file** — đọc bình thường bằng MNE (`1,512,740` mẫu), có `2` marker `boundary` và chứa **`775` stimuli**.
> 7. **Cảnh báo XAI (RQ3 - Motor Confound)**: **`75.25%`** số trial `Target Hit` (trên 39 subjects sạch) có Reaction Time bấm phím ngón giữa tay phải rơi đúng vào cửa sổ P300 **`300–600 ms`**! Nếu model học vào điện thế vận động bên vỏ não trái (`C3`, `CP3`, `CCP3h`) thay vì sóng P300 vùng đỉnh (`Pz`, `CPz`), kết quả XAI sẽ bị nhiễm confound vận động.

## 1. Cấu trúc BIDS & Tóm tắt Metadata gốc

### 1.1. Thông tin Dataset & Tham số Thu thập
- **Tên dataset**: The Nencki-Symfonia EEG/ERP dataset (`BIDSVersion: 1.0.2`, `License: CC0`)
- **Công bố gốc**: Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. *The Nencki-Symfonia electroencephalography/event-related potential dataset: Multiple cognitive tasks and resting-state data collected in a sample of healthy adults*. GigaScience. 2022;11:giac015.
- **Thiết bị & Hệ thống đo** (`task-oddball_eeg.json`):
  - Máy đo: `Brain Products GmbH actiCHamp` (mũ `Brain Products GmbH`)
  - Tần số lấy mẫu gốc: `1000 Hz` | Bộ lọc phần cứng Lowpass: `280 Hz` | Điện lưới: `50 Hz`
  - Reference gốc: **`FCz`** | Ground: **`FPz`**
  - Thiết kế thực nghiệm: 3 kích thích (`p` Frequent ~76%, `þ` Target ~12%, `Þ` Distractor ~12%), tổng chuẩn **660 stimuli/subject**, thời gian hiển thị `200 ms`, ISI ngẫu nhiên `1200–1600 ms`, tổng thời lượng ~22 phút (gồm 4 block nghỉ giữa chừng). Người tham gia bấm phím **`2`** trên bàn phím số bằng **ngón giữa tay phải** khi thấy Target `þ`.
- **Lưu ý nhãn trong `task-oddball_events.json`**:
  - `S  5`: Standard/Frequent stimulus (`trial_type = stimulus`)
  - `S  6`: Target stimulus (`trial_type = stimulus`)
  - `S  7`: Deviant/Distractor stimulus (`trial_type = stimulus`)
  - `S  1`: Response button (`trial_type = response` — lưu ý file JSON ghi mô tả `'Response: button no. 1'` theo mã trigger BrainVision dù phím vật lý là phím `'2'`)
  - `S 10`: Break (nghỉ giữa block) | `S 12`: Beginning of the run (`trial_type = n/a`)

### 1.2. Tóm tắt Nhân khẩu học (`participants.tsv`)
- Tổng số dòng: `42` (`sub-01` .. `sub-42`), `90` cột (nhân khẩu học + bảng hỏi tâm lý UMACL/UWIST & ARSQ).
- Giới tính: Nữ (`F`) = `22`, Nam (`M`) = `20`.
- Độ tuổi: `24.62 ± 4.12` tuổi (khoảng `20–34` tuổi).
- Dùng thuốc trong ngày đo (`medication use=1`): `5` người | Dùng caffeine (`caffeine uptake=1`): `17` người.
- **Lưu ý parse `participants.tsv`**: Có `2` ô bảng hỏi chứa chuỗi mảng thay vì số nguyên: `sub-04:ARSQ_4=[3,4], sub-12:ARSQ_10=[1,4]`.

### 1.3. Tổng hợp Ghi chú Tác giả & Lịch sử Sửa đổi (`README`, `CHANGES`, Bài báo GigaScience 2022)
- **Từ `README` & `CHANGES`**:
  - `15/06/2023` (v1.0.0): Chuyển đơn vị cột `onset` trong tất cả `*_events.tsv` từ mili-giây (`ms`) sang **giây (`s`)**.
  - `10/2024` (v1.0.2): Sửa logfiles trong `/sourcedata` cho tác vụ MSIT & SRT (`sub-01`..`sub-03`).
  - `02/2025` (v1.0.3): Sửa marker cho tác vụ **REST** của `sub-20`, `sub-23` (không ảnh hưởng đến `task-oddball`).
- **Từ bài báo gốc (Dzianok et al., 2022, *GigaScience*, giac015)**:
  - **Loại trừ hành vi (Table 5)**: Tác giả loại **3 subjects (`sub-02`, `sub-14`, `sub-17`)** khỏi phân tích hành vi Oddball vì họ bấm nút cho cả kích thích Distractor (`S  7`) với tỷ lệ False Alarm là `98.77%` (`sub-02`), `97.53%` (`sub-14`), và `76.54%` (`sub-17`) — chứng tỏ hiểu nhầm hướng dẫn thực nghiệm.
  - **Sự cố kỹ thuật trong phiên ghi**: Tác giả ghi nhận **5 subjects (`sub-13`, `sub-23`, `sub-24`, `sub-37`, `sub-38`)** có tổng số kích thích khác mức chuẩn 660 (từ 659 đến 775) do lỗi kỹ thuật thu thập được khắc phục/khởi động lại ngay trong phiên ghi.

## 2. Độ đầy đủ File của 42 Subjects & Giải mã `sub-38`

- **Độ đầy đủ file `task-oddball`**: **`42/42` subjects** có đủ cả 5 file bắt buộc (`.vhdr`, `.eeg`, `.vmrk`, `_events.tsv`, `_channels.tsv`).
- **Kiểm tra toàn vẹn nhị phân `.eeg`**: **`42/42` subjects** khớp tuyệt đối từng byte với công thức `DataPoints × 127 × 4` (`IEEE_FLOAT_32`).
- **File tọa độ điện cực (`_electrodes.tsv` / `_coordsystem.json`)**: Tìm thấy **`0` file** trong thư mục local `data/raw/oddball_data/`. *(Giải thích: Các file tọa độ CapTrak `.bvct` nằm trong thư mục `/sourcedata/sub-XX/coords/` trên S3 OpenNeuro và không nằm trong nhánh `sub-XX/eeg/` khi script tải lọc theo `'task-oddball' in fname`.)*

### 2.1. Kiểm tra Chuyên sâu `sub-38` (Giải đáp nghi vấn 'thiếu dữ liệu')

- **Khả năng đọc bằng MNE-Python**: `mne.io.read_raw_brainvision()` đọc **THÀNH CÔNG 100%** (`readable_by_mne = True`).
- **Thông số thực tế của `sub-38`**:
  - Số mẫu thực tế (`n_times`): **`1,512,740` mẫu** tại `sfreq = 1000 Hz` (thời lượng `1512.74 s` ≈ `25.21 phút`).
  - Dung lượng `.eeg` thực tế: **`768,471,920` bytes** (khớp tuyệt đối `768,471,920` bytes theo header `.vhdr`).
  - Kiểm tra giá trị mẫu đầu / vùng boundary / cuối bằng MNE: `head_finite=True`, `mid_boundary_finite=True`, `tail_finite=True` (hoàn toàn không có NaN/Inf hay cắt cụt).
- **Đối chiếu `.vmrk` vs `_events.tsv` của `sub-38`**:
  - Tệp `.vmrk` có tổng cộng `896` dòng marker: gồm `2` `Time 0` tại mẫu `1`, `1` marker `Stimulus,empty` tại mẫu `1`, `3` marker `New Segment` (tại mẫu `1`, `1,178,725`, `1,179,901`), `2` marker `Stimulus,boundary` (tại mẫu `1,179,901`), và `888` marker sự kiện task (`583` `S  5`, `96` `S  6`, `96` `S  7`, `105` `S  1`, `4` `S 10`, `4` `S 12`).
  - Tệp `_events.tsv` có `889` dòng (`888` marker sự kiện task + `1` dòng `empty` ở `onset = 0.001 s`), tức đã **ẩn các marker `New Segment` và `boundary`**. Sai lệch thời điểm tối đa giữa `_events.tsv` và `.vmrk`: `1.000 ms` (lệch 1 sample `0.001s` do quy đổi 1-based sang giây).
- **Nguyên nhân dẫn tới ghi chú lỗi trước đây**:
  1. Trong bài báo gốc, tác giả liệt kê `sub-38` vào nhóm 5 subject gặp lỗi thu thập kỹ thuật vì phiên ghi bị **gián đoạn ở phút ~19.6 (`1178.7–1179.9 s`)**, sau đó được khởi động lại block bổ sung và ghép nối (`New Segment` + `boundary`) dẫn đến tổng số stimuli tăng từ `660` lên **`775`** (`583` S5 + `96` S6 + `96` S7). Ngoài ra, `sub-38` từng bị nhầm lẫn là thiếu dữ liệu oddball trong các commit ban đầu của nhóm trước khi cập nhật lại `download_oddball.py`.
  2. **Kết luận về `sub-38`**: Dữ liệu `task-oddball` của `sub-38` **KHÔNG bị mất**, hoàn toàn đọc được bằng MNE và hành vi đạt chất lượng cao (`Hit Rate = 100.0%`, `FA_Dist = 8.3%`), chỉ cần lưu ý **không cắt epoch xuyên qua điểm boundary `1178.7–1179.9 s`**.

## 3. Audit Tín hiệu EEG & Hệ kênh 128 (Montage Check)

### 3.1. Thông số Chung & Bẫy 127 vs 128 Kênh
- **Tần số lấy mẫu (`sfreq`)**: Đồng nhất **`1000.0 Hz`** trên cả 42/42 subjects.
- **Số kênh trong file**: Đồng nhất **`127` kênh**, toàn bộ đều mang loại **`EEG`** (`0` kênh EOG/ECG/MISC/STIM riêng trong `_channels.tsv` và `.vhdr`).
- **Giải thích 127 vs 128 kênh**: Hệ thống mũ là actiCHamp 128 điện cực, nhưng kênh thứ 128 là **`FCz`** được sử dụng làm **điện cực tham chiếu phần cứng (online reference)** nên không lưu thành kênh riêng trong file raw. Khi thực hiện Average Re-referencing, nhóm có thể tùy chọn gọi `mne.add_reference_channels(raw, ref_channels=['FCz'])` để khôi phục đủ 128 kênh trước `set_eeg_reference('average')`.
- **Độ dài bản ghi**: Trung bình `1368.2 ± 69.8 s` (`~22.80 phút`), ngắn nhất `21.95 phút` (`sub-19`), dài nhất `28.25 phút` (`sub-08`).
- **Danh sách kênh `bad` trong `_channels.tsv`**: Không có cột `status` (`good`/`bad`) trong bất kỳ file `_channels.tsv` nào (chỉ có 3 cột `name`, `type`, `units`). Tất cả 42 file `_channels.tsv` đều dùng mã hóa **`Latin-1`** (`µV` = byte `0xb5`), bắt buộc truyền `encoding='latin1'` khi đọc bằng Python/Pandas.

### 3.2. Tính Đồng nhất Tên Kênh, Thứ tự Kênh & Đối chiếu Montage Topomap
- **Tập hợp 127 tên kênh**: Đồng nhất **100%** trên cả 42/42 subjects (`set_mismatches = 0`).
- **Bẫy đảo thứ tự kênh**: Phát hiện **`3/42` subjects** bị hoán đổi thứ tự kênh trong `.vhdr` so với `sub-01`:
  - `sub-13`: hoán đổi `2` vị trí (idx 120 (FCC6h vs FFT8h), idx 123 (FFT8h vs FCC6h)).
  - `sub-41`: hoán đổi `2` vị trí (idx 4 (FT9 vs FC5), idx 5 (FC5 vs FT9)).
  - `sub-42`: hoán đổi `35` vị trí (idx 92 (FFC2h vs AFF2h), idx 93 (AFF2h vs F10), idx 94 (F10 vs AFp1), idx 95 (AFp1 vs AFF5h)...).
- **Đối chiếu với `mne.channels.make_standard_montage('standard_1005')`** *(lưu ý MNE khuyến nghị `colin27_1005` thay `standard_1005` từ MNE 1.14)*:
  - Số kênh khớp trực tiếp từ tên gốc: **`123 / 127` kênh**.
  - **4 kênh không khớp trực tiếp**: `TP9 LEFT EAR, TP10 RIGHT EAR, O9, O10`.
    1. `TP9 LEFT EAR` → đổi thành **`TP9`** (khớp `1005`).
    2. `TP10 RIGHT EAR` → đổi thành **`TP10`** (khớp `1005`).
    3. `O9` → trong `standard_1005`/`colin27_1005` của MNE tên chuẩn là **`I1`** (kiểm chứng tọa độ 3D của `O9` trong `standard_1020` và `I1` trong `standard_1005` sai khác = **`0.000000 m`** — trùng khít tuyệt đối!).
    4. `O10` → trong `standard_1005`/`colin27_1005` của MNE tên chuẩn là **`I2`** (sai khác tọa độ = **`0.000000 m`**).
  - Sau khi áp dụng đủ 4 alias trên: **`127 / 127` kênh (100%)** có đầy đủ tọa độ 3D (`unmatched = []`). *(Khuyến nghị sửa ngay `CHANNEL_ALIASES` trong `src/preprocessing/pipeline.py` dòng 105–108 để bổ sung `'O9': 'I1'` và `'O10': 'I2'`).*

### 3.3. Kiểm tra Tín hiệu Lỗi (NaN, Inf, Flatline, Noisy Channels, Saturation) trên toàn 42 Subjects
- **Giá trị `NaN` / `Inf`**: `NaN` = `0/42` subjects, `Inf` = `0/42` subjects (toàn bộ 42 file sạch tuyệt đối NaN/Inf).
- **Kênh phẳng (`std < 0.1 µV`)**: Phát hiện ở `0/42` subjects (không phát hiện kênh phẳng trên toàn cửa sổ task).
- **Biên độ Std trung vị trong cửa sổ task (`task_median_std_uv`) vs Toàn file (`median_std_uv`)**:
  - Trong cửa sổ task thực tế (`[first_stim - 2s, last_stim + 2s]`): `task_median_std_uv` dao động từ `108.6 µV` (`sub-22`) đến `972.1 µV` (`sub-32`).
  - **Bẫy rút mũ / bão hòa cuối file (`sub-08`)**: Trên toàn file `sub-08`, `median_std_uv = 291,895.27 µV` (`max_abs = 820,810.9 µV`), gấp >1000 lần mức trong cửa sổ task (`task_median_std_uv = 200.85 µV`) do sau khi task kết thúc ở `1278.6s`, tín hiệu nhảy step khổng lồ tại `1446.21s` rồi phẳng lì đến `1694.9s`. Kiểm chứng với MNE: nếu chạy `raw.filter(0.5, 40)` không `crop()`, `sub-08` bị ringing hỏng nặng (`filtered median_std = 3,080.82 µV`); sau khi `raw.crop()` theo cửa sổ task `[2.1s, 1283.6s]`, `filtered median_std` trở về **`9.82 µV`** (hoàn toàn sạch!). Ngoài ra `sub-35` cũng có jump cuối file (`5,467.5 µV` tại `1378.47s`, sau `last_stim = 1278.4s`).
  - **Làm rõ trường hợp `sub-32`**: `sub-32` có `median_std_uv = 1025.04 µV` trên tín hiệu thô (cao hơn đa số các subject khác ~200–350 µV do DC drift/trở kháng cao toàn đầu), nhưng có `0` kênh vượt ngưỡng MAD z-score > 5.0 và sau bandpass `0.5–40 Hz` đạt `median_std = 12.44 µV`.
- **Nguy cơ kênh `O9`/`O10` nhiễm vào Average Reference**: Có `3` subjects có kênh `O9` hoặc `O10` bị nhiễu rất nặng trên raw (`sub-16, sub-17, sub-41` — ví dụ `sub-41` có `O10 = 7,960.97 µV`, `sub-16` có `O10 = 7,212.39 µV`, `sub-17` có `O9 = 1,672.49 µV`). Do chưa map `O9->I1` và `O10->I2`, pipeline cũ bỏ qua phát hiện bad channel trên 2 kênh này khiến nhiễu cực lớn bị cộng vào Average Reference!
- **Top subjects có nhiều kênh nhiễu (`MAD z-score > 5.0` trên tín hiệu thô)**:
  - `sub-18`: `8` kênh noisy (`Fp1(1667.03uV/z=11.69);Fp2(1524.38uV/z=10.58);AF7(1296.24uV/z=8.81);F6(820.01uV/z=5.1);AF8(1414.07uV/z=9.72);AFF1h(4421.65uV/z=33.14);AFF6h(830.09uV/z=5.18);AFp2(7801.35uV/z=59.45)`), `task_median_std = 161.56 µV`, `max_ptp = 19274.0 µV` (`Fp2`).
  - `sub-17`: `8` kênh noisy (`Fz(1573.65uV/z=7.39);CP1(1226.05uV/z=5.47);AF3(1767.99uV/z=8.46);PO7(1264.23uV/z=5.68);AF4(1370.57uV/z=6.27);O9(1672.49uV/z=7.94);FFC6h(1166.58uV/z=5.14);AFp1(1470.98uV/z=6.82)`), `task_median_std = 227.23 µV`, `max_ptp = 6323.0 µV` (`AF3`).
  - `sub-37`: `6` kênh noisy (`TP10 RIGHT EAR(1593.52uV/z=5.46);FFC1h(1730.39uV/z=6.03);CCP1h(1527.67uV/z=5.18);P10(2182.02uV/z=7.93);FCC1h(2075.54uV/z=7.48);FCC2h(3267.98uV/z=12.48)`), `task_median_std = 283.67 µV`, `max_ptp = 8972.3 µV` (`P10`).
  - `sub-24`: `6` kênh noisy (`CP1(2425.39uV/z=6.4);C1(2886.81uV/z=7.84);F9(4939.29uV/z=14.27);FCC3h(4811.42uV/z=13.87);FTT9h(2362.08uV/z=6.2);CCP3h(7489.7uV/z=22.25)`), `task_median_std = 354.79 µV`, `max_ptp = 25154.4 µV` (`FTT9h`).
  - `sub-11`: `6` kênh noisy (`CP2(1291.29uV/z=5.18);FC3(1257.73uV/z=5.01);PO4(1876.89uV/z=8.08);CPP4h(2479.71uV/z=11.06);FCC5h(1632.55uV/z=6.87);FTT9h(4412.87uV/z=20.63)`), `task_median_std = 232.56 µV`, `max_ptp = 19727.4 µV` (`CPP4h`).
  - `sub-38`: `6` kênh noisy (`C3(2139.92uV/z=6.99);CP1(4272.98uV/z=15.56);AFF1h(1702.92uV/z=5.23);OI1h(5075.69uV/z=18.79);CPP1h(2802.38uV/z=9.65);POO1(2067.6uV/z=6.7)`), `task_median_std = 389.63 µV`, `max_ptp = 21693.7 µV` (`OI1h`).
  - `sub-27`: `5` kênh noisy (`F8(1413.58uV/z=7.1);FFC1h(1764.45uV/z=9.2);FFC5h(1790.29uV/z=9.35);OI2h(1762.77uV/z=9.19);FFC4h(1119.11uV/z=5.33)`), `task_median_std = 210.47 µV`, `max_ptp = 6447.3 µV` (`F8`).
  - `sub-20`: `5` kênh noisy (`P4(498.18uV/z=5.56);CP2(572.44uV/z=6.78);CPz(879.12uV/z=11.84);FCC1h(489.35uV/z=5.41);FFC4h(494.87uV/z=5.51)`), `task_median_std = 158.56 µV`, `max_ptp = 4410.8 µV` (`Fp1`).
  - `sub-19`: `5` kênh noisy (`P3(1198.22uV/z=5.19);F1(1454.05uV/z=6.63);AFF1h(1689.65uV/z=7.95);FFC3h(2076.72uV/z=10.13);CPP2h(3845.52uV/z=20.07)`), `task_median_std = 268.88 µV`, `max_ptp = 13011.0 µV` (`CPP2h`).
  - `sub-36`: `4` kênh noisy (`P3(1709.7uV/z=6.76);AF3(1835.07uV/z=7.42);FCC3h(1967.9uV/z=8.13);CCP5h(1907.75uV/z=7.81)`), `task_median_std = 400.53 µV`, `max_ptp = 7802.9 µV` (`AF3`).

## 4. Audit Events, Triggers & Lọc Hành vi

### 4.1. Cấu trúc Cột & Giá trị Unique trong `*_events.tsv` và `.vmrk`
- **Các cột trong tất cả 42 file `_events.tsv`**: `onset` (giây), `duration` (toàn bộ là `n/a`), `event_type`, `trial_type`. **Không có sẵn cột `correct` hay `reaction_time`** — bắt buộc phải tự tính từ cặp `Stimulus -> Response (S  1)`!
- **Các giá trị `event_type` xuất hiện toàn tập**:
  - Chuẩn Oddball: `S  5` (Frequent Standard), `S  6` (Target `þ`), `S  7` (Distractor `Þ`), `S  1` (Response), `S 10` (Break), `S 12` (Beginning of run).
  - Bất thường trong `_events.tsv`: `empty` (tại `onset=0.001s` ở `sub-37`, `sub-38`), và `R 15`, `S 14`, `S 15` (tại `onset=1346.74–1347.03s` ở cuối bản ghi `sub-13`).
  - **Marker ẩn trong `.vmrk` nhưng KHÔNG có trong `_events.tsv`**:
    - **5 subjects (`sub-15, sub-23, sub-24, sub-37, sub-38`)** có marker `Stimulus,boundary` và `New Segment` giữa bản ghi trong `.vmrk` (do tạm dừng/ghép nối). Ngoài ra `sub-13`, `sub-41`, `sub-42` có `1` dòng `Mk2=Stimulus,,1,0,0,0` (mô tả rỗng tại mẫu `1`).
    - Sai lệch thời điểm `onset` giữa `_events.tsv` và `.vmrk`: tối đa `1.000 ms` trên toàn tập (chênh đúng `1 sample = 1.0 ms` do file BIDS `_events.tsv` quy đổi trực tiếp số thứ tự mẫu 1-based chia `1000`, trong khi MNE dùng chỉ số 0-based).

### 4.2. Quy tắc Lọc Hành vi (Behavioral Filtering) & Thống kê Toàn Dataset
Áp dụng tiêu chuẩn ghép cặp nối tiếp trong mỗi block (không ghép xuyên qua stimulus kế tiếp, `S 10`, `S 12` hoặc `boundary`):
- **Frequent (`S  5`) hợp lệ (Correct Rejection - CR)**: Không có `S  1` giữa thời điểm kích thích và kích thích/ngắt quãng tiếp theo.
- **Target (`S  6`) hợp lệ (Hit)**: Có `S  1` đầu tiên sau kích thích với `100 ms <= RT <= 1500 ms` trước kích thích/ngắt quãng tiếp theo.
- **Loại bỏ**: Toàn bộ Distractor (`S  7`), toàn bộ Target Miss (`no response` / `RT > 1500 ms`), Target Anticipation (`RT < 100 ms`), và Frequent False Alarm (bấm nhầm nút khi thấy `S  5`).

| Tập đối tượng | Frequent Raw (`S  5`) | Target Raw (`S  6`) | Distractor Raw (`S  7`) | Frequent Clean (CR) | Target Clean (Hit) | Tổng Epoch Sạch | Tỷ lệ `Frequent : Target` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Toàn bộ 42 subjects** | 21,018 | 3,422 | 3,425 | **20,935** | **3,245** | **24,180** | **6.45 : 1** |
| **39 subjects sạch hành vi** *(loại `sub-02, 14, 17`)* | 19,524 | 3,179 | 3,182 | **19,475** | **3,002** | **22,477** | **6.49 : 1** |

### 4.3. Phát hiện Các Subject Bất thường về Số Trial & Hành vi
- **Subject hiểu sai luật bấm nhầm Distractor (`S  7`) cực cao**:
  - **`sub-02`**: `FA_Dist = 98.77%` (`80/81` lần bấm cho Distractor!), `Hit Rate = 100.00%`, `FA_Freq = 3.21%`. -> **Cần loại bỏ** vì đối tượng phản ứng với cả `Þ` và `þ` (không còn là phân biệt Target thuần túy; đồng thời tạo nhiễu nhãn nghiêm trọng cho P3b vs P3a/Novelty P3).
  - **`sub-14`**: `FA_Dist = 97.53%` (`79/81` lần bấm cho Distractor!), `Hit Rate = 100.00%`, `FA_Freq = 1.61%`. -> **Cần loại bỏ** vì đối tượng phản ứng với cả `Þ` và `þ` (không còn là phân biệt Target thuần túy; đồng thời tạo nhiễu nhãn nghiêm trọng cho P3b vs P3a/Novelty P3).
  - **`sub-17`**: `FA_Dist = 76.54%` (`62/81` lần bấm cho Distractor!), `Hit Rate = 100.00%`, `FA_Freq = 2.01%`. -> **Cần loại bỏ** vì đối tượng phản ứng với cả `Þ` và `þ` (không còn là phân biệt Target thuần túy; đồng thời tạo nhiễu nhãn nghiêm trọng cho P3b vs P3a/Novelty P3).
- **Subject có tổng số stimuli khác mức chuẩn 660**:
  - `sub-23`: Tổng `692` stimuli (`S5=517, S6=86, S7=89`), boundary trong `.vmrk` = `1`.
  - `sub-24`: Tổng `659` stimuli (`S5=497, S6=81, S7=81`), boundary trong `.vmrk` = `1`.
  - `sub-37`: Tổng `659` stimuli (`S5=497, S6=81, S7=81`), boundary trong `.vmrk` = `2`.
  - `sub-38`: Tổng `775` stimuli (`S5=583, S6=96, S7=96`), boundary trong `.vmrk` = `2`.
- **Cảnh báo `8` subjects có Hit Rate Target < 90% (hoặc bấm nhầm Distractor mức trung bình 25–40%)**:
  - `sub-07`: `Hit Rate = 60.49%` (`49/81` Target sạch, Miss=`32`, `FA_Dist = 29.63%`, `Imbalance = 10.06:1`).
  - `sub-01`: `Hit Rate = 76.54%` (`62/81` Target sạch, Miss=`19`, `FA_Dist = 37.04%`, `Imbalance = 7.81:1`).
  - `sub-25`: `Hit Rate = 77.78%` (`63/81` Target sạch, Miss=`18`, `FA_Dist = 7.41%`, `Imbalance = 7.90:1`).
  - `sub-10`: `Hit Rate = 82.72%` (`67/81` Target sạch, Miss=`14`, `FA_Dist = 39.51%`, `Imbalance = 7.37:1`).
  - `sub-30`: `Hit Rate = 83.95%` (`68/81` Target sạch, Miss=`13`, `FA_Dist = 1.23%`, `Imbalance = 7.32:1`).
  - `sub-08`: `Hit Rate = 85.19%` (`69/81` Target sạch, Miss=`12`, `FA_Dist = 14.81%`, `Imbalance = 7.19:1`).
  - `sub-09`: `Hit Rate = 87.65%` (`71/81` Target sạch, Miss=`10`, `FA_Dist = 8.64%`, `Imbalance = 7.01:1`).
  - `sub-29`: `Hit Rate = 88.89%` (`72/81` Target sạch, Miss=`9`, `FA_Dist = 24.69%`, `Imbalance = 6.89:1`).
- **Tỷ lệ mất cân bằng lớp (`Frequent : Target`) từng subject**: dao động từ `6.06:1` (`sub-38`) đến `10.06:1` (`sub-07`) trên 39 subjects sạch (trung bình `6.55 ± 0.75 : 1`).

## 5. Audit Reaction Time (RT) & Nguy cơ Nhiễu cho Bài toán XAI (RQ3)

### 5.1. Thống kê Phân bố Reaction Time của Target Hit
Tính trên tất cả **`3,002` trials Target Hit** của 39 subjects sạch hành vi (và `3,245` trials trên toàn 42 subjects):

| Chỉ số RT (Target Hit) | 39 Subjects Sạch Hành vi | Toàn bộ 42 Subjects |
|---|---:|---:|
| Mean ± Std (ms) | **540.55 ± 119.91 ms** | 533.61 ± 120.59 ms |
| Median [Q25–Q75] (ms) | **525.00 [458.0–599.0] ms** | 518.00 [449.0–594.0] ms |
| Min – Max (ms) | `310.0 – 1449.0 ms` | `208.0 – 1449.0 ms` |
| RT trong `[100, 300) ms` | `0.00%` | `0.12%` |
| **RT rơi vào cửa sổ P300 `[300, 600] ms`** | **`75.25%`** | **`76.58%`** |
| RT `> 600 ms` | `24.75%` | `23.30%` |

### 5.2. Đánh giá Nguy cơ Nhiễu Vận động (Motor Confound) & Nhiễu Chớp mắt (EOG) cho XAI
1. **Nguy cơ Motor Confound cực lớn**: Có tới **`75.25%`** các trial Target có thời điểm bấm phím rơi trực tiếp vào cửa sổ P300 (`300–600 ms`, trung vị `525.0 ms`). Vì người tham gia bấm phím **`2`** bằng **ngón giữa tay phải** chỉ ở lớp `Target` (không bấm ở lớp `Frequent`), tín hiệu vùng vỏ não vận động sơ cấp bán cầu trái (`C3, CP3, FCC3h, CCP3h, CCP1h, FC3`) sẽ mang xung chuẩn bị & thực thi vận động (Motor Potential / Lateralized Readiness Potential) chồng lấp về mặt thời gian với sóng P300 (`Pz, CPz, Cz, P1, P2, CP1, CP2`).
   - **Khuyến nghị kiểm chứng XAI (RQ3)**: Khi trực quan hóa Topomap attribution (Saliency / Integrated Gradients / SHAP / DeepLIFT) trong cửa sổ `300–600 ms`, cần thực hiện 2 phép kiểm chứng bắt buộc:
     - *(a) Kiểm tra tính đối xứng đỉnh vs lệch trái*: Sóng P300 chuẩn có phân bố đỉnh-trung tâm đối xứng quanh `Pz/CPz/Cz`; nếu heatmap XAI sáng rực bất thường tại `C3/CP3/FCC3h` gần thời điểm `RT`, mô hình đang khai thác **Motor Confound** thay vì đáp ứng nhận thức P300.
     - *(b) Kiểm chứng theo phân vị RT (RT-stratified ERP & XAI)*: Tách các trial Target thành nhóm `Fast RT (< 450 ms)` vs `Slow RT (> 550 ms)` hoặc thử cắt cửa sổ phân loại sớm `[-0.2, 0.45] s` để xem đặc trưng quan trọng dịch chuyển theo RT hay giữ nguyên ở pha P300.
2. **Kiểm chứng nhiễu chớp mắt (EOG)**: Dataset **KHÔNG có kênh EOG chuyên dụng**, nhưng có đầy đủ hệ thống kênh trán cực độ phân giải cao: **`Fp1, Fp2, AF7, AF3, AF8, AF4, AFp1, AFp2`**. Nhóm nên:
   - Dùng cặp `Fp1, Fp2` (hoặc `AFp1, AFp2`) làm kênh EOG surrogate trong `ica.find_bads_eog()` (lưu ý chỉ dùng để tính điểm tương quan EOG, không xóa bỏ kênh khỏi tensor sau khi chạy `ica.apply()`).
   - Trong XAI, kiểm tra tỷ trọng attribution tại cụm kênh trán (`Fp1`, `Fp2`, `AFp1`, `AFp2`, `AF7`, `AF8`) trên cả dữ liệu *có ICA* vs *không ICA* (`X_no_ica_no_reject`) để chứng minh mô hình không ăn gian bằng phản xạ chớp mắt sau kích thích Target.

## 6. Ước lượng Tài nguyên & Đề xuất Data Contract cho Nhóm 5 Người

### 6.1. Ước lượng Dung lượng Bộ nhớ Tập Epoch Sạch (Chưa tính tỷ lệ loại biên độ ICA/Artifact)
Dựa trên tổng số epoch sạch hành vi thực tế (`22,477` trials cho **39 subjects sạch**, và `24,180` trials nếu giữ cả 42 subjects):

| Tần số lấy mẫu | Cửa sổ Epoch | Số mẫu/epoch | 39 Subs (127 ch, `float32`) | 39 Subs (127 ch, `float64`) | 39 Subs (128 ch +FCz, `float32`) | 42 Subs (127 ch, `float32`) |
|---:|---|---:|---:|---:|---:|---:|
| **1000 Hz** | `[-0.2, 0.8]s` | 1001 | **10.645 GB** | 21.290 GB | 10.729 GB | 11.451 GB |
| **1000 Hz** | `[-0.2, 1.0]s` | 1201 | **12.772 GB** | 25.543 GB | 12.872 GB | 13.739 GB |
| **250 Hz** | `[-0.2, 0.8]s` | 251 | **2.669 GB** | 5.338 GB | 2.690 GB | 2.871 GB |
| **250 Hz** | `[-0.2, 1.0]s` | 301 | **3.201 GB** | 6.402 GB | 3.226 GB | 3.443 GB |
| **128 Hz** | `[-0.2, 0.8]s` | 129 | **1.372 GB** | 2.744 GB | 1.383 GB | 1.476 GB |
| **128 Hz** | `[-0.2, 1.0]s` | 155 | **1.648 GB** | 3.297 GB | 1.661 GB | 1.773 GB |

> [!TIP]
> **Khuyến nghị cấu hình chuẩn**: Dùng `sfreq = 250 Hz`, cửa sổ `[-0.2, 0.8] s` (`251` mẫu/epoch), lưu kiểu `np.float32` (**`2.669 GB`** cho `22,477` epochs của 39 subjects sạch hành vi) — vừa giữ trọn băng thông P300 (`0.1–30/40 Hz`, Nyquist `125 Hz`), vừa đủ nhẹ để train EEGNet/ShallowFBCSPNet và chạy PyTorch Captum / SHAP trên GPU/RAM phổ thông 16 GB.

### 6.2. Data Contract Chính thức cho Nhóm 5 Người

#### A. Quy tắc Chọn / Loại Subject (Cohort Policy)
- **Tier 1 — Loại BẮT BUỘC (`3` subjects)**: **`sub-02`, `sub-14`, `sub-17`** (lỗi hành vi hiểu nhầm quy tắc, bấm nút cho Distractor `76.5%–98.8%`). Tập chuẩn mặc định của toàn nhóm là **39 subjects** (`total_clean_binary = 22,477` epochs sạch hành vi trước bước loại biên độ).
- **Tier 2 — Cờ cảnh báo chất lượng & Ablation Sensitivity (Giữ mặc định ở tập 39, kiểm chứng độ nhạy khi chạy mô hình)**:
  - **`sub-08` & `sub-35`**: BẮT BUỘC gọi **`raw.crop(tmin=max(0, first_stim - 5.0), tmax=min(raw.times[-1], last_stim + 5.0))` TRƯỚC KHI chạy `filter()` và `ICA`** để loại bỏ đoạn tháo mũ / bão hòa ở cuối file (`sub-08` tại `1446.21s`).
  - **`sub-07` (`Hit=60.5%`, `49` Targets), `sub-01` (`Hit=76.5%`, `FA_Dist=37.0%`), `sub-10` (`Hit=82.7%`, `FA_Dist=39.5%`), `sub-25` (`Hit=77.8%`)**: Nhóm có độ tập trung thấp hoặc bấm nhầm Distractor mức trung bình — nên giữ trong báo cáo chính (39 subjects) nhưng chạy thêm một nhánh kiểm chứng độ nhạy (*Sensitivity Analysis*) nếu kết quả Cross-subject biến động mạnh ở các subject này.
  - **`sub-15`, `sub-23`, `sub-24`, `sub-37`, `sub-38`**: Có đoạn ghép nối `boundary` trong `.vmrk` (`sub-23` có `692` stimuli, `sub-38` có `775` stimuli). Cần đọc `.vhdr/.vmrk` qua MNE (MNE tự tạo annotation `BAD boundary` / `EDGE boundary` từ `.vmrk`) để tự động drop các epoch chạm ranh giới ghép nối.
  - **`sub-32` & các subject nhiều kênh nhiễu (`sub-17, 18, 11, 24, 37, 38`)**: Theo dõi tỷ lệ kênh nội suy và tỷ lệ giữ epoch sau ICA/reject; nếu một subject mất `> 30%` epoch hoặc có `> 15%` kênh xấu sau lọc dải tần `0.5–40 Hz` mới cân nhắc loại ở bước Preprocessing QC.
  - **`sub-13`**: Có 3 marker kỹ thuật lạ ở cuối file (`R 15`, `S 14`, `S 15` tại `1346.7s`) — không ảnh hưởng tới bất kỳ stimulus Oddball nào nếu chỉ trích xuất `S  5`, `S  6`.

#### B. Quy tắc Cắt gọn, Chuẩn hóa Kênh, Montage & Đồng bộ Thứ tự (Code Mẫu)
```python
# 0. QUAN TRỌNG: Cắt gọn về cửa sổ task thực tế TRƯỚC khi Filter & ICA (cứu sub-08, sub-35)
ev_df = pd.read_csv(events_tsv_path, sep='\t')
stim_onsets = ev_df.loc[ev_df['event_type'].isin(['S  5', 'S  6', 'S  7']), 'onset'].astype(float)
raw.crop(tmin=max(0.0, stim_onsets.min() - 5.0), tmax=min(raw.times[-1], stim_onsets.max() + 5.0))

# 1. Chuẩn hóa tên kênh đầy đủ (khắc phục lỗi thiếu O9/O10 trong colin27_1005)
CHANNEL_ALIASES = {
    'TP9 LEFT EAR': 'TP9',
    'TP10 RIGHT EAR': 'TP10',
    'O9': 'I1',
    'O10': 'I2',
}
raw.rename_channels({k: v for k, v in CHANNEL_ALIASES.items() if k in raw.ch_names})

# 2. Gán montage chuẩn (trên MNE >= 1.14 dùng 'colin27_1005'; bản cũ hỗ trợ 'standard_1005')
montage = mne.channels.make_standard_montage('colin27_1005')
raw.set_montage(montage, on_missing='raise')  # Đảm bảo 127/127 kênh đều có tọa độ 3D

# 3. Luôn chuẩn hóa thứ tự kênh về canonical_order (của sub-01 sau alias) trước khi xuất Tensor
raw.reorder_channels(CANONICAL_127_CHANNELS)
```

#### C. Quy tắc Lọc Event Hành vi Chính xác (Code Mẫu Copy-Paste)
```python
def extract_behaviorally_clean_events(events_tsv_path: Path, sfreq: float = 1000.0,
                                      rt_min: float = 0.100, rt_max: float = 1.500):
    """Trả về mảng MNE events (N, 3) chỉ gồm Frequent CR (0) và Target Hit (1)."""
    df = pd.read_csv(events_tsv_path, sep='\t')
    stim_codes = {'S  5', 'S  6', 'S  7'}
    break_codes = {'S 10', 'S 12'}
    records = list(zip(df['onset'].astype(float), df['event_type'].astype(str)))
    stim_idx = [i for i, (_, c) in enumerate(records) if c in stim_codes]
    clean_events, rts_ms = [], []
    for k, i in enumerate(stim_idx):
        t_stim, code = records[i]
        j_end = stim_idx[k + 1] if k + 1 < len(stim_idx) else len(records)
        resps = []
        for j in range(i + 1, j_end):
            t_j, c_j = records[j]
            if c_j in break_codes:
                break
            if c_j == 'S  1':
                resps.append(t_j - t_stim)
        sample = int(round(t_stim * sfreq))
        if code == 'S  5' and not any(r <= rt_max for r in resps):
            clean_events.append([sample, 0, 0])   # Label 0: Frequent CR
        elif code == 'S  6' and resps and (rt_min <= resps[0] <= rt_max):
            clean_events.append([sample, 0, 1])   # Label 1: Target Hit
            rts_ms.append(resps[0] * 1000.0)
    return np.asarray(clean_events, dtype=np.int64), np.asarray(rts_ms, dtype=np.float64)
```

#### D. Giao thức Chia Tập Chuẩn (Validation Splits) & Chống Data Leakage
1. **Kịch bản 1 — Within-Subject Classification**:
   - Chạy riêng trên từng subject với **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** (đảm bảo tỷ lệ ~6:1 giữa Frequent/Target ở mỗi fold; chú ý `sub-XX` không xáo trộn rò rỉ giữa các fold vì các epoch đã tách rời, nhưng có thể cân nhắc `StratifiedGroupKFold` theo block nếu muốn kiểm tra drift theo thời gian).
   - **Chống Leakage**: Mọi tham số chuẩn hóa (`StandardScaler`, `CSP`, `xDAWN`, `PCA`, `class_weight`) **chỉ được `.fit()` trên tập Train của từng fold**, tuyệt đối không fit trên toàn bộ dữ liệu trước khi chia fold.
2. **Kịch bản 2 — Cross-Subject Classification (Tổng quát hóa sang người mới)**:
   - Sử dụng **`StratifiedGroupKFold(n_splits=5)`** với `groups=subject_id` (hoặc **LOSO - `LeaveOneGroupOut()`** cho 39 folds để đánh giá chuẩn cuối cùng).
   - **Cấm tuyệt đối**: Chia random toàn bộ `22,477` trials mà không nhóm theo `subject_id` (sẽ gây rò rỉ chữ ký sinh học từng cá nhân giữa Train và Test, đẩy AUC lên ảo).
   - **Metric đánh giá bắt buộc**: Vì mất cân bằng lớp `6.49 : 1`, KHÔNG dùng Accuracy thô làm metric chính; bắt buộc báo cáo **`Balanced Accuracy`**, **`ROC-AUC`**, **`PR-AUC`**, và **`F1-score (Target class)`**.

## 7. Bảng Tổng hợp Toàn bộ 42 Subjects (Trích yếu)
Chi tiết đầy đủ 65 cột cho từng subject được lưu tại [`reports/audit_per_subject.csv`](audit_per_subject.csv).

| Subject | Size (MB) | Phút | Kênh lỗi (Flat/Noisy) | S5 Raw | S6 Raw | S7 Raw | S5 Clean | S6 Clean | Ratio (F:T) | Hit (%) | FA_Dist (%) | RT Mean (ms) | RT in P300 (%) | Trạng thái / Ghi chú |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sub-01 | 664.4 | 22.86 | 0/0 | 498 | 81 | 81 | 484 | 62 | 7.81:1 | 76.5% | 37.0% | 571.5 | 64.5% | LOW_HIT_RATE(76.5%) |
| sub-02 | 652.1 | 22.43 | 0/4 | 498 | 81 | 81 | 482 | 81 | 5.95:1 | 100.0% | 98.8% | 385.2 | 95.1% | EXCLUDE_BEHAVIOR_FA_DIST; NOISY_CHS(4) |
| sub-03 | 644.0 | 22.15 | 0/0 | 498 | 81 | 81 | 497 | 81 | 6.14:1 | 100.0% | 12.3% | 507.7 | 91.4% | OK |
| sub-04 | 642.5 | 22.10 | 0/4 | 498 | 81 | 81 | 498 | 79 | 6.30:1 | 97.5% | 6.2% | 552.9 | 78.5% | NOISY_CHS(4) |
| sub-05 | 640.9 | 22.05 | 0/0 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 9.9% | 428.6 | 95.1% | OK |
| sub-06 | 644.3 | 22.16 | 0/4 | 498 | 81 | 81 | 497 | 78 | 6.37:1 | 96.3% | 3.7% | 579.6 | 66.7% | NOISY_CHS(4) |
| sub-07 | 650.3 | 22.37 | 0/3 | 498 | 81 | 81 | 493 | 49 | 10.06:1 | 60.5% | 29.6% | 686.4 | 42.9% | LOW_HIT_RATE(60.5%); NOISY_CHS(3) |
| sub-08 | 821.1 | 28.25 | 0/0 | 498 | 81 | 81 | 496 | 69 | 7.19:1 | 85.2% | 14.8% | 561.3 | 71.0% | LOW_HIT_RATE(85.2%); TAIL_SATURATION_CAP_REMOVAL |
| sub-09 | 642.5 | 22.10 | 0/1 | 498 | 81 | 81 | 498 | 71 | 7.01:1 | 87.7% | 8.6% | 592.2 | 60.6% | LOW_HIT_RATE(87.7%); NOISY_CHS(1) |
| sub-10 | 652.2 | 22.43 | 0/1 | 498 | 81 | 81 | 494 | 67 | 7.37:1 | 82.7% | 39.5% | 545.8 | 77.6% | LOW_HIT_RATE(82.7%); NOISY_CHS(1) |
| sub-11 | 646.9 | 22.25 | 0/6 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 1.2% | 417.6 | 97.5% | HIGH_NOISY_CHS(6) |
| sub-12 | 639.2 | 21.99 | 0/1 | 498 | 81 | 81 | 497 | 80 | 6.21:1 | 98.8% | 6.2% | 541.7 | 82.5% | NOISY_CHS(1) |
| sub-13 | 661.1 | 22.74 | 0/1 | 498 | 81 | 81 | 497 | 81 | 6.14:1 | 100.0% | 6.2% | 599.7 | 53.1% | REORDER_CHANNELS; NOISY_CHS(1); WEIRD_MARKERS |
| sub-14 | 647.9 | 22.29 | 0/1 | 498 | 81 | 81 | 490 | 81 | 6.05:1 | 100.0% | 97.5% | 477.9 | 92.6% | EXCLUDE_BEHAVIOR_FA_DIST; NOISY_CHS(1) |
| sub-15 | 678.6 | 23.34 | 0/4 | 498 | 81 | 81 | 498 | 80 | 6.22:1 | 98.8% | 9.9% | 502.2 | 82.5% | BOUNDARY(1); NOISY_CHS(4) |
| sub-16 | 645.1 | 22.19 | 0/3 | 498 | 81 | 81 | 497 | 80 | 6.21:1 | 98.8% | 3.7% | 567.6 | 76.2% | NOISY_CHS(3) |
| sub-17 | 646.0 | 22.22 | 0/8 | 498 | 81 | 81 | 488 | 81 | 6.02:1 | 100.0% | 76.5% | 480.6 | 91.4% | EXCLUDE_BEHAVIOR_FA_DIST; HIGH_NOISY_CHS(8) |
| sub-18 | 652.0 | 22.43 | 0/8 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 4.9% | 600.0 | 54.3% | HIGH_NOISY_CHS(8) |
| sub-19 | 638.2 | 21.95 | 0/5 | 498 | 81 | 81 | 498 | 75 | 6.64:1 | 92.6% | 2.5% | 562.6 | 69.3% | HIGH_NOISY_CHS(5) |
| sub-20 | 649.5 | 22.34 | 0/5 | 498 | 81 | 81 | 498 | 80 | 6.22:1 | 98.8% | 3.7% | 626.1 | 48.8% | HIGH_NOISY_CHS(5) |
| sub-21 | 651.5 | 22.41 | 0/3 | 498 | 81 | 81 | 498 | 79 | 6.30:1 | 97.5% | 1.2% | 603.3 | 62.0% | NOISY_CHS(3) |
| sub-22 | 680.0 | 23.39 | 0/2 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 1.2% | 585.9 | 63.0% | NOISY_CHS(2) |
| sub-23 | 687.9 | 23.66 | 0/3 | 517 | 86 | 89 | 517 | 82 | 6.30:1 | 95.3% | 6.7% | 438.6 | 96.3% | BOUNDARY(1); STIM_COUNT_692; NOISY_CHS(3) |
| sub-24 | 653.9 | 22.49 | 0/6 | 497 | 81 | 81 | 495 | 74 | 6.69:1 | 91.4% | 17.3% | 548.0 | 73.0% | BOUNDARY(1); STIM_COUNT_659; HIGH_NOISY_CHS(6) |
| sub-25 | 734.1 | 25.25 | 0/2 | 498 | 81 | 81 | 498 | 63 | 7.90:1 | 77.8% | 7.4% | 491.4 | 93.7% | LOW_HIT_RATE(77.8%); NOISY_CHS(2) |
| sub-26 | 645.2 | 22.20 | 0/2 | 498 | 81 | 81 | 497 | 77 | 6.45:1 | 95.1% | 7.4% | 499.4 | 85.7% | NOISY_CHS(2) |
| sub-27 | 662.8 | 22.80 | 0/5 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 3.7% | 463.7 | 95.1% | HIGH_NOISY_CHS(5) |
| sub-28 | 650.5 | 22.38 | 0/4 | 498 | 81 | 81 | 496 | 81 | 6.12:1 | 100.0% | 1.2% | 563.9 | 77.8% | NOISY_CHS(4) |
| sub-29 | 645.6 | 22.21 | 0/0 | 498 | 81 | 81 | 496 | 72 | 6.89:1 | 88.9% | 24.7% | 492.9 | 83.3% | LOW_HIT_RATE(88.9%) |
| sub-30 | 654.2 | 22.50 | 0/0 | 498 | 81 | 81 | 498 | 68 | 7.32:1 | 84.0% | 1.2% | 631.2 | 41.2% | LOW_HIT_RATE(84.0%) |
| sub-31 | 643.8 | 22.15 | 0/0 | 498 | 81 | 81 | 497 | 79 | 6.29:1 | 97.5% | 9.9% | 496.3 | 92.4% | OK |
| sub-32 | 660.1 | 22.71 | 0/0 | 498 | 81 | 81 | 498 | 79 | 6.30:1 | 97.5% | 6.2% | 650.6 | 39.2% | OK |
| sub-33 | 639.8 | 22.01 | 0/0 | 498 | 81 | 81 | 497 | 81 | 6.14:1 | 100.0% | 6.2% | 491.1 | 92.6% | OK |
| sub-34 | 652.8 | 22.45 | 0/3 | 498 | 81 | 81 | 497 | 79 | 6.29:1 | 97.5% | 8.6% | 529.1 | 81.0% | NOISY_CHS(3) |
| sub-35 | 672.8 | 23.14 | 0/1 | 498 | 81 | 81 | 497 | 80 | 6.21:1 | 98.8% | 18.5% | 446.6 | 96.2% | NOISY_CHS(1) |
| sub-36 | 690.9 | 23.77 | 0/4 | 498 | 81 | 81 | 496 | 79 | 6.28:1 | 97.5% | 1.2% | 521.3 | 86.1% | NOISY_CHS(4) |
| sub-37 | 677.1 | 23.29 | 0/6 | 497 | 81 | 81 | 497 | 75 | 6.63:1 | 92.6% | 9.9% | 489.2 | 92.0% | BOUNDARY(2); STIM_COUNT_659; HIGH_NOISY_CHS(6); WEIRD_MARKERS |
| sub-38 | 732.9 | 25.21 | 0/6 | 583 | 96 | 96 | 582 | 96 | 6.06:1 | 100.0% | 8.3% | 568.4 | 68.8% | BOUNDARY(2); STIM_COUNT_775; HIGH_NOISY_CHS(6); WEIRD_MARKERS |
| sub-39 | 646.4 | 22.24 | 0/1 | 498 | 81 | 81 | 498 | 81 | 6.15:1 | 100.0% | 2.5% | 580.3 | 64.2% | NOISY_CHS(1) |
| sub-40 | 704.2 | 24.22 | 0/4 | 498 | 81 | 81 | 493 | 81 | 6.09:1 | 100.0% | 14.8% | 475.2 | 90.1% | NOISY_CHS(4) |
| sub-41 | 655.5 | 22.55 | 0/2 | 498 | 81 | 81 | 498 | 79 | 6.30:1 | 97.5% | 12.3% | 664.0 | 35.4% | REORDER_CHANNELS; NOISY_CHS(2) |
| sub-42 | 640.3 | 22.03 | 0/1 | 498 | 81 | 81 | 498 | 80 | 6.22:1 | 98.8% | 4.9% | 480.1 | 96.2% | REORDER_CHANNELS; NOISY_CHS(1) |
