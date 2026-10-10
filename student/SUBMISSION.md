# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Bản nháp có hỗ trợ AI; học viên cần đọc, kiểm tra và tự giải thích trước khi nộp.
> Chưa sẵn sàng nộp: còn thiếu segment Waymo và kết quả chạy thật.
> Weights đã tải, nạp model và kiểm tra inference thành công.

## Thông tin học viên

- Họ tên: Phạm Minh Hiếu (suy ra từ tên repo; cần học viên xác nhận)
- MSSV: 2A202602919
- Email: hieup3503@gmail.com (theo cấu hình Git của repo)
- Link repo (fork): https://github.com/BenKei-19/K4-L2L3-DAY23-PhamMinhHieu-2A202602919-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Chờ commit cuối sau khi chạy đủ dữ liệu

## Tóm tắt kết quả

- Cấu hình dự kiến: `fusion_mode=compare`, `frames=[0,198]`, `seed=0`, segment `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`. Đây là cấu hình chuẩn bị, chưa phải kết quả thực nghiệm.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: Chưa có; chờ chạy Waymo.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa có; chờ chạy Waymo.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa có; chờ chạy Waymo.
- Giải thích khác biệt hai mode: Chưa thể kết luận khi chưa có artifacts. Khi chạy xong, so sánh RMSE cùng matches, ghost và miss; RMSE thấp trên ít cặp ghép chưa chứng minh tracker tốt hơn. Tính thêm `precision_track=matches/(matches+ghost_track_frames)`, `coverage=matches/det_tp` và độ chênh `rmse_fused-rmse_lidar`.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

Trên PowerShell, dùng môi trường `.venv` đã chuẩn bị:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest student/tests -q
.\.venv\Scripts\fusion-run-lab.exe --config student/config/paths.yaml --fusion compare --seed 0
.\.venv\Scripts\python.exe -X utf8 tools/check_submission.py
```

Trước khi chạy, tải dữ liệu theo [data/README.md](../data/README.md), đặt segment
vào `data/Waymo/` và weights vào
`data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth`.
`student/config/paths.yaml` đã được chuẩn bị cho frame 0–198 và được Git bỏ qua.

### Trạng thái chuẩn bị dữ liệu (10/10/2026)

- Đã truy cập thư mục Drive của lớp và tìm đúng segment mặc định. Khi tải,
  Google Drive trả lỗi: **Too many users have viewed or downloaded this file recently**.
  Chưa có segment trên máy để chạy CP5.
- Kho Waymo gốc mà [Udacity dẫn tới](https://github.com/udacity/nd013-c2-fusion-starter#waymo-open-dataset-files)
  trả HTTP 403 `AccessDenied` cho truy cập không xác thực; cần tài khoản có quyền.
- Đã tải weights từ [link gốc Udacity](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view),
  giải nén `fpn_resnet_18_epoch_300.pth` (50.984.463 byte) vào đúng thư mục local.
  Nạp state dict và inference với BEV rỗng `(1,3,608,608)` thành công, trả 0 detection.
  Đây chỉ là kiểm tra kỹ thuật trên đầu vào tổng hợp, không phải kết quả Waymo.
- Khi giới hạn Drive được gỡ hoặc có bản segment do giảng viên cung cấp, đặt
  file đúng tên vào `data/Waymo/`, rồi chạy lệnh compare ở trên. Không cần tải lại weights.
- Đã thử lệnh `--fusion compare --seed 0`: runner nạp model thành công và dừng
  tại `WaymoDataFileReader` với `FileNotFoundError` của segment mặc định.
  Lần thử này không tạo metrics/log chấm điểm.

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR đo vị trí `(x,y,z)` trong hệ cảm biến nên `z` có kích thước 3×1,
   `R` là 3×3 theo đơn vị m² và `H` là 3×6. Camera đo pixel `(u,v)` nên `z`
   là 2×1, `R=diag(sigma_cam_i²,sigma_cam_j²)` theo pixel² và `H` là 2×6.
   LiDAR dùng phép đổi tọa độ tuyến tính; camera dùng phép chiếu pinhole phi tuyến
   và Jacobian tại trạng thái hiện tại. Cả hai dùng `gamma=z-h(x)`,
   `S=H P Hᵀ+R`, `K=P Hᵀ S⁻¹` trong `workspace/kalman.py`.
   Code giải hệ tuyến tính để tính K và dùng dạng Joseph để cập nhật covariance.

2. Vì sao cần gating Mahalanobis trước khi gán?

   Gating loại các cặp đo–track không phù hợp trước khi gán greedy, giúp tránh
   cập nhật track bằng đo của xe khác. `d²=gammaᵀ S⁻¹ gamma` chuẩn hóa residual
   theo độ bất định; cùng sai lệch vị trí nhưng P lớn thì d² thường nhỏ hơn.
   Ngưỡng là `chi2.ppf(gating_threshold, dim_meas)`, tương ứng 3 chiều LiDAR
   hoặc 2 chiều camera. `association_cost_matrix` kiểm tra FOV trước khi tính
   khoảng cách, để không chiếu camera tại điểm ngoài vùng hợp lệ.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Đây là track-then-fuse: giữ một danh sách track cho mỗi mode, predict một lần
   mỗi frame, gán và update LiDAR, quản lý vòng đời, rồi gán và update camera
   trong mode fused. Thứ tự này nằm trong `platform/fusion_lab/scripts/run_lab.py`,
   hàm `run`; `associate_and_update` không gọi predict.
   Chưa có log thực nghiệm để trích dẫn. Khi có `grade_run.log`, đối chiếu record
   lidar/fused cùng frame để so sánh kết quả; log tổng hợp không ghi từng lần
   predict/update nên cần dẫn code để xác minh thứ tự xử lý.
   Lab giả định hai cảm biến trong một Waymo Frame có cùng thời điểm logic,
   `Measurement.t=frame_index*dt`, nên không predict lại giữa hai lượt update.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Phép đổi tọa độ sai làm pixel dự đoán h(x) lệch có hệ thống so với z.
   Residual có thể giữ thiên lệch theo một hướng qua nhiều frame; d² tăng và
   nhiều đo bị gating loại. Nếu sai lệch vẫn lọt cổng, update có thể kéo trạng thái
   track khỏi vị trí đúng, tăng sai số fused hoặc gây gán nhầm. Đây là phân tích
   lý thuyết; chưa chạy thí nghiệm lệch calibration để khẳng định số liệu.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi danh sách đo rỗng, không thể lấy sensor từ phần tử đầu. Sensor tường minh
   cho phép vẫn gọi `manager.manage_tracks` để lượt LiDAR trừ score cho track
   bị miss trong FOV và xóa track khi cần. Lượt camera rỗng không tác động tồn tại.
   LiDAR có vị trí 3D để khởi tạo; camera chỉ có đo 2D để tinh chỉnh track đã có.
   `TrackManager` chỉ gọi helper score/init/delete cho LiDAR, tránh tăng score
   hai lần trong một frame hoặc xóa track khi không có quan sát camera.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Khởi tạo `score=1/window`, trạng thái `initialized`, vận tốc bằng 0;
   covariance vị trí là R xoay sang vehicle frame, covariance vận tốc lấy bình
   phương sigma_p44/55/66. Hit LiDAR cộng `1/window`, tối đa 1; hit của track
   chưa confirmed chuyển sang tentative nếu chưa vượt ngưỡng. Khi
   `score>confirmed_threshold` thì confirmed. Miss LiDAR trong FOV trừ
   `1/window`, giữ trạng thái confirmed. Xóa nếu Pxx hoặc Pyy vượt max_P,
   hoặc confirmed có score nhỏ hơn delete_threshold, hoặc chưa confirmed có
   score ≤ 0. Với cấu hình hiện tại: window=6, confirmed_threshold=0.8,
   delete_threshold=0.6, max_P=9 m². Các điều kiện biên dùng đúng dấu >, <, ≤
   như `workspace/track_management.py`; camera không xét xóa.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): OpenAI Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): Đọc yêu cầu, cài đặt toàn bộ hàm Part E–H, chuẩn bị môi trường/cấu hình, kiểm thử và soạn bản nháp giải thích trong báo cáo.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex đối chiếu công thức/API và chạy bộ test gốc với `PYTHONUTF8=1`: **128 passed**, không failed/xfailed/skipped (10/10/2026). `fusion-run-lab --help` chạy được, `git diff --check` không báo lỗi. Weights tải từ nguồn gốc, nạp model và inference tổng hợp thành công. Chưa có kết quả Waymo do thiếu segment. Học viên cần tự đọc lại code, xác nhận câu trả lời và hiểu từng hàm trước khi nộp.

Lệnh kiểm thử thực tế: `$env:PYTHONUTF8='1'; .venv/Scripts/python.exe -m pytest student/tests -q --basetemp=.venv/pytest_tmp`.
UTF-8 cần được bật trên Windows vì test kiểm tra bài nộp tạo file có tiếng Việt.
Các cảnh báo hiện có liên quan `np.matrix` theo API đề và các ca cố ý dùng tọa độ
không hợp lệ; không có test nào bị bỏ qua.
Code E–H được lưu thành bốn commit CP1–CP4 sau khi kiểm thử, trong phiên làm việc
có hỗ trợ AI; lịch sử này không phải bằng chứng học viên tự làm trong giờ lab.
CP5 chưa hoàn tất vì segment bị giới hạn tải. Báo cáo hiện là bản nháp chuẩn bị,
chưa có commit nộp cuối cùng hoặc artifacts để xác nhận sẵn sàng nộp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên code được cung cấp
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
