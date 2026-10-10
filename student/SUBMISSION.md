# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Bài làm có hỗ trợ AI; học viên cần đọc, kiểm tra và tự giải thích trước khi nộp.
> Đã chạy Waymo compare trên frame 0–198, seed 0; số liệu lấy từ artifacts do runner sinh ra.

## Thông tin học viên

- Họ tên: Phạm Minh Hiếu
- MSSV: 2A202602919
- Email: hieup3503@gmail.com (theo cấu hình Git của repo)
- Link repo (fork): https://github.com/BenKei-19/K4-L2L3-DAY23-PhamMinhHieu-2A202602919-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Lấy commit cuối bằng `git rev-parse HEAD` sau khi commit/push; dùng hash đó trên LMS.

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: `[0, 198]` (199 frame/mode); `seed`: `0`; segment: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- Detection: `precision=0.970093458`, `recall=0.700404858`, `tp=519`, `fp=16`, `fn=222`.

| Chỉ số | LiDAR | Fused |
|---|---:|---:|
| `rmse` | 0.15032270040636328 | 0.1358667802927652 |
| `matches` | 502 | 502 |
| `sum_sq_err` | 11.343650957245547 | 9.26681055753551 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.522613065326633 | 2.522613065326633 |
| `precision_track` | 1.000000000 | 1.000000000 |
| `coverage` | 0.967244701 | 0.967244701 |

Camera làm RMSE thay đổi từ 0.150323 m xuống 0.135867 m
(`fused-lidar=-0.014456 m`). Số cặp ghép lần lượt 502 và 502; ghost 0 và 0; miss 239 và 239. Cần đọc các chỉ số này cùng nhau vì RMSE chỉ tính trên các track confirmed ghép được GT.

Trong lần chạy này, camera cải thiện độ chính xác vị trí của các cặp ghép mà
không thay đổi tổng matches/ghost/miss. Ví dụ tại frame 100, cả hai mode có 3
confirmed tracks ghép được 3 GT, ghost=miss=0; `sum_sq_err` giảm từ
0.06014079837659239 xuống 0.04875269926563258 m². Bằng chứng nằm ở
`grade_run.log`, dòng 101 (lidar) và dòng 300 (fused).
Miss vẫn là 239 ở cả hai mode; camera không tạo track mới để bù các xe LiDAR
không phát hiện hoặc track chưa confirmed. Camera trong lab lấy đo từ nhãn FRONT
có nhiễu, nên kết quả này chỉ đánh giá cơ chế fusion với nguồn đo mô phỏng đó.

Đối chiếu rubric: RMSE LiDAR ≤ 0.45 m: đạt; RMSE fused ≤ 0.45 m: đạt; độ chênh fused−LiDAR ≤ 0.05 m: đạt. Cả hai mode có precision_track=1 ≥ 0.75 và coverage≈0.9672 ≥ 0.70, tính theo đúng mẫu số của đề.

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

Dữ liệu và weights do học viên tải từ Drive của lớp. Cấu hình local `student/config/paths.yaml` trỏ tới thư mục Downloads đã cung cấp và dùng frame 0–198; file cấu hình được Git bỏ qua. Nếu chạy trên máy khác, tạo cấu hình từ `paths.example.yaml` rồi sửa đường dẫn và đặt `frame_end: 198`.

Weights `fpn_resnet_18_epoch_300.pth` có SHA-256 `d20d2576b51ec3b069bb57c9203df7cb637c241f5751af8a9fd15916425acabc`, trùng với bản từ link Udacity gốc. Lần chạy compare đọc đúng segment mặc định và sinh đủ 6 artifacts, 398 record trong log gộp (199/mode). Các record được kiểm tra lại bằng `validate_metrics_records`; tổng detection giống nhau ở hai mode.

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
   `artifacts/grade_run.log` có một record/mode/frame, gồm đủ frame 0–198
   cho lidar và fused. Log ghi kết quả tổng hợp; thứ tự predict/update
   được xác minh từ hàm `run`, không suy ra chỉ từ RMSE trong log.
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
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex đối chiếu công thức/API và chạy bộ test gốc với `PYTHONUTF8=1`: **128 passed**, không failed/xfailed/skipped (10/10/2026). `fusion-run-lab --help` chạy được, `git diff --check` không báo lỗi. Weights tải từ nguồn gốc, nạp model và inference tổng hợp thành công. Đã chạy Waymo compare, seed 0, frame 0–198 và đối chiếu metrics với toàn bộ JSONL. Học viên cần tự đọc lại code, xác nhận câu trả lời và hiểu từng hàm trước khi nộp.

Lệnh kiểm thử thực tế: `$env:PYTHONUTF8='1'; .venv/Scripts/python.exe -m pytest student/tests -q --basetemp=.venv/pytest_tmp`.
UTF-8 cần được bật trên Windows vì test kiểm tra bài nộp tạo file có tiếng Việt.
Các cảnh báo hiện có liên quan `np.matrix` theo API đề và các ca cố ý dùng tọa độ
không hợp lệ; không có test nào bị bỏ qua.
Code E–H được lưu thành bốn commit CP1–CP4 sau khi kiểm thử, trong phiên làm việc
có hỗ trợ AI; lịch sử này không phải bằng chứng học viên tự làm trong giờ lab.
CP5 đã hoàn tất bằng lần chạy dữ liệu thật sau khi học viên cung cấp segment.
CP6 gồm báo cáo, kiểm tra bài nộp và push; học viên nộp link và commit hash trên LMS.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên code được cung cấp
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push code, artifacts và báo cáo lên repo.
- [ ] Học viên nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md)).
