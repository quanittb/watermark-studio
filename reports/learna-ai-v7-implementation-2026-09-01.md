# Báo cáo triển khai và kiểm thử Learna AI V7

## Phạm vi

Triển khai trực tiếp trên nhánh `main` của Watermark Studio, giữ nguyên source video,
golden reference và các output cũ. Pipeline Best-quality chỉ nhận
`CalibrationProfileV7` ở trạng thái `READY`; profile V3–V6 vẫn được đọc để xem
diagnostics nhưng không được render final.

## Checkpoint Git

- Checkpoint trước khi sửa: `11482702cef42a543ae91814d59f1eeb1f7462b6`.
- Milestone V7 hiện tại: `0d5a3a42761e7296c8ad09a49ef004994b58eda9`.
- Không tạo branch mới, không reset/rebase/pull và không stage cache tạm.

## Thay đổi đã thực hiện

1. Runtime preflight trong Rust: kiểm tra Python, import `cv2`/`numpy`/`torch`,
   FFmpeg/ffprobe, CUDA/VRAM và model ProPainter; lỗi trả mã có cấu trúc và UI
   Settings hiển thị rõ nguyên nhân.
2. Entrypoint production `calibrate_trajectory_v7.py` và `quality_qa_v7.py`;
   Best-quality không còn gọi nhầm script legacy.
3. Calibration V7: local NCC refinement dày theo active interval, holdout
   deterministic, gate inlier/residual/gap/refined coverage; không dùng mốc frame
   periodic hardcode làm quỹ đạo cuối.
4. Strict JSON: `allow_nan=False`, kiểm tra finite và ghi profile atomically.
5. Render retry theo range frame lỗi: sao chép draft làm input tạm, chỉ re-render
   difficult range với mask mở rộng tối đa 2 px, sau đó QA lại.
6. Quality report V7: QA toàn bộ frame `maskRequired`, phân biệt frame ngoài scan
   range, thêm coverage contract và contact sheet Source/Trajectory/Mask/Output/
   Difference.
7. UI Settings: card runtime health, Python/import/model/FFmpeg và danh sách lỗi
   có hướng dẫn; giữ nguyên luồng Review/Queue/History.

## Build gate

- TypeScript/Vite: PASS.
- Rust test: 51 PASS, 2 integration test IGNORE (cần video/runtime thật).
- `cargo fmt --check`: PASS.
- Clippy `-D warnings`: PASS.
- Python syntax (`py_compile`): PASS.
- Python regression: 25 PASS bằng `C:\Python314\python.exe` (OpenCV headless đã có).
- Tauri executable Windows x64: PASS tại
  `src-tauri/target/release/watermark-studio.exe`.
- MSI bundling: BLOCKED ở WiX `light.exe`; không ảnh hưởng executable đã build.

## Kiểm thử UI và ba video yêu cầu

Đã chuẩn bị executable để chạy các case:

- `C:\Users\quant\Dropbox\PC\Downloads\14\_7 (117).mp4`
- `C:\Users\quant\Dropbox\PC\Downloads\14\_7 (121).mp4`
- `C:\Users\quant\Dropbox\PC\Downloads\clip\_test.mp4`

Computer Use không khởi tạo được trong môi trường hiện tại, lỗi lặp lại:
`failed to write kernel assets: The system cannot find the path specified (os error 3)`.
Vì vậy chưa có thao tác click/import/calibration/queue nào được ghi nhận là test UI
thành công; không dùng script để giả lập kết quả output. Đây là blocker kiểm thử
thực tế, cần khởi động lại Computer Use helper trên máy desktop rồi chạy đúng flow
UI.

Runtime backend hiện cũng phát hiện venv ProPainter cũ trỏ tới Python 3.11 đã bị
xóa; Python 3.14 chỉ dùng được cho unit test, chưa đủ tương thích để thay thế
runtime ProPainter. Khi runtime chưa được sửa, app phải giữ trạng thái `MISCONFIGURED` và khóa
Calibration/Render thay vì tạo output không đáng tin cậy.

## Kết quả chất lượng

Chưa tuyên bố output final cho ba video vì chưa vượt được hai điều kiện bắt buộc:
(1) Computer Use UI regression chưa chạy do helper lỗi; (2) Python 3.11/cv2/torch
runtime chưa sẵn sàng. Khi hai blocker được xử lý, mỗi video phải sinh draft
`.review.mp4`, `*.qa.v7.json`, `*.qa.v7.png` và chỉ promote final nếu
`maskApplicationCoverageInRange = 1.0`, `residualPassCoverageInRange = 1.0`,
`failedFrames = []`, `unmeasurableFrames = []`.

## Bước tiếp theo

1. Khôi phục Python 3.11 tương thích với venv hoặc cấu hình runtime mới có đủ
   `cv2`, `numpy`, `torch` và model ProPainter; xác nhận trong Settings → Processing
   đến khi Runtime `READY`.
2. Khởi động lại Computer Use helper, chạy UI regression ba video theo thứ tự
   `clip_test` → `(117)` → `(121)`; lưu output và QA artifacts vào báo cáo này.
3. Chỉ sau khi cả ba QA pass mới tạo commit nghiệm thu cuối
   `feat: complete validated Learna AI multi-trajectory pipeline`.
