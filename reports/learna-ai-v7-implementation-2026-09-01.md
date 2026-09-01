# Báo cáo triển khai và kiểm thử Learna AI V7

## Phạm vi

Triển khai trực tiếp trên nhánh `main` của Watermark Studio, giữ nguyên source video,
golden reference và các output cũ. Pipeline Best-quality chỉ nhận
`CalibrationProfileV7` ở trạng thái `READY`; profile V3–V6 vẫn được đọc để xem
diagnostics nhưng không được render final.

## Checkpoint Git

- Checkpoint trước khi sửa: `11482702cef42a543ae91814d59f1eeb1f7462b6`.
- Milestone V7 hiện tại: `0d5a3a42761e7296c8ad09a49ef004994b58eda9`.
- Commit QA/retry: `926571fe5cfcd18df68c1658036bd841f2af865e`.
- Commit regression: `0292f6bcb6a292713671ea7d6117772343f41499`.
- Commit workspace hygiene mới nhất: `cea0e7faabff62728616da1b8082d8dfd0d9ee47`.
- Commit chống crash JSON candidate: `94d17ae`.
- Commit chuẩn hóa metric không đo được thành `null`: `ff14757`.
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
- Executable đã được rebuild sau khi đổi artifact QA sang đuôi `.qa.v7.*`.

## Kiểm thử UI và ba video yêu cầu

Đã mở Watermark Studio bằng Computer Use và chạy flow thật trên executable
`D:\\rustProject\\watermark-studio\\src-tauri\\target\\debug\\watermark-studio.exe`
cho các case:

- `C:\Users\quant\Dropbox\PC\Downloads\14\_7 (117).mp4`
- `C:\Users\quant\Dropbox\PC\Downloads\14\_7 (121).mp4`
- `C:\Users\quant\Dropbox\PC\Downloads\clip\_test.mp4`

Kết quả UI (không dùng script để giả lập output):

- `(117)`: import thành công (2473 frame). Auto calibration hoàn tất không crash
  JSON nhưng bị `NEEDS_REVIEW`: 213 hard-direct, 100% sampled path, residual p95
  6,33 px; còn 3 cụm ROI và holdout/inlier gate chưa đạt. Queue bị khóa.
- `(121)`: import thành công (4582 frame). Auto calibration hoàn tất không crash
  JSON nhưng bị `NEEDS_REVIEW`: 138 hard-direct, 89% evidence, residual p95
  57,16 px; còn 6 cụm ROI, trajectory/holdout gate chưa đạt. Queue bị khóa.
- `clip_test`: chạy lại V7 trên 904 frame, kết quả `NEEDS_REVIEW`: 37 hard-direct,
  36% evidence, residual p95 2,32 px; holdout p95 77,77 px và coverage refined
  chưa đạt. Queue bị khóa.

Các project và trạng thái được giữ lại khi mở lại app. Không có video final mới
được tạo vì pipeline fail-closed; đây là hành vi đúng khi profile V7 chưa vượt
quality gate. Runtime health trong Settings hiển thị `READY` (Python 3.11.9,
cv2/numpy/torch, CUDA GTX 1650 4 GB, model và FFmpeg sẵn sàng).

## Kết quả chất lượng

Chưa tuyên bố output final cho ba video vì cả ba calibration V7 đều chưa vượt
quality gate holdout/trajectory/refined coverage. Do đó UI không cho Queue/Render
và không sinh draft giả. Khi gate đạt, mỗi video phải sinh draft
`.review.mp4`, `*.qa.v7.json`, `*.qa.v7.png` và chỉ promote final nếu
`maskApplicationCoverageInRange = 1.0`, `residualPassCoverageInRange = 1.0`,
`failedFrames = []`, `unmeasurableFrames = []`.

## Bước tiếp theo

1. Trong Review, xử lý các cụm ROI V7 còn thiếu theo thứ tự ưu tiên; mỗi video
   vẫn bị giới hạn tối đa ba ROI mới trước khi chuyển `NEEDS_REVIEW` có cấu trúc.
2. Sau khi profile V7 đạt holdout/refined-coverage gate, Queue render qua UI và
   lưu draft, QA report/contact sheet; hiện chưa có output final mới.
3. Chỉ sau khi cả ba QA pass mới tạo commit nghiệm thu cuối
   `feat: complete validated Learna AI multi-trajectory pipeline`.
