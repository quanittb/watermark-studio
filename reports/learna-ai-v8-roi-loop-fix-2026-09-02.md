# Watermark Studio — bounded Learna AI calibration report

Ngày: 2026-09-02  
Branch: `main`  
Checkpoint trước thay đổi: `70adae172b43d086704133e3b3e65e7f5d0b96a7`

## Mục tiêu

Khóa vòng lặp yêu cầu ROI, giữ evidence sau reload/calibration lại, tách phát hiện
watermark khỏi quỹ đạo được chọn và cho phép kết thúc có kiểm soát bằng
`READY` hoặc review draft. Best-quality vẫn fail-closed: không profile đạt gate
thì không được gọi là final.

## Đã thực hiện

- Nâng entry point Best-quality và QA lên Calibration/Quality V8.
- Lưu `roiEvidence` ở `project.json` và profile; merge theo frame, đồng thời
  migrate evidence cũ nằm trong profile V7.
- ROI được đưa vào candidate graph như seed mềm; không còn thay toàn bộ global
  track bằng track ROI.
- `activeIntervals` được tạo từ activity evidence độc lập rồi hợp nhất với track;
  frame ngoài `scanRange` vẫn passthrough.
- Bổ sung refinement source-resolution stride 1 và một pass recovery bán kính rộng
  cho failed ranges, thay vì buộc người dùng vẽ hàng chục ROI.
- Hợp nhất các review range giao nhau/gần nhau trước khi hiển thị.
- Giới hạn ROI mới ở tối đa 3 frame. Sau khi đủ budget, không phát sinh danh sách
  ROI mới; profile chuyển sang `NEEDS_REVIEW_DRAFT` nếu gate chưa đạt.
- Queue ghi nhận `allowReviewDraft`; draft chỉ được giữ ở `.review.mp4`, sau đó QA
  mới có quyền promote thành final.
- QA V8 đọc profile draft để kiểm tra thực tế, đồng thời vẫn hard-gate residual,
  seam, patch, flicker và metadata.
- UI đổi stepper sang Validate V8, ẩn nút vẽ thêm sau khi hết budget và đồng bộ
  thông báo queue/draft.

## Kiểm tra đã chạy

- Python `py_compile`: **PASS** cho detector, wrapper V8, pipeline và QA.
- `npm run build`: **PASS** (TypeScript + Vite).
- `cargo test --manifest-path src-tauri/Cargo.toml`: **PASS**, 51 test pass,
  2 integration test được đánh dấu ignore theo yêu cầu môi trường.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --check`: **PASS**.
- `cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings`:
  **PASS**.
- `git diff --check`: không có lỗi nội dung; chỉ còn cảnh báo chuyển newline
  CRLF do môi trường Windows.

## Blocker môi trường hiện tại

Chưa thể chạy calibration/render thật qua UI trong lượt này vì runtime AI chưa
đạt preflight:

- `D:\propainter-watermark-venv\Scripts\python.exe` tồn tại nhưng launcher trỏ
  tới Python 3.11 đã bị xóa nên không khởi chạy được.
- Python hệ thống `C:\Python314\python.exe` không có module `cv2`; vì vậy
  `python -m unittest discover -s tools -p "test_*.py"` dừng ở lỗi import,
  không phải lỗi assertion của detector.

Settings → Processing phải trỏ tới Python 3.10/3.11 thực sự có `cv2`, `numpy`,
`torch`, CUDA và model ProPainter trước khi chạy regression UI.

## Cách nghiệm thu sau khi runtime READY

1. Mở app, import `clip_test.mp4`, chạy `Tự động tìm, hiệu chỉnh và kiểm chứng`;
   kỳ vọng 0 ROI mới, profile V8 READY và QA toàn bộ 904 frame so với golden.
2. Import `14_7 (117).mp4` và `14_7 (121).mp4`; evidence cũ được giữ, project mới
   chỉ được hỏi tối đa 3 ROI trong một dialog. Sau đó kết thúc READY hoặc tạo
   review draft, không lặp lần 2.
3. Queue qua UI, kiểm tra stage PREPARING → INFERENCING → ENCODING → VERIFYING,
   mở QA/contact sheet trong History và chỉ coi output là final khi failed và
   unmeasurable frames đều rỗng.

## Giới hạn còn lại

Chưa thể báo hai video thật đạt final hoặc đưa ra hash/output mới khi Python AI
runtime chưa chạy được. Bản vá đã khóa đường logic và UX; việc xác nhận chất
lượng hình ảnh ProPainter cần thực hiện sau khi sửa runtime, qua chính Queue/UI.
