# Watermark Studio V9 — báo cáo tái xây pipeline Learna AI

## Phạm vi

- Nhánh: `codex/learna-ai-v9-quality-rebuild`
- Checkpoint trước sửa: `a026b0b4cbb4e13241cd57d7d4b13364142e5915`
- Milestone detector: `b8cab4f`
- QA độc lập/full-frame: `5f5e425`
- UI acceptance + QA reuse: `cf935da`
- UI wording polish: `3ad7ec1`
- Fallback/QA safety validation: `0645deb`
- Không push remote; source video và golden output không bị ghi đè.

## Thay đổi đã thực hiện

1. Calibration V9 dùng canonical Learna AI, scan đa scale, activity interval độc lập, trajectory riêng từng video và frameData theo từng frame.
2. Render ProPainter FP32 theo crop động 512×288 trên profile Safe 4 GB; frame inactive được passthrough.
3. QA V9 quét lại toàn khung hình, đối chiếu residual với detector nguồn, kiểm tra manifest/decode/SSIM/flicker và fail-closed.
4. Nếu QA còn residual, chạy một lượt fallback badge QuanPH có plate opaque phủ toàn bbox + uncertainty margin; chỉ promote sau QA lần hai đạt.
5. UI Best-quality có nút một lần `Tự động tìm & hiệu chỉnh`, dialog 9 bước, Queue/History stage và thông báo rõ review draft/final.
6. Runtime error được rút gọn thành chẩn đoán có ích thay vì traceback dài; profile JSON được ghi strict finite/atomic.
7. QA/badge lượt hai tái sử dụng detection nguồn đã xác minh từ report lượt một (có guard source path + frame count), giảm việc quét detector trùng lặp mà vẫn quét độc lập toàn bộ output.
8. Badge opaque không còn được miễn residual một cách mù quáng: manifest phải có `appliedBoxes` giao phủ ít nhất 80% glyph source; badge đặt sai vị trí bị đánh `badge_does_not_cover_source` và không thể promote final.
9. Fallback trajectory bị ngắt hoặc không bắt đầu từ sample đầu tiên sẽ trả `fallbackPathStatus=UNRESOLVED`; renderer không dùng lại bbox V9 cũ để che nhầm background.

## Runtime và build

- Python kiểm thử: `D:\propainter-watermark-venv\Scripts\python.exe` (3.11.9, OpenCV 4.11, NumPy 1.26.4, Torch 2.0.1+cu117, CUDA khả dụng).
- GPU: NVIDIA GTX 1650, 4 GB; profile Safe 288×512, crop động 512×288, core 60, context 8.
- Python unit tests: 38 pass.
- TypeScript/Vite build: pass.
- Rust: 51 pass, 2 ignored.
- `cargo fmt --check`: pass.
- Clippy `-D warnings`: pass.
- Tauri Windows x64 release: pass (MSI/NSIS).

## QA backend đã xác nhận trên clip test

Input: `C:\Users\quant\Dropbox\PC\Downloads\clip_test.mp4` (904 frame, 1080×1920, 30.008 fps).

- Draft ProPainter ban đầu còn residual và được giữ ở `.review.mp4`.
- Fallback opaque QuanPH đã che các frame/range mà inpaint không đủ sạch.
- QA V9 full-frame lần hai: `status=PASSED`, `decodedFrames=904`, `activeFrames=850`, `maskApplicationCoverage=1.0`, `residualPassCoverage=1.0`, `oldLearnaResidualDetections=0`, `failedFrames=[]`, `unmeasurableFrames=[]`.
- `minOutsideMaskSsim=0.97058`, `maxTemporalFlicker=0.10147`.
- QA report: `C:\Users\quant\AppData\Local\Temp\watermark-studio-v9\clip-v9.hybrid2.review.qa.v9.final.json`.
- Contact sheet: `C:\Users\quant\AppData\Local\Temp\watermark-studio-v9\clip-v9.hybrid2.review.qa.v9.final.png`.
- Bản kiểm tra này dùng badge opaque ở các frame khó; không tuyên bố khôi phục pixel nền tuyệt đối.

Artifact đã sao chép sang acceptance root (không ghi đè golden):

- Output: `C:\Users\quant\Dropbox\PC\Downloads\output\v9-acceptance\clip_test_watermark_removed_best_v9.mp4`
- QA: `C:\Users\quant\Dropbox\PC\Downloads\output\v9-acceptance\clip_test_watermark_removed_best_v9.qa.v9.json`
- Contact sheet: `C:\Users\quant\Dropbox\PC\Downloads\output\v9-acceptance\clip_test_watermark_removed_best_v9.qa.v9.png`
- Render manifest/badge manifest nằm cùng thư mục.
- SHA-256 source: `63F9A001077577743CD5BC2D4FC0FBFE9C134EEA1F4EB1BB366A765C16D12B9F`.
- SHA-256 output: `11CB0A30E27A471D4ABD250BDCFD2542B9FBF985050DB158F710E692731252E1`.
- FFmpeg full decode: 0 lỗi; output H.264 1080×1920, 904 frame, `7232/241` fps, AAC 48 kHz.
- Kiểm tra trực quan frame 530 xác nhận plate QuanPH opaque phủ toàn bộ Learna AI cũ; source frame 530 giữ nguyên.

## Kiểm thử qua Tauri UI

Đã mở Watermark Studio bằng Computer Use, vào Review của `clip_test.mp4`, bấm `Tự động tìm & hiệu chỉnh`, xác nhận dialog 9 bước hoàn tất và nhận kết quả `NEEDS_REVIEW_DRAFT` không yêu cầu ROI. Sau đó bấm `Đưa vào hàng đợi`; Queue hiển thị `PREPARING → INFERENCING → ENCODING → VERIFYING`, phạm vi `0–903`, và chỉ chạy một job GPU. QA lượt đầu phát hiện residual, UI tự chuyển fallback QuanPH opaque; QA lượt hai đạt và History hiển thị `COMPLETED · 100%`. Output UI: `C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_4.mp4`; đã mở action History `Mở output` và `Xem QA`.

## Giới hạn còn phải nghiệm thu

- Clip test đã hoàn tất qua UI và được kiểm tra artifact trong History.
- Cần chạy regression UI cho các video `14_7 (117)`, `14_7 (121)`, `14_7 (132)`; video dài chỉ được công nhận final khi QA V9 full-frame pass.
- Kiểm tra an toàn fallback trên `14_7 (132)` cho thấy candidate graph hiện không tạo được một path nối liên tục bắt đầu từ frame 0. Bản kiểm tra `14_7 (132)_fallback_safe_check.mp4` có manifest `fallbackPathStatus=UNRESOLVED` và không áp badge; đây là kết quả đúng theo fail-closed, không phải output final.
- Cần thay thuật toán fallback sparse hiện tại bằng dense local refinement/trajectory model có holdout trước khi có thể nghiệm thu 132, 117 và 121. Không được dùng lại fallback path tĩnh hoặc nâng threshold để biến chúng thành `COMPLETED`.
- Nếu profile không đủ khả năng định vị, hệ thống giữ `.review.mp4`/`NEEDS_REVIEW` hoặc che opaque theo residual; không hạ gate và không tự đoán background.
