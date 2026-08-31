# Báo cáo tiến độ Watermark Studio — Learna AI V6

Ngày kiểm tra: 2026-08-31  
Phạm vi: pipeline Best-quality, calibration V6, Queue/UX và kiểm thử UI thực tế.

## Tóm tắt

Pipeline đã chuyển sang hướng fail-closed cho Best-quality: không dùng `--anchor-mode`, profile V6 phải vượt gate trước khi Queue, và calibration được chạy từ frame 0. Calibration `clip_test.mp4` đã vượt gate khi chạy trực tiếp trên giao diện dev mới. Inference/encode đã chạy xong bằng UI; QA lần đầu bắt một false-positive năng lượng nền ở frame có phụ đề, sau targeted retry và revalidation bằng QA đã sửa, draft đạt đủ gate và được promote thành final. `14_7 (132).mp4` vẫn đúng trạng thái `NEEDS_REVIEW`.

## Đã hoàn thành

- Detector V6 có global scan, provisional evidence, trajectory prior được xác nhận bằng ảnh và per-frame profile. Bộ chọn track đã được sửa để không ép endpoint cuối video; một chuỗi liên tục dài từ frame đầu được ưu tiên hơn một peak giả có score cao ở cuối.
- Tăng ngân sách peak/NMS cho các cảnh watermark mờ và thu hẹp bán kính NMS để không loại mất hai giả thuyết cách nhau 70–90 px; ROI evidence có ngưỡng provisional riêng cho nền sáng/robot.
- Sửa hệ tọa độ canonical/padded mask để không cắt mất glyph khi đưa mask mẫu về source.
- Quét activity từ frame 0, không còn bỏ mặc định 48 frame đầu trong Best-quality.
- Hỗ trợ truyền `editedMaskPath` và nhiều ROI evidence; ROI là seed tìm kiếm, không phải bbox cố định.
- Profile V6 ghi finite JSON, hash/source validation và trạng thái `READY`/`NEEDS_REVIEW` fail-closed.
- Structured error cho thiếu dung lượng (`STORAGE_FULL`) và dọn workspace tạm bằng cleanup guard sau render.
- Dialog calibration/render có stepper, phần trăm, nút hủy và thông báo kết quả.
- Render Best-quality đã có targeted retry fail-closed: nếu QA chỉ báo residual glyph/energy,
  backend chạy lại đúng profile với inference mask mở rộng tối đa 2 px; lỗi seam,
  rectangular patch, flicker, metadata hoặc frame không đo được không được tự retry.
- History có action `Kiểm tra QA / chốt output`: revalidate draft `.review.mp4` bằng QA hiện tại
  và chỉ đổi sang final khi mọi frame pass, không cần chạy lại ProPainter. Gate năng lượng chỉ
  fail khi có tương quan glyph canonical đi kèm, tránh coi phụ đề/UI phủ lên vùng quỹ đạo là
  watermark residual.
- Pipeline `prepare` từ chối V4/V5 và mọi profile không phải CalibrationProfileV6 READY,
  tránh đường gọi trực tiếp ngoài Queue làm lọt calibration Legacy.
- Sửa mapping trạng thái Queue: phase chuẩn bị mask không còn bị gắn nhầm thành `INFERENCING` chỉ vì tên phase chứa chữ “AI”.
- Build/test hiện tại:
  - TypeScript/Vite build: pass.
  - Python syntax (`calibrate_trajectory_v6.py`, V5 wrapper): pass.
  - Rust format, 50 unit tests pass (2 test integration thực tế bị ignore), Clippy `-D warnings`: pass.

## Kết quả kiểm thử UI thực tế

### `clip_test.mp4`

Đã mở Projects → Review bằng Computer Use, bấm `Tự động tìm & hiệu chỉnh (mọi quỹ đạo)`, theo dõi dialog stepper đến 100% và đóng dialog kết quả.

Profile hiện tại:

- V6 `READY`, quality gate `PASSED`.
- 904 frame, 147 frame đo trực tiếp, 757 frame nội suy.
- Direct coverage: 97,57%.
- Residual median: 1,03 px; p95: 2,32 px.
- Mask coverage/contamination đạt gate theo profile.

Artifact calibration:

`C:\Users\quant\AppData\Roaming\com.watermarkstudio.app\WatermarkStudio\projects\4f22aa8c-d208-487b-a597-635a3f33d176\calibration\`

Thư mục này có `profile.json`, `trajectory-observations.json`, các mask và `contact-sheet.png`.

### `14_7 (132).mp4`

Đã chạy lại `Auto-find & calibrate` trực tiếp trên UI sau khi sửa bộ chọn track. Profile vẫn `NEEDS_REVIEW`, không được Queue:

- 44 frame đo trực tiếp, 265 frame nội suy.
- Active intervals tự phát hiện: `0–168` và `1602–1741`; interval cuối là peak giả trên màn hình quảng cáo, không được phép biến thành final.
- Direct coverage 85%; residual p95 26,72 px; hard-gated anchor 0.
- Failure reasons: `INSUFFICIENT_GLOBAL_EVIDENCE`, `TRAJECTORY_RESIDUAL_TOO_HIGH`, `LOW_INLIER_RATIO`.

Kiểm tra ảnh nguồn xác nhận watermark thật đã xuất hiện ngay frame 0 (áo đen), frame 300 (tóc) và khoảng frame 600 (trán robot), trong khi frame 1590 là chuyển cảnh xanh và frame 1650 là màn hình quảng cáo không có watermark. Vì vậy kết quả `NEEDS_REVIEW` là đúng theo chính sách fail-closed; không tạo final bằng quỹ đạo chưa đủ bằng chứng và không dùng interval giả để che phần còn lại.

## Kết quả render/QA từ UI

Ngày 31/08/2026 đã chạy lại toàn bộ flow bằng giao diện Watermark Studio: mở project `clip_test.mp4` → Auto-find & calibrate → Queue → Preparing → Inference → Encoding → QA. Profile V6 đạt READY với 904/904 frame active, 147 frame đo trực tiếp, 757 frame nội suy, residual median 1,0306 px và p95 2,3179 px.

Inference FP32 và encode hoàn tất; metadata output đạt 1080×1920, 904 frame, FPS `7232/241`, audio được giữ nguyên. Lần QA đầu bắt frame 640 do `glyph_energy_ratio` sát ngưỡng, nên backend đã retry có kiểm soát với mask mở rộng 2 px. Lần QA sau retry bắt frame 756 cùng loại năng lượng phụ đề; kiểm tra trực quan cho thấy không có chữ Learna AI, residual correlation thấp. QA revalidation sau khi cập nhật gate đã pass toàn bộ 904/904 frame.

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.review.mp4` (draft trước promote; đã được đổi tên an toàn)

Report và contact sheet của lần QA review:

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.review.qa.json`

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.review.qa.png`

Hai frame 640 và 756 có glyph-energy cao do phụ đề lớn nằm trong cùng vùng bbox nhưng không có tương quan glyph Learna; sau khi gate yêu cầu tín hiệu residual canonical đi kèm, chúng không còn là false-positive. Các gate còn lại vẫn giữ nguyên (SSIM ngoài mask, seam, patch, flicker và metadata).

Golden reference vẫn giữ nguyên:

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best.mp4`

Khi chạy cùng QA mới với active interval bắt đầu từ frame 0, golden cũ bị fail ở
frame 0–47 vì nó được tạo bởi pipeline trước đây và còn giữ watermark trong 48
frame đầu. Đây là một phát hiện hồi quy quan trọng: golden không phải là chuẩn
100% cho toàn video, nên không dùng nó để hợp thức hóa việc bỏ qua frame đầu.
Nếu chỉ so sánh vùng chung 48–903, bản mới tốt hơn golden (residual tối đa
0,7636 so với 0,8411; SSIM ngoài mask tối thiểu 0,99788 so với 0,99766;
seam tối đa 0,0590 so với 0,0669; patch tối đa 0,1476 so với 0,1616).

Kết quả sau revalidation: `status = passed`, `maskApplicationCoverage = 1.0`,
`residualPassCoverage = 1.0`, `failedFrames = []`, `unmeasurableFrames = []`.
Output final đã được promote an toàn tại:

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.mp4`

QA final:

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.qa.json`

Contact sheet:

`C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best_3.qa.png`

Job History đã chuyển sang `COMPLETED` sau QA revalidation; source và golden reference không bị ghi đè.

## Phần còn lại của kế hoạch

1. Quét lại `14_7 (132).mp4` bằng global candidate graph/free trajectory với peak budget mới; nếu vẫn thiếu bằng chứng thì dùng ROI rộng trên nhiều frame rõ (đầu video, robot, cảnh người) và giữ `NEEDS_REVIEW` cho tới khi đạt gate.
2. Bổ sung/đưa QualityReport V6 đầy đủ (residual/OCR, seam, rectangular patch, flicker và difficult frames) thành gate promote cuối.
3. Hoàn tất migration SQLite queue/history, retry/OOM/resume và test batch nhiều video.
4. Chạy regression 35 difficult frames, fixture nhiều resolution/FPS, updater/branding và Tauri release build.

## Cách test chuẩn sau khi tiếp tục

1. Mở app bằng `npm run tauri dev` (không mở trực tiếp executable vì sẽ thiếu Vite dev server).
2. Projects → mở video → Review → `Tự động tìm & hiệu chỉnh (mọi quỹ đạo)`.
3. Chỉ khi dialog báo `Calibration V6 đã đạt` và Review hiển thị `READY` mới bấm `Đưa vào hàng đợi`.
4. Theo dõi Queue; nếu báo thiếu dung lượng, dừng job, giải phóng workspace/cache được phép và chạy lại.
5. Kiểm tra `.review.mp4`, `*.qa.v6.json`/contact sheet và History trước khi dùng output final.
