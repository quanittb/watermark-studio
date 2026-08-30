export type Language = "vi" | "en";

const dictionary = {
  vi: {
    projects: "Dự án", review: "Duyệt", queue: "Hàng đợi", history: "Lịch sử", settings: "Cài đặt",
    videoLibrary: "Thư viện video", reviewCalibration: "Duyệt và hiệu chỉnh", renderQueue: "Hàng đợi render", jobHistory: "Lịch sử job",
    importVideos: "Import nhiều video", findSamples: "Tìm mẫu tốt nhất", findSamplesAll: "Tự động tìm mẫu (6 phase)", autoCalibrate: "Tự động tìm & hiệu chỉnh (mọi quỹ đạo)", calibrating: "Đang tìm và khớp quỹ đạo…", scanning: "Đang quét…", scanningAllPhases: "Đang quét đủ 6 phase quỹ đạo…", queueRender: "Đưa vào hàng đợi", findAlternatives: "Tìm 5 mẫu thay thế (6 phase)", alternativesHint: "Chưa mẫu nào đạt chuẩn theo mắt? Lượt thay thế sẽ quét lại đủ 6 phase, loại ±72 frame và scene vừa xem.", roiFallbackActive: "ROI fallback đang bật: kéo vùng tương đối trên preview, không cần khoanh sát từng glyph.", roiFit: "Khớp quỹ đạo theo ROI này", viewQa: "Xem QA", contactSheet: "Contact sheet", openOutput: "Mở output", openFolder: "Mở thư mục",
    general: "Chung", processing: "Xử lý", updates: "Cập nhật", advanced: "Nâng cao", about: "Giới thiệu",
    language: "Ngôn ngữ", outputFolder: "Thư mục output dùng chung", outputName: "Tên file output",
    noValidSample: "Không tìm thấy sample hợp lệ sau khi quét đủ 6 phase. Save/Render vẫn bị khóa.", scanAnotherPhase: "Quét lại đủ 6 phase",
    saveCalibration: "Lưu Calibration V4 đã xác minh", maskEditor: "Chỉnh mask", add: "Thêm", erase: "Xóa", resetMask: "Khôi phục auto-mask",
    currentFile: "FILE HIỆN TẠI", noVideo: "Chưa chọn video", importAndReview: "Import và duyệt lần lượt", importBatch: "Import video batch", remove: "Xóa", refresh: "Làm mới", noJobs: "Chưa có job. Hãy xác nhận sample trước khi đưa vào hàng đợi.",
    framePreview: "XEM TRƯỚC FRAME", noVideoSelected: "Chưa chọn video", fit: "Vừa khung", resetView: "Đặt lại khung nhìn", dropVideo: "Thả video vào đây", clickChoose: "hoặc bấm để chọn file", previewZoom: "Zoom preview", outputFileName: "Tên file output", chooseOutput: "Chọn thư mục output", changeOutput: "Đổi thư mục output", addLabel: "Thêm label sau khi xóa", checkNow: "Kiểm tra ngay", refreshGpu: "Quét lại GPU", cudaAvailable: "CUDA khả dụng", cudaUnavailable: "CUDA không khả dụng", aiProfile: "Profile AI", notSupported: "Không hỗ trợ", context: "Context", gpuQueue: "1 job GPU",
  },
  en: {
    projects: "Projects", review: "Review", queue: "Queue", history: "History", settings: "Settings",
    videoLibrary: "Video library", reviewCalibration: "Review and calibration", renderQueue: "Render queue", jobHistory: "Job history",
    importVideos: "Import videos", findSamples: "Find best samples", findSamplesAll: "Auto-find samples (6 phases)", autoCalibrate: "Auto-find & calibrate (all trajectories)", calibrating: "Finding and fitting trajectory…", scanning: "Scanning…", scanningAllPhases: "Scanning all 6 trajectory phases…", queueRender: "Queue render", findAlternatives: "Find 5 alternatives (6 phases)", alternativesHint: "Not satisfied with these samples? The next pass scans all 6 phases, excludes ±72 frames and the scenes already shown.", roiFallbackActive: "ROI fallback is enabled: draw a relative region in the preview; tight glyph tracing is not required.", roiFit: "Fit trajectory from this ROI", viewQa: "View QA", contactSheet: "Contact sheet", openOutput: "Open output", openFolder: "Open folder",
    general: "General", processing: "Processing", updates: "Updates", advanced: "Advanced", about: "About",
    language: "Language", outputFolder: "Shared output folder", outputName: "Output file name",
    noValidSample: "No valid sample was found after scanning all 6 phases. Save and Render remain locked.", scanAnotherPhase: "Rescan all 6 phases",
    saveCalibration: "Save verified Calibration V4", maskEditor: "Mask Editor", add: "Add", erase: "Erase", resetMask: "Reset auto-mask",
    currentFile: "CURRENT FILE", noVideo: "No video selected", importAndReview: "Import and review sequentially", importBatch: "Import video batch", remove: "Remove", refresh: "Refresh", noJobs: "No jobs yet. Confirm a sample before queueing.",
    framePreview: "FRAME PREVIEW", noVideoSelected: "No video selected", fit: "Fit", resetView: "Reset view", dropVideo: "Drop a video here", clickChoose: "or click to choose a file", previewZoom: "Preview zoom", outputFileName: "Output file name", chooseOutput: "Choose output folder", changeOutput: "Change output folder", addLabel: "Add a label after removal", checkNow: "Check now", refreshGpu: "Refresh GPU", cudaAvailable: "CUDA available", cudaUnavailable: "CUDA unavailable", aiProfile: "AI profile", notSupported: "Not supported", context: "Context", gpuQueue: "1 GPU job",
  },
} as const;

export type TranslationKey = keyof typeof dictionary.vi;
export function translate(language: Language, key: TranslationKey): string {
  return dictionary[language][key];
}
