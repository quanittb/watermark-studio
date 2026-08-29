export type Language = "vi" | "en";

const dictionary = {
  vi: {
    projects: "Dự án", review: "Duyệt", queue: "Hàng đợi", history: "Lịch sử", settings: "Cài đặt",
    videoLibrary: "Thư viện video", reviewCalibration: "Duyệt và hiệu chỉnh", renderQueue: "Hàng đợi render", jobHistory: "Lịch sử job",
    importVideos: "Import nhiều video", findSamples: "Tìm mẫu tốt nhất", scanning: "Đang quét…", queueRender: "Đưa vào hàng đợi",
    general: "Chung", processing: "Xử lý", updates: "Cập nhật", advanced: "Nâng cao", about: "Giới thiệu",
    language: "Ngôn ngữ", outputFolder: "Thư mục output dùng chung", outputName: "Tên file output",
    noValidSample: "Không tìm thấy sample hợp lệ ở phase này. Save/Render vẫn bị khóa.", scanAnotherPhase: "Quét phase khác",
    saveCalibration: "Lưu Calibration V3 đã xác minh", maskEditor: "Chỉnh mask", add: "Thêm", erase: "Xóa", resetMask: "Khôi phục auto-mask",
    currentFile: "FILE HIỆN TẠI", noVideo: "Chưa chọn video", importAndReview: "Import và duyệt lần lượt", importBatch: "Import video batch", remove: "Xóa", refresh: "Làm mới", noJobs: "Chưa có job. Hãy xác nhận sample trước khi đưa vào hàng đợi.",
    framePreview: "XEM TRƯỚC FRAME", noVideoSelected: "Chưa chọn video", fit: "Vừa khung", resetView: "Đặt lại khung nhìn", dropVideo: "Thả video vào đây", clickChoose: "hoặc bấm để chọn file", previewZoom: "Zoom preview", outputFileName: "Tên file output", chooseOutput: "Chọn thư mục output", changeOutput: "Đổi thư mục output", addLabel: "Thêm label sau khi xóa", checkNow: "Kiểm tra ngay", refreshGpu: "Quét lại GPU", cudaAvailable: "CUDA khả dụng", cudaUnavailable: "CUDA không khả dụng", aiProfile: "Profile AI", notSupported: "Không hỗ trợ", context: "Context", gpuQueue: "1 job GPU",
  },
  en: {
    projects: "Projects", review: "Review", queue: "Queue", history: "History", settings: "Settings",
    videoLibrary: "Video library", reviewCalibration: "Review and calibration", renderQueue: "Render queue", jobHistory: "Job history",
    importVideos: "Import videos", findSamples: "Find best samples", scanning: "Scanning…", queueRender: "Queue render",
    general: "General", processing: "Processing", updates: "Updates", advanced: "Advanced", about: "About",
    language: "Language", outputFolder: "Shared output folder", outputName: "Output file name",
    noValidSample: "No valid sample was found in this phase. Save and Render remain locked.", scanAnotherPhase: "Scan another phase",
    saveCalibration: "Save verified Calibration V3", maskEditor: "Mask Editor", add: "Add", erase: "Erase", resetMask: "Reset auto-mask",
    currentFile: "CURRENT FILE", noVideo: "No video selected", importAndReview: "Import and review sequentially", importBatch: "Import video batch", remove: "Remove", refresh: "Refresh", noJobs: "No jobs yet. Confirm a sample before queueing.",
    framePreview: "FRAME PREVIEW", noVideoSelected: "No video selected", fit: "Fit", resetView: "Reset view", dropVideo: "Drop a video here", clickChoose: "or click to choose a file", previewZoom: "Preview zoom", outputFileName: "Output file name", chooseOutput: "Choose output folder", changeOutput: "Change output folder", addLabel: "Add a label after removal", checkNow: "Check now", refreshGpu: "Refresh GPU", cudaAvailable: "CUDA available", cudaUnavailable: "CUDA unavailable", aiProfile: "AI profile", notSupported: "Not supported", context: "Context", gpuQueue: "1 GPU job",
  },
} as const;

export type TranslationKey = keyof typeof dictionary.vi;
export function translate(language: Language, key: TranslationKey): string {
  return dictionary[language][key];
}
