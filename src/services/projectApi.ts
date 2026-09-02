import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import type { AppError, BoundingBox, FrameResult, RemovalConfig, RoiEvidenceRecord, ScanRange, WatermarkProject } from '../types/project';

export type BestQualitySample = {
  frame: number;
  timestampSeconds: number;
  bbox: BoundingBox;
  maskCoverage: number;
  maskPeak: number;
  backgroundComplexity: number;
  temporalInstability: number;
  glyphCorrelation: number;
  glyphIou: number;
  contamination: number;
  temporalPassCount: number;
  score: number;
  sceneSignature: string;
  previewPath: string;
  maskPath: string;
  editorMaskPath: string;
  roiFallback?: boolean;
};
export type RoiHint = { x: number; y: number; width: number; height: number; frame?: number };
export type FocusPreview = { frame: number; timestampSeconds: number; path: string; crop: BoundingBox };
export type JobStatus = 'IMPORTED' | 'SCANNING' | 'AWAITING_REVIEW' | 'READY' | 'QUEUED' | 'PREPARING' | 'INFERENCING' | 'ENCODING' | 'VERIFYING' | 'COMPLETED' | 'NEEDS_REVIEW' | 'FAILED' | 'CANCELED' | 'INTERRUPTED';
export type JobRecord = { id: string; projectId: string; sourceName: string; outputRoot: string | null; outputName: string | null; outputPath: string | null; scanRange: ScanRange | null; status: JobStatus; stage: string; progress: number; batchProgress: number; currentFrame: number | null; currentChunk: number | null; elapsedSeconds: number | null; etaSeconds: number | null; replacementConfig: unknown | null; hardwareProfile: string | null; allowReviewDraft?: boolean; attempt: number; qaReportPath: string | null; contactSheetPath: string | null; errorCode: string | null; error: string | null; createdAt: string; updatedAt: string };
export type HardwareProfile = { gpuName: string; vramMb: number; cudaAvailable: boolean; supported: boolean; tier: 'UNSUPPORTED' | 'SAFE' | 'BALANCED' | 'HIGH' | 'MAX'; width: number; height: number; coreLength: number; context: number };
export type RuntimeHealth = {
  pythonPath: string | null;
  pythonVersion: string | null;
  ffmpegPath: string | null;
  ffprobePath: string | null;
  cudaAvailable: boolean;
  gpuName: string | null;
  vramMb: number;
  imports: { cv2: boolean; numpy: boolean; torch: boolean };
  propainterModelReady: boolean;
  workspaceRoot: string;
  freeWorkspaceBytes: number | null;
  status: 'READY' | 'MISCONFIGURED' | 'UNSUPPORTED';
  problems: string[];
};

export async function chooseVideoPath(): Promise<string | null> {
  const selected = await open({ multiple: false, directory: false, filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'webm', 'm4v'] }] });
  return typeof selected === 'string' ? selected : null;
}

export async function chooseVideoPaths(): Promise<string[]> {
  const selected = await open({ multiple: true, directory: false, filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'webm', 'm4v'] }] });
  return Array.isArray(selected) ? selected : typeof selected === 'string' ? [selected] : [];
}

export async function chooseOutputDirectory(): Promise<string | null> {
  const selected = await open({ directory: true, multiple: false, recursive: true });
  return typeof selected === 'string' ? selected : null;
}

export function openVideo(path: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('open_video', { path });
}

export function getProject(projectId: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('get_project', { projectId });
}
export function listProjects(): Promise<WatermarkProject[]> { return invoke<WatermarkProject[]>('list_projects'); }
export function removeProject(projectId: string): Promise<void> { return invoke<void>('remove_project', { projectId }); }

export function extractPreviewFrame(projectId: string, frame: number): Promise<FrameResult> {
  return invoke<FrameResult>('extract_preview_frame', { projectId, frame });
}

export function extractFocusPreview(projectId: string, frame: number, bbox: BoundingBox): Promise<FocusPreview> {
  return invoke<FocusPreview>('extract_focus_preview', { request: { projectId, frame, bbox } });
}

export function saveWatermarkAnchor(projectId: string, frame: number, timestampSeconds: number, bbox: BoundingBox, label: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('save_watermark_anchor', {
    request: { projectId, frame, timestampSeconds, bbox, label: label.trim() || null },
  });
}

export function analyzeTrack(projectId: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('analyze_track', { projectId });
}

export function retrackTrack(projectId: string, frame: number): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('retrack_track', { projectId, frame });
}

export function cancelTracking(): Promise<void> {
  return invoke<void>('cancel_tracking');
}

export function interpolateTrackingRange(projectId: string, startFrame: number, endFrame: number): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('interpolate_tracking_range', { projectId, startFrame, endFrame });
}

export function saveManualAnchor(projectId: string, frame: number, timestampSeconds: number, bbox: BoundingBox): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('save_manual_anchor', { request: { projectId, frame, timestampSeconds, bbox } });
}

export function acceptTrackingFrame(projectId: string, frame: number): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('accept_tracking_frame', { projectId, frame });
}

export function markOccludedRange(projectId: string, startFrame: number, endFrame: number): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('mark_occluded_range', { projectId, startFrame, endFrame });
}

export function saveRemovalConfig(projectId: string, config: RemovalConfig): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('save_removal_config', { projectId, config });
}

export function chooseReplacementPath(): Promise<string | null> {
  return open({ multiple: false, directory: false, filters: [{ name: 'PNG image', extensions: ['png'] }] }).then((selected) => typeof selected === 'string' ? selected : null);
}

export type RenderResult = { outputPath: string; mode: RemovalConfig['mode']; qaReportPath?: string };
export type BestQualityReplacement = { kind: 'text' | 'image'; text: string; imagePath: string | null; placement: 'follow' | 'fixed'; fixedX: number; fixedY: number; scale: number; opacity: number };

export function renderVideo(projectId: string, config: RemovalConfig): Promise<RenderResult> {
  return invoke<RenderResult>('render_video', { request: { projectId, config } });
}

export function renderBestQualityVideo(projectId: string, replacement: BestQualityReplacement | null, outputRoot: string | null = null, outputName: string | null = null, allowReviewDraft = false): Promise<RenderResult> {
  return invoke<RenderResult>('render_best_quality_video', { request: { projectId, replacement, outputRoot, outputName, allowReviewDraft } });
}

export function listJobs(): Promise<JobRecord[]> { return invoke<JobRecord[]>('list_jobs'); }
export function revalidateReviewJob(jobId: string): Promise<JobRecord> { return invoke<JobRecord>('revalidate_review_job', { jobId }); }
export function enqueueBestQualityJob(projectId: string, outputRoot: string | null, outputName: string | null, replacement: BestQualityReplacement | null, allowReviewDraft = false): Promise<JobRecord> { return invoke<JobRecord>('enqueue_best_quality_job', { request: { projectId, outputRoot, outputName, replacement, allowReviewDraft } }); }
export function cancelJob(jobId: string): Promise<void> { return invoke<void>('cancel_job', { jobId }); }
export function regenJob(jobId: string): Promise<JobRecord> { return invoke<JobRecord>('regen_job', { jobId }); }

export function suggestBestQualitySamples(projectId: string, options: { scanRound: number; excludeFrames: number[]; excludeSceneSignatures: string[]; roi?: RoiHint | null; anchorFrame?: number; scanRange?: ScanRange | null }): Promise<BestQualitySample[]> {
  return invoke<BestQualitySample[]>('suggest_best_quality_samples', { request: { projectId, ...options } });
}

export function createCalibrationProfile(projectId: string, sample: BestQualitySample, editedMaskPath: string | null = null): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('create_calibration_profile', { request: { projectId, sample, editedMaskPath } });
}
export function autoCalibrateBestQuality(projectId: string, roi: RoiHint | null = null, editedMaskPath: string | null = null, roiEvidence: RoiHint[] = [], scanRange: ScanRange | null = null): Promise<WatermarkProject> {
  const evidence = roiEvidence
    .filter((item): item is RoiHint & { frame: number } => typeof item.frame === "number")
    .map((item) => ({ frame: item.frame, bbox: { x: item.x, y: item.y, width: item.width, height: item.height } }));
  return invoke<WatermarkProject>('auto_calibrate_best_quality', { request: { projectId, roi, editedMaskPath, roiEvidence: evidence, scanRange } });
}

export function persistedRoiEvidence(project: WatermarkProject | null): Array<RoiHint & { frame: number }> {
  // V8 stores evidence at project level.  Older V7 projects may only have it
  // nested in the calibration profile, so use that as a migration source
  // without treating frame numbers alone as precise geometry.
  const records = project?.roiEvidence?.length
    ? project.roiEvidence
    : (project?.calibration?.roiEvidence ?? []);
  return records.filter((item): item is RoiEvidenceRecord & { frame: number } =>
    Number.isFinite(item.frame) && Number.isFinite(item.bbox?.x) && Number.isFinite(item.bbox?.y) &&
    Number.isFinite(item.bbox?.width) && Number.isFinite(item.bbox?.height),
  ).map((item) => ({ ...item.bbox, frame: item.frame }));
}
export function saveCalibrationMaskEdit(projectId: string, pngBytes: number[]): Promise<string> {
  return invoke<string>('save_calibration_mask_edit', { request: { projectId, pngBytes } });
}

export function readProjectAssetBytes(projectId: string, asset: string): Promise<number[]> {
  return invoke<number[]>('read_project_asset_bytes', { request: { projectId, asset } });
}

export function detectHardware(): Promise<HardwareProfile> { return invoke<HardwareProfile>('detect_hardware'); }
export function detectRuntimeHealth(): Promise<RuntimeHealth> { return invoke<RuntimeHealth>('detect_runtime_health'); }

export function cancelRender(): Promise<void> {
  return invoke<void>('cancel_render');
}

export function getProjectAssetPath(projectId: string, asset: string): Promise<string> {
  return invoke<string>('get_project_asset_path', { projectId, asset });
}

export function getErrorMessage(error: unknown): string {
  if (isAppError(error)) {
    const stage = error.stage ? ` [${error.stage}]` : '';
    const compact = (message: string) => message.length > 600 ? `${message.slice(0, 600)}…` : message;
    switch (error.code) {
      case 'FFMPEG_NOT_FOUND': return 'FFmpeg hoặc ffprobe chưa được cài đặt hoặc chưa có trong PATH.';
      case 'UNSUPPORTED_VIDEO': return 'Định dạng video chưa được hỗ trợ. Hãy chọn MP4, MOV, MKV, WEBM hoặc M4V.';
      case 'INVALID_BOUNDING_BOX': return 'Vùng watermark phải nằm hoàn toàn trong video và có kích thước tối thiểu 8 × 8 pixel.';
      case 'FFPROBE_FAILED': return 'Không thể đọc metadata của video này.';
      case 'FFMPEG_FAILED': return `${compact(error.message)}${stage}`;
      case 'STORAGE_FULL': return `${compact(error.message)} Hãy giải phóng dung lượng ở ổ workspace rồi chạy lại.${stage}`;
      case 'RUNTIME_NOT_READY': return `${compact(error.message)} Mở Settings → Processing để sửa runtime trước khi chạy.${stage}`;
      case 'OPERATION_CANCELLED': return 'Tác vụ đã được hủy.';
      case 'CALIBRATION_CORRUPT': return `${error.message} Hãy mở Review và chạy lại Calibration V8.`;
      case 'INVALID_SCAN_RANGE': return `${error.message} Hãy chọn lại phạm vi frame hợp lệ trong Review.`;
      case 'ROI_OUTSIDE_SCAN_RANGE': return 'ROI evidence nằm ngoài phạm vi quét hiện tại. Hãy mở rộng phạm vi hoặc chọn ROI trong khoảng đã đặt.';
      case 'INVALID_REQUEST': {
        if (/stale|quality gate/i.test(error.message)) return `${error.message} Profile chưa READY; hãy xem diagnostics và chạy lại Auto-find & calibrate.`;
        if (/profile hash|mask hash|fingerprint/i.test(error.message)) return `${error.message} Không dùng lại profile sau khi source/mask thay đổi; hãy regenerate V8.`;
        return error.message;
      }
      case 'QUALITY_NEEDS_REVIEW': return `${error.message} Draft và QA vẫn được giữ để mở lại trong History.`;
      default: return `${compact(error.message)}${stage}`;
    }
  }
  if (error instanceof Error) return error.message;
  return 'Đã xảy ra lỗi không xác định.';
}

function isAppError(error: unknown): error is AppError {
  return typeof error === 'object' && error !== null && 'code' in error && 'message' in error;
}
