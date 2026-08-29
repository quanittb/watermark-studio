import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import type { AppError, BoundingBox, FrameResult, RemovalConfig, WatermarkProject } from '../types/project';

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
export type RoiHint = { x: number; y: number; width: number; height: number };
export type FocusPreview = { frame: number; timestampSeconds: number; path: string; crop: BoundingBox };
export type JobStatus = 'IMPORTED' | 'SCANNING' | 'AWAITING_REVIEW' | 'READY' | 'QUEUED' | 'PREPARING' | 'INFERENCING' | 'ENCODING' | 'VERIFYING' | 'COMPLETED' | 'NEEDS_REVIEW' | 'FAILED' | 'CANCELED' | 'INTERRUPTED';
export type JobRecord = { id: string; projectId: string; sourceName: string; outputRoot: string | null; outputName: string | null; outputPath: string | null; status: JobStatus; stage: string; progress: number; batchProgress: number; currentFrame: number | null; currentChunk: number | null; elapsedSeconds: number | null; etaSeconds: number | null; replacementConfig: unknown | null; hardwareProfile: string | null; attempt: number; qaReportPath: string | null; contactSheetPath: string | null; errorCode: string | null; error: string | null; createdAt: string; updatedAt: string };
export type HardwareProfile = { gpuName: string; vramMb: number; cudaAvailable: boolean; supported: boolean; tier: 'UNSUPPORTED' | 'SAFE' | 'BALANCED' | 'HIGH' | 'MAX'; width: number; height: number; coreLength: number; context: number };

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

export function renderBestQualityVideo(projectId: string, replacement: BestQualityReplacement | null, outputRoot: string | null = null, outputName: string | null = null): Promise<RenderResult> {
  return invoke<RenderResult>('render_best_quality_video', { request: { projectId, replacement, outputRoot, outputName } });
}

export function listJobs(): Promise<JobRecord[]> { return invoke<JobRecord[]>('list_jobs'); }
export function enqueueBestQualityJob(projectId: string, outputRoot: string | null, outputName: string | null, replacement: BestQualityReplacement | null): Promise<JobRecord> { return invoke<JobRecord>('enqueue_best_quality_job', { request: { projectId, outputRoot, outputName, replacement } }); }
export function cancelJob(jobId: string): Promise<void> { return invoke<void>('cancel_job', { jobId }); }
export function regenJob(jobId: string): Promise<JobRecord> { return invoke<JobRecord>('regen_job', { jobId }); }

export function suggestBestQualitySamples(projectId: string, options: { scanRound: number; excludeFrames: number[]; excludeSceneSignatures: string[]; roi?: RoiHint | null; anchorFrame?: number }): Promise<BestQualitySample[]> {
  return invoke<BestQualitySample[]>('suggest_best_quality_samples', { request: { projectId, ...options } });
}

export function createCalibrationProfile(projectId: string, sample: BestQualitySample, editedMaskPath: string | null = null): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('create_calibration_profile', { request: { projectId, sample, editedMaskPath } });
}
export function saveCalibrationMaskEdit(projectId: string, pngBytes: number[]): Promise<string> {
  return invoke<string>('save_calibration_mask_edit', { request: { projectId, pngBytes } });
}

export function readProjectAssetBytes(projectId: string, asset: string): Promise<number[]> {
  return invoke<number[]>('read_project_asset_bytes', { request: { projectId, asset } });
}

export function detectHardware(): Promise<HardwareProfile> { return invoke<HardwareProfile>('detect_hardware'); }

export function cancelRender(): Promise<void> {
  return invoke<void>('cancel_render');
}

export function getProjectAssetPath(projectId: string, asset: string): Promise<string> {
  return invoke<string>('get_project_asset_path', { projectId, asset });
}

export function getErrorMessage(error: unknown): string {
  if (isAppError(error)) {
    switch (error.code) {
      case 'FFMPEG_NOT_FOUND': return 'FFmpeg hoặc ffprobe chưa được cài đặt hoặc chưa có trong PATH.';
      case 'UNSUPPORTED_VIDEO': return 'Định dạng video chưa được hỗ trợ. Hãy chọn MP4, MOV, MKV, WEBM hoặc M4V.';
      case 'INVALID_BOUNDING_BOX': return 'Vùng watermark phải nằm hoàn toàn trong video và có kích thước tối thiểu 8 × 8 pixel.';
      case 'FFPROBE_FAILED': return 'Không thể đọc metadata của video này.';
      case 'FFMPEG_FAILED': return 'Không thể trích xuất frame hoặc template đã chọn.';
      case 'OPERATION_CANCELLED': return 'Tác vụ đã được hủy.';
      default: return error.message;
    }
  }
  if (error instanceof Error) return error.message;
  return 'Đã xảy ra lỗi không xác định.';
}

function isAppError(error: unknown): error is AppError {
  return typeof error === 'object' && error !== null && 'code' in error && 'message' in error;
}
