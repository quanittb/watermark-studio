import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import type { AppError, BoundingBox, FrameResult, RemovalConfig, WatermarkProject } from '../types/project';

export async function chooseVideoPath(): Promise<string | null> {
  const selected = await open({ multiple: false, directory: false, filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'mkv', 'webm', 'm4v'] }] });
  return typeof selected === 'string' ? selected : null;
}

export function openVideo(path: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('open_video', { path });
}

export function getProject(projectId: string): Promise<WatermarkProject> {
  return invoke<WatermarkProject>('get_project', { projectId });
}

export function extractPreviewFrame(projectId: string, frame: number): Promise<FrameResult> {
  return invoke<FrameResult>('extract_preview_frame', { projectId, frame });
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

export function renderVideo(projectId: string, config: RemovalConfig): Promise<{ outputPath: string; mode: RemovalConfig['mode'] }> {
  return invoke<{ outputPath: string; mode: RemovalConfig['mode'] }>('render_video', { request: { projectId, config } });
}

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
