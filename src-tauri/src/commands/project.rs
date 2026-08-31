use crate::error::{AppError, AppErrorDto};
use crate::jobs::{self, JobRecord, JobStatus};
use crate::media::{best_quality, ffmpeg, ffprobe};
use crate::media::{mask, render};
use crate::project::model::{
    AnchorFrame, AnchorType, BoundingBox, FrameResult, ManualAnchor, Project, RemovalConfig,
    ScanRange, TemplatePaths, TrackingFrame, TrackingStatus, WatermarkConfig,
};
use crate::project::service;
use crate::tracking;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

const PROJECT_VERSION: u32 = 1;
const DEFAULT_TEMPLATE_PADDING: u32 = 4;

#[derive(Default)]
pub struct AppState {
    pub tracking_cancel: Arc<AtomicBool>,
    pub render_cancel: Arc<AtomicBool>,
    /// ProPainter uses all available VRAM on the target GTX 1650. Every
    /// project has an isolated workspace, but GPU jobs must be serialized.
    pub best_quality_gpu_lock: Arc<Mutex<()>>,
    pub job_worker_running: Arc<Mutex<bool>>,
    /// Serializes read-modify-write operations on the queue store so a
    /// cancel/regen cannot be overwritten by a worker progress update.
    pub job_store_lock: Arc<Mutex<()>>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnqueueBestQualityRequest {
    pub project_id: String,
    pub output_root: Option<String>,
    pub output_name: Option<String>,
    pub replacement: Option<BestQualityReplacement>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationProgress {
    pub phase: String,
    pub current_frame: u64,
    pub total_frames: u64,
    pub progress: f64,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManualAnchorRequest {
    pub project_id: String,
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderVideoRequest {
    pub project_id: String,
    pub config: Option<RemovalConfig>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderVideoResult {
    pub output_path: String,
    pub mode: crate::project::model::RemovalMode,
    pub qa_report_path: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BestQualityRenderRequest {
    pub project_id: String,
    pub replacement: Option<BestQualityReplacement>,
    pub output_root: Option<String>,
    pub output_name: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BestQualityReplacement {
    pub kind: String,
    pub text: Option<String>,
    pub image_path: Option<String>,
    pub placement: String,
    pub fixed_x: f64,
    pub fixed_y: f64,
    pub scale: f64,
    pub opacity: f64,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BestQualitySamplesRequest {
    pub project_id: String,
    #[serde(default)]
    pub scan_round: u32,
    #[serde(default)]
    pub exclude_frames: Vec<u64>,
    #[serde(default)]
    pub exclude_scene_signatures: Vec<String>,
    #[serde(default)]
    pub roi: Option<BoundingBox>,
    #[serde(default)]
    pub anchor_frame: Option<u64>,
    #[serde(default)]
    pub scan_range: Option<ScanRange>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateCalibrationRequest {
    pub project_id: String,
    pub sample: best_quality::BestQualitySample,
    pub edited_mask_path: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdaptiveCalibrationRequest {
    pub project_id: String,
    /// Optional inclusive range in which Best-quality searches for WTM.
    /// Absent means the complete source video.
    #[serde(default)]
    pub scan_range: Option<ScanRange>,
    #[serde(default)]
    pub roi: Option<BoundingBox>,
    /// Frame at which the user drew the broad ROI.  It is evidence for the
    /// detector, not a fixed render position.
    #[serde(default)]
    pub roi_frame: Option<u64>,
    /// Optional canonical mask edited in the Review Mask Editor.  V6 keeps
    /// the source geometry/trajectory independent from this mask, but uses it
    /// for glyph scoring and the inference/blend mask artifacts.
    #[serde(default)]
    pub edited_mask_path: Option<String>,
    /// Optional broad ROI evidence collected on multiple representative
    /// frames.  Each item is used only on its own frame as a seed; the
    /// trajectory remains free and is learned globally.
    #[serde(default)]
    pub roi_evidence: Vec<RoiEvidence>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RoiEvidence {
    pub frame: u64,
    pub bbox: BoundingBox,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveCalibrationMaskRequest {
    pub project_id: String,
    pub png_bytes: Vec<u8>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadProjectAssetRequest {
    pub project_id: String,
    pub asset: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusPreviewRequest {
    pub project_id: String,
    pub frame: u64,
    pub bbox: BoundingBox,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusPreviewResult {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub path: String,
    pub crop: BoundingBox,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveWatermarkAnchorRequest {
    pub project_id: String,
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
    pub label: Option<String>,
}

#[tauri::command]
pub async fn open_video(app: AppHandle, path: String) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || open_video_sync(&app, &path))
        .await
        .map_err(|error| AppError::Io(format!("Opening video task failed: {error}")))?
        .map_err(Into::into)
}

#[tauri::command]
pub async fn get_project(app: AppHandle, project_id: String) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &project_id)?;
        // Older projects may have marked long geometric interpolations as
        // resolved. Recompute the queue on load so an unverified bbox cannot
        // silently reach rendering after an app upgrade.
        if let Some(tracking) = project.tracking.as_mut() {
            normalize_interpolated_confidence(&mut tracking.frames);
            tracking.problem_ranges = tracking::service::group_problem_ranges(&tracking.frames);
        }
        let normalized_path = normalize_canonical_path(PathBuf::from(&project.source.path));
        let normalized_path = normalized_path.to_string_lossy().to_string();
        if normalized_path != project.source.path || project.tracking.is_some() {
            project.source.path = normalized_path;
            let directory = service::project_directory(&app_data_dir, &project.id)?;
            service::save_project_atomic(&directory, &project)?;
        }
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Loading project task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn list_projects(app: AppHandle) -> Result<Vec<Project>, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        service::list_projects(&app_data_dir)
    })
    .await
    .map_err(|error| AppError::Io(format!("Listing projects failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn remove_project(
    app: AppHandle,
    state: State<'_, AppState>,
    project_id: String,
) -> Result<(), AppErrorDto> {
    let store_lock = Arc::clone(&state.job_store_lock);
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let _store_guard = store_lock
            .lock()
            .map_err(|_| AppError::Io("Job queue is unavailable.".to_string()))?;
        let jobs = jobs::load(&app_data_dir)?;
        if jobs.iter().any(|job| {
            job.project_id == project_id
                && matches!(
                    job.status,
                    JobStatus::Queued
                        | JobStatus::Preparing
                        | JobStatus::Inferencing
                        | JobStatus::Encoding
                        | JobStatus::Verifying
                )
        }) {
            return Err(AppError::InvalidRequest(
                "Cannot remove a project with a queued or running job.".to_string(),
            ));
        }
        service::remove_project_workspace(&app_data_dir, &project_id)
    })
    .await
    .map_err(|error| AppError::Io(format!("Removing project failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn extract_preview_frame(
    app: AppHandle,
    project_id: String,
    frame: u64,
) -> Result<FrameResult, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let project = service::load_project(&app_data_dir, &project_id)?;
        if frame >= project.video.frame_count {
            return Err(AppError::InvalidRequest(
                "Preview frame is outside the video.".to_string(),
            ));
        }
        let timestamp_seconds = frame_to_timestamp(frame, project.video.fps);
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        let output_path = directory.join("cache").join(format!("preview-{frame}.png"));
        fs::create_dir_all(output_path.parent().ok_or_else(|| {
            AppError::InvalidRequest("Invalid preview output path.".to_string())
        })?)?;
        ffmpeg::ensure_tools_available()?;
        ffmpeg::extract_frame(
            Path::new(&project.source.path),
            timestamp_seconds,
            &output_path,
        )?;
        Ok::<FrameResult, AppError>(FrameResult {
            frame,
            timestamp_seconds,
            path: output_path.to_string_lossy().to_string(),
        })
    })
    .await
    .map_err(|error| AppError::Io(format!("Preview extraction task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn extract_focus_preview(
    app: AppHandle,
    request: FocusPreviewRequest,
) -> Result<FocusPreviewResult, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let project = service::load_project(&app_data_dir, &request.project_id)?;
        if request.frame >= project.video.frame_count {
            return Err(AppError::InvalidRequest(
                "Focus frame is outside the video.".to_string(),
            ));
        }
        validate_bbox(&request.bbox, project.video.width, project.video.height)?;
        let crop = focus_crop(&request.bbox, project.video.width, project.video.height);
        let timestamp_seconds = frame_to_timestamp(request.frame, project.video.fps);
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        let output_path = directory.join("cache").join(format!(
            "focus-{}-{}-{}.png",
            request.frame,
            crop.x.round(),
            crop.y.round()
        ));
        fs::create_dir_all(output_path.parent().ok_or_else(|| {
            AppError::InvalidRequest("Invalid focus-preview output path.".to_string())
        })?)?;
        ffmpeg::ensure_tools_available()?;
        ffmpeg::crop_image_from_video(
            Path::new(&project.source.path),
            timestamp_seconds,
            &crop,
            &output_path,
        )?;
        Ok::<FocusPreviewResult, AppError>(FocusPreviewResult {
            frame: request.frame,
            timestamp_seconds,
            path: output_path.to_string_lossy().to_string(),
            crop,
        })
    })
    .await
    .map_err(|error| AppError::Io(format!("Focus preview task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn save_watermark_anchor(
    app: AppHandle,
    request: SaveWatermarkAnchorRequest,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || save_watermark_anchor_sync(&app, request))
        .await
        .map_err(|error| AppError::Io(format!("Saving anchor task failed: {error}")))?
        .map_err(Into::into)
}

#[tauri::command]
pub async fn create_calibration_profile(
    app: AppHandle,
    state: State<'_, AppState>,
    request: CreateCalibrationRequest,
) -> Result<Project, AppErrorDto> {
    let cancel = Arc::clone(&state.render_cancel);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &request.project_id)?;
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        let edited_mask = request.edited_mask_path.as_deref().map(Path::new);
        let calibration = best_quality::create_calibration_profile(
            &directory,
            &project,
            &request.sample,
            edited_mask,
            &cancel,
        )?;
        project.watermark.label = Some("Learna AI".to_string());
        project.watermark.anchor = Some(AnchorFrame {
            frame: request.sample.frame,
            timestamp_seconds: request.sample.timestamp_seconds,
            bbox: request.sample.bbox.clone(),
        });
        project.calibration = Some(calibration.clone());
        if let Some(templates) = project.watermark.templates.as_mut() {
            templates.mask = Some(calibration.mask_path.clone());
        }
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Calibration task failed: {error}")))?
    .map_err(Into::into)
}

/// Starts the Best-quality adaptive route.  The command intentionally returns
/// a project even when the trajectory gate is NEEDS_REVIEW so the UI can show
/// diagnostics and offer a wider ROI retry without fabricating a READY profile.
#[tauri::command]
pub async fn auto_calibrate_best_quality(
    app: AppHandle,
    state: State<'_, AppState>,
    request: AdaptiveCalibrationRequest,
) -> Result<Project, AppErrorDto> {
    let cancel = Arc::clone(&state.render_cancel);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &request.project_id)?;
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        let scan_range = request.scan_range.or_else(|| {
            project
                .calibration
                .as_ref()
                .and_then(|profile| profile.scan_range)
        });
        let calibration = best_quality::create_adaptive_calibration_profile(
            &directory,
            &project,
            request.roi.as_ref(),
            request.roi_frame,
            request.edited_mask_path.as_deref(),
            &request.roi_evidence,
            scan_range,
            &cancel,
        )?;
        project.watermark.label = Some("Learna AI".to_string());
        project.calibration = Some(calibration.clone());
        if let Some(templates) = project.watermark.templates.as_mut() {
            templates.mask = Some(calibration.mask_path.clone());
        }
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Adaptive calibration task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn save_calibration_mask_edit(
    app: AppHandle,
    request: SaveCalibrationMaskRequest,
) -> Result<String, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        if request.png_bytes.len() > 2_000_000 {
            return Err(AppError::InvalidRequest(
                "Edited mask is unexpectedly large.".to_string(),
            ));
        }
        let decoded = image::load_from_memory(&request.png_bytes).map_err(|error| {
            AppError::InvalidRequest(format!("Edited mask is not a valid PNG: {error}"))
        })?;
        if decoded.width() < 32 || decoded.height() < 16 {
            return Err(AppError::InvalidRequest(
                "Edited mask dimensions are invalid.".to_string(),
            ));
        }
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let directory = service::project_directory(&app_data_dir, &request.project_id)?;
        let output = directory.join("calibration").join("edited_mask.png");
        fs::create_dir_all(
            output
                .parent()
                .ok_or_else(|| AppError::Io("Invalid calibration directory.".to_string()))?,
        )?;
        decoded
            .to_luma8()
            .save(&output)
            .map_err(|error| AppError::Io(error.to_string()))?;
        Ok::<String, AppError>(output.to_string_lossy().to_string())
    })
    .await
    .map_err(|error| AppError::Io(format!("Saving mask edit failed: {error}")))?
    .map_err(Into::into)
}

fn open_video_sync(app: &AppHandle, path: &str) -> Result<Project, AppError> {
    let selected_path = Path::new(path);
    if !selected_path.is_file() {
        return Err(AppError::VideoNotFound);
    }
    if !is_supported_video(selected_path) {
        return Err(AppError::UnsupportedVideo);
    }

    ffmpeg::ensure_tools_available()?;
    let source_path = normalize_canonical_path(fs::canonicalize(selected_path)?);
    let video = ffprobe::probe_video(&source_path)?;
    let file_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| AppError::InvalidRequest("Video file name is invalid.".to_string()))?
        .to_string();
    let project = Project {
        version: PROJECT_VERSION,
        id: Uuid::new_v4().to_string(),
        source: crate::project::model::SourceVideo {
            path: source_path.to_string_lossy().to_string(),
            file_name,
        },
        video,
        watermark: WatermarkConfig {
            template_padding: DEFAULT_TEMPLATE_PADDING,
            ..Default::default()
        },
        calibration: None,
        anchors: Vec::new(),
        tracking: None,
        removal: None,
    };

    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    service::create_project_workspace(&app_data_dir, &project)?;
    Ok(project)
}

fn save_watermark_anchor_sync(
    app: &AppHandle,
    request: SaveWatermarkAnchorRequest,
) -> Result<Project, AppError> {
    if !request.timestamp_seconds.is_finite() || request.timestamp_seconds < 0.0 {
        return Err(AppError::InvalidRequest(
            "Invalid anchor timestamp.".to_string(),
        ));
    }

    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    let mut project = service::load_project(&app_data_dir, &request.project_id)?;
    let timestamp_frame = timestamp_to_frame(request.timestamp_seconds, project.video.fps);
    if request.frame >= project.video.frame_count || timestamp_frame >= project.video.frame_count {
        return Err(AppError::InvalidRequest(
            "Anchor frame is outside the video.".to_string(),
        ));
    }
    validate_bbox(&request.bbox, project.video.width, project.video.height)?;
    let source_path = Path::new(&project.source.path);
    if !source_path.is_file() {
        return Err(AppError::VideoNotFound);
    }

    ffmpeg::ensure_tools_available()?;
    let directory = service::project_directory(&app_data_dir, &project.id)?;
    let anchor_path = directory.join("frames").join("anchor.png");
    let template_path = directory.join("templates").join("original.png");
    let grayscale_path = directory.join("templates").join("grayscale.png");
    let high_contrast_path = directory.join("templates").join("high_contrast.png");
    let padded_bbox = clamp_and_pad_bbox(
        &request.bbox,
        project.video.width,
        project.video.height,
        project.watermark.template_padding,
    )?;

    ffmpeg::extract_frame(source_path, request.timestamp_seconds, &anchor_path)?;
    ffmpeg::crop_image_from_video(
        source_path,
        request.timestamp_seconds,
        &padded_bbox,
        &template_path,
    )?;
    ffmpeg::generate_template_variant(&template_path, "format=gray", &grayscale_path)?;
    ffmpeg::generate_template_variant(
        &template_path,
        "format=gray,eq=contrast=1.8:brightness=0.05",
        &high_contrast_path,
    )?;
    let mask_path = directory.join("templates").join("mask.png");
    mask::generate_template_mask(&template_path, &mask_path)?;

    project.watermark.label = request.label.filter(|label| !label.trim().is_empty());
    // A new sample invalidates the previous full-video calibration. It will
    // be rebuilt on the next Best-quality render.
    project.calibration = None;
    project.watermark.anchor = Some(AnchorFrame {
        frame: request.frame,
        timestamp_seconds: request.timestamp_seconds,
        bbox: request.bbox.clone(),
    });
    project.watermark.templates = Some(TemplatePaths {
        original: "templates/original.png".to_string(),
        grayscale: "templates/grayscale.png".to_string(),
        high_contrast: "templates/high_contrast.png".to_string(),
        mask: Some("templates/mask.png".to_string()),
    });

    let manual_anchor = ManualAnchor {
        frame: request.frame,
        timestamp_seconds: request.timestamp_seconds,
        bbox: request.bbox,
        anchor_type: AnchorType::Initial,
        locked: true,
    };
    if let Some(anchor) = project
        .anchors
        .iter_mut()
        .find(|anchor| anchor.frame == manual_anchor.frame)
    {
        *anchor = manual_anchor;
    } else {
        project.anchors.push(manual_anchor);
    }

    service::save_project_atomic(&directory, &project)?;
    Ok(project)
}

#[tauri::command]
pub async fn analyze_track(
    app: AppHandle,
    state: State<'_, AppState>,
    project_id: String,
) -> Result<Project, AppErrorDto> {
    let cancel = Arc::clone(&state.tracking_cancel);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let callback_cancel = Arc::clone(&cancel);
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        tracking::service::analyze_project(
            &app_data_dir,
            &project_id,
            &cancel,
            move |phase, current, total| {
                let progress = OperationProgress {
                    phase: phase.to_string(),
                    current_frame: current,
                    total_frames: total,
                    progress: if total == 0 {
                        0.0
                    } else {
                        current as f64 / total as f64
                    },
                };
                let _ = app.emit("operation-progress", progress);
                !callback_cancel.load(Ordering::Relaxed)
            },
        )
    })
    .await
    .map_err(|error| AppError::Io(format!("Tracking task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn retrack_track(
    app: AppHandle,
    state: State<'_, AppState>,
    project_id: String,
    frame: u64,
) -> Result<Project, AppErrorDto> {
    let cancel = Arc::clone(&state.tracking_cancel);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let callback_cancel = Arc::clone(&cancel);
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        tracking::service::retrack_project(
            &app_data_dir,
            &project_id,
            frame,
            &cancel,
            move |phase, current, total| {
                let progress = OperationProgress {
                    phase: phase.to_string(),
                    current_frame: current,
                    total_frames: total,
                    progress: if total == 0 {
                        0.0
                    } else {
                        current as f64 / total as f64
                    },
                };
                let _ = app.emit("operation-progress", progress);
                !callback_cancel.load(Ordering::Relaxed)
            },
        )
    })
    .await
    .map_err(|error| AppError::Io(format!("Retracking task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub fn cancel_tracking(state: State<'_, AppState>) {
    state.tracking_cancel.store(true, Ordering::Relaxed);
}

#[tauri::command]
pub async fn interpolate_tracking_range(
    app: AppHandle,
    project_id: String,
    start_frame: u64,
    end_frame: u64,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &project_id)?;
        let tracking = project.tracking.as_mut().ok_or_else(|| {
            AppError::InvalidRequest("Analyze the track before interpolating.".to_string())
        })?;
        if start_frame > end_frame || end_frame >= tracking.frames.len() as u64 {
            return Err(AppError::InvalidRequest(
                "Invalid interpolation range.".to_string(),
            ));
        }
        let mut anchors = project.anchors.clone();
        anchors.sort_by_key(|anchor| anchor.frame);
        let left = anchors
            .iter()
            .rfind(|anchor| anchor.frame <= start_frame)
            .cloned()
            .ok_or_else(|| {
                AppError::InvalidRequest(
                    "A trusted anchor before the range is required.".to_string(),
                )
            })?;
        let right = anchors
            .iter()
            .find(|anchor| anchor.frame >= end_frame && anchor.frame != left.frame)
            .cloned()
            .ok_or_else(|| {
                AppError::InvalidRequest(
                    "A trusted anchor after the range is required.".to_string(),
                )
            })?;
        tracking::service::interpolate_between_anchors(
            &mut tracking.frames,
            &left,
            &right,
            start_frame,
            end_frame,
        )?;
        tracking.problem_ranges = tracking::service::group_problem_ranges(&tracking.frames);
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Interpolation task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn save_manual_anchor(
    app: AppHandle,
    request: ManualAnchorRequest,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || save_manual_anchor_sync(&app, request))
        .await
        .map_err(|error| AppError::Io(format!("Saving manual anchor task failed: {error}")))?
        .map_err(Into::into)
}

fn save_manual_anchor_sync(
    app: &AppHandle,
    request: ManualAnchorRequest,
) -> Result<Project, AppError> {
    if !request.timestamp_seconds.is_finite() || request.timestamp_seconds < 0.0 {
        return Err(AppError::InvalidRequest(
            "Invalid anchor timestamp.".to_string(),
        ));
    }
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    let mut project = service::load_project(&app_data_dir, &request.project_id)?;
    if request.frame >= project.video.frame_count
        || timestamp_to_frame(request.timestamp_seconds, project.video.fps)
            >= project.video.frame_count
    {
        return Err(AppError::InvalidRequest(
            "Anchor frame is outside the video.".to_string(),
        ));
    }
    validate_bbox(&request.bbox, project.video.width, project.video.height)?;
    let manual = ManualAnchor {
        frame: request.frame,
        timestamp_seconds: request.timestamp_seconds,
        bbox: request.bbox.clone(),
        anchor_type: AnchorType::Manual,
        locked: true,
    };
    if let Some(existing) = project
        .anchors
        .iter_mut()
        .find(|anchor| anchor.frame == manual.frame)
    {
        *existing = manual;
    } else {
        project.anchors.push(manual);
    }
    if let Some(tracking) = project.tracking.as_mut() {
        if let Some(frame) = tracking.frames.get_mut(request.frame as usize) {
            frame.bbox = request.bbox;
            frame.timestamp_seconds = request.timestamp_seconds;
            frame.confidence = 1.0;
            frame.status = TrackingStatus::Manual;
            frame.source = crate::project::model::TrackingSource::Manual;
            frame.locked = true;
        }
        tracking.problem_ranges = tracking::service::group_problem_ranges(&tracking.frames);
    }
    let directory = service::project_directory(&app_data_dir, &project.id)?;
    service::save_project_atomic(&directory, &project)?;
    Ok(project)
}

#[tauri::command]
pub async fn accept_tracking_frame(
    app: AppHandle,
    project_id: String,
    frame: u64,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &project_id)?;
        let (timestamp_seconds, bbox) = {
            let tracking = project.tracking.as_mut().ok_or_else(|| {
                AppError::InvalidRequest("Analyze the track before accepting a frame.".to_string())
            })?;
            let tracked = tracking.frames.get_mut(frame as usize).ok_or_else(|| {
                AppError::InvalidRequest("Frame is outside the tracking data.".to_string())
            })?;
            tracked.status = TrackingStatus::Manual;
            tracked.source = crate::project::model::TrackingSource::Manual;
            tracked.confidence = 1.0;
            tracked.locked = true;
            (tracked.timestamp_seconds, tracked.bbox.clone())
        };
        let accepted_anchor = ManualAnchor {
            frame,
            timestamp_seconds,
            bbox,
            anchor_type: AnchorType::Manual,
            locked: true,
        };
        if let Some(anchor) = project
            .anchors
            .iter_mut()
            .find(|anchor| anchor.frame == frame)
        {
            *anchor = accepted_anchor;
        } else {
            project.anchors.push(accepted_anchor);
        }
        if let Some(tracking) = project.tracking.as_mut() {
            tracking.problem_ranges = tracking::service::group_problem_ranges(&tracking.frames);
        }
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Accept frame task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn mark_occluded_range(
    app: AppHandle,
    project_id: String,
    start_frame: u64,
    end_frame: u64,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &project_id)?;
        {
            let tracking = project.tracking.as_mut().ok_or_else(|| {
                AppError::InvalidRequest(
                    "Analyze the track before marking occluded frames.".to_string(),
                )
            })?;
            if start_frame > end_frame || end_frame >= tracking.frames.len() as u64 {
                return Err(AppError::InvalidRequest(
                    "Invalid occluded frame range.".to_string(),
                ));
            }
            for frame in &mut tracking.frames[start_frame as usize..=end_frame as usize] {
                frame.status = TrackingStatus::Occluded;
                frame.source = crate::project::model::TrackingSource::Occluded;
                frame.confidence = 0.0;
                frame.locked = false;
                frame.scores.motion_smoothness = Some(0.0);
            }
            tracking.problem_ranges = tracking::service::group_problem_ranges(&tracking.frames);
        }
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Marking occluded range failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn save_removal_config(
    app: AppHandle,
    project_id: String,
    config: RemovalConfig,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let mut project = service::load_project(&app_data_dir, &project_id)?;
        project.removal = Some(config);
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        service::save_project_atomic(&directory, &project)?;
        Ok::<Project, AppError>(project)
    })
    .await
    .map_err(|error| AppError::Io(format!("Saving removal config task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn render_video(
    app: AppHandle,
    state: State<'_, AppState>,
    request: RenderVideoRequest,
) -> Result<RenderVideoResult, AppErrorDto> {
    let cancel = Arc::clone(&state.render_cancel);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let project = service::load_project(&app_data_dir, &request.project_id)?;
        let config = request
            .config
            .or(project.removal.clone())
            .unwrap_or_default();
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        ffmpeg::ensure_tools_available()?;
        let output = render::render_project(
            &directory,
            &project,
            &config,
            &cancel,
            |phase, current, total| {
                let progress = OperationProgress {
                    phase: phase.to_string(),
                    current_frame: current,
                    total_frames: total,
                    progress: if total == 0 {
                        0.0
                    } else {
                        current as f64 / total as f64
                    },
                };
                let _ = app.emit("operation-progress", progress);
            },
        )?;
        Ok::<RenderVideoResult, AppError>(RenderVideoResult {
            output_path: output.to_string_lossy().to_string(),
            mode: config.mode,
            qa_report_path: None,
        })
    })
    .await
    .map_err(|error| AppError::Io(format!("Render task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn render_best_quality_video(
    app: AppHandle,
    state: State<'_, AppState>,
    request: BestQualityRenderRequest,
) -> Result<RenderVideoResult, AppErrorDto> {
    let cancel = Arc::clone(&state.render_cancel);
    let gpu_lock = Arc::clone(&state.best_quality_gpu_lock);
    cancel.store(false, Ordering::Relaxed);
    tauri::async_runtime::spawn_blocking(move || {
        let _gpu_job = gpu_lock
            .lock()
            .map_err(|_| AppError::Io("Best-quality GPU queue is unavailable.".to_string()))?;
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let project = service::load_project(&app_data_dir, &request.project_id)?;
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        ffmpeg::ensure_tools_available()?;
        let output = best_quality::render_best_quality(
            &directory,
            &project,
            request.replacement.as_ref(),
            request.output_root.as_deref(),
            request.output_name.as_deref(),
            &cancel,
            |phase, current, total| {
                let progress = OperationProgress {
                    phase: phase.to_string(),
                    current_frame: current,
                    total_frames: total,
                    progress: if total == 0 {
                        0.0
                    } else {
                        current as f64 / total as f64
                    },
                };
                let _ = app.emit("operation-progress", progress);
            },
        )?;
        // Persist the exact profile used for this render so reopening the
        // project never falls back to the legacy anchor mask.
        let mut persisted_project = project.clone();
        if let Ok(calibration) = best_quality::calibration_metadata(&directory, &persisted_project)
        {
            persisted_project.calibration = Some(calibration.clone());
            if let Some(templates) = persisted_project.watermark.templates.as_mut() {
                templates.mask = Some(calibration.mask_path);
            }
            service::save_project_atomic(&directory, &persisted_project)?;
        }
        Ok::<RenderVideoResult, AppError>(RenderVideoResult {
            qa_report_path: Some(
                best_quality::qa_report_path(&output)
                    .to_string_lossy()
                    .to_string(),
            ),
            output_path: output.to_string_lossy().to_string(),
            mode: crate::project::model::RemovalMode::AutoBest,
        })
    })
    .await
    .map_err(|error| AppError::Io(format!("Best-quality render task failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn list_jobs(app: AppHandle) -> Result<Vec<JobRecord>, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        // Startup recovery is performed once by `jobs::open_database`.  This
        // command is polled by the UI every few seconds, so mutating active
        // jobs here would incorrectly mark a healthy render as INTERRUPTED.
        let records = jobs::load(&app_data_dir)?;
        Ok::<Vec<JobRecord>, AppError>(records)
    })
    .await
    .map_err(|error| AppError::Io(format!("Listing jobs failed: {error}")))?
    .map_err(Into::into)
}

/// Re-validates a completed review draft with the current QA rules.  This is
/// useful when a QA classifier is corrected after a long GPU render: the
/// existing draft is promoted only after the same fail-closed checks pass.
#[tauri::command]
pub async fn revalidate_review_job(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
) -> Result<JobRecord, AppErrorDto> {
    let gpu_lock = Arc::clone(&state.best_quality_gpu_lock);
    let store_lock = Arc::clone(&state.job_store_lock);
    let cancel = Arc::clone(&state.render_cancel);
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let record = {
            let _store_guard = store_lock
                .lock()
                .map_err(|_| AppError::Io("Job queue is unavailable.".to_string()))?;
            jobs::load(&app_data_dir)?
                .into_iter()
                .find(|item| item.id == job_id)
                .ok_or_else(|| AppError::InvalidRequest("Job not found.".to_string()))?
        };
        if record.status != JobStatus::NeedsReview {
            return Err(AppError::InvalidRequest(
                "Only a NEEDS_REVIEW job can be revalidated.".to_string(),
            ));
        }
        let review_output = record.output_path.clone().ok_or_else(|| {
            AppError::InvalidRequest(
                "This job has no review draft to revalidate; render it again.".to_string(),
            )
        })?;
        let project = service::load_project(&app_data_dir, &record.project_id)?;
        let directory = service::project_directory(&app_data_dir, &record.project_id)?;
        let _gpu_job = gpu_lock
            .lock()
            .map_err(|_| AppError::Io("Best-quality GPU queue is unavailable.".to_string()))?;
        cancel.store(false, Ordering::Relaxed);
        let output = best_quality::revalidate_review_output(
            &directory,
            &project,
            Path::new(&review_output),
            &cancel,
            |phase, current, total| {
                let _ = app.emit(
                    "operation-progress",
                    OperationProgress {
                        phase: phase.to_string(),
                        current_frame: current,
                        total_frames: total,
                        progress: if total == 0 {
                            0.0
                        } else {
                            current as f64 / total as f64
                        },
                    },
                );
            },
        )?;
        let _store_guard = store_lock
            .lock()
            .map_err(|_| AppError::Io("Job queue is unavailable.".to_string()))?;
        let mut records = jobs::load(&app_data_dir)?;
        let updated = records
            .iter_mut()
            .find(|item| item.id == job_id)
            .ok_or_else(|| {
                AppError::InvalidRequest("Job disappeared from the queue.".to_string())
            })?;
        updated.status = JobStatus::Completed;
        updated.stage = "Completed after QA revalidation".to_string();
        updated.progress = 1.0;
        updated.output_path = Some(output.to_string_lossy().to_string());
        updated.qa_report_path = Some(
            best_quality::qa_report_path(&output)
                .to_string_lossy()
                .to_string(),
        );
        updated.contact_sheet_path = Some(
            output
                .with_extension("qa.png")
                .to_string_lossy()
                .to_string(),
        );
        updated.error_code = None;
        updated.error = None;
        updated.updated_at = jobs::chrono_like_now();
        let result = updated.clone();
        jobs::save(&app_data_dir, &records)?;
        Ok::<JobRecord, AppError>(result)
    })
    .await
    .map_err(|error| AppError::Io(format!("QA revalidation failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub async fn enqueue_best_quality_job(
    app: AppHandle,
    state: State<'_, AppState>,
    request: EnqueueBestQualityRequest,
) -> Result<JobRecord, AppErrorDto> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    let project =
        service::load_project(&app_data_dir, &request.project_id).map_err(AppErrorDto::from)?;
    let project_directory =
        service::project_directory(&app_data_dir, &project.id).map_err(AppErrorDto::from)?;
    best_quality::validate_calibration(&project_directory, &project).map_err(AppErrorDto::from)?;
    let hardware = best_quality::detect_hardware();
    if !hardware.supported {
        return Err(AppError::InvalidRequest(
            "Best-quality final requires at least 4 GB NVIDIA CUDA VRAM.".to_string(),
        )
        .into());
    }
    let _store_guard = state
        .job_store_lock
        .lock()
        .map_err(|_| AppErrorDto::from(AppError::Io("Job queue is unavailable.".to_string())))?;
    let mut records = jobs::load(&app_data_dir).map_err(AppErrorDto::from)?;
    let mut record = jobs::new_record(
        project.id.clone(),
        project.source.file_name.clone(),
        request.output_root,
        JobStatus::Queued,
    );
    record.output_name = request.output_name;
    record.scan_range = project
        .calibration
        .as_ref()
        .and_then(|profile| profile.scan_range);
    record.hardware_profile = Some(format!(
        "{} · {} · {} MB",
        hardware.tier, hardware.gpu_name, hardware.vram_mb
    ));
    record.replacement_config = request
        .replacement
        .map(serde_json::to_value)
        .transpose()
        .map_err(AppError::from)
        .map_err(AppErrorDto::from)?;
    records.push(record.clone());
    jobs::save(&app_data_dir, &records).map_err(AppErrorDto::from)?;
    drop(_store_guard);
    start_job_worker(app, state);
    Ok(record)
}

#[tauri::command]
pub async fn cancel_job(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
) -> Result<(), AppErrorDto> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    let _store_guard = state
        .job_store_lock
        .lock()
        .map_err(|_| AppErrorDto::from(AppError::Io("Job queue is unavailable.".to_string())))?;
    let mut records = jobs::load(&app_data_dir).map_err(AppErrorDto::from)?;
    if let Some(record) = records.iter_mut().find(|record| record.id == job_id) {
        if matches!(
            record.status,
            JobStatus::Preparing
                | JobStatus::Inferencing
                | JobStatus::Encoding
                | JobStatus::Verifying
        ) {
            state.render_cancel.store(true, Ordering::Relaxed);
        } else if matches!(
            record.status,
            JobStatus::Queued | JobStatus::AwaitingReview | JobStatus::Scanning
        ) {
            record.status = JobStatus::Canceled;
            record.updated_at = jobs::chrono_like_now();
            jobs::save(&app_data_dir, &records).map_err(AppErrorDto::from)?;
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn regen_job(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
) -> Result<JobRecord, AppErrorDto> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| AppError::Io(error.to_string()))?;
    let _store_guard = state
        .job_store_lock
        .lock()
        .map_err(|_| AppErrorDto::from(AppError::Io("Job queue is unavailable.".to_string())))?;
    let mut records = jobs::load(&app_data_dir).map_err(AppErrorDto::from)?;
    let record = records
        .iter_mut()
        .find(|record| record.id == job_id)
        .ok_or_else(|| AppError::InvalidRequest("Job not found.".to_string()))?;
    if matches!(
        record.status,
        JobStatus::Queued
            | JobStatus::Preparing
            | JobStatus::Inferencing
            | JobStatus::Encoding
            | JobStatus::Verifying
    ) {
        return Err(AppError::InvalidRequest(
            "A queued or running job cannot be regenerated. Cancel it first.".to_string(),
        )
        .into());
    }
    let project_dir =
        service::project_directory(&app_data_dir, &record.project_id).map_err(AppErrorDto::from)?;
    let mut project =
        service::load_project(&app_data_dir, &record.project_id).map_err(AppErrorDto::from)?;
    project.calibration = None;
    if let Some(templates) = project.watermark.templates.as_mut() {
        templates.mask = Some("templates/mask.png".to_string());
    }
    let calibration_dir = project_dir.join("calibration");
    if calibration_dir.is_dir() {
        fs::remove_dir_all(calibration_dir).map_err(AppError::from)?;
    }
    service::save_project_atomic(&project_dir, &project).map_err(AppErrorDto::from)?;
    record.status = JobStatus::AwaitingReview;
    record.stage = "Rescan samples".to_string();
    record.progress = 0.0;
    record.error = None;
    record.updated_at = jobs::chrono_like_now();
    let result = record.clone();
    jobs::save(&app_data_dir, &records).map_err(AppErrorDto::from)?;
    Ok(result)
}

/// Starts the serialized queue worker during app startup as well as after a
/// new enqueue.  This lets QUEUED jobs continue after a normal app restart;
/// `jobs::open_database` has already converted jobs that were mid-stage into
/// INTERRUPTED before the worker begins.
pub fn start_pending_job_worker(app: AppHandle, state: State<'_, AppState>) {
    start_job_worker(app, state);
}

fn start_job_worker(app: AppHandle, state: State<'_, AppState>) {
    let running = Arc::clone(&state.job_worker_running);
    let gpu_lock = Arc::clone(&state.best_quality_gpu_lock);
    let store_lock = Arc::clone(&state.job_store_lock);
    let render_cancel = Arc::clone(&state.render_cancel);
    let mut guard = match running.lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };
    if *guard {
        return;
    }
    *guard = true;
    drop(guard);
    tauri::async_runtime::spawn_blocking(move || {
        loop {
            let app_data_dir = match app.path().app_data_dir() {
                Ok(path) => path,
                Err(_) => break,
            };
            let (records, index) = match store_lock.lock() {
                Ok(_store_guard) => {
                    let mut records = match jobs::load(&app_data_dir) {
                        Ok(records) => records,
                        Err(_) => break,
                    };
                    let Some(index) = records
                        .iter()
                        .position(|record| record.status == JobStatus::Queued)
                    else {
                        break;
                    };
                    records[index].status = JobStatus::Preparing;
                    records[index].stage = "Preparing calibration".to_string();
                    records[index].updated_at = jobs::chrono_like_now();
                    if jobs::save(&app_data_dir, &records).is_err() {
                        break;
                    }
                    (records, index)
                }
                Err(_) => break,
            };
            let job_id = records[index].id.clone();
            render_cancel.store(false, Ordering::Relaxed);
            let result = (|| -> Result<PathBuf, AppError> {
                let _gpu_job = gpu_lock
                    .lock()
                    .map_err(|_| AppError::Io("GPU queue unavailable.".to_string()))?;
                let project = service::load_project(&app_data_dir, &records[index].project_id)?;
                let directory = service::project_directory(&app_data_dir, &project.id)?;
                ffmpeg::ensure_tools_available()?;
                let replacement = records[index]
                    .replacement_config
                    .clone()
                    .map(serde_json::from_value::<BestQualityReplacement>)
                    .transpose()?;
                best_quality::render_best_quality(
                    &directory,
                    &project,
                    replacement.as_ref(),
                    records[index].output_root.as_deref(),
                    records[index].output_name.as_deref(),
                    &render_cancel,
                    |phase, current, total| {
                        if let Ok(_store_guard) = store_lock.lock() {
                            if let Ok(mut latest) = jobs::load(&app_data_dir) {
                                if let Some(job) = latest.iter_mut().find(|job| job.id == job_id) {
                                    job.stage = phase.to_string();
                                    // Check preparation before the generic AI label: the mask
                                    // preparation phase also contains "AI" in its user-facing
                                    // text, but must remain PREPARING in Queue/History.
                                    job.status =
                                        if phase.contains("QA") || phase.contains("Decoding") {
                                            JobStatus::Verifying
                                        } else if phase.contains("Preparing")
                                            || phase.contains("Validating")
                                        {
                                            JobStatus::Preparing
                                        } else if phase.contains("AI") {
                                            JobStatus::Inferencing
                                        } else {
                                            JobStatus::Encoding
                                        };
                                    job.progress = if total == 0 {
                                        0.0
                                    } else {
                                        current as f64 / total as f64
                                    };
                                    job.current_frame = Some(current);
                                    job.updated_at = jobs::chrono_like_now();
                                    let _ = jobs::save(&app_data_dir, &latest);
                                }
                            }
                        }
                        let _ = app.emit(
                            "operation-progress",
                            OperationProgress {
                                phase: phase.to_string(),
                                current_frame: current,
                                total_frames: total,
                                progress: if total == 0 {
                                    0.0
                                } else {
                                    current as f64 / total as f64
                                },
                            },
                        );
                    },
                )
            })();
            if let Ok(_store_guard) = store_lock.lock() {
                let mut latest = jobs::load(&app_data_dir).unwrap_or_default();
                if let Some(job) = latest.iter_mut().find(|job| job.id == job_id) {
                    match result {
                        Ok(output) => {
                            if let Ok(mut persisted_project) =
                                service::load_project(&app_data_dir, &job.project_id)
                            {
                                if let Ok(directory) =
                                    service::project_directory(&app_data_dir, &job.project_id)
                                {
                                    if let Ok(calibration) = best_quality::calibration_metadata(
                                        &directory,
                                        &persisted_project,
                                    ) {
                                        persisted_project.calibration = Some(calibration.clone());
                                        if let Some(templates) =
                                            persisted_project.watermark.templates.as_mut()
                                        {
                                            templates.mask = Some(calibration.mask_path);
                                        }
                                        let _ = service::save_project_atomic(
                                            &directory,
                                            &persisted_project,
                                        );
                                    }
                                }
                            }
                            job.status = JobStatus::Completed;
                            job.stage = "Completed".to_string();
                            job.progress = 1.0;
                            job.output_path = Some(output.to_string_lossy().to_string());
                            job.qa_report_path = Some(
                                best_quality::qa_report_path(&output)
                                    .to_string_lossy()
                                    .to_string(),
                            );
                            job.contact_sheet_path = Some(
                                output
                                    .with_extension("qa.png")
                                    .to_string_lossy()
                                    .to_string(),
                            );
                            job.error = None;
                        }
                        Err(error) if matches!(error, AppError::OperationCancelled) => {
                            job.status = JobStatus::Canceled;
                            job.stage = "Canceled".to_string();
                            job.error = Some(error.to_string());
                        }
                        Err(AppError::QualityNeedsReview(report)) => {
                            job.status = JobStatus::NeedsReview;
                            job.stage = "Quality review required".to_string();
                            job.output_path = report
                                .strip_suffix(".qa.json")
                                .map(|value| format!("{value}.mp4"));
                            job.qa_report_path = Some(report.clone());
                            job.contact_sheet_path = report
                                .strip_suffix(".qa.json")
                                .map(|value| format!("{value}.qa.png"));
                            job.error_code = Some("QUALITY_NEEDS_REVIEW".to_string());
                            job.error = Some(format!("Review QA report: {report}"));
                        }
                        Err(error) => {
                            job.status = JobStatus::Failed;
                            job.stage = "Failed".to_string();
                            job.error = Some(error.to_string());
                        }
                    }
                    job.updated_at = jobs::chrono_like_now();
                }
                let _ = jobs::save(&app_data_dir, &latest);
            }
        }
        if let Ok(mut guard) = running.lock() {
            *guard = false;
        }
    });
}

#[tauri::command]
pub async fn suggest_best_quality_samples(
    app: AppHandle,
    request: BestQualitySamplesRequest,
) -> Result<Vec<best_quality::BestQualitySample>, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let project = service::load_project(&app_data_dir, &request.project_id)?;
        let directory = service::project_directory(&app_data_dir, &project.id)?;
        ffmpeg::ensure_tools_available()?;
        best_quality::find_best_samples(
            &directory,
            &project,
            best_quality::BestQualityScanOptions {
                scan_round: request.scan_round,
                excluded_frames: &request.exclude_frames,
                excluded_scene_signatures: &request.exclude_scene_signatures,
                roi: request.roi.as_ref(),
                anchor_frame: request.anchor_frame.unwrap_or(0),
                scan_range: request.scan_range,
            },
            |current, total| {
                let progress = OperationProgress {
                    phase: if request.scan_round == 0 {
                        "Scanning all six Learna AI trajectory phases".to_string()
                    } else {
                        format!(
                            "Scanning all six phases for alternatives (pass {})",
                            request.scan_round + 1
                        )
                    },
                    current_frame: current,
                    total_frames: total,
                    progress: if total == 0 {
                        0.0
                    } else {
                        current as f64 / total as f64
                    },
                };
                let _ = app.emit("operation-progress", progress);
            },
        )
    })
    .await
    .map_err(|error| AppError::Io(format!("Best-quality sample scan failed: {error}")))?
    .map_err(Into::into)
}

#[tauri::command]
pub fn detect_hardware() -> best_quality::HardwareProfile {
    best_quality::detect_hardware()
}

#[tauri::command]
pub fn detect_runtime_health() -> best_quality::RuntimeHealth {
    best_quality::detect_runtime_health()
}

#[tauri::command]
pub fn cancel_render(state: State<'_, AppState>) {
    state.render_cancel.store(true, Ordering::Relaxed);
}

#[tauri::command]
pub async fn get_project_asset_path(
    app: AppHandle,
    project_id: String,
    asset: String,
) -> Result<String, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let directory = service::project_directory(&app_data_dir, &project_id)?;
        let relative = Path::new(&asset);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(AppError::InvalidRequest(
                "Invalid project asset path.".to_string(),
            ));
        }
        let path = directory.join(relative);
        if !path.is_file() {
            return Err(AppError::InvalidRequest(
                "Project asset was not found.".to_string(),
            ));
        }
        Ok(path.to_string_lossy().to_string())
    })
    .await
    .map_err(|error| AppError::Io(format!("Asset path task failed: {error}")))?
    .map_err(Into::into)
}

/// Reads a project-owned image as bytes so the WebView can create a same-origin
/// data URL. Drawing a `tauri://`/`asset://` image directly onto a canvas can
/// taint it and make `toBlob()` fail during mask editing.
#[tauri::command]
pub async fn read_project_asset_bytes(
    app: AppHandle,
    request: ReadProjectAssetRequest,
) -> Result<Vec<u8>, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data_dir = app
            .path()
            .app_data_dir()
            .map_err(|error| AppError::Io(error.to_string()))?;
        let directory = service::project_directory(&app_data_dir, &request.project_id)?;
        let requested = PathBuf::from(&request.asset);
        let path = if requested.is_absolute() {
            let canonical_directory = fs::canonicalize(&directory)?;
            let canonical_requested = fs::canonicalize(&requested).map_err(|_| {
                AppError::InvalidRequest("Project asset was not found.".to_string())
            })?;
            if !canonical_requested.starts_with(&canonical_directory) {
                return Err(AppError::InvalidRequest(
                    "Invalid project asset path.".to_string(),
                ));
            }
            canonical_requested
        } else {
            if requested
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                return Err(AppError::InvalidRequest(
                    "Invalid project asset path.".to_string(),
                ));
            }
            directory.join(requested)
        };
        if !path.is_file() {
            return Err(AppError::InvalidRequest(
                "Project asset was not found.".to_string(),
            ));
        }
        fs::read(path).map_err(AppError::from)
    })
    .await
    .map_err(|error| AppError::Io(format!("Reading project asset failed: {error}")))?
    .map_err(Into::into)
}

pub fn validate_bbox(
    bbox: &BoundingBox,
    source_width: u32,
    source_height: u32,
) -> Result<(), AppError> {
    if !bbox.x.is_finite()
        || !bbox.y.is_finite()
        || !bbox.width.is_finite()
        || !bbox.height.is_finite()
        || bbox.x < 0.0
        || bbox.y < 0.0
        || bbox.width < 8.0
        || bbox.height < 8.0
        || bbox.x + bbox.width > f64::from(source_width)
        || bbox.y + bbox.height > f64::from(source_height)
    {
        return Err(AppError::InvalidBoundingBox);
    }
    Ok(())
}

pub fn clamp_and_pad_bbox(
    bbox: &BoundingBox,
    source_width: u32,
    source_height: u32,
    padding: u32,
) -> Result<BoundingBox, AppError> {
    validate_bbox(bbox, source_width, source_height)?;
    let max_x = f64::from(source_width);
    let max_y = f64::from(source_height);
    let padding = f64::from(padding);
    let x = (bbox.x - padding).max(0.0).floor();
    let y = (bbox.y - padding).max(0.0).floor();
    let right = (bbox.x + bbox.width + padding).min(max_x).ceil();
    let bottom = (bbox.y + bbox.height + padding).min(max_y).ceil();
    Ok(BoundingBox {
        x,
        y,
        width: (right - x).max(1.0),
        height: (bottom - y).max(1.0),
    })
}

fn focus_crop(bbox: &BoundingBox, source_width: u32, source_height: u32) -> BoundingBox {
    let width = (bbox.width * 3.0).max(520.0).min(f64::from(source_width));
    let height = (bbox.height * 4.0).max(300.0).min(f64::from(source_height));
    let center_x = bbox.x + bbox.width / 2.0;
    let center_y = bbox.y + bbox.height / 2.0;
    BoundingBox {
        x: (center_x - width / 2.0).clamp(0.0, f64::from(source_width) - width),
        y: (center_y - height / 2.0).clamp(0.0, f64::from(source_height) - height),
        width,
        height,
    }
}

pub fn frame_to_timestamp(frame: u64, fps: f64) -> f64 {
    if fps > 0.0 && fps.is_finite() {
        frame as f64 / fps
    } else {
        0.0
    }
}

pub fn timestamp_to_frame(timestamp_seconds: f64, fps: f64) -> u64 {
    if timestamp_seconds > 0.0 && fps > 0.0 && timestamp_seconds.is_finite() && fps.is_finite() {
        (timestamp_seconds * fps).round() as u64
    } else {
        0
    }
}

fn is_supported_video(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "mov" | "mkv" | "webm" | "m4v")
    )
}

/// Windows canonicalize may return an extended-length path (`\\?\\...`),
/// which is valid for native APIs but is not accepted consistently by the
/// WebView asset URL conversion. Persist the conventional absolute path.
fn normalize_canonical_path(path: PathBuf) -> PathBuf {
    let value = path.to_string_lossy();
    if let Some(rest) = value.strip_prefix("\\\\?\\UNC\\") {
        return PathBuf::from(format!("\\\\{rest}"));
    }
    if let Some(rest) = value.strip_prefix("\\\\?\\") {
        return PathBuf::from(rest);
    }
    path
}

fn normalize_interpolated_confidence(frames: &mut [TrackingFrame]) {
    for frame in frames {
        if frame.status == TrackingStatus::Interpolated {
            // Interpolation contains no image observation, so persisted values
            // from older builds must never be presented as measured confidence.
            frame.confidence = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_rejects_out_of_bounds_boxes() {
        let valid = BoundingBox {
            x: 620.0,
            y: 421.0,
            width: 220.0,
            height: 58.0,
        };
        assert!(validate_bbox(&valid, 1080, 1920).is_ok());
        assert!(validate_bbox(
            &BoundingBox {
                x: -1.0,
                ..valid.clone()
            },
            1080,
            1920
        )
        .is_err());
        assert!(validate_bbox(&BoundingBox { x: 900.0, ..valid }, 1080, 1920).is_err());
        assert!(validate_bbox(
            &BoundingBox {
                width: 7.0,
                height: 8.0,
                ..valid
            },
            1080,
            1920
        )
        .is_err());
    }

    #[test]
    fn clamps_padding_to_source_bounds() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 12.0,
            width: 10.0,
            height: 11.0,
        };
        let padded = clamp_and_pad_bbox(&bbox, 100, 100, 4).unwrap();
        assert_eq!(padded.x, 6.0);
        assert_eq!(padded.y, 8.0);
        assert_eq!(padded.width, 18.0);
        assert_eq!(padded.height, 19.0);
    }

    #[test]
    fn converts_frame_and_timestamp() {
        assert_eq!(frame_to_timestamp(84, 30.0), 2.8);
        assert_eq!(timestamp_to_frame(2.8, 30.0), 84);
    }

    #[test]
    fn normalizes_windows_extended_length_paths() {
        let normalized = normalize_canonical_path(PathBuf::from(r"\\?\C:\videos\test.mp4"));
        assert_eq!(normalized, PathBuf::from(r"C:\videos\test.mp4"));
    }

    #[test]
    fn clears_legacy_confidence_from_interpolated_frames() {
        let mut frames = vec![TrackingFrame {
            frame: 1,
            timestamp_seconds: 1.0 / 30.0,
            bbox: BoundingBox {
                x: 10.0,
                y: 20.0,
                width: 30.0,
                height: 12.0,
            },
            confidence: 0.75,
            status: TrackingStatus::Interpolated,
            source: crate::project::model::TrackingSource::Interpolated,
            locked: false,
            scores: crate::project::model::TrackingScores {
                template: 0.0,
                highpass: 0.0,
                edge: 0.0,
                motion: 0.0,
                position: 0.0,
                size: 1.0,
                optical_flow: None,
                forward_backward: None,
                motion_smoothness: Some(1.0),
                match_margin: None,
            },
        }];

        normalize_interpolated_confidence(&mut frames);

        assert_eq!(frames[0].confidence, 0.0);
        assert_eq!(frames[0].status, TrackingStatus::Interpolated);
    }
}
