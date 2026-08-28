use crate::error::{AppError, AppErrorDto};
use crate::media::{ffmpeg, ffprobe};
use crate::media::{mask, render};
use crate::project::model::{
    AnchorFrame, AnchorType, BoundingBox, FrameResult, ManualAnchor, Project, RemovalConfig,
    TemplatePaths, TrackingFrame, TrackingStatus, WatermarkConfig,
};
use crate::project::service;
use crate::tracking;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

const PROJECT_VERSION: u32 = 1;
const DEFAULT_TEMPLATE_PADDING: u32 = 4;

#[derive(Default)]
pub struct AppState {
    pub tracking_cancel: Arc<AtomicBool>,
    pub render_cancel: Arc<AtomicBool>,
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
pub async fn save_watermark_anchor(
    app: AppHandle,
    request: SaveWatermarkAnchorRequest,
) -> Result<Project, AppErrorDto> {
    tauri::async_runtime::spawn_blocking(move || save_watermark_anchor_sync(&app, request))
        .await
        .map_err(|error| AppError::Io(format!("Saving anchor task failed: {error}")))?
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
        })
    })
    .await
    .map_err(|error| AppError::Io(format!("Render task failed: {error}")))?
    .map_err(Into::into)
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
