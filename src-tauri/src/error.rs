use serde::Serialize;
use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Video file was not found.")]
    VideoNotFound,
    #[error("This video format is not supported.")]
    UnsupportedVideo,
    #[error("FFmpeg was not found. Install FFmpeg and ffprobe, then try again.")]
    FfmpegNotFound,
    #[error("ffprobe failed: {0}")]
    FfprobeFailed(String),
    #[error("ffmpeg failed: {0}")]
    FfmpegFailed(String),
    #[error("Storage capacity is insufficient: {0}")]
    StorageFull(String),
    #[error("The watermark bounding box is invalid.")]
    InvalidBoundingBox,
    #[error("Project was not found.")]
    ProjectNotFound,
    #[error("I/O error: {0}")]
    Io(String),
    #[error("JSON error: {0}")]
    Json(String),
    #[error("Calibration profile is corrupt: {0}")]
    CalibrationCorrupt(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Invalid scan range: {0}")]
    InvalidScanRange(String),
    #[error("ROI evidence is outside the selected scan range.")]
    RoiOutsideScanRange,
    #[error("Operation cancelled.")]
    OperationCancelled,
    #[error("Quality review is required: {0}")]
    QualityNeedsReview(String),
}

impl From<io::Error> for AppError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value.to_string())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppErrorDto {
    pub code: String,
    pub message: String,
    /// Stable processing stage for the UI dialog; clients should not need to
    /// parse the human-readable message to decide where to resume.
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
}

impl From<AppError> for AppErrorDto {
    fn from(error: AppError) -> Self {
        let code = match &error {
            AppError::VideoNotFound => "VIDEO_NOT_FOUND",
            AppError::UnsupportedVideo => "UNSUPPORTED_VIDEO",
            AppError::FfmpegNotFound => "FFMPEG_NOT_FOUND",
            AppError::FfprobeFailed(_) => "FFPROBE_FAILED",
            AppError::FfmpegFailed(_) => "FFMPEG_FAILED",
            AppError::StorageFull(_) => "STORAGE_FULL",
            AppError::InvalidBoundingBox => "INVALID_BOUNDING_BOX",
            AppError::ProjectNotFound => "PROJECT_NOT_FOUND",
            AppError::Io(_) => "IO_ERROR",
            AppError::Json(_) => "JSON_ERROR",
            AppError::CalibrationCorrupt(_) => "CALIBRATION_CORRUPT",
            AppError::InvalidRequest(_) => "INVALID_REQUEST",
            AppError::InvalidScanRange(_) => "INVALID_SCAN_RANGE",
            AppError::RoiOutsideScanRange => "ROI_OUTSIDE_SCAN_RANGE",
            AppError::OperationCancelled => "OPERATION_CANCELLED",
            AppError::QualityNeedsReview(_) => "QUALITY_NEEDS_REVIEW",
        };

        let stage = match &error {
            AppError::VideoNotFound | AppError::UnsupportedVideo | AppError::FfmpegNotFound => {
                "VALIDATE_SOURCE"
            }
            AppError::FfprobeFailed(_) => "VALIDATE_SOURCE",
            AppError::InvalidBoundingBox => "REVIEW",
            AppError::CalibrationCorrupt(_)
            | AppError::InvalidRequest(_)
            | AppError::InvalidScanRange(_)
            | AppError::RoiOutsideScanRange => "CALIBRATION",
            AppError::FfmpegFailed(message)
                if message.to_ascii_lowercase().contains("encoding") =>
            {
                "ENCODING"
            }
            AppError::StorageFull(_) => "STORAGE",
            AppError::FfmpegFailed(_) => "PROCESSING",
            AppError::QualityNeedsReview(_) => "VERIFYING",
            AppError::OperationCancelled => "CANCELED",
            AppError::ProjectNotFound | AppError::Io(_) | AppError::Json(_) => "STORAGE",
        };
        Self {
            code: code.to_string(),
            message: error.to_string(),
            stage: stage.to_string(),
            artifact_path: None,
        }
    }
}
