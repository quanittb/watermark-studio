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
    #[error("The watermark bounding box is invalid.")]
    InvalidBoundingBox,
    #[error("Project was not found.")]
    ProjectNotFound,
    #[error("I/O error: {0}")]
    Io(String),
    #[error("JSON error: {0}")]
    Json(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Operation cancelled.")]
    OperationCancelled,
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
}

impl From<AppError> for AppErrorDto {
    fn from(error: AppError) -> Self {
        let code = match &error {
            AppError::VideoNotFound => "VIDEO_NOT_FOUND",
            AppError::UnsupportedVideo => "UNSUPPORTED_VIDEO",
            AppError::FfmpegNotFound => "FFMPEG_NOT_FOUND",
            AppError::FfprobeFailed(_) => "FFPROBE_FAILED",
            AppError::FfmpegFailed(_) => "FFMPEG_FAILED",
            AppError::InvalidBoundingBox => "INVALID_BOUNDING_BOX",
            AppError::ProjectNotFound => "PROJECT_NOT_FOUND",
            AppError::Io(_) => "IO_ERROR",
            AppError::Json(_) => "JSON_ERROR",
            AppError::InvalidRequest(_) => "INVALID_REQUEST",
            AppError::OperationCancelled => "OPERATION_CANCELLED",
        };

        Self {
            code: code.to_string(),
            message: error.to_string(),
        }
    }
}
