use crate::error::AppError;
use crate::project::model::BoundingBox;
use std::io::{ErrorKind, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

#[derive(Debug, Clone)]
pub struct GrayFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct RawVideoDecoder {
    child: Child,
    stdout: std::process::ChildStdout,
    frame_size: usize,
}

pub struct RawVideoEncoder {
    child: Child,
    stdin: ChildStdin,
}

pub fn ensure_tools_available() -> Result<(), AppError> {
    for tool in ["ffmpeg", "ffprobe"] {
        let result = Command::new(tool).arg("-version").output();
        match result {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                let diagnostic = clean_stderr(&output.stderr);
                return Err(AppError::FfmpegFailed(format!(
                    "Unable to run {tool}: {diagnostic}"
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(AppError::FfmpegNotFound);
            }
            Err(error) => {
                return Err(AppError::FfmpegFailed(format!(
                    "Unable to run {tool}: {error}"
                )));
            }
        }
    }
    Ok(())
}

pub fn extract_frame(
    video_path: &Path,
    timestamp_seconds: f64,
    output_path: &Path,
) -> Result<(), AppError> {
    run_ffmpeg([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid video path.".to_string()))?,
        "-ss",
        &format_timestamp(timestamp_seconds),
        "-frames:v",
        "1",
        "-fps_mode",
        "vfr",
        output_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid output path.".to_string()))?,
    ])
}

pub fn crop_image_from_video(
    video_path: &Path,
    timestamp_seconds: f64,
    bbox: &BoundingBox,
    output_path: &Path,
) -> Result<(), AppError> {
    let crop = format!(
        "crop={}:{}:{}:{}",
        bbox.width.round(),
        bbox.height.round(),
        bbox.x.round(),
        bbox.y.round()
    );
    run_ffmpeg([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid video path.".to_string()))?,
        "-ss",
        &format_timestamp(timestamp_seconds),
        "-vf",
        &crop,
        "-frames:v",
        "1",
        output_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid output path.".to_string()))?,
    ])
}

pub fn generate_template_variant(
    input_path: &Path,
    filter: &str,
    output_path: &Path,
) -> Result<(), AppError> {
    run_ffmpeg([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid template path.".to_string()))?,
        "-vf",
        filter,
        "-frames:v",
        "1",
        output_path
            .to_str()
            .ok_or_else(|| AppError::InvalidRequest("Invalid output path.".to_string()))?,
    ])
}

pub fn read_analysis_frames(
    video_path: &Path,
    source_width: u32,
    source_height: u32,
    analysis_long_edge: u32,
) -> Result<Vec<GrayFrame>, AppError> {
    let (width, height) = analysis_dimensions(source_width, source_height, analysis_long_edge);
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(video_path)
        .args([
            "-an",
            "-vsync",
            "0",
            "-vf",
            &format!("scale={width}:{height}:flags=lanczos"),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(map_spawn_error)?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::FfmpegFailed("FFmpeg stdout was unavailable.".to_string()))?;
    let frame_size = (width * height) as usize;
    let mut frames = Vec::new();
    loop {
        let mut pixels = vec![0u8; frame_size];
        let first_read = stdout.read(&mut pixels)?;
        if first_read == 0 {
            break;
        }
        if first_read < frame_size {
            stdout.read_exact(&mut pixels[first_read..])?;
        }
        frames.push(GrayFrame {
            width,
            height,
            pixels,
        });
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(AppError::FfmpegFailed(clean_stderr(&output.stderr)));
    }
    if frames.is_empty() {
        return Err(AppError::FfmpegFailed(
            "FFmpeg returned no video frames.".to_string(),
        ));
    }
    Ok(frames)
}

pub fn analysis_dimensions(
    source_width: u32,
    source_height: u32,
    analysis_long_edge: u32,
) -> (u32, u32) {
    let long_edge = analysis_long_edge.max(64);
    let source_long_edge = source_width.max(source_height).max(1);
    let scale = f64::from(long_edge) / f64::from(source_long_edge);
    let width = (f64::from(source_width) * scale).round().max(1.0) as u32;
    let height = (f64::from(source_height) * scale).round().max(1.0) as u32;
    (width, height)
}

pub fn open_raw_decoder(
    video_path: &Path,
    width: u32,
    height: u32,
) -> Result<RawVideoDecoder, AppError> {
    let mut child = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(video_path)
        .args(["-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(map_spawn_error)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::FfmpegFailed("FFmpeg stdout was unavailable.".to_string()))?;
    Ok(RawVideoDecoder {
        child,
        stdout,
        frame_size: (width * height * 3) as usize,
    })
}

impl RawVideoDecoder {
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, AppError> {
        let mut frame = vec![0u8; self.frame_size];
        let first = self.stdout.read(&mut frame)?;
        if first == 0 {
            return Ok(None);
        }
        self.stdout.read_exact(&mut frame[first..])?;
        Ok(Some(frame))
    }

    pub fn finish(self) -> Result<(), AppError> {
        let output = self.child.wait_with_output()?;
        if output.status.success() {
            Ok(())
        } else {
            Err(AppError::FfmpegFailed(clean_stderr(&output.stderr)))
        }
    }
}

pub fn open_raw_encoder(
    width: u32,
    height: u32,
    fps: f64,
    output_path: &Path,
) -> Result<RawVideoEncoder, AppError> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
        ])
        .arg(format!("{width}x{height}"))
        .args([
            "-r",
            &format!("{fps:.12}"),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(map_spawn_error)?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| AppError::FfmpegFailed("FFmpeg stdin was unavailable.".to_string()))?;
    Ok(RawVideoEncoder { child, stdin })
}

impl RawVideoEncoder {
    pub fn write_frame(&mut self, frame: &[u8]) -> Result<(), AppError> {
        self.stdin.write_all(frame).map_err(|error| {
            if error.kind() == ErrorKind::BrokenPipe {
                AppError::FfmpegFailed("FFmpeg encoder closed unexpectedly.".to_string())
            } else {
                AppError::Io(error.to_string())
            }
        })
    }

    pub fn finish(self) -> Result<(), AppError> {
        drop(self.stdin);
        let output = self.child.wait_with_output()?;
        if output.status.success() {
            Ok(())
        } else {
            Err(AppError::FfmpegFailed(clean_stderr(&output.stderr)))
        }
    }
}

fn run_ffmpeg<const N: usize>(args: [&str; N]) -> Result<(), AppError> {
    let output = Command::new("ffmpeg")
        .args(args)
        .output()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                AppError::FfmpegNotFound
            } else {
                AppError::FfmpegFailed(error.to_string())
            }
        })?;

    if output.status.success() {
        Ok(())
    } else {
        let message = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(AppError::FfmpegFailed(if message.is_empty() {
            "FFmpeg did not provide diagnostic output.".to_string()
        } else {
            message.chars().take(500).collect()
        }))
    }
}

fn map_spawn_error(error: std::io::Error) -> AppError {
    if error.kind() == ErrorKind::NotFound {
        AppError::FfmpegNotFound
    } else {
        AppError::FfmpegFailed(error.to_string())
    }
}

fn clean_stderr(stderr: &[u8]) -> String {
    let message = String::from_utf8_lossy(stderr).trim().to_string();
    if message.is_empty() {
        "FFmpeg did not provide diagnostic output.".to_string()
    } else {
        message.chars().take(500).collect()
    }
}

fn format_timestamp(timestamp_seconds: f64) -> String {
    format!("{:.6}", timestamp_seconds.max(0.0))
}
