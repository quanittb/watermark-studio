use crate::error::AppError;
use crate::project::model::VideoMetadata;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    pix_fmt: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    duration: Option<String>,
    nb_frames: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

pub fn probe_video(path: &Path) -> Result<VideoMetadata, AppError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(path)
        .output()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                AppError::FfmpegNotFound
            } else {
                AppError::FfprobeFailed(error.to_string())
            }
        })?;

    if !output.status.success() {
        return Err(AppError::FfprobeFailed(clean_stderr(&output.stderr)));
    }

    let parsed: ProbeOutput = serde_json::from_slice(&output.stdout)
        .map_err(|error| AppError::FfprobeFailed(format!("Invalid ffprobe JSON: {error}")))?;
    let stream = parsed
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| AppError::FfprobeFailed("No video stream was found.".to_string()))?;

    let width = stream
        .width
        .filter(|value| *value > 0)
        .ok_or_else(|| AppError::FfprobeFailed("Video width is unavailable.".to_string()))?;
    let height = stream
        .height
        .filter(|value| *value > 0)
        .ok_or_else(|| AppError::FfprobeFailed("Video height is unavailable.".to_string()))?;
    let fps = stream
        .avg_frame_rate
        .as_deref()
        .and_then(parse_frame_rate)
        .or_else(|| stream.r_frame_rate.as_deref().and_then(parse_frame_rate))
        .ok_or_else(|| AppError::FfprobeFailed("Video frame rate is unavailable.".to_string()))?;
    let duration_seconds = stream
        .duration
        .as_deref()
        .and_then(parse_duration)
        .or_else(|| {
            parsed
                .format
                .as_ref()
                .and_then(|format| format.duration.as_deref())
                .and_then(parse_duration)
        })
        .ok_or_else(|| AppError::FfprobeFailed("Video duration is unavailable.".to_string()))?;
    let frame_count = stream
        .nb_frames
        .as_deref()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| (duration_seconds * fps).round().max(1.0) as u64);

    Ok(VideoMetadata {
        width,
        height,
        duration_seconds,
        fps,
        frame_count,
        codec: stream.codec_name.clone(),
        pixel_format: stream.pix_fmt.clone(),
    })
}

fn parse_duration(value: &str) -> Option<f64> {
    let duration = value.parse::<f64>().ok()?;
    (duration.is_finite() && duration >= 0.0).then_some(duration)
}

pub fn parse_frame_rate(value: &str) -> Option<f64> {
    let mut parts = value.split('/');
    let numerator = parts.next()?.parse::<f64>().ok()?;
    let denominator = parts.next()?.parse::<f64>().ok()?;
    if parts.next().is_some()
        || !numerator.is_finite()
        || !denominator.is_finite()
        || denominator == 0.0
    {
        return None;
    }

    let fps = numerator / denominator;
    (fps.is_finite() && fps > 0.0).then_some(fps)
}

fn clean_stderr(stderr: &[u8]) -> String {
    let message = String::from_utf8_lossy(stderr).trim().to_string();
    if message.is_empty() {
        "ffprobe did not provide diagnostic output.".to_string()
    } else {
        message.chars().take(500).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::parse_frame_rate;

    #[test]
    fn parses_common_rational_frame_rates() {
        assert!((parse_frame_rate("30000/1001").unwrap() - 29.970029).abs() < 0.000001);
        assert_eq!(parse_frame_rate("30/1"), Some(30.0));
        assert_eq!(parse_frame_rate("25/1"), Some(25.0));
    }

    #[test]
    fn rejects_invalid_frame_rates() {
        assert_eq!(parse_frame_rate("0/1"), None);
        assert_eq!(parse_frame_rate("30/0"), None);
        assert_eq!(parse_frame_rate("30"), None);
    }
}
