use crate::commands::project::BestQualityReplacement;
use crate::error::AppError;
use crate::media::ffmpeg;
use crate::project::model::{
    BoundingBox, CalibrationPreset, CalibrationProfile, CalibrationQuality, CalibrationStatus,
    Project, SourceFingerprint,
};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const SUPPORTED_WIDTH: u32 = 1080;
const SUPPORTED_HEIGHT: u32 = 1920;
// Best-quality V6 discovers the active interval from calibration.  The old
// fixed 48-frame start is retained only by the legacy path/tests.
const LEGACY_FIRST_WATERMARK_FRAME: u64 = 48;
const DEFAULT_PROPAINTER_PYTHON: &str = r"D:\propainter-watermark-venv\Scripts\python.exe";
const DEFAULT_PROPAINTER_ROOT: &str = r"D:\propainter-watermark-work";
const DEFAULT_WORK_ROOT: &str = r"D:\watermark-studio-ai-work";
const CALIBRATION_SCRIPT: &str = "calibrate_best_quality.py";
const ADAPTIVE_CALIBRATION_SCRIPT: &str = "calibrate_trajectory_v6.py";
const FIND_SAMPLES_SCRIPT: &str = "find_learna_samples.py";
const AUDIT_SCRIPT: &str = "audit_watermark_detection.py";
const MIN_MASK_COVERAGE: u64 = 350;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BestQualitySample {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
    pub mask_coverage: u64,
    pub mask_peak: u8,
    pub background_complexity: f64,
    pub temporal_instability: f64,
    pub glyph_correlation: f64,
    pub glyph_iou: f64,
    pub contamination: f64,
    pub temporal_pass_count: u8,
    pub score: f64,
    pub scene_signature: String,
    pub preview_path: String,
    pub mask_path: String,
    pub editor_mask_path: String,
    #[serde(default)]
    pub roi_fallback: bool,
    #[serde(default)]
    pub trajectory_phase_offset: i32,
}

pub struct BestQualityScanOptions<'a> {
    pub scan_round: u32,
    pub excluded_frames: &'a [u64],
    pub excluded_scene_signatures: &'a [String],
    pub roi: Option<&'a BoundingBox>,
    pub anchor_frame: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_mb: u64,
    pub cuda_available: bool,
    pub supported: bool,
    pub tier: String,
    pub width: u32,
    pub height: u32,
    pub core_length: u32,
    pub context: u32,
}

pub fn detect_hardware() -> HardwareProfile {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let parsed = output
        .ok()
        .filter(|value| value.status.success())
        .and_then(|value| {
            let line = String::from_utf8_lossy(&value.stdout)
                .lines()
                .next()?
                .to_string();
            let mut fields = line.split(',').map(str::trim);
            Some((
                fields.next()?.to_string(),
                fields.next()?.parse::<u64>().ok()?,
            ))
        });
    let cuda_available = parsed.is_some();
    let (gpu_name, vram_mb) = parsed.unwrap_or_else(|| ("CUDA GPU not detected".to_string(), 0));
    let (tier, width, height, core_length, context) = match vram_mb {
        12_000.. => ("MAX", 576, 1024, 120, 16),
        8_000.. => ("HIGH", 432, 768, 90, 12),
        6_000.. => ("BALANCED", 384, 672, 72, 10),
        4_000.. => ("SAFE", 288, 512, 60, 8),
        _ => ("UNSUPPORTED", 0, 0, 0, 0),
    };
    HardwareProfile {
        gpu_name,
        vram_mb,
        cuda_available,
        supported: vram_mb >= 4_000,
        tier: tier.to_string(),
        width,
        height,
        core_length,
        context,
    }
}

/// Runs the verified full-frame ProPainter path for the known repeating Learna
/// AI watermark. This deliberately has a narrow compatibility contract: an
/// unsupported watermark must fail before it can modify unrelated pixels.
pub fn render_best_quality<F>(
    project_directory: &Path,
    project: &Project,
    replacement: Option<&BestQualityReplacement>,
    output_root: Option<&str>,
    output_name: Option<&str>,
    cancel: &AtomicBool,
    mut progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(&str, u64, u64),
{
    validate_project(project_directory, project)?;
    let python = configured_path(
        "WATERMARK_STUDIO_PROPAINTER_PYTHON",
        DEFAULT_PROPAINTER_PYTHON,
    );
    let propainter_root =
        configured_path("WATERMARK_STUDIO_PROPAINTER_ROOT", DEFAULT_PROPAINTER_ROOT);
    let workspace_root = configured_path("WATERMARK_STUDIO_WORK_ROOT", DEFAULT_WORK_ROOT);
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let pipeline = repo_root
        .join("tools")
        .join("propainter_periodic_pipeline.py");
    let chunks = repo_root.join("tools").join("run_propainter_chunks.py");
    let qa_script = repo_root.join("tools").join("quality_qa_v4.py");

    require_file(&python, "ProPainter Python")?;
    require_file(
        &propainter_root.join("inference_propainter.py"),
        "ProPainter inference",
    )?;
    require_file(&pipeline, "best-quality pipeline")?;
    require_file(&chunks, "best-quality chunk runner")?;
    require_file(&qa_script, "Best-quality QA script")?;
    validate_replacement(replacement)?;

    let profile_path = validate_calibration_profile(project_directory, project)?;
    let profile_value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&profile_path)?).map_err(|error| {
            AppError::InvalidRequest(format!("Calibration profile is invalid JSON: {error}"))
        })?;
    let first_frame = profile_value
        .get("firstWatermarkFrame")
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
        .min(project.video.frame_count.saturating_sub(1));
    let last_profile_frame = profile_value
        .get("lastWatermarkFrame")
        .and_then(|value| value.as_u64())
        .unwrap_or_else(|| project.video.frame_count.saturating_sub(1));
    let hardware = detect_hardware();
    if !hardware.supported {
        return Err(AppError::InvalidRequest(
            "Best-quality final requires an NVIDIA CUDA GPU with at least 4 GB VRAM.".to_string(),
        ));
    }

    let workspace = workspace_root.join(&project.id);
    let result_root = workspace.join("results");
    let output = next_output_path(project, output_root, output_name)?;
    let output_stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| AppError::InvalidRequest("Invalid output file name.".to_string()))?;
    let draft = output.with_file_name(format!("{output_stem}.review.mp4"));
    let project_json = project_directory.join("project.json");
    let last_frame = project.video.frame_count.saturating_sub(1);

    progress("Validating best-quality render", 0, 5);
    check_cancel(cancel)?;
    progress("Preparing full-resolution AI masks", 1, 5);
    run_process(
        Command::new(&python)
            .arg(&pipeline)
            .arg("prepare")
            .arg(&project_json)
            .arg("-")
            .arg(&workspace)
            .arg("--profile")
            .arg(&profile_path)
            .arg("--start-frame")
            .arg(first_frame.to_string())
            .arg("--end-frame")
            .arg(last_profile_frame.min(last_frame).to_string())
            .arg("--full-frame"),
        cancel,
        "Preparing AI masks",
    )?;

    progress("Running temporal AI restoration", 2, 5);
    if let Err(error) = run_propainter_chunks(
        &python,
        &chunks,
        &workspace,
        &result_root,
        &propainter_root,
        &hardware,
        cancel,
        &mut progress,
    ) {
        let message = error.to_string().to_ascii_lowercase();
        if !message.contains("out of memory") && !message.contains("cuda oom") {
            return Err(error);
        }
        let fallback = lower_hardware_profile(&hardware).ok_or(error)?;
        progress(
            &format!("CUDA OOM; retrying once with {} profile", fallback.tier),
            2,
            5,
        );
        run_propainter_chunks(
            &python,
            &chunks,
            &workspace,
            &result_root,
            &propainter_root,
            &fallback,
            cancel,
            &mut progress,
        )?;
    }

    progress("Encoding final full-resolution video", 3, 5);
    let mut composite = Command::new(&python);
    composite
        .arg(&pipeline)
        .arg("composite")
        .arg(&project_json)
        .arg(&workspace)
        .arg(result_root.join("merged-frames"))
        .arg(&draft);
    append_replacement_arguments(&mut composite, replacement);
    run_process(&mut composite, cancel, "Encoding final video")?;
    if !draft.is_file() {
        return Err(AppError::FfmpegFailed(
            "Best-quality pipeline completed without creating an output video.".to_string(),
        ));
    }
    progress("Decoding final output for QA", 4, 5);
    check_cancel(cancel)?;
    ffmpeg::verify_video_decode(&draft)?;
    run_quality_qa(&python, &qa_script, project, &profile_path, &draft, cancel)?;
    let draft_report = qa_report_path(&draft);
    let draft_sheet = draft.with_extension("qa.png");
    fs::rename(&draft, &output)?;
    if draft_report.is_file() {
        fs::rename(draft_report, qa_report_path(&output))?;
    }
    if draft_sheet.is_file() {
        fs::rename(draft_sheet, output.with_extension("qa.png"))?;
    }

    // These are generated cache frames, never user source or final output.
    // They are expensive to retain (several GB) and can always be recreated.
    let _ = fs::remove_dir_all(&workspace);
    progress("Best-quality render complete", 5, 5);
    Ok(output)
}

pub fn qa_report_path(output: &Path) -> PathBuf {
    output.with_extension("qa.json")
}

pub fn calibration_metadata(
    project_directory: &Path,
    project: &Project,
) -> Result<CalibrationProfile, AppError> {
    let profile_path = project_directory.join("calibration").join("profile.json");
    let profile: serde_json::Value = serde_json::from_str(&fs::read_to_string(&profile_path)?)
        .map_err(|error| {
            AppError::CalibrationCorrupt(format!(
            "non-standard JSON value or truncated write; regenerate CalibrationProfileV6 ({error})"
        ))
        })?;
    let profile_frame_count = profile
        .get("frameCount")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let mask_pixels = profile
        .get("maskPixels")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            profile
                .get("qualityGate")
                .and_then(|value| value.get("maskPixels"))
                .and_then(|value| value.as_u64())
        })
        .unwrap_or(0);
    let profile_version = profile
        .get("version")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let trajectory_ready = !matches!(profile_version, 5 | 6)
        || (profile
            .get("trajectoryGate")
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            == Some("PASSED")
            && profile.get("trajectoryModel").is_some());
    let is_ready = matches!(profile_version, 4..=6)
        && profile_frame_count == project.video.frame_count
        && mask_pixels >= MIN_MASK_COVERAGE
        && profile.get("status").and_then(|value| value.as_str()) == Some("READY")
        && profile
            .get("qualityGate")
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            == Some("PASSED")
        && profile
            .get("maskSha256")
            .and_then(|value| value.as_str())
            .is_some()
        && trajectory_ready;
    let status = if is_ready {
        CalibrationStatus::Ready
    } else if profile
        .get("version")
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
        < 4
    {
        CalibrationStatus::Stale
    } else {
        CalibrationStatus::NeedsReview
    };
    let reliable_frames = profile
        .get("reliableFrames")
        .and_then(|value| value.as_u64())
        .or_else(|| {
            profile
                .get("qualityGate")
                .and_then(|value| value.get("reliableFrames"))
                .and_then(|value| value.as_u64())
        })
        .or_else(|| {
            profile
                .get("qualityGate")
                .and_then(|value| value.get("measuredFrames"))
                .and_then(|value| value.as_u64())
        })
        .unwrap_or_else(|| {
            profile
                .get("confidence")
                .and_then(|value| value.as_array())
                .map(|values| {
                    values
                        .iter()
                        .filter(|value| value.as_f64().unwrap_or(0.0) >= 0.30)
                        .count() as u64
                })
                .unwrap_or(0)
        });
    let frame_count = profile
        .get("frameCount")
        .and_then(|value| value.as_u64())
        .unwrap_or(project.video.frame_count);
    let mask_path = profile
        .get("maskPath")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            AppError::InvalidRequest("Calibration profile has no mask path.".to_string())
        })?;
    Ok(CalibrationProfile {
        version: profile
            .get("version")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as u32,
        preset: match profile.get("preset").and_then(|value| value.as_str()) {
            Some("GENERAL_MOVING") => CalibrationPreset::GeneralMoving,
            Some("LEARNA_AI_ADAPTIVE") => CalibrationPreset::LearnaAiAdaptive,
            _ => CalibrationPreset::LearnaAiPeriodic,
        },
        detector_version: profile
            .get("detectorVersion")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        source_fingerprint: profile
            .get("sourceFingerprint")
            .and_then(|value| serde_json::from_value::<SourceFingerprint>(value.clone()).ok()),
        profile_path: "calibration/profile.json".to_string(),
        mask_path: mask_path.to_string(),
        canonical_mask_path: profile
            .get("canonicalMaskPath")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        auto_mask_path: profile
            .get("autoMaskPath")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        blend_mask_path: profile
            .get("blendMaskPath")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        brush_delta_path: profile
            .get("brushDeltaPath")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        route: profile
            .get("route")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        trajectory_model: profile.get("trajectoryModel").cloned(),
        difficult_frames: profile
            .get("difficultFrames")
            .and_then(|value| value.as_array())
            .map(|frames| frames.iter().filter_map(|value| value.as_u64()).collect())
            .unwrap_or_default(),
        contact_sheet_path: profile
            .get("contactSheetPath")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        mask_hash: profile
            .get("maskSha256")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        profile_hash: profile
            .get("profileSha256")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        sample_frame: profile
            .get("sampleFrame")
            .and_then(|value| value.as_u64())
            .or_else(|| project.watermark.anchor.as_ref().map(|anchor| anchor.frame))
            .unwrap_or(0),
        frame_count,
        quality: CalibrationQuality {
            status,
            reliable_frames,
            low_confidence_frames: profile
                .get("qualityGate")
                .and_then(|value| value.get("interpolatedFrames"))
                .and_then(|value| value.as_u64())
                .unwrap_or_else(|| frame_count.saturating_sub(reliable_frames)),
            mask_pixels,
            glyph_coverage: profile
                .get("qualityGate")
                .and_then(|value| value.get("glyphCoverage"))
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            contamination: profile
                .get("qualityGate")
                .and_then(|value| value.get("contamination"))
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0),
            large_holes: profile
                .get("qualityGate")
                .and_then(|value| value.get("largeHoles"))
                .and_then(|value| value.as_u64())
                .unwrap_or(u64::MAX)
                .min(u64::from(u32::MAX)) as u32,
        },
    })
}

fn run_quality_qa(
    python: &Path,
    qa_script: &Path,
    project: &Project,
    profile_path: &Path,
    output: &Path,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    let report = qa_report_path(output);
    let contact_sheet = output.with_extension("qa.png");
    check_cancel(cancel)?;
    let result = Command::new(python)
        .arg(qa_script)
        .arg(&project.source.path)
        .arg(output)
        .arg(profile_path)
        .arg(&report)
        .arg(&contact_sheet)
        .output()
        .map_err(|error| AppError::Io(format!("Visual quality QA could not start: {error}")))?;
    check_cancel(cancel)?;
    if result.status.code() == Some(2) && report.is_file() {
        return Err(AppError::QualityNeedsReview(
            report.to_string_lossy().to_string(),
        ));
    }
    if !result.status.success() {
        return Err(AppError::FfmpegFailed(format!(
            "Visual quality QA failed: {}",
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }
    let body = fs::read_to_string(&report)?;
    let parsed: serde_json::Value = serde_json::from_str(&body)
        .map_err(|error| AppError::Io(format!("Unable to parse visual QA report: {error}")))?;
    if parsed.get("status").and_then(|value| value.as_str()) != Some("passed") {
        return Err(AppError::QualityNeedsReview(
            report.to_string_lossy().to_string(),
        ));
    }
    Ok(())
}

/// Runs the Learna AI canonical-glyph detector. Only hard-gate candidates are
/// returned to Review; rejected texture/background crops stay in diagnostics.
pub fn find_best_samples<F>(
    project_directory: &Path,
    project: &Project,
    options: BestQualityScanOptions<'_>,
    mut progress: F,
) -> Result<Vec<BestQualitySample>, AppError>
where
    F: FnMut(u64, u64),
{
    validate_layout(project)?;
    let candidates_directory = project_directory.join("cache").join(format!(
        "best-quality-candidates-scan-{}",
        options.scan_round
    ));
    fs::create_dir_all(&candidates_directory)?;
    let python = configured_path(
        "WATERMARK_STUDIO_PROPAINTER_PYTHON",
        DEFAULT_PROPAINTER_PYTHON,
    );
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let detector = repo_root.join("tools").join(FIND_SAMPLES_SCRIPT);
    require_file(&python, "Detector Python")?;
    require_file(&detector, "Learna AI detector")?;
    // The detector now evaluates all six stride phases in one process.  Keep
    // the progress contract phase-based so the UI does not appear finished
    // while the remaining phases are still being scanned.
    progress(0, 6);
    let mut command = Command::new(&python);
    command
        .arg(&detector)
        .arg(project_directory.join("project.json"))
        .arg(&candidates_directory)
        .arg("--scan-round")
        .arg(options.scan_round.to_string())
        // Match the verified regression path: one click evaluates every
        // six-frame sampling phase, so the cleanest glyph evidence cannot be
        // missed merely because it falls between phase-0 samples.
        .arg("--all-phases")
        .arg("--exclude-frames")
        .arg(serde_json::to_string(options.excluded_frames)?)
        .arg("--exclude-signatures")
        .arg(serde_json::to_string(options.excluded_scene_signatures)?);
    if let Some(roi) = options.roi {
        command
            .arg("--roi-json")
            .arg(serde_json::to_string(roi)?)
            .arg("--anchor-frame")
            .arg(options.anchor_frame.to_string());
    }
    let output = command
        .output()
        .map_err(|error| AppError::Io(format!("Learna AI detector could not start: {error}")))?;
    if !output.status.success() {
        return Err(AppError::FfmpegFailed(format!(
            "Learna AI detector failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let candidates: Vec<BestQualitySample> =
        serde_json::from_slice(&output.stdout).map_err(|error| {
            AppError::Io(format!(
                "Unable to parse Learna AI detector result: {error}"
            ))
        })?;
    progress(6, 6);
    Ok(candidates)
}

fn validate_project(project_directory: &Path, project: &Project) -> Result<(), AppError> {
    // Best-quality is profile-driven; a legacy manual anchor is optional and
    // must never be required for the final route.
    validate_layout(project)?;
    let _ = project_directory;
    Ok(())
}

pub fn create_calibration_profile(
    project_directory: &Path,
    project: &Project,
    sample: &BestQualitySample,
    edited_mask: Option<&Path>,
    cancel: &AtomicBool,
) -> Result<CalibrationProfile, AppError> {
    validate_layout(project)?;
    if sample.frame >= project.video.frame_count
        || sample.glyph_correlation < 0.65
        || sample.glyph_iou < 0.55
        || sample.contamination > 0.20
        || sample.temporal_pass_count < 3
    {
        return Err(AppError::InvalidRequest(
            "The selected sample did not pass the Learna AI hard gate.".to_string(),
        ));
    }
    let python = configured_path(
        "WATERMARK_STUDIO_PROPAINTER_PYTHON",
        DEFAULT_PROPAINTER_PYTHON,
    );
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let calibration_script = repo_root.join("tools").join(CALIBRATION_SCRIPT);
    let audit_script = repo_root.join("tools").join(AUDIT_SCRIPT);
    require_file(&python, "Calibration Python")?;
    require_file(&calibration_script, "Best-quality calibration script")?;
    require_file(&audit_script, "Trajectory audit script")?;
    let profile_path = project_directory.join("calibration").join("profile.json");
    if let Some(parent) = profile_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let audit_directory = project_directory
        .join("calibration")
        .join("trajectory-audit");
    fs::create_dir_all(&audit_directory)?;
    let phase_shift = sample.trajectory_phase_offset.rem_euclid(360);
    let mut audit = Command::new(&python);
    audit
        .arg(&audit_script)
        .arg(project_directory.join("project.json"))
        .arg(&audit_directory)
        .arg("--template-frame")
        .arg(sample.frame.to_string())
        .arg("--template-bbox-json")
        .arg(serde_json::to_string(&sample.bbox)?)
        .arg("--phase-shift")
        .arg(phase_shift.to_string())
        .arg("--all-frames");
    run_process(
        &mut audit,
        cancel,
        "Analyzing the full-video watermark trajectory",
    )?;
    let audit_path = audit_directory.join("all-matches.json");
    let mut command = Command::new(&python);
    command
        .arg(&calibration_script)
        .arg(project_directory.join("project.json"))
        .arg(&profile_path)
        .arg("--sample-frame")
        .arg(sample.frame.to_string())
        .arg("--bbox-json")
        .arg(serde_json::to_string(&sample.bbox)?)
        .arg("--candidate-json")
        .arg(serde_json::to_string(sample)?);
    command
        .arg("--audit-json")
        .arg(&audit_path)
        .arg("--route")
        .arg(if sample.roi_fallback {
            "ROI_FALLBACK"
        } else {
            "AUTO_FIND"
        });
    if let Some(mask) = edited_mask {
        command.arg("--edited-mask").arg(mask);
    }
    run_process(&mut command, cancel, "Creating CalibrationProfileV4")?;
    if !profile_path.is_file() {
        return Err(AppError::FfmpegFailed(
            "Calibration completed without creating a profile.".to_string(),
        ));
    }
    let metadata = calibration_metadata(project_directory, project)?;
    if metadata.version != 4 || metadata.quality.status != CalibrationStatus::Ready {
        return Err(AppError::InvalidRequest(
            "CalibrationProfileV4 did not pass its quality gate.".to_string(),
        ));
    }
    Ok(metadata)
}

/// Runs the adaptive Best-quality calibration.  Unlike the legacy sample
/// route, this searches the full video with the canonical glyph and fits a
/// trajectory model without assuming the 360-frame prior.
pub fn create_adaptive_calibration_profile(
    project_directory: &Path,
    project: &Project,
    roi: Option<&BoundingBox>,
    roi_frame: Option<u64>,
    cancel: &AtomicBool,
) -> Result<CalibrationProfile, AppError> {
    validate_layout(project)?;
    let python = configured_path(
        "WATERMARK_STUDIO_PROPAINTER_PYTHON",
        DEFAULT_PROPAINTER_PYTHON,
    );
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let script = repo_root.join("tools").join(ADAPTIVE_CALIBRATION_SCRIPT);
    require_file(&python, "Calibration Python")?;
    require_file(&script, "Adaptive calibration script")?;
    let profile_path = project_directory.join("calibration").join("profile.json");
    if let Some(parent) = profile_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let route = if roi.is_some() {
        "ROI_FALLBACK"
    } else {
        "AUTO_GLOBAL_TEMPLATE"
    };
    let mut command = Command::new(&python);
    command
        .arg(&script)
        .arg(project_directory.join("project.json"))
        .arg(&profile_path)
        .arg("--route")
        .arg(route);
    if let Some(roi) = roi {
        command.arg("--roi-json").arg(serde_json::to_string(roi)?);
    }
    if let Some(frame) = roi_frame {
        command.arg("--roi-frame").arg(frame.to_string());
    }
    run_process(&mut command, cancel, "Adaptive Learna AI calibration")?;
    if !profile_path.is_file() {
        return Err(AppError::FfmpegFailed(
            "Adaptive calibration completed without creating a profile.".to_string(),
        ));
    }
    calibration_metadata(project_directory, project)
}

fn validate_calibration_profile(
    project_directory: &Path,
    project: &Project,
) -> Result<PathBuf, AppError> {
    let profile_path = project_directory.join("calibration").join("profile.json");
    if !profile_path.is_file() {
        return Err(AppError::InvalidRequest(
            "Best-quality render requires a confirmed CalibrationProfileV6. Run Auto-find & calibrate in Review first."
                .to_string(),
        ));
    }
    let body = fs::read_to_string(&profile_path)?;
    let mut profile: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
        AppError::CalibrationCorrupt(format!(
            "non-standard JSON numbers or truncated write; regenerate V6 ({error})"
        ))
    })?;
    let metadata = calibration_metadata(project_directory, project)?;
    if metadata.version != 6 || metadata.quality.status != CalibrationStatus::Ready {
        return Err(AppError::InvalidRequest(
            "Calibration is stale or did not pass the V6 quality gate. Regenerate it in Review."
                .to_string(),
        ));
    }
    let fingerprint = metadata.source_fingerprint.as_ref().ok_or_else(|| {
        AppError::InvalidRequest("Calibration has no source fingerprint.".to_string())
    })?;
    let source = Path::new(&project.source.path);
    let current_size = source.metadata()?.len();
    if fingerprint.frame_count != project.video.frame_count
        || fingerprint.width != project.video.width
        || fingerprint.height != project.video.height
        || fingerprint.size_bytes != current_size
        || fingerprint.sha256 != sha256_file(source)?
    {
        return Err(AppError::InvalidRequest(
            "Calibration source fingerprint does not match this video.".to_string(),
        ));
    }
    let mask_path = project_directory.join(&metadata.mask_path);
    if !mask_path.is_file() || metadata.mask_hash != sha256_file(&mask_path)? {
        return Err(AppError::InvalidRequest(
            "Calibration mask hash is missing or has changed.".to_string(),
        ));
    }
    let stored_profile_hash = profile
        .get("profileSha256")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    if let Some(object) = profile.as_object_mut() {
        object.remove("profileSha256");
    }
    let computed_profile_hash = profile_sha256(&profile)?;
    if stored_profile_hash.is_empty() || stored_profile_hash != computed_profile_hash {
        return Err(AppError::InvalidRequest(
            "Calibration profile hash is missing or has changed.".to_string(),
        ));
    }
    Ok(profile_path)
}

fn profile_sha256(profile: &serde_json::Value) -> Result<String, AppError> {
    let mut canonical = String::new();
    write_canonical_json(profile, &mut canonical)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

/// Matches `json.dumps(value, sort_keys=True, separators=(",", ":"),
/// ensure_ascii=False)` used by the Python calibration writer. Hashing the
/// canonical representation keeps profiles portable across runtimes.
fn write_canonical_json(value: &serde_json::Value, output: &mut String) -> Result<(), AppError> {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => output.push_str(&value.to_string()),
        serde_json::Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                write_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

pub fn validate_calibration(
    project_directory: &Path,
    project: &Project,
) -> Result<CalibrationProfile, AppError> {
    validate_calibration_profile(project_directory, project)?;
    calibration_metadata(project_directory, project)
}

fn sha256_file(path: &Path) -> Result<String, AppError> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn lower_hardware_profile(current: &HardwareProfile) -> Option<HardwareProfile> {
    let (tier, width, height, core_length, context) = match current.tier.as_str() {
        "MAX" => ("HIGH", 432, 768, 90, 12),
        "HIGH" => ("BALANCED", 384, 672, 72, 10),
        "BALANCED" => ("SAFE", 288, 512, 60, 8),
        _ => return None,
    };
    Some(HardwareProfile {
        gpu_name: current.gpu_name.clone(),
        vram_mb: current.vram_mb,
        cuda_available: current.cuda_available,
        supported: true,
        tier: tier.to_string(),
        width,
        height,
        core_length,
        context,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_propainter_chunks(
    python: &Path,
    chunks: &Path,
    workspace: &Path,
    result_root: &Path,
    propainter_root: &Path,
    hardware: &HardwareProfile,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(&str, u64, u64),
) -> Result<(), AppError> {
    let mut child = Command::new(python)
        .arg(chunks)
        .arg(workspace)
        .arg(result_root)
        .arg(propainter_root)
        .arg(python)
        .arg("--width")
        .arg(hardware.width.to_string())
        .arg("--height")
        .arg(hardware.height.to_string())
        .arg("--core-length")
        .arg(hardware.core_length.to_string())
        .arg("--context")
        .arg(hardware.context.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            AppError::Io(format!("Running AI restoration could not start: {error}"))
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Io("AI restoration stdout unavailable.".to_string()))?;
    // Drain stderr concurrently.  ProPainter writes progress bars and model
    // diagnostics there; waiting until process exit to read a full pipe can
    // deadlock once the Windows pipe buffer fills on a long video.
    let stderr = child.stderr.take();
    let stderr_buffer = Arc::new(Mutex::new(String::new()));
    let stderr_buffer_writer = Arc::clone(&stderr_buffer);
    thread::spawn(move || {
        if let Some(pipe) = stderr {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                if let Ok(mut buffer) = stderr_buffer_writer.lock() {
                    if buffer.len() < 16_384 {
                        buffer.push_str(&line);
                        buffer.push('\n');
                    }
                }
            }
        }
    });
    let (sender, receiver) = mpsc::channel::<String>();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = sender.send(line);
        }
    });
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AppError::OperationCancelled);
        }
        while let Ok(line) = receiver.try_recv() {
            if let Some((current, total)) = parse_chunk_progress(&line) {
                progress("Running temporal AI restoration", current, total);
            }
        }
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                progress("Running temporal AI restoration", 1, 1);
                return Ok(());
            }
            Ok(Some(status)) => {
                let stderr = stderr_buffer
                    .lock()
                    .map(|buffer| buffer.trim().to_string())
                    .unwrap_or_default();
                return Err(AppError::FfmpegFailed(format!(
                    "Running AI restoration failed with exit code {}. {}",
                    status
                        .code()
                        .map_or("unknown".to_string(), |code| code.to_string()),
                    stderr.trim()
                )));
            }
            Ok(None) => thread::sleep(Duration::from_millis(250)),
            Err(error) => {
                return Err(AppError::Io(format!(
                    "AI restoration status check failed: {error}"
                )))
            }
        }
    }
}

fn parse_chunk_progress(line: &str) -> Option<(u64, u64)> {
    let marker = "merged ";
    let start = line.find(marker)? + marker.len();
    let values = line[start..].split_once('/')?;
    let current = values.0.trim().parse().ok()?;
    let total = values.1.trim().parse().ok()?;
    Some((current, total))
}

fn validate_replacement(replacement: Option<&BestQualityReplacement>) -> Result<(), AppError> {
    let Some(replacement) = replacement else {
        return Ok(());
    };
    if !matches!(replacement.kind.as_str(), "text" | "image") {
        return Err(AppError::InvalidRequest(
            "Unsupported Best-quality replacement kind.".to_string(),
        ));
    }
    if !matches!(replacement.placement.as_str(), "follow" | "fixed") {
        return Err(AppError::InvalidRequest(
            "Replacement placement must be follow or fixed.".to_string(),
        ));
    }
    if !(0.1..=4.0).contains(&replacement.scale) || !(0.0..=1.0).contains(&replacement.opacity) {
        return Err(AppError::InvalidRequest(
            "Replacement scale or opacity is outside the supported range.".to_string(),
        ));
    }
    match replacement.kind.as_str() {
        "text"
            if replacement
                .text
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none() =>
        {
            Err(AppError::InvalidRequest(
                "Enter replacement text before rendering.".to_string(),
            ))
        }
        "image" => {
            let image = replacement
                .image_path
                .as_deref()
                .map(Path::new)
                .filter(|path| path.is_file());
            if image.is_none() {
                Err(AppError::InvalidRequest(
                    "Choose a valid transparent PNG replacement before rendering.".to_string(),
                ))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

fn append_replacement_arguments(
    command: &mut Command,
    replacement: Option<&BestQualityReplacement>,
) {
    let Some(replacement) = replacement else {
        return;
    };
    command
        .arg("--replacement-kind")
        .arg(&replacement.kind)
        .arg("--replacement-placement")
        .arg(&replacement.placement)
        .arg("--replacement-fixed-x")
        .arg(replacement.fixed_x.to_string())
        .arg("--replacement-fixed-y")
        .arg(replacement.fixed_y.to_string())
        .arg("--replacement-scale")
        .arg(replacement.scale.to_string())
        .arg("--replacement-opacity")
        .arg(replacement.opacity.to_string());
    if let Some(text) = replacement.text.as_deref() {
        command.arg("--replacement-text").arg(text);
    }
    if let Some(path) = replacement.image_path.as_deref() {
        command.arg("--replacement-image").arg(path);
    }
}

fn validate_layout(project: &Project) -> Result<(), AppError> {
    if project.video.width != SUPPORTED_WIDTH || project.video.height != SUPPORTED_HEIGHT {
        return Err(AppError::InvalidRequest(format!(
            "Best-quality AI currently supports the verified {SUPPORTED_WIDTH}x{SUPPORTED_HEIGHT} Learna AI video layout; this video is {}x{}.",
            project.video.width, project.video.height
        )));
    }
    if project.video.frame_count <= LEGACY_FIRST_WATERMARK_FRAME {
        return Err(AppError::InvalidRequest(
            "The video is too short for the supported best-quality workflow.".to_string(),
        ));
    }
    if !Path::new(&project.source.path).is_file() {
        return Err(AppError::VideoNotFound);
    }
    Ok(())
}

#[cfg(test)]
fn candidate_frames(frame_count: u64, anchor_frame: u64, scan_round: u32) -> Vec<u64> {
    let end = frame_count.saturating_sub(1);
    // Each alternative pass samples a different phase of the same trajectory.
    // A fixed scan was useful for reproducibility, but it meant a user who
    // rejected five visually poor backgrounds would receive the same five
    // cards forever. Prime-ish phase offsets traverse all 24 frame phases
    // before repeating.
    let phase_offset = (u64::from(scan_round) * 7) % 24;
    let first = LEGACY_FIRST_WATERMARK_FRAME + phase_offset;
    let mut frames: Vec<u64> = (first..=end).step_by(24).collect();
    if scan_round == 0 {
        frames.push(anchor_frame);
        frames.push(end);
    }
    frames.sort_unstable();
    frames.dedup();
    frames
}

fn configured_path(variable: &str, fallback: &str) -> PathBuf {
    env::var_os(variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(fallback))
}

fn require_file(path: &Path, label: &str) -> Result<(), AppError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(AppError::InvalidRequest(format!(
            "{label} was not found at {}. Configure its path before using Best-quality render.",
            path.display()
        )))
    }
}

fn next_output_path(
    project: &Project,
    configured_root: Option<&str>,
    configured_name: Option<&str>,
) -> Result<PathBuf, AppError> {
    let source = Path::new(&project.source.path);
    let parent = source.parent().ok_or_else(|| {
        AppError::InvalidRequest("The source video does not have an output folder.".to_string())
    })?;
    let source_stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| AppError::InvalidRequest("The source video name is invalid.".to_string()))?;
    let requested_name = configured_name
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(name) = requested_name {
        validate_output_file_name(name)?;
    }
    let stem = requested_name
        .map(|name| name.strip_suffix(".mp4").unwrap_or(name).to_string())
        .unwrap_or_else(|| format!("{source_stem}_watermark_removed_best"));
    let output_directory = configured_root
        .filter(|root| !root.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| parent.join("output"));
    if !output_directory.is_absolute() {
        return Err(AppError::InvalidRequest(
            "The output folder must be an absolute path.".to_string(),
        ));
    }
    fs::create_dir_all(&output_directory)?;
    for index in 0..10_000_u32 {
        let suffix = if index == 0 {
            String::new()
        } else {
            format!("_{index}")
        };
        let candidate = output_directory.join(format!("{stem}{suffix}.mp4"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(AppError::Io(
        "Unable to allocate a unique best-quality output file name.".to_string(),
    ))
}

fn validate_output_file_name(name: &str) -> Result<(), AppError> {
    if name.len() > 180
        || Path::new(name).file_name().and_then(|value| value.to_str()) != Some(name)
        || name.chars().any(|character| {
            matches!(
                character,
                '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
            )
        })
        || name.chars().any(char::is_control)
        || name.ends_with('.')
        || name.ends_with(' ')
    {
        return Err(AppError::InvalidRequest(
            "Output name must be a valid file name without a path or reserved characters."
                .to_string(),
        ));
    }
    let stem = name
        .strip_suffix(".mp4")
        .or_else(|| name.strip_suffix(".MP4"))
        .unwrap_or(name);
    let reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if stem.is_empty()
        || reserved
            .iter()
            .any(|value| stem.eq_ignore_ascii_case(value))
    {
        return Err(AppError::InvalidRequest(
            "Output name is reserved by Windows. Choose another name.".to_string(),
        ));
    }
    Ok(())
}

fn run_process(command: &mut Command, cancel: &AtomicBool, phase: &str) -> Result<(), AppError> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| AppError::Io(format!("{phase} could not start: {error}")))?;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(AppError::OperationCancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => {
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                return Err(AppError::FfmpegFailed(format!(
                    "{phase} failed with exit code {}. {}",
                    status
                        .code()
                        .map_or("unknown".to_string(), |code| code.to_string()),
                    stderr.trim()
                )));
            }
            Ok(None) => thread::sleep(Duration::from_millis(250)),
            Err(error) => {
                return Err(AppError::Io(format!(
                    "{phase} status check failed: {error}"
                )))
            }
        }
    }
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), AppError> {
    if cancel.load(Ordering::Relaxed) {
        Err(AppError::OperationCancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternative_scan_uses_a_different_trajectory_phase() {
        let first = candidate_frames(904, LEGACY_FIRST_WATERMARK_FRAME, 0);
        let alternative = candidate_frames(904, LEGACY_FIRST_WATERMARK_FRAME, 1);

        assert!(first.contains(&LEGACY_FIRST_WATERMARK_FRAME));
        assert!(!alternative.contains(&LEGACY_FIRST_WATERMARK_FRAME));
        assert!(first.iter().any(|frame| !alternative.contains(frame)));
        assert!(alternative.iter().any(|frame| !first.contains(frame)));
    }

    #[test]
    fn profile_hash_matches_python_canonical_json() {
        let value = serde_json::json!({"z":1,"a":{"b":1.0,"a":[true,0.25]},"path":"calibration/inference_mask.png"});
        assert_eq!(
            profile_sha256(&value).unwrap(),
            "7e23cbf28474f7880dc364f345fd876eef0631aed5e8e6080ec6635f0efd82f8"
        );
    }

    #[test]
    fn output_file_name_rejects_paths_and_windows_reserved_names() {
        assert!(validate_output_file_name("nested\\output").is_err());
        assert!(validate_output_file_name("CON.mp4").is_err());
        assert!(validate_output_file_name("clip:final.mp4").is_err());
        assert!(validate_output_file_name("clip_final.mp4").is_ok());
    }

    #[test]
    fn output_path_is_incremented_without_overwriting_existing_file() {
        let directory =
            std::env::temp_dir().join(format!("watermark-studio-best-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("clip.mp4");
        fs::write(&source, []).unwrap();
        let project = Project {
            version: 1,
            id: "test-project".to_string(),
            source: crate::project::model::SourceVideo {
                path: source.to_string_lossy().to_string(),
                file_name: "clip.mp4".to_string(),
            },
            video: crate::project::model::VideoMetadata {
                width: 1080,
                height: 1920,
                duration_seconds: 1.0,
                fps: 30.0,
                frame_count: 60,
                codec: None,
                pixel_format: None,
            },
            watermark: Default::default(),
            calibration: None,
            anchors: Vec::new(),
            tracking: None,
            removal: None,
        };
        let first = next_output_path(&project, None, None).unwrap();
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::write(&first, []).unwrap();
        let second = next_output_path(&project, None, None).unwrap();
        assert!(second.ends_with("clip_watermark_removed_best_1.mp4"));
        let _ = fs::remove_dir_all(directory);
    }
}
