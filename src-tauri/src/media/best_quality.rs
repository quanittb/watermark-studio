use crate::commands::project::BestQualityReplacement;
use crate::error::AppError;
use crate::media::ffmpeg;
use crate::project::model::{
    BoundingBox, CalibrationPreset, CalibrationProfile, CalibrationQuality, CalibrationStatus,
    Project, RoiEvidenceRecord, ScanRange, SourceFingerprint, TrajectoryGateSummary,
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
const ADAPTIVE_CALIBRATION_SCRIPT: &str = "calibrate_trajectory_v9.py";
const FIND_SAMPLES_SCRIPT: &str = "find_learna_samples.py";
const AUDIT_SCRIPT: &str = "audit_watermark_detection.py";
const MIN_MASK_COVERAGE: u64 = 350;
const QUALITY_QA_SCRIPT: &str = "quality_qa_v9.py";
const OPAQUE_BADGE_SCRIPT: &str = "apply_opaque_badge.py";

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
    pub scan_range: Option<ScanRange>,
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

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeHealth {
    pub python_path: Option<String>,
    pub python_version: Option<String>,
    pub ffmpeg_path: Option<String>,
    pub ffprobe_path: Option<String>,
    pub cuda_available: bool,
    pub gpu_name: Option<String>,
    pub vram_mb: u64,
    pub imports: RuntimeImports,
    pub propainter_model_ready: bool,
    pub workspace_root: String,
    pub free_workspace_bytes: Option<u64>,
    pub status: String,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeImports {
    pub cv2: bool,
    pub numpy: bool,
    pub torch: bool,
}

#[derive(Debug, Clone)]
struct PythonRuntime {
    executable: PathBuf,
    site_packages: Option<PathBuf>,
}

/// Resolve a usable interpreter even when a copied Windows venv launcher has
/// become stale.  The project machine keeps the 3.11 packages in
/// `D:\propainter-watermark-venv\Lib\site-packages`, while the venv
/// `Scripts\python.exe` can fail to start after its original Python install
/// moved.  Running the matching base interpreter with that site-packages
/// directory is equivalent for this read-only processing runtime and avoids a
/// misleading `PYTHON_IMPORT_FAILED` preflight result.
fn resolve_python_runtime() -> PythonRuntime {
    let configured = configured_path(
        "WATERMARK_STUDIO_PROPAINTER_PYTHON",
        DEFAULT_PROPAINTER_PYTHON,
    );
    let venv_root = configured
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    if let Some(root) = venv_root {
        let cfg = root.join("pyvenv.cfg");
        if let Ok(contents) = fs::read_to_string(&cfg) {
            let home = contents
                .lines()
                .find_map(|line| line.trim().strip_prefix("home ="))
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from);
            if let Some(home) = home {
                let base = if cfg!(windows) {
                    home.join("python.exe")
                } else {
                    home.join("bin").join("python")
                };
                let packages = root.join("Lib").join("site-packages");
                if base.is_file() && packages.is_dir() {
                    return PythonRuntime {
                        executable: base,
                        site_packages: Some(packages),
                    };
                }
            }
        }
    }
    PythonRuntime {
        executable: configured,
        site_packages: None,
    }
}

fn python_command(runtime: &PythonRuntime) -> Command {
    let mut command = Command::new(&runtime.executable);
    if let Some(site_packages) = runtime.site_packages.as_ref() {
        command.env("PYTHONPATH", site_packages);
    }
    command
}

fn find_binary(name: &str) -> Option<String> {
    Command::new("where")
        .arg(name)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .map(str::trim)
                .map(str::to_string)
        })
        .filter(|path| !path.is_empty())
}

/// Performs a side-effect-free preflight so calibration/render failures are
/// actionable instead of surfacing later as a generic child-process error.
pub fn detect_runtime_health() -> RuntimeHealth {
    let runtime = resolve_python_runtime();
    let python = runtime.executable.clone();
    let propainter_root =
        configured_path("WATERMARK_STUDIO_PROPAINTER_ROOT", DEFAULT_PROPAINTER_ROOT);
    let workspace_root = configured_path("WATERMARK_STUDIO_WORK_ROOT", DEFAULT_WORK_ROOT);
    let mut problems = Vec::new();
    let mut python_version = None;
    let mut imports = RuntimeImports {
        cv2: false,
        numpy: false,
        torch: false,
    };
    if !python.is_file() {
        problems.push("PYTHON_RUNTIME_MISSING".to_string());
    } else {
        let probe = python_command(&runtime)
            .args(["-c", "import sys, json; mods={m: __import__(m) is not None for m in ('cv2','numpy','torch')}; print(json.dumps({'version':sys.version.split()[0], 'mods':mods}))"])
            .output();
        match probe {
            Ok(output) if output.status.success() => {
                if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                    python_version = value
                        .get("version")
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    if let Some(mods) = value.get("mods").and_then(|v| v.as_object()) {
                        imports.cv2 = mods.get("cv2").and_then(|v| v.as_bool()).unwrap_or(false);
                        imports.numpy =
                            mods.get("numpy").and_then(|v| v.as_bool()).unwrap_or(false);
                        imports.torch =
                            mods.get("torch").and_then(|v| v.as_bool()).unwrap_or(false);
                    }
                } else {
                    problems.push("PYTHON_IMPORT_FAILED".to_string());
                }
            }
            _ => problems.push("PYTHON_IMPORT_FAILED".to_string()),
        }
    }
    if (!imports.cv2 || !imports.numpy || !imports.torch)
        && python.is_file()
        && !problems.iter().any(|item| item == "PYTHON_IMPORT_FAILED")
    {
        problems.push("PYTHON_IMPORT_FAILED".to_string());
    }
    let ffmpeg_path = find_binary("ffmpeg");
    let ffprobe_path = find_binary("ffprobe");
    if ffmpeg_path.is_none() || ffprobe_path.is_none() {
        problems.push("FFMPEG_MISSING".to_string());
    }
    let propainter_model_ready = propainter_root.join("inference_propainter.py").is_file();
    if !propainter_model_ready {
        problems.push("PROPAINTER_MODEL_MISSING".to_string());
    }
    let hardware = detect_hardware();
    if !hardware.supported {
        problems.push("CUDA_UNAVAILABLE".to_string());
    }
    let status = if problems.is_empty() {
        "READY"
    } else if !hardware.supported {
        "UNSUPPORTED"
    } else {
        "MISCONFIGURED"
    };
    RuntimeHealth {
        python_path: python
            .is_file()
            .then(|| python.to_string_lossy().to_string()),
        python_version,
        ffmpeg_path,
        ffprobe_path,
        cuda_available: hardware.cuda_available,
        gpu_name: (hardware.gpu_name != "CUDA GPU not detected").then_some(hardware.gpu_name),
        vram_mb: hardware.vram_mb,
        imports,
        propainter_model_ready,
        workspace_root: workspace_root.to_string_lossy().to_string(),
        free_workspace_bytes: None,
        status: status.to_string(),
        problems,
    }
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
#[allow(clippy::too_many_arguments)]
pub fn render_best_quality<F>(
    project_directory: &Path,
    project: &Project,
    replacement: Option<&BestQualityReplacement>,
    output_root: Option<&str>,
    output_name: Option<&str>,
    _allow_review_draft: bool,
    cancel: &AtomicBool,
    mut progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(&str, u64, u64),
{
    validate_project(project_directory, project)?;
    let runtime_health = detect_runtime_health();
    if runtime_health.status != "READY" {
        return Err(AppError::RuntimeNotReady(format!(
            "{}; configure Python/FFmpeg/CUDA in Settings before starting Best-quality",
            runtime_health.problems.join(", ")
        )));
    }
    let python_runtime = resolve_python_runtime();
    let python = python_runtime.executable.clone();
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
    let qa_script = repo_root.join("tools").join(QUALITY_QA_SCRIPT);

    require_file(&python, "ProPainter Python")?;
    require_file(
        &propainter_root.join("inference_propainter.py"),
        "ProPainter inference",
    )?;
    require_file(&pipeline, "best-quality pipeline")?;
    require_file(&chunks, "best-quality chunk runner")?;
    require_file(&qa_script, "Best-quality QA script")?;
    validate_replacement(replacement)?;

    let profile_path =
        validate_calibration_profile(project_directory, project, _allow_review_draft)?;
    // V9 frameData is complete for the source; inactive ranges are explicit
    // passthrough rows, therefore rendering keeps all source frames.
    let first_frame = 0;
    let last_profile_frame = project.video.frame_count.saturating_sub(1);
    let hardware = detect_hardware();
    if !hardware.supported {
        return Err(AppError::InvalidRequest(
            "Best-quality final requires an NVIDIA CUDA GPU with at least 4 GB VRAM.".to_string(),
        ));
    }

    let workspace = workspace_root.join(&project.id);
    // A failed/canceled attempt can leave tens of thousands of PNGs.  Keep
    // the cleanup guard alive for every return path so a later retry cannot
    // silently exhaust the workspace volume.
    let _workspace_cleanup = WorkspaceCleanup(workspace.clone());
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
    render_attempt(
        &python_runtime,
        &pipeline,
        &chunks,
        &project_json,
        &profile_path,
        &workspace,
        &result_root,
        &propainter_root,
        &hardware,
        first_frame,
        last_profile_frame.min(last_frame),
        &draft,
        replacement,
        cancel,
        0,
        None,
        &mut progress,
    )?;
    progress("Decoding final output for QA", 4, 5);
    check_cancel(cancel)?;
    ffmpeg::verify_video_decode(&draft)?;
    if let Err(first_qa_error) = run_quality_qa(
        &python_runtime,
        &qa_script,
        project,
        &profile_path,
        &draft,
        cancel,
    ) {
        // A failed inpaint is not silently promoted.  Instead perform one
        // deterministic, opaque badge pass over the exact frames identified
        // by the independent full-frame QA.  This guarantees that gaps
        // between glyphs cannot reveal Learna AI underneath.
        progress(
            "QA còn residual Learna AI; đang che kín các frame lỗi bằng badge QuanPH",
            4,
            5,
        );
        let report = qa_report_path(&draft);
        let badge_script = repo_root.join("tools").join(OPAQUE_BADGE_SCRIPT);
        let badge_asset = repo_root.join("assets").join("quanph_watermark_v1.png");
        require_file(&badge_script, "opaque QuanPH fallback")?;
        let hybrid = output.with_file_name(format!("{output_stem}.hybrid.review.mp4"));
        let mut badge = python_command(&python_runtime);
        badge
            .arg(&badge_script)
            .arg(&project.source.path)
            .arg(&draft)
            .arg(&profile_path)
            .arg(&report)
            .arg(&hybrid)
            .arg("--badge")
            .arg(&badge_asset);
        run_process(&mut badge, cancel, "Applying opaque QuanPH fallback")?;
        if !hybrid.is_file() {
            return Err(first_qa_error);
        }
        let draft_manifest =
            draft.with_file_name(format!("{output_stem}.review.render-manifest.json"));
        let hybrid_manifest =
            hybrid.with_file_name(format!("{output_stem}.hybrid.review.render-manifest.json"));
        if draft_manifest.is_file() {
            fs::copy(&draft_manifest, &hybrid_manifest)?;
        }
        fs::remove_file(&draft)?;
        fs::rename(&hybrid, &draft)?;
        let _ = fs::remove_file(&draft_manifest);
        if hybrid_manifest.is_file() {
            fs::rename(hybrid_manifest, draft_manifest)?;
        }
        let hybrid_badge_manifest =
            hybrid.with_file_name(format!("{output_stem}.hybrid.review.badge-manifest.json"));
        let draft_badge_manifest =
            draft.with_file_name(format!("{output_stem}.review.badge-manifest.json"));
        if hybrid_badge_manifest.is_file() {
            fs::rename(hybrid_badge_manifest, draft_badge_manifest)?;
        }
        ffmpeg::verify_video_decode(&draft)?;
        run_quality_qa(
            &python_runtime,
            &qa_script,
            project,
            &profile_path,
            &draft,
            cancel,
        )?;
    }
    let draft_report = qa_report_path(&draft);
    let draft_sheet = qa_contact_sheet_path(&draft);
    fs::rename(&draft, &output)?;
    if draft_report.is_file() {
        fs::rename(draft_report, qa_report_path(&output))?;
    }
    if draft_sheet.is_file() {
        fs::rename(draft_sheet, qa_contact_sheet_path(&output))?;
    }
    let draft_manifest = draft.with_file_name(format!("{output_stem}.review.render-manifest.json"));
    let final_manifest = output.with_file_name(format!("{output_stem}.render-manifest.json"));
    if draft_manifest.is_file() {
        fs::rename(draft_manifest, final_manifest)?;
    }
    let draft_badge_manifest =
        draft.with_file_name(format!("{output_stem}.review.badge-manifest.json"));
    let final_badge_manifest = output.with_file_name(format!("{output_stem}.badge-manifest.json"));
    if draft_badge_manifest.is_file() {
        fs::rename(draft_badge_manifest, final_badge_manifest)?;
    }

    progress("Best-quality render complete", 5, 5);
    Ok(output)
}

struct WorkspaceCleanup(PathBuf);

impl Drop for WorkspaceCleanup {
    fn drop(&mut self) {
        // Generated cache frames are reproducible and must never be confused
        // with source/output artifacts.  Ignore an already-removed directory.
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[allow(clippy::too_many_arguments)]
fn render_attempt(
    python_runtime: &PythonRuntime,
    pipeline: &Path,
    chunks: &Path,
    project_json: &Path,
    profile_path: &Path,
    workspace: &Path,
    result_root: &Path,
    propainter_root: &Path,
    hardware: &HardwareProfile,
    first_frame: u64,
    last_frame: u64,
    draft: &Path,
    replacement: Option<&BestQualityReplacement>,
    cancel: &AtomicBool,
    mask_dilate: u8,
    source_override: Option<&Path>,
    progress: &mut impl FnMut(&str, u64, u64),
) -> Result<(), AppError> {
    progress("Preparing full-resolution AI masks", 1, 5);
    let mut prepare = python_command(python_runtime);
    prepare
        .arg(pipeline)
        .arg("prepare")
        .arg(project_json)
        .arg("-")
        .arg(workspace)
        .arg("--profile")
        .arg(profile_path)
        .arg("--start-frame")
        .arg(first_frame.to_string())
        .arg("--end-frame")
        .arg(last_frame.to_string())
        .arg("--dynamic-crop")
        .arg("--crop-width")
        .arg("512")
        .arg("--crop-height")
        .arg("288");
    if let Some(video) = source_override {
        prepare.arg("--video").arg(video);
    }
    if mask_dilate > 0 {
        prepare.arg("--mask-dilate").arg(mask_dilate.to_string());
    }
    run_process(&mut prepare, cancel, "Preparing AI masks")?;

    progress("Running temporal AI restoration", 2, 5);
    if let Err(error) = run_propainter_chunks(
        python_runtime,
        chunks,
        workspace,
        result_root,
        propainter_root,
        hardware,
        cancel,
        progress,
    ) {
        let message = error.to_string().to_ascii_lowercase();
        if !message.contains("out of memory") && !message.contains("cuda oom") {
            return Err(error);
        }
        let fallback = lower_hardware_profile(hardware).ok_or(error)?;
        progress(
            &format!("CUDA OOM; retrying once with {} profile", fallback.tier),
            2,
            5,
        );
        run_propainter_chunks(
            python_runtime,
            chunks,
            workspace,
            result_root,
            propainter_root,
            &fallback,
            cancel,
            progress,
        )?;
    }

    progress("Encoding final full-resolution video", 3, 5);
    let mut composite = python_command(python_runtime);
    composite
        .arg(pipeline)
        .arg("composite")
        .arg(project_json)
        .arg(workspace)
        .arg(result_root.join("merged-frames"))
        .arg(draft);
    if let Some(video) = source_override {
        composite.arg("--video").arg(video);
    }
    append_replacement_arguments(&mut composite, replacement);
    run_process(&mut composite, cancel, "Encoding final video")?;
    if !draft.is_file() {
        return Err(AppError::FfmpegFailed(
            "Best-quality pipeline completed without creating an output video.".to_string(),
        ));
    }
    Ok(())
}

pub fn qa_report_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    output.with_file_name(format!("{stem}.qa.v9.json"))
}

pub fn qa_contact_sheet_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    output.with_file_name(format!("{stem}.qa.v9.png"))
}

/// Re-checks an existing `.review.mp4` with the current QA implementation and
/// promotes it only when every required frame passes.  This is intentionally
/// separate from rendering so a QA-rule fix (for example, distinguishing a
/// subtitle from residual glyph energy) can safely revalidate a completed
/// draft without consuming another GPU run.
pub fn revalidate_review_output<F>(
    project_directory: &Path,
    project: &Project,
    review_output: &Path,
    cancel: &AtomicBool,
    mut progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(&str, u64, u64),
{
    validate_project(project_directory, project)?;
    if !review_output.is_absolute() || !review_output.is_file() {
        return Err(AppError::InvalidRequest(
            "Review output was not found; render the job again before promotion.".to_string(),
        ));
    }
    let file_name = review_output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| AppError::InvalidRequest("Review output name is invalid.".to_string()))?;
    if !file_name.ends_with(".review.mp4") {
        return Err(AppError::InvalidRequest(
            "Only a .review.mp4 draft can be promoted after QA.".to_string(),
        ));
    }
    // A review draft is intentionally eligible for QA re-validation.  The
    // normal render/queue path remains fail-closed, while this history action
    // must be able to promote a draft after a later QA rule or artifact fix.
    let profile_path = validate_calibration_profile(project_directory, project, true)?;
    let python_runtime = resolve_python_runtime();
    let python = python_runtime.executable.clone();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let qa_script = repo_root.join("tools").join(QUALITY_QA_SCRIPT);
    require_file(&python, "ProPainter Python")?;
    require_file(&qa_script, "Best-quality QA script")?;
    progress("Re-validating review output", 1, 2);
    check_cancel(cancel)?;
    run_quality_qa(
        &python_runtime,
        &qa_script,
        project,
        &profile_path,
        review_output,
        cancel,
    )?;

    let review_stem = review_output
        .file_stem()
        .and_then(|value| value.to_str())
        .and_then(|value| value.strip_suffix(".review"))
        .ok_or_else(|| AppError::InvalidRequest("Review output name is invalid.".to_string()))?;
    let parent = review_output
        .parent()
        .ok_or_else(|| AppError::InvalidRequest("Review output folder is invalid.".to_string()))?;
    let final_output = collision_safe_final_path(parent, review_stem)?;
    let review_report = qa_report_path(review_output);
    let review_sheet = qa_contact_sheet_path(review_output);
    fs::rename(review_output, &final_output)?;
    if review_report.is_file() {
        fs::rename(review_report, qa_report_path(&final_output))?;
    }
    if review_sheet.is_file() {
        fs::rename(review_sheet, qa_contact_sheet_path(&final_output))?;
    }
    let review_manifest =
        review_output.with_file_name(format!("{}.review.render-manifest.json", review_stem));
    let final_manifest = final_output.with_file_name(format!("{review_stem}.render-manifest.json"));
    if review_manifest.is_file() {
        fs::rename(review_manifest, final_manifest)?;
    }
    let review_badge_manifest =
        review_output.with_file_name(format!("{review_stem}.review.badge-manifest.json"));
    let final_badge_manifest =
        final_output.with_file_name(format!("{review_stem}.badge-manifest.json"));
    if review_badge_manifest.is_file() {
        fs::rename(review_badge_manifest, final_badge_manifest)?;
    }
    progress("Review output promoted after QA", 2, 2);
    Ok(final_output)
}

fn collision_safe_final_path(parent: &Path, stem: &str) -> Result<PathBuf, AppError> {
    for index in 0..10_000_u32 {
        let suffix = if index == 0 {
            String::new()
        } else {
            format!("_{index}")
        };
        let candidate = parent.join(format!("{stem}{suffix}.mp4"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(AppError::Io(
        "Unable to allocate a unique promoted output file name.".to_string(),
    ))
}

pub fn calibration_metadata(
    project_directory: &Path,
    project: &Project,
) -> Result<CalibrationProfile, AppError> {
    let profile_path = project_directory.join("calibration").join("profile.json");
    let profile: serde_json::Value = serde_json::from_str(&fs::read_to_string(&profile_path)?)
        .map_err(|error| {
            AppError::CalibrationCorrupt(format!(
            "non-standard JSON value or truncated write; regenerate CalibrationProfileV9 ({error})"
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
    let trajectory_ready = !matches!(profile_version, 5..=9)
        || (profile
            .get("trajectoryGate")
            .and_then(|value| value.get("status"))
            .and_then(|value| value.as_str())
            == Some("PASSED")
            && profile.get("trajectoryModel").is_some());
    let is_ready = profile_version == 9
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
        scan_range: profile
            .get("scanRange")
            .and_then(|value| serde_json::from_value::<ScanRange>(value.clone()).ok()),
        trajectory_gate: profile
            .get("trajectoryGate")
            .and_then(|value| serde_json::from_value::<TrajectoryGateSummary>(value.clone()).ok()),
        trajectory_model: profile.get("trajectoryModel").cloned(),
        difficult_frames: profile
            .get("difficultFrames")
            .and_then(|value| value.as_array())
            .map(|frames| frames.iter().filter_map(|value| value.as_u64()).collect())
            .unwrap_or_default(),
        roi_evidence_frames: profile
            .get("roiEvidenceFrames")
            .and_then(|value| value.as_array())
            .map(|frames| frames.iter().filter_map(|value| value.as_u64()).collect())
            .unwrap_or_default(),
        roi_evidence: profile
            .get("roiEvidence")
            .and_then(|value| serde_json::from_value::<Vec<RoiEvidenceRecord>>(value.clone()).ok())
            .unwrap_or_default(),
        roi_budget_used: profile
            .get("roiBudgetUsed")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as u32,
        roi_budget_max: profile
            .get("roiBudgetMax")
            .and_then(|value| value.as_u64())
            .unwrap_or(3) as u32,
        outcome: profile
            .get("outcome")
            .and_then(|value| value.as_str())
            .map(str::to_string),
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
    python_runtime: &PythonRuntime,
    qa_script: &Path,
    project: &Project,
    profile_path: &Path,
    output: &Path,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    let report = qa_report_path(output);
    let contact_sheet = qa_contact_sheet_path(output);
    check_cancel(cancel)?;
    let result = python_command(python_runtime)
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
    let python_runtime = resolve_python_runtime();
    let python = python_runtime.executable.clone();
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| AppError::Io("Unable to locate the application workspace.".to_string()))?;
    let detector = repo_root.join("tools").join(FIND_SAMPLES_SCRIPT);
    require_file(&python, "Detector Python")?;
    require_file(&detector, "Learna AI detector")?;
    let scan_range = normalize_scan_range(project, options.scan_range)?;
    if options.roi.is_some()
        && (options.anchor_frame < scan_range.start_frame
            || options.anchor_frame > scan_range.end_frame)
    {
        return Err(AppError::RoiOutsideScanRange);
    }
    // The detector now evaluates all six stride phases in one process.  Keep
    // the progress contract phase-based so the UI does not appear finished
    // while the remaining phases are still being scanned.
    progress(0, 6);
    let mut command = python_command(&python_runtime);
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
    command
        .arg("--scan-start-frame")
        .arg(scan_range.start_frame.to_string())
        .arg("--scan-end-frame")
        .arg(scan_range.end_frame.to_string());
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
    let runtime = detect_runtime_health();
    if runtime.status != "READY" {
        return Err(AppError::RuntimeNotReady(format!(
            "{}; configure Python/FFmpeg/CUDA in Settings before calibration",
            runtime.problems.join(", ")
        )));
    }
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
    let python_runtime = resolve_python_runtime();
    let python = python_runtime.executable.clone();
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
    let mut audit = python_command(&python_runtime);
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
    let mut command = python_command(&python_runtime);
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
#[allow(clippy::too_many_arguments)]
pub fn create_adaptive_calibration_profile(
    project_directory: &Path,
    project: &Project,
    roi: Option<&BoundingBox>,
    roi_frame: Option<u64>,
    edited_mask_path: Option<&str>,
    roi_evidence: &[crate::commands::project::RoiEvidence],
    scan_range: Option<ScanRange>,
    cancel: &AtomicBool,
) -> Result<CalibrationProfile, AppError> {
    validate_layout(project)?;
    let runtime = detect_runtime_health();
    if runtime.status != "READY" {
        return Err(AppError::RuntimeNotReady(format!(
            "{}; configure Python/FFmpeg/CUDA in Settings before calibration",
            runtime.problems.join(", ")
        )));
    }
    let scan_range = normalize_scan_range(project, scan_range)?;
    if roi_evidence.iter().any(|evidence| {
        evidence.frame < scan_range.start_frame || evidence.frame > scan_range.end_frame
    }) {
        return Err(AppError::RoiOutsideScanRange);
    }
    let python_runtime = resolve_python_runtime();
    let python = python_runtime.executable.clone();
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
    // A review may collect several broad ROI evidence frames and then move
    // the playhead away before starting calibration.  Treat that evidence as
    // the ROI route even when there is no live selection on the current frame;
    // otherwise the backend silently falls back to AUTO_GLOBAL_TEMPLATE and
    // discards the user's additional evidence for the quality gate.
    let route = if roi.is_some() || !roi_evidence.is_empty() {
        "ROI_FALLBACK"
    } else {
        "AUTO_GLOBAL_TEMPLATE"
    };
    let mut command = python_command(&python_runtime);
    command
        .arg(&script)
        .arg(project_directory.join("project.json"))
        .arg(&profile_path)
        .arg("--route")
        .arg(route);
    command
        .arg("--scan-start-frame")
        .arg(scan_range.start_frame.to_string())
        .arg("--scan-end-frame")
        .arg(scan_range.end_frame.to_string());
    if let Some(roi) = roi {
        command.arg("--roi-json").arg(serde_json::to_string(roi)?);
    }
    if let Some(frame) = roi_frame {
        command.arg("--roi-frame").arg(frame.to_string());
    }
    if let Some(mask) = edited_mask_path {
        command.arg("--edited-mask").arg(mask);
    }
    if !roi_evidence.is_empty() {
        command
            .arg("--roi-evidence-json")
            .arg(serde_json::to_string(roi_evidence)?);
    }
    run_process(&mut command, cancel, "Adaptive Learna AI calibration")?;
    if !profile_path.is_file() {
        return Err(AppError::FfmpegFailed(
            "Adaptive calibration completed without creating a profile.".to_string(),
        ));
    }
    calibration_metadata(project_directory, project)
}

fn normalize_scan_range(
    project: &Project,
    range: Option<ScanRange>,
) -> Result<ScanRange, AppError> {
    let last_frame = project.video.frame_count.saturating_sub(1);
    let resolved = range.unwrap_or(ScanRange {
        start_frame: 0,
        end_frame: last_frame,
    });
    if project.video.frame_count == 0
        || resolved.start_frame > resolved.end_frame
        || resolved.end_frame >= project.video.frame_count
    {
        return Err(AppError::InvalidScanRange(format!(
            "Invalid scan range {}–{}; expected 0–{} with start <= end.",
            resolved.start_frame, resolved.end_frame, last_frame
        )));
    }
    Ok(resolved)
}

fn validate_calibration_profile(
    project_directory: &Path,
    project: &Project,
    _allow_review_draft: bool,
) -> Result<PathBuf, AppError> {
    let profile_path = project_directory.join("calibration").join("profile.json");
    if !profile_path.is_file() {
        return Err(AppError::InvalidRequest(
            "Best-quality render requires a confirmed CalibrationProfileV9. Run automatic calibration in Review first."
                .to_string(),
        ));
    }
    let body = fs::read_to_string(&profile_path)?;
    let mut profile: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
        AppError::CalibrationCorrupt(format!(
            "non-standard JSON numbers or truncated write; regenerate V9 ({error})"
        ))
    })?;
    let metadata = calibration_metadata(project_directory, project)?;
    let review_draft =
        profile.get("outcome").and_then(|value| value.as_str()) == Some("NEEDS_REVIEW_DRAFT");
    // A terminal draft may be rendered for QA/fallback inspection, but the
    // output is promoted only after the independent V9 QA gate passes.
    let status_allowed = metadata.quality.status == CalibrationStatus::Ready
        || (_allow_review_draft && review_draft);
    if metadata.version != 9 || !status_allowed {
        return Err(AppError::InvalidRequest(
            "Calibration is stale or did not pass the V9 quality gate. Regenerate it in Review."
                .to_string(),
        ));
    }
    let scan_range = metadata.scan_range.ok_or_else(|| {
        AppError::InvalidRequest(
            "Calibration profile has no scan range; regenerate it in Review.".to_string(),
        )
    })?;
    if scan_range.start_frame > scan_range.end_frame
        || scan_range.end_frame >= project.video.frame_count
    {
        return Err(AppError::InvalidScanRange(
            "Calibration profile scan range is outside this source video.".to_string(),
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
    validate_calibration_profile(project_directory, project, false)?;
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
    python_runtime: &PythonRuntime,
    chunks: &Path,
    workspace: &Path,
    result_root: &Path,
    propainter_root: &Path,
    hardware: &HardwareProfile,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(&str, u64, u64),
) -> Result<(), AppError> {
    let mut child = python_command(python_runtime)
        .arg(chunks)
        .arg(workspace)
        .arg(result_root)
        .arg(propainter_root)
        .arg(&python_runtime.executable)
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
                let stderr_summary = summarize_process_stderr(&stderr);
                let details = format!(
                    "{phase} failed with exit code {}. {}",
                    status
                        .code()
                        .map_or("unknown".to_string(), |code| code.to_string()),
                    stderr_summary
                );
                let lower = details.to_ascii_lowercase();
                if lower.contains("no space left on device")
                    || lower.contains("libpng error: write error")
                    || lower.contains("not enough space")
                {
                    return Err(AppError::StorageFull(format!(
                        "{phase} không đủ dung lượng ở workspace tạm. Hãy giải phóng thêm dung lượng ở ổ workspace rồi thử lại."
                    )));
                }
                return Err(AppError::FfmpegFailed(details));
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

/// Python/OpenCV tools often emit a several-hundred-line traceback.  Exposing
/// that whole buffer in a modal hides the actionable reason and can make the
/// UI appear frozen. Keep the structured error code/last exception while
/// retaining a short diagnostic tail for support logs.
fn summarize_process_stderr(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.is_empty() {
        return "no diagnostic output".to_string();
    }
    let mut selected: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|line| {
            line.contains("INVALID_")
                || line.contains("NO_VALID_")
                || line.contains("RuntimeError")
                || line.contains("ValueError")
                || line.contains("FileNotFoundError")
                || line.contains("CUDA")
                || line.contains("out of memory")
                || line.contains("libpng")
        })
        .collect();
    if selected.is_empty() {
        selected = lines.iter().rev().take(2).copied().collect();
        selected.reverse();
    } else if selected.len() > 3 {
        selected = selected.split_off(selected.len() - 3);
    }
    selected.join(" | ")
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
    fn scan_range_defaults_to_full_video_and_rejects_invalid_bounds() {
        let project = Project {
            version: 1,
            id: "scan-range-test".to_string(),
            source: crate::project::model::SourceVideo {
                path: "test.mp4".to_string(),
                file_name: "test.mp4".to_string(),
            },
            video: crate::project::model::VideoMetadata {
                width: 1080,
                height: 1920,
                duration_seconds: 1.0,
                fps: 30.0,
                frame_count: 10,
                codec: None,
                pixel_format: None,
            },
            watermark: Default::default(),
            calibration: None,
            anchors: Vec::new(),
            tracking: None,
            removal: None,
            roi_evidence: Vec::new(),
        };
        assert_eq!(
            normalize_scan_range(&project, None).unwrap(),
            ScanRange {
                start_frame: 0,
                end_frame: 9
            }
        );
        assert!(normalize_scan_range(
            &project,
            Some(ScanRange {
                start_frame: 8,
                end_frame: 7
            })
        )
        .is_err());
        assert!(normalize_scan_range(
            &project,
            Some(ScanRange {
                start_frame: 0,
                end_frame: 10
            })
        )
        .is_err());
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
            roi_evidence: Vec::new(),
        };
        let first = next_output_path(&project, None, None).unwrap();
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::write(&first, []).unwrap();
        let second = next_output_path(&project, None, None).unwrap();
        assert!(second.ends_with("clip_watermark_removed_best_1.mp4"));
        let _ = fs::remove_dir_all(directory);
    }
}
