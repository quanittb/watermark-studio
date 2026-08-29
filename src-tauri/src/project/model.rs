use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub version: u32,
    pub id: String,
    pub source: SourceVideo,
    pub video: VideoMetadata,
    pub watermark: WatermarkConfig,
    /// Versioned calibration used by the Best-quality pipeline. Legacy
    /// anchors/tracking remain available for backwards compatibility.
    #[serde(default)]
    pub calibration: Option<CalibrationProfile>,
    #[serde(default)]
    pub anchors: Vec<ManualAnchor>,
    #[serde(default)]
    pub tracking: Option<TrackingData>,
    #[serde(default)]
    pub removal: Option<RemovalConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationProfile {
    pub version: u32,
    pub preset: CalibrationPreset,
    #[serde(default)]
    pub detector_version: Option<String>,
    #[serde(default)]
    pub source_fingerprint: Option<SourceFingerprint>,
    pub profile_path: String,
    pub mask_path: String,
    #[serde(default)]
    pub canonical_mask_path: Option<String>,
    #[serde(default)]
    pub auto_mask_path: Option<String>,
    #[serde(default)]
    pub blend_mask_path: Option<String>,
    #[serde(default)]
    pub brush_delta_path: Option<String>,
    #[serde(default)]
    pub mask_hash: String,
    pub profile_hash: String,
    pub sample_frame: u64,
    pub frame_count: u64,
    pub quality: CalibrationQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceFingerprint {
    pub sha256: String,
    pub size_bytes: u64,
    pub frame_count: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CalibrationPreset {
    LearnaAiPeriodic,
    GeneralMoving,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationQuality {
    pub status: CalibrationStatus,
    pub reliable_frames: u64,
    pub low_confidence_frames: u64,
    pub mask_pixels: u64,
    #[serde(default)]
    pub glyph_coverage: f64,
    #[serde(default)]
    pub contamination: f64,
    #[serde(default)]
    pub large_holes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CalibrationStatus {
    Ready,
    Stale,
    NeedsReview,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceVideo {
    pub path: String,
    pub file_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub duration_seconds: f64,
    pub fps: f64,
    pub frame_count: u64,
    pub codec: Option<String>,
    pub pixel_format: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WatermarkConfig {
    pub label: Option<String>,
    pub anchor: Option<AnchorFrame>,
    pub templates: Option<TemplatePaths>,
    #[serde(default = "default_template_padding")]
    pub template_padding: u32,
}

fn default_template_padding() -> u32 {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManualAnchor {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
    pub anchor_type: AnchorType,
    pub locked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AnchorType {
    Initial,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackingData {
    pub config: TrackingConfig,
    pub frames: Vec<TrackingFrame>,
    pub problem_ranges: Vec<ProblemRange>,
    pub analyzed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct TrackingConfig {
    pub analysis_long_edge: u32,
    pub local_search_radius: u32,
    pub accept_threshold: f64,
    pub weak_threshold: f64,
    pub global_search_threshold: f64,
    pub optical_flow_radius: u32,
    pub smoothing_alpha: f64,
    pub max_frame_displacement: f64,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            analysis_long_edge: 720,
            local_search_radius: 48,
            accept_threshold: 0.64,
            weak_threshold: 0.54,
            global_search_threshold: 0.42,
            optical_flow_radius: 3,
            smoothing_alpha: 0.20,
            max_frame_displacement: 90.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackingFrame {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
    pub confidence: f64,
    pub status: TrackingStatus,
    pub source: TrackingSource,
    pub locked: bool,
    pub scores: TrackingScores,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrackingStatus {
    AutoGood,
    AutoWeak,
    NeedReview,
    Manual,
    Interpolated,
    /// The watermark is not observable in this frame; rendering must be a no-op.
    Occluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrackingSource {
    Forward,
    Backward,
    Fused,
    Manual,
    Interpolated,
    Occluded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackingScores {
    pub template: f64,
    pub highpass: f64,
    pub edge: f64,
    pub motion: f64,
    pub position: f64,
    pub size: f64,
    #[serde(default)]
    pub optical_flow: Option<f64>,
    pub forward_backward: Option<f64>,
    pub motion_smoothness: Option<f64>,
    #[serde(default)]
    pub match_margin: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProblemRange {
    pub start_frame: u64,
    pub end_frame: u64,
    pub worst_frame: u64,
    pub min_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct RemovalConfig {
    pub mode: RemovalMode,
    pub mask_padding: u32,
    pub feather_radius: u32,
    pub replacement_path: Option<String>,
    pub replacement_scale: f64,
    pub replacement_opacity: f64,
    pub replacement_offset_x: f64,
    pub replacement_offset_y: f64,
    pub temporal_window_before: u32,
    pub temporal_window_after: u32,
    pub max_temporal_candidates: u32,
    pub restoration_roi_padding: u32,
    pub artifact_threshold: f64,
    pub fallback_policy: FallbackPolicy,
    pub inpaint_variant: InpaintVariant,
}

impl Default for RemovalConfig {
    fn default() -> Self {
        Self {
            mode: RemovalMode::Blur,
            mask_padding: 4,
            feather_radius: 3,
            replacement_path: None,
            replacement_scale: 1.0,
            replacement_opacity: 1.0,
            replacement_offset_x: 0.0,
            replacement_offset_y: 0.0,
            temporal_window_before: 12,
            temporal_window_after: 12,
            max_temporal_candidates: 10,
            restoration_roi_padding: 32,
            artifact_threshold: 0.25,
            fallback_policy: FallbackPolicy::TemporalInpaintBlur,
            inpaint_variant: InpaintVariant::Iterative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemovalMode {
    Replacement,
    Blur,
    Inpaint,
    TemporalRestore,
    AutoBest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FallbackPolicy {
    TemporalInpaintBlur,
    InpaintBlur,
    BlurOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InpaintVariant {
    Iterative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnchorFrame {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplatePaths {
    pub original: String,
    pub grayscale: String,
    pub high_contrast: String,
    #[serde(default)]
    pub mask: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameResult {
    pub frame: u64,
    pub timestamp_seconds: f64,
    pub path: String,
}
