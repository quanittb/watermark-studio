export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface VideoMetadata {
  width: number;
  height: number;
  durationSeconds: number;
  fps: number;
  frameCount: number;
  codec: string | null;
  pixelFormat: string | null;
}

export interface WatermarkAnchor {
  frame: number;
  timestampSeconds: number;
  bbox: BoundingBox;
}

export interface TemplatePaths {
  original: string;
  grayscale: string;
  highContrast: string;
  mask: string | null;
}

export interface WatermarkConfig {
  label: string | null;
  anchor: WatermarkAnchor | null;
  templates: TemplatePaths | null;
  templatePadding: number;
}

export type CalibrationPreset = 'LEARNA_AI_PERIODIC' | 'LEARNA_AI_ADAPTIVE' | 'GENERAL_MOVING';
export type CalibrationStatus = 'READY' | 'STALE' | 'NEEDS_REVIEW' | 'FAILED';
export interface ScanRange {
  startFrame: number;
  endFrame: number;
}
export type RenderPolicy = 'REMOVE_THEN_COVER' | 'COVER_ALL_FAST' | 'REMOVE_ONLY';
export type FrameDisposition = 'INPAINT' | 'COVER_BADGE' | 'PASSTHROUGH';
export interface CalibrationProfileV9 {
  sourceFingerprint: { sha256: string; sizeBytes: number; frameCount: number; width: number; height: number };
  activeIntervals: Array<{ startFrame: number; endFrame: number }>;
  frameData: Array<{ frame: number; bbox: BoundingBox; maskRequired: boolean; positionSource: string; uncertaintyPx?: number }>;
  detectorVersion: string;
  profileSha256: string;
  status: 'READY' | 'NEEDS_REVIEW';
}
export interface CalibrationProfile {
  version: number;
  preset: CalibrationPreset;
  route?: 'AUTO_GLOBAL_TEMPLATE' | 'AUTO_ROI_TEMPLATE' | 'ROI_FALLBACK' | 'AUTO_FIND' | string | null;
  scanRange?: ScanRange | null;
  scanRangeSemantics?: 'inclusive' | string;
  excludedFrameCount?: number;
  outsideRangePolicy?: 'PASSTHROUGH_WARN' | string;
  detectorVersion: string | null;
  validationVersion?: string | null;
  sourceFingerprint: { sha256: string; sizeBytes: number; frameCount: number; width: number; height: number } | null;
  profilePath: string;
  maskPath: string;
  canonicalMaskPath: string | null;
  editedMaskPath?: string | null;
  editedMaskSha256?: string | null;
  autoMaskPath: string | null;
  blendMaskPath: string | null;
  brushDeltaPath: string | null;
  maskHash: string;
  profileHash: string;
  trajectoryModel?: {
    type?: string;
    source?: string;
    periodicPrior?: string;
    maxInterpolationGap?: number;
    segments?: Array<{ startFrame: number; x: number; y: number; scale: number }>;
  } | null;
  difficultFrames?: number[];
  contactSheetPath?: string | null;
  activeIntervals?: Array<{ startFrame: number; endFrame: number }>;
  roiEvidenceFrames?: number[];
  roiEvidence?: RoiEvidenceRecord[];
  roiBudgetUsed?: number;
  roiBudgetMax?: number;
  outcome?: 'READY' | 'AUTO_REFINEMENT_REQUIRED' | 'AWAITING_ROI_BATCH' | 'NEEDS_REVIEW_DRAFT' | 'FAILED_RUNTIME' | string;
  frameData?: Array<{
    frame: number;
    bbox: BoundingBox;
    visibility: boolean;
    confidence: number;
    occlusion: boolean;
    maskRequired: boolean;
    positionSource: string;
    scale: number;
    opacity: number;
    uncertaintyPx?: number;
    sceneId?: number;
  }>;
  trajectoryGate?: {
    status: 'PASSED' | 'FAILED';
    inlierRatio?: number;
    residualMedian?: number | null;
    residualP95?: number | null;
    directCoverage?: number | null;
    validatedCoverage?: number | null;
    confirmedCoverage?: number | null;
    measuredCoverage?: number | null;
    hardMeasuredFrames?: number;
    roiEvidenceFrames?: number;
    maxInterpolationGap?: number;
    maxObservationGap?: number;
    rawResidualMedian?: number | null;
    rawResidualP95?: number | null;
    residualFitSource?: 'CONFIRMED_CONTROL_PATH' | 'SELECTED_CANDIDATE_PATH' | string;
    residualFitFrames?: number;
    failureReasons?: string[];
    reviewRangesSuppressed?: boolean;
    rawReviewRanges?: Array<{
      startFrame: number;
      endFrame: number;
      suggestedFrames?: number[];
      reason?: string;
    }>;
    refinedFrames?: number;
    refinedCoverage?: number | null;
    holdout?: {
      count?: number;
      trainingCount?: number;
      median?: number | null;
      p95?: number | null;
      inlierRatio?: number;
      reason?: string | null;
    } | null;
    holdoutMedian?: number | null;
    holdoutP95?: number | null;
    holdoutInlierRatio?: number | null;
    reviewRanges?: Array<{
      startFrame: number;
      endFrame: number;
      suggestedFrames?: number[];
      reason?: string;
    }>;
  };
  sampleFrame: number;
  frameCount: number;
  quality: {
    status: CalibrationStatus;
    reliableFrames: number;
    lowConfidenceFrames: number;
    maskPixels: number;
    glyphCoverage: number;
    contamination: number;
    largeHoles: number;
  };
}

export type AnchorType = 'INITIAL' | 'MANUAL';

export interface ManualAnchor {
  frame: number;
  timestampSeconds: number;
  bbox: BoundingBox;
  anchorType: AnchorType;
  locked: boolean;
}

export interface WatermarkProject {
  version: number;
  id: string;
  source: { path: string; fileName: string };
  video: VideoMetadata;
  watermark: WatermarkConfig;
  calibration: CalibrationProfile | null;
  anchors: ManualAnchor[];
  tracking: TrackingData | null;
  removal: RemovalConfig | null;
  roiEvidence?: RoiEvidenceRecord[];
}

export interface RoiEvidenceRecord {
  id: string;
  frame: number;
  bbox: BoundingBox;
  source: 'USER' | 'MIGRATED_V7' | 'AUTO_RECOMMENDED' | string;
  createdAt: string;
  attemptId: string;
  acceptedByDetector?: boolean;
  rejectionReason?: string | null;
}

export type TrackingStatus = 'AUTO_GOOD' | 'AUTO_WEAK' | 'NEED_REVIEW' | 'MANUAL' | 'INTERPOLATED' | 'OCCLUDED';
export type TrackingSource = 'FORWARD' | 'BACKWARD' | 'FUSED' | 'MANUAL' | 'INTERPOLATED' | 'OCCLUDED';

export interface TrackingScores {
  template: number;
  highpass: number;
  edge: number;
  motion: number;
  position: number;
  size: number;
  opticalFlow: number | null;
  forwardBackward: number | null;
  motionSmoothness: number | null;
  matchMargin: number | null;
}

export interface TrackingFrame {
  frame: number;
  timestampSeconds: number;
  bbox: BoundingBox;
  confidence: number;
  status: TrackingStatus;
  source: TrackingSource;
  locked: boolean;
  scores: TrackingScores;
}

export interface ProblemRange {
  startFrame: number;
  endFrame: number;
  worstFrame: number;
  minConfidence: number;
}

export interface TrackingData {
  config: {
    analysisLongEdge: number;
    localSearchRadius: number;
    acceptThreshold: number;
    weakThreshold: number;
    globalSearchThreshold: number;
    opticalFlowRadius: number;
    smoothingAlpha: number;
    maxFrameDisplacement: number;
  };
  frames: TrackingFrame[];
  problemRanges: ProblemRange[];
  analyzedAt: string | null;
}

export type RemovalMode = 'REPLACEMENT' | 'BLUR' | 'INPAINT' | 'TEMPORAL_RESTORE' | 'AUTO_BEST';
export type FallbackPolicy = 'TEMPORAL_INPAINT_BLUR' | 'INPAINT_BLUR' | 'BLUR_ONLY';
export type InpaintVariant = 'ITERATIVE';

export interface RemovalConfig {
  mode: RemovalMode;
  maskPadding: number;
  featherRadius: number;
  replacementPath: string | null;
  replacementScale: number;
  replacementOpacity: number;
  replacementOffsetX: number;
  replacementOffsetY: number;
  temporalWindowBefore: number;
  temporalWindowAfter: number;
  maxTemporalCandidates: number;
  restorationRoiPadding: number;
  artifactThreshold: number;
  fallbackPolicy: FallbackPolicy;
  inpaintVariant: InpaintVariant;
}

export interface FrameResult {
  frame: number;
  timestampSeconds: number;
  path: string;
}

export interface AppError {
  code: string;
  message: string;
  stage?: string;
  artifactPath?: string | null;
}
