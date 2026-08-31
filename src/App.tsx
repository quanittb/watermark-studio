import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";
import {
  acceptTrackingFrame,
  cancelJob,
  cancelRender,
  cancelTracking,
  chooseOutputDirectory,
  chooseReplacementPath,
  chooseVideoPath,
  chooseVideoPaths,
  autoCalibrateBestQuality,
  detectHardware,
  enqueueBestQualityJob,
  extractFocusPreview,
  getErrorMessage,
  getProject,
  interpolateTrackingRange,
  listJobs,
  listProjects,
  markOccludedRange,
  openVideo,
  regenJob,
  revalidateReviewJob,
  removeProject,
  readProjectAssetBytes,
  renderVideo,
  retrackTrack,
  saveManualAnchor,
  saveCalibrationMaskEdit,
  saveRemovalConfig,
  saveWatermarkAnchor,
  suggestBestQualitySamples,
} from "./services/projectApi";
import type {
  BestQualityReplacement,
  BestQualitySample,
  FocusPreview,
  JobRecord,
  RoiHint,
} from "./services/projectApi";
import type {
  BoundingBox,
  RemovalConfig,
  ScanRange,
  TrackingFrame,
  WatermarkProject,
} from "./types/project";
import "./styles.css";
import { translate } from "./i18n";

type LoadingTask =
  "opening" | "saving" | "tracking" | "sampling" | "calibrating" | "rendering" | null;
type WorkspaceMode = "best" | "legacy";
type AppRoute = "projects" | "review" | "queue" | "history" | "settings";
type SettingsTab = "general" | "processing" | "updates" | "advanced" | "about";
type Point = { x: number; y: number };
type ContentRect = { left: number; top: number; width: number; height: number };
type OperationProgress = {
  phase: string;
  currentFrame: number;
  totalFrames: number;
  progress: number;
};
type OperationDialogState = {
  task: Exclude<LoadingTask, null>;
  status: "running" | "success" | "error";
  title: string;
  detail: string;
  reviewRanges?: Array<{
    startFrame: number;
    endFrame: number;
    suggestedFrames: number[];
    reason?: string;
  }>;
};

const defaultRemoval: RemovalConfig = {
  mode: "BLUR",
  maskPadding: 4,
  featherRadius: 3,
  replacementPath: null,
  replacementScale: 1,
  replacementOpacity: 1,
  replacementOffsetX: 0,
  replacementOffsetY: 0,
  temporalWindowBefore: 12,
  temporalWindowAfter: 12,
  maxTemporalCandidates: 10,
  restorationRoiPadding: 32,
  artifactThreshold: 0.25,
  fallbackPolicy: "TEMPORAL_INPAINT_BLUR",
  inpaintVariant: "ITERATIVE",
};

// Match the calibration convergence guard in calibrate_trajectory_v6.py.  It
// is intentionally only a review-hint guard: a saturated but inaccurate path
// remains NEEDS_REVIEW and cannot be queued or rendered.
const ROI_SATURATION_MIN_EVIDENCE = 24;
const ROI_SATURATION_MIN_CONFIRMED_COVERAGE = 0.15;
const ROI_SATURATION_MIN_PATH_COVERAGE = 0.70;
const defaultBestReplacement: BestQualityReplacement = {
  kind: "text",
  text: "",
  imagePath: null,
  placement: "follow",
  fixedX: 0,
  fixedY: 0,
  scale: 1,
  opacity: 1,
};

function normalizeRemoval(config: RemovalConfig | null): RemovalConfig {
  return { ...defaultRemoval, ...(config ?? {}) };
}

function Icon({ children }: { children: ReactNode }) {
  return <span className="icon">{children}</span>;
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainingSeconds.toFixed(3).padStart(6, "0")}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function formatJobError(error: string): string {
  const normalized = error.replace(/\s+/g, " ").trim();
  if (/No space left on device|libpng error: Write Error/i.test(normalized)) {
    return "Không đủ dung lượng tạm khi tạo mask AI. Hãy kiểm tra workspace/cache rồi thử Regen.";
  }
  if (normalized.length <= 280) return normalized;
  return `${normalized.slice(0, 277).trimEnd()}…`;
}

function formatCalibrationReviewRanges(
  ranges?: Array<{
    startFrame: number;
    endFrame: number;
    suggestedFrames?: number[];
  }>,
): string {
  if (!ranges?.length) return "";
  return ranges
    .slice(0, 8)
    .map((range) => {
      const fallback = Math.round((range.startFrame + range.endFrame) / 2);
      const frames = (range.suggestedFrames?.length ? range.suggestedFrames : [fallback]).slice(0, 3);
      return `${range.startFrame}–${range.endFrame} → ưu tiên ${frames[0]}${frames.length > 1 ? ` (dự phòng ${frames.slice(1).join("/")})` : ""}`;
    })
    .join("; ");
}

function roiEvidenceStorageKey(projectId: string): string {
  return `watermark-studio:roi-evidence:${projectId}`;
}

function scanRangeStorageKey(projectId: string): string {
  return `watermark-studio:scan-range:${projectId}`;
}

function readStoredScanRange(project: WatermarkProject): ScanRange {
  const lastFrame = Math.max(0, project.video.frameCount - 1);
  const profileRange = project.calibration?.scanRange;
  const stored = (() => {
    try {
      const raw = window.localStorage.getItem(scanRangeStorageKey(project.id));
      return raw ? JSON.parse(raw) as Partial<ScanRange> : null;
    } catch {
      return null;
    }
  })();
  const startFrame = Number(profileRange?.startFrame ?? stored?.startFrame ?? 0);
  const endFrame = Number(profileRange?.endFrame ?? stored?.endFrame ?? lastFrame);
  if (!Number.isInteger(startFrame) || !Number.isInteger(endFrame) || startFrame < 0 || endFrame < startFrame || endFrame > lastFrame) {
    return { startFrame: 0, endFrame: lastFrame };
  }
  return { startFrame, endFrame };
}

function readStoredRoiEvidence(projectId: string): Array<RoiHint & { frame: number }> {
  try {
    const raw = window.localStorage.getItem(roiEvidenceStorageKey(projectId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is RoiHint & { frame: number } => {
      if (!item || typeof item !== "object") return false;
      const value = item as Record<string, unknown>;
      return Number.isFinite(value.frame)
        && Number.isFinite(value.x)
        && Number.isFinite(value.y)
        && Number.isFinite(value.width)
        && Number.isFinite(value.height)
        && Number(value.width) >= 1
        && Number(value.height) >= 1;
    });
  } catch {
    return [];
  }
}

async function openArtifact(path: string): Promise<void> {
  const lowerPath = path.toLowerCase();
  const openWith = lowerPath.endsWith(".json") ? "notepad.exe" : lowerPath.endsWith(".png") ? "mspaint.exe" : undefined;
  await openPath(path, openWith);
}

function sourceToPercent(value: number, sourceSize: number): number {
  return sourceSize > 0 ? (value / sourceSize) * 100 : 0;
}

function bytesToDataUrl(bytes: number[], mime = "image/png"): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.slice(offset, offset + chunkSize));
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

function trackingColor(status: TrackingFrame["status"]): string {
  switch (status) {
    case "AUTO_GOOD":
      return "#77d99a";
    case "AUTO_WEAK":
      return "#e3bd70";
    case "NEED_REVIEW":
      return "#d97676";
    case "MANUAL":
      return "#8fb8ff";
    case "INTERPOLATED":
      return "#b89be8";
    case "OCCLUDED":
      return "#8b8f9c";
  }
}

function trackingSegments(
  frames: TrackingFrame[],
): Array<{ start: number; end: number; status: TrackingFrame["status"] }> {
  if (!frames.length) return [];
  const segments: Array<{
    start: number;
    end: number;
    status: TrackingFrame["status"];
  }> = [];
  let start = 0;
  for (let index = 1; index <= frames.length; index += 1) {
    if (
      index === frames.length ||
      frames[index].status !== frames[start].status
    ) {
      segments.push({ start, end: index - 1, status: frames[start].status });
      start = index;
    }
  }
  return segments;
}

function ReplacementPreview({
  replacement,
  previewUrl,
  crop,
  target,
}: {
  replacement: BestQualityReplacement | null;
  previewUrl: string | null;
  crop: BoundingBox;
  target: BoundingBox;
}) {
  if (!replacement) return null;
  const follows = replacement.placement === "follow";
  const x = follows ? target.x : replacement.fixedX;
  const y = follows ? target.y : replacement.fixedY;
  const width = target.width * replacement.scale;
  const style = {
    left: `${((x - crop.x) / crop.width) * 100}%`,
    top: `${((y - crop.y) / crop.height) * 100}%`,
    width: `${(width / crop.width) * 100}%`,
    opacity: replacement.opacity,
  };
  return (
    <div
      className="replacement-preview"
      style={style}
      title="Replacement placement preview"
    >
      {replacement.kind === "image" && previewUrl ? (
        <img src={previewUrl} alt="Replacement PNG preview" />
      ) : (
        <span>{replacement.text || "Text preview"}</span>
      )}
    </div>
  );
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoFrameRef = useRef<HTMLDivElement>(null);
  const selectionSurfaceRef = useRef<HTMLDivElement>(null);
  const selectionStartRef = useRef<Point | null>(null);
  const selectionFrameRef = useRef<number | null>(null);
  const lockedBestSampleFrameRef = useRef<number | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskDrawingRef = useRef(false);
  const maskLoadTokenRef = useRef(0);
  const focusPreviewTokenRef = useRef(0);
  const [project, setProject] = useState<WatermarkProject | null>(null);
  const [loadingTask, setLoadingTask] = useState<LoadingTask>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [scanRange, setScanRange] = useState<ScanRange>({ startFrame: 0, endFrame: 0 });
  const [selectionMode, setSelectionMode] = useState(false);
  const [selection, setSelection] = useState<BoundingBox | null>(null);
  const [contentRect, setContentRect] = useState<ContentRect | null>(null);
  const [label, setLabel] = useState("Learna AI");
  const [playing, setPlaying] = useState(false);
  const [message, setMessage] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<OperationProgress | null>(null);
  const [removal, setRemoval] = useState<RemovalConfig>(defaultRemoval);
  const [bestQualitySamples, setBestQualitySamples] = useState<
    BestQualitySample[]
  >([]);
  const [selectedBestQualitySample, setSelectedBestQualitySample] =
    useState<BestQualitySample | null>(null);
  const [bestSampleScanRound, setBestSampleScanRound] = useState(0);
  const [rejectedBestSampleFrames, setRejectedBestSampleFrames] = useState<
    number[]
  >([]);
  const [rejectedSceneSignatures, setRejectedSceneSignatures] = useState<string[]>([]);
  const [roiFallbackArmed, setRoiFallbackArmed] = useState(false);
  const [roiEvidence, setRoiEvidence] = useState<Array<RoiHint & { frame: number }>>([]);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("best");
  const [bestReplacement, setBestReplacement] =
    useState<BestQualityReplacement | null>(null);
  const [focusPreview, setFocusPreview] = useState<FocusPreview | null>(null);
  const [inspectionMode, setInspectionMode] = useState(false);
  const [inspectionZoom, setInspectionZoom] = useState(1);
  const [outputRoot, setOutputRoot] = useState(
    () => window.localStorage.getItem("watermark-studio:output-root") ?? "",
  );
  const [outputName, setOutputName] = useState("");
  const [language, setLanguage] = useState<"vi" | "en">(
    () =>
      (window.localStorage.getItem("watermark-studio:language") as
        "vi" | "en" | null) ?? (navigator.language.toLowerCase().startsWith("vi") ? "vi" : "en"),
  );
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [projects, setProjects] = useState<WatermarkProject[]>([]);
  const [route, setRoute] = useState<AppRoute>(() =>
    window.location.hash.startsWith("#/review")
      ? "review"
      : (window.location.hash.slice(2).split("/")[0] as AppRoute) || "projects",
  );
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("general");
  const [hardware, setHardware] = useState<Awaited<
    ReturnType<typeof detectHardware>
  > | null>(null);
  const [brushMode, setBrushMode] = useState<"add" | "erase">("add");
  const [brushSize, setBrushSize] = useState(6);
  const [maskOpacity, setMaskOpacity] = useState(0.65);
  const [maskUndo, setMaskUndo] = useState<string[]>([]);
  const [maskRedo, setMaskRedo] = useState<string[]>([]);
  const [maskEditorReady, setMaskEditorReady] = useState(false);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const [operationDialog, setOperationDialog] = useState<OperationDialogState | null>(null);
  // Queue progress events can arrive while an adaptive calibration is still
  // running.  Calibration currently reports its result atomically, so keep
  // those stale queue values out of the calibration dialog instead of
  // showing a misleading percentage/step.
  const operationTaskRef = useRef<LoadingTask>(null);

  const videoUrl = project ? convertFileSrc(project.source.path) : null;
  const currentTracking =
    project?.tracking?.frames.find((frame) => frame.frame === currentFrame) ??
    null;
  const trackingNeedsReview =
    currentTracking &&
    ["AUTO_WEAK", "NEED_REVIEW", "INTERPOLATED"].includes(
      currentTracking.status,
    );
  // Keep provisional coordinates visible for diagnosis, but style them as
  // unverified and require a fresh user-drawn selection before saving.
  const displaySelection =
    selection ??
    currentTracking?.bbox ??
    project?.watermark.anchor?.bbox ??
    null;
  const displayIsProvisional = Boolean(!selection && trackingNeedsReview);
  const unresolvedCount = project?.tracking?.problemRanges.length ?? 0;
  const problemRangeSummary =
    project?.tracking?.problemRanges
      .map((range) =>
        range.startFrame === range.endFrame
          ? `${range.startFrame}`
          : `${range.startFrame}–${range.endFrame}`,
      )
      .join(", ") ?? "";
  const calibrationEvidenceFrames = new Set<number>([
    ...(project?.calibration?.roiEvidenceFrames ?? []),
    ...roiEvidence.map((item) => item.frame),
  ]);
  const calibrationGate = project?.calibration?.trajectoryGate;
  const calibrationRoiCount = calibrationGate?.roiEvidenceFrames ?? calibrationEvidenceFrames.size;
  const calibrationConfirmedCoverage = calibrationGate?.confirmedCoverage ?? 0;
  const calibrationMeasuredCoverage = calibrationGate?.measuredCoverage ?? 0;
  const calibrationResidualP95 = calibrationGate?.residualP95 ?? null;
  const calibrationResidualTolerance = project
    ? 3 * project.video.width / 1080
    : 3;
  const calibrationMaxGap = calibrationGate?.maxObservationGap
    ?? calibrationGate?.maxInterpolationGap
    ?? 0;
  const roiReviewSaturated = Boolean(
    calibrationRoiCount >= ROI_SATURATION_MIN_EVIDENCE
    && calibrationConfirmedCoverage >= ROI_SATURATION_MIN_CONFIRMED_COVERAGE
    && calibrationMeasuredCoverage >= ROI_SATURATION_MIN_PATH_COVERAGE
    && (
      calibrationMaxGap > 18
      || (calibrationResidualP95 != null && calibrationResidualP95 > calibrationResidualTolerance)
    ),
  );
  // Once the profile has broad evidence but still has a poor fit, showing the
  // same weak clusters again only creates an endless manual-ROI loop.  The
  // backend records the raw ranges in diagnostics; the Review UI hides them
  // and points to automatic trajectory refinement instead.
  const calibrationReviewRanges = roiReviewSaturated
    ? []
    : (calibrationGate?.reviewRanges ?? []).filter(
      (range) => !Array.from(calibrationEvidenceFrames).some(
        (frame) => frame >= range.startFrame && frame <= range.endFrame,
      ),
    );
  const pendingCalibrationReviewRanges = calibrationReviewRanges;
  const hasNextReviewProblem = workspaceMode === "legacy"
    ? unresolvedCount > 0
    : pendingCalibrationReviewRanges.length > 0;
  const isBusy = loadingTask !== null;
  const inspectionTarget =
    focusPreview?.frame === currentFrame ? displaySelection : null;
  const focusPreviewUrl = focusPreview
    ? convertFileSrc(focusPreview.path)
    : null;
  const bestReplacementPreviewUrl =
    bestReplacement?.kind === "image" && bestReplacement.imagePath
      ? convertFileSrc(bestReplacement.imagePath)
      : null;
  const visibleJobs = jobs.filter((job) => route === "queue"
    ? ["QUEUED", "PREPARING", "INFERENCING", "ENCODING", "VERIFYING"].includes(job.status)
    : ["COMPLETED", "NEEDS_REVIEW", "FAILED", "CANCELED", "INTERRUPTED"].includes(job.status));
  const orderedJobs = route === "history"
    ? [...visibleJobs].sort((left, right) => Number(right.updatedAt) - Number(left.updatedAt))
    : visibleJobs;
  const t = (key: Parameters<typeof translate>[1]) => translate(language, key);
  const dialogSteps = operationDialog?.task === "calibrating"
    ? ["Validate source", "Validate scan range", "Global template scan", "Fit trajectory", "Refine active frames", "Build consensus mask", "Validate V6"]
    : operationDialog?.task === "sampling"
      ? ["Read source", "Scan candidates", "Score glyph", "Build contact sheet"]
      : ["Prepare profile", "ProPainter FP32", "Encode output", "Verify QA"];
  const dialogProgress = progress && operationDialog?.status === "running"
    ? Math.round(clamp(progress.progress, 0, 1) * 100)
    : operationDialog?.status === "success" || operationDialog?.status === "error" ? 100 : 0;
  const dialogActiveStep = operationDialog?.status === "success"
    ? dialogSteps.length - 1
    : progress && operationDialog?.status === "running"
      ? Math.min(dialogSteps.length - 1, Math.floor(clamp(progress.progress, 0, 0.999) * dialogSteps.length))
      : operationDialog?.status === "error" ? Math.max(0, dialogSteps.length - 1) : 0;

  const refreshHardware = async () => {
    try {
      setHardware(await detectHardware());
      setMessage(language === "vi" ? "Đã cập nhật thông tin phần cứng." : "Hardware information refreshed.");
    } catch (hardwareError) {
      setError(getErrorMessage(hardwareError));
    }
  };

  const checkForUpdatesNow = async () => {
    if (isBusy) return;
    setError(null);
    setMessage(language === "vi" ? "Đang kiểm tra bản cập nhật…" : "Checking for updates…");
    try {
      const update = await check();
      if (!update) {
        setMessage(language === "vi" ? "Bạn đang dùng phiên bản mới nhất." : "You are using the latest version.");
        return;
      }
      const latestJobs = await listJobs();
      if (latestJobs.some((job) => ["PREPARING", "INFERENCING", "ENCODING", "VERIFYING"].includes(job.status))) {
        setMessage(language === "vi" ? "Có bản cập nhật mới; sẽ cài sau khi hàng đợi GPU rảnh." : "An update is available; installation waits until the GPU queue is idle.");
        return;
      }
      if (!window.confirm(language === "vi" ? `Watermark Studio ${update.version} đã sẵn sàng. Cài đặt ngay?` : `Watermark Studio ${update.version} is available. Install now?`)) return;
      await update.downloadAndInstall();
      await relaunch();
    } catch (updateError) {
      setError(getErrorMessage(updateError));
      setMessage(language === "vi" ? "Không thể kiểm tra bản cập nhật." : "Unable to check for updates.");
    }
  };

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void listen<OperationProgress>("operation-progress", (event) => {
      if (disposed) return;
      const activeTask = operationTaskRef.current;
      if (!activeTask || activeTask === "calibrating") return;
      // Queue workers share the same event channel as Review operations.  A
      // background render must not overwrite the sample-scan dialog (and a
      // stale scan event must not move the render stepper backwards).
      const phase = event.payload.phase.toLowerCase();
      if (activeTask === "sampling" && !/(scan|sample|candidate|phase)/.test(phase)) return;
      if (activeTask === "rendering" && /(scan|sample|candidate)/.test(phase)) return;
      setProgress(event.payload);
    }).then((stop) => {
      if (disposed) stop();
      else unlisten = stop;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    const onHashChange = () => {
      const segment = window.location.hash.slice(2).split("/")[0];
      setRoute(
        (["projects", "review", "queue", "history", "settings"].includes(
          segment,
        )
          ? segment
          : "projects") as AppRoute,
      );
    };
    window.addEventListener("hashchange", onHashChange);
    if (!window.location.hash) window.location.hash = "#/projects";
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    void listProjects()
      .then(setProjects)
      .catch(() => undefined);
    void detectHardware()
      .then(setHardware)
      .catch(() => undefined);
  }, []);

  const navigate = (next: AppRoute, projectId?: string) => {
    window.location.hash =
      next === "review" && projectId ? `#/review/${projectId}` : `#/${next}`;
    setRoute(next);
  };

  useEffect(() => {
    const lastCheck = Number(
      window.localStorage.getItem("watermark-studio:last-update-check") ?? 0,
    );
    if (Date.now() - lastCheck < 24 * 60 * 60 * 1000) return;
    window.localStorage.setItem(
      "watermark-studio:last-update-check",
      String(Date.now()),
    );
    const timer = window.setTimeout(() => {
      void check()
        .then(async (update) => {
          if (!update || loadingTask !== null) return;
          const latestJobs = await listJobs();
          if (latestJobs.some((job) => ["PREPARING", "INFERENCING", "ENCODING", "VERIFYING"].includes(job.status))) {
            setMessage("Update available; installation is postponed until the GPU queue is idle.");
            return;
          }
          const accepted = window.confirm(
            `Watermark Studio ${update.version} is available. Install now?`,
          );
          if (!accepted) return;
          await update.downloadAndInstall();
          await relaunch();
        })
        .catch(() => {
          /* Offline/dev builds simply skip update checks. */
        });
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [loadingTask]);

  useEffect(() => {
    if (route !== "queue" && route !== "history") return;
    let disposed = false;
    const refresh = () => {
      void listJobs()
        .then((next) => {
          if (!disposed) setJobs(next);
        })
        .catch(() => undefined);
    };
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [route]);

  useEffect(() => {
    const hashParts = window.location.hash.slice(2).split("/");
    const hashProjectId = hashParts[0] === "review" ? hashParts[1] : null;
    const projectId =
      hashProjectId ??
      window.localStorage.getItem("watermark-studio:last-project-id");
    if (!projectId) return;
    let disposed = false;
    setLoadingTask("opening");
    setMessage("Loading saved project…");
    void getProject(projectId)
      .then((savedProject) => {
        if (disposed) return;
        setProject(savedProject);
        setScanRange(readStoredScanRange(savedProject));
        setRoiEvidence(readStoredRoiEvidence(savedProject.id));
        setSelection(
          savedProject.tracking
            ? null
            : (savedProject.watermark.anchor?.bbox ?? null),
        );
        setCurrentFrame(savedProject.watermark.anchor?.frame ?? 0);
        setCurrentTime(savedProject.watermark.anchor?.timestampSeconds ?? 0);
        setLabel(savedProject.watermark.label ?? "Learna AI");
        setRemoval(normalizeRemoval(savedProject.removal));
        const defaultOutputName = `${savedProject.source.fileName.replace(/\.[^.]+$/, "")}_watermark_removed_best.mp4`;
        setOutputName(window.localStorage.getItem(`watermark-studio:output-name:${savedProject.id}`) ?? defaultOutputName);
        setMessage("Saved project loaded.");
      })
      .catch(() => {
        if (!disposed && !hashProjectId)
          window.localStorage.removeItem("watermark-studio:last-project-id");
        if (!disposed && hashProjectId) {
          setError(language === "vi" ? "Project trong đường dẫn không còn tồn tại." : "The project in this URL is no longer available.");
          navigate("projects");
        }
      })
      .finally(() => {
        if (!disposed) setLoadingTask(null);
      });
    return () => {
      disposed = true;
    };
  }, []);

  const measureContentRect = useCallback(() => {
    const frame = videoFrameRef.current;
    const video = videoRef.current;
    if (
      !frame ||
      !video ||
      !project ||
      video.clientWidth === 0 ||
      video.clientHeight === 0
    )
      return;
    const frameBounds = frame.getBoundingClientRect();
    const videoBounds = video.getBoundingClientRect();
    const sourceRatio = project.video.width / project.video.height;
    const contentWidth = Math.min(
      video.clientWidth,
      video.clientHeight * sourceRatio,
    );
    const contentHeight = contentWidth / sourceRatio;
    setContentRect({
      left:
        (video.clientWidth - contentWidth) / 2 +
        (videoBounds.left - frameBounds.left),
      top:
        (video.clientHeight - contentHeight) / 2 +
        (videoBounds.top - frameBounds.top),
      width: contentWidth,
      height: contentHeight,
    });
  }, [project]);

  useEffect(() => {
    const frame = videoFrameRef.current;
    if (!frame) return;
    const observer = new ResizeObserver(measureContentRect);
    observer.observe(frame);
    return () => observer.disconnect();
  }, [measureContentRect]);

  const setVideoFrame = (
    frame: number,
    preserveBestSampleSelection = false,
  ) => {
    if (!project || !videoRef.current) return;
    const lastFrame = Math.max(0, project.video.frameCount - 1);
    const nextFrame = clamp(Math.round(frame), 0, lastFrame);
    const nextTime = nextFrame / project.video.fps;
    videoRef.current.currentTime = nextTime;
    if (!preserveBestSampleSelection) {
      lockedBestSampleFrameRef.current = null;
      selectionFrameRef.current = null;
      setSelection(null);
      setInspectionMode(false);
    }
    setCurrentFrame(nextFrame);
    setCurrentTime(nextTime);
  };

  const setVideoTime = (time: number) => {
    if (!project || !videoRef.current || !Number.isFinite(time)) return;
    const nextTime = clamp(time, 0, project.video.durationSeconds);
    videoRef.current.currentTime = nextTime;
    lockedBestSampleFrameRef.current = null;
    selectionFrameRef.current = null;
    setSelection(null);
    setInspectionMode(false);
    setCurrentTime(nextTime);
    setCurrentFrame(
      clamp(
        Math.round(nextTime * project.video.fps),
        0,
        Math.max(0, project.video.frameCount - 1),
      ),
    );
  };

  const onVideoLoadedMetadata = () => {
    setVideoLoadError(null);
    measureContentRect();
    if (!project || !videoRef.current) return;
    const initialFrame = project.watermark.anchor?.frame ?? 0;
    const initialTime =
      project.watermark.anchor?.timestampSeconds ??
      initialFrame / project.video.fps;
    videoRef.current.currentTime = initialTime;
    setCurrentFrame(initialFrame);
    setCurrentTime(initialTime);
  };

  const onVideoTimeUpdate = () => {
    if (!project || !videoRef.current) return;
    const time = videoRef.current.currentTime;
    setCurrentTime(time);
    const frame = clamp(
      Math.round(time * project.video.fps),
      0,
      Math.max(0, project.video.frameCount - 1),
    );
    setCurrentFrame(frame);
    // Keep a confirmed candidate box visible while the seek settles on that
    // exact frame. Any navigation to another frame clears the stale box.
    if (selectionFrameRef.current !== null && selectionFrameRef.current !== frame) {
      selectionFrameRef.current = null;
      setSelection(null);
    } else if (!selectionMode && lockedBestSampleFrameRef.current !== frame) {
      setSelection(null);
    }
  };

  const activateProject = (nextProject: WatermarkProject) => {
    window.localStorage.setItem(
      "watermark-studio:last-project-id",
      nextProject.id,
    );
    setProject(nextProject);
    setScanRange(readStoredScanRange(nextProject));
    setRoiEvidence(readStoredRoiEvidence(nextProject.id));
    setSelection(null);
    selectionFrameRef.current = null;
    lockedBestSampleFrameRef.current = null;
    setContentRect(null);
    setCurrentFrame(nextProject.watermark.anchor?.frame ?? 0);
    setCurrentTime(nextProject.watermark.anchor?.timestampSeconds ?? 0);
    setLabel(nextProject.watermark.label ?? "Learna AI");
    setRemoval(normalizeRemoval(nextProject.removal));
    const defaultOutputName = `${nextProject.source.fileName.replace(/\.[^.]+$/, "")}_watermark_removed_best.mp4`;
    setOutputName(window.localStorage.getItem(`watermark-studio:output-name:${nextProject.id}`) ?? defaultOutputName);
    setBestQualitySamples([]);
    setSelectedBestQualitySample(null);
    focusPreviewTokenRef.current += 1;
    setFocusPreview(null);
    setInspectionMode(false);
    setSelectionMode(false);
    setPlaying(false);
    setVideoLoadError(null);
    navigate("review", nextProject.id);
  };

  const chooseAndOpenVideo = async () => {
    setError(null);
    try {
      const selectedPath = await chooseVideoPath();
      if (!selectedPath) return;
      setLoadingTask("opening");
      setMessage("Opening video and reading metadata…");
      const nextProject = await openVideo(selectedPath);
      setProjects((current) => [
        ...current.filter((item) => item.id !== nextProject.id),
        nextProject,
      ]);
      activateProject(nextProject);
      setMessage(
        "Video loaded. Run Auto-find & calibrate to create a verified Calibration V6.",
      );
    } catch (openError) {
      setError(getErrorMessage(openError));
      setMessage("Could not open video");
    } finally {
      setLoadingTask(null);
    }
  };

  const importVideos = async () => {
    setError(null);
    let paths: string[];
    try {
      paths = await chooseVideoPaths();
    } catch (dialogError) {
      setError(getErrorMessage(dialogError));
      return;
    }
    if (paths.length === 0) return;
    setLoadingTask("opening");
    const imported: WatermarkProject[] = [];
    try {
      for (let index = 0; index < paths.length; index += 1) {
        setMessage(`Importing ${index + 1}/${paths.length}…`);
        imported.push(await openVideo(paths[index]));
      }
      setProjects(await listProjects());
      if (imported[0]) activateProject(imported[0]);
      setMessage(
        `Imported ${imported.length} video(s). Review them before queueing.`,
      );
    } catch (importError) {
      setError(getErrorMessage(importError));
    } finally {
      setLoadingTask(null);
    }
  };

  const openLibraryProject = async (projectId: string) => {
    setLoadingTask("opening");
    setError(null);
    try {
      activateProject(await getProject(projectId));
    } catch (openError) {
      setError(getErrorMessage(openError));
    } finally {
      setLoadingTask(null);
    }
  };

  const removeLibraryProject = async (item: WatermarkProject) => {
    if (project?.id === item.id && isBusy) {
      setError(language === "vi" ? "Không thể xóa project khi tác vụ đang chạy." : "This project cannot be removed while an operation is running.");
      return;
    }
    try {
      const currentJobs = await listJobs();
      const hasActiveJob = currentJobs.some((job) =>
        job.projectId === item.id &&
        ["QUEUED", "PREPARING", "INFERENCING", "ENCODING", "VERIFYING"].includes(job.status),
      );
      if (hasActiveJob) {
        setError(language === "vi" ? "Project đang có job trong hàng đợi hoặc đang render. Hãy hủy job trước." : "This project has a queued or running job. Cancel it before removing the project.");
        return;
      }
    } catch (jobsError) {
      setError(getErrorMessage(jobsError));
      return;
    }
    if (
      !window.confirm(
        `Remove ${item.source.fileName} from the library? Source and output files will be kept.`,
      )
    )
      return;
    setError(null);
    try {
      await removeProject(item.id);
      setProjects((current) =>
        current.filter((projectItem) => projectItem.id !== item.id),
      );
      if (project?.id === item.id) {
        setProject(null);
        window.localStorage.removeItem("watermark-studio:last-project-id");
        navigate("projects");
      }
      setMessage(language === "vi" ? "Đã xóa project khỏi thư viện (source/output được giữ nguyên)." : "Project removed from the library; source/output files were kept.");
    } catch (removeError) {
      setError(getErrorMessage(removeError));
    }
  };

  const pointFromPointer = (
    event: ReactPointerEvent<HTMLDivElement>,
  ): Point | null => {
    const surface = selectionSurfaceRef.current;
    if (!surface) return null;
    const bounds = surface.getBoundingClientRect();
    return {
      x: clamp(event.clientX - bounds.left, 0, bounds.width),
      y: clamp(event.clientY - bounds.top, 0, bounds.height),
    };
  };

  const sourceBoxFromPoints = (
    start: Point,
    end: Point,
  ): BoundingBox | null => {
    if (!project || !contentRect) return null;
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    return {
      x: (left / contentRect.width) * project.video.width,
      y: (top / contentRect.height) * project.video.height,
      width:
        (Math.abs(end.x - start.x) / contentRect.width) * project.video.width,
      height:
        (Math.abs(end.y - start.y) / contentRect.height) * project.video.height,
    };
  };

  const onSelectionPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!selectionMode || isBusy) return;
    const point = pointFromPointer(event);
    if (!point) return;
    // A moving video can clear a freshly drawn ROI on the next timeupdate.
    // Freeze the frame before starting a drag so the selected coordinates stay
    // tied to the image the user is actually inspecting.
    videoRef.current?.pause();
    setPlaying(false);
    selectionStartRef.current = point;
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelection(null);
  };

  const onSelectionPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = selectionStartRef.current;
    const point = pointFromPointer(event);
    if (!start || !point) return;
    setSelection(sourceBoxFromPoints(start, point));
  };

  const onSelectionPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const start = selectionStartRef.current;
    const point = pointFromPointer(event);
    selectionStartRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    if (!start || !point) return;
    const nextSelection = sourceBoxFromPoints(start, point);
    if (
      nextSelection &&
      nextSelection.width >= 8 &&
      nextSelection.height >= 8
    ) {
      setSelection(nextSelection);
      selectionFrameRef.current = currentFrame;
      setMessage(
        "Watermark selected. Review source coordinates, then save the anchor.",
      );
    } else {
      setSelection(null);
      setMessage(
        "Selection is too small. Minimum size is 8 × 8 source pixels.",
      );
    }
  };

  const saveAnchor = async () => {
    const bboxToSave = project?.tracking ? selection : displaySelection;
    if (!project || !bboxToSave || isBusy) return;
    setError(null);
    setLoadingTask("saving");
    setMessage(
      project.tracking
        ? "Saving manual correction…"
        : "Saving anchor and extracting templates…",
    );
    try {
      const updatedProject = project.tracking
        ? await saveManualAnchor(
            project.id,
            currentFrame,
            currentTime,
            bboxToSave,
          )
        : await saveWatermarkAnchor(
            project.id,
            currentFrame,
            currentTime,
            bboxToSave,
            label,
          );
      setProject(updatedProject);
      setSelection(
        project.tracking
          ? null
          : (updatedProject.watermark.anchor?.bbox ?? null),
      );
      setSelectionMode(false);
      setMessage(
        project.tracking
          ? "Manual frame saved and locked. Click Re-track section."
          : "Anchor saved. Templates and mask are ready.",
      );
    } catch (saveError) {
      setError(getErrorMessage(saveError));
      setMessage("Could not save anchor");
    } finally {
      setLoadingTask(null);
    }
  };

  const runRetrack = async () => {
    if (!project || !project.tracking || isBusy) return;
    setError(null);
    setLoadingTask("tracking");
    setProgress(null);
    setMessage(`Re-tracking around frame ${currentFrame}…`);
    try {
      const updatedProject = await retrackTrack(project.id, currentFrame);
      setProject(updatedProject);
      setSelection(null);
      setMessage(
        `Section re-tracked. ${updatedProject.tracking?.problemRanges.length ?? 0} problem range(s) remain.`,
      );
    } catch (trackingError) {
      setError(getErrorMessage(trackingError));
      setMessage("Re-track failed");
    } finally {
      setLoadingTask(null);
      setProgress(null);
    }
  };

  const cancelBusy = () => {
    if (loadingTask === "tracking") void cancelTracking();
    if (loadingTask === "rendering" || loadingTask === "calibrating") void cancelRender();
  };

  const openCalibrationReviewRange = (range: {
    startFrame: number;
    endFrame: number;
    suggestedFrames?: number[];
  }) => {
    const frame = range.suggestedFrames?.[0]
      ?? Math.round((range.startFrame + range.endFrame) / 2);
    videoRef.current?.pause();
    setPlaying(false);
    setVideoFrame(frame);
    setInspectionMode(false);
    setFocusPreview(null);
    setSelection(null);
    selectionFrameRef.current = null;
    setRoiFallbackArmed(true);
    setSelectionMode(true);
    setMessage(
      language === "vi"
        ? `Frame review ROI ${frame} (${range.startFrame}–${range.endFrame}). Vẽ ROI rộng rồi bấm thêm evidence; Next problem sẽ chuyển sang cụm tiếp theo.`
        : `ROI review frame ${frame} (${range.startFrame}–${range.endFrame}). Draw a broad ROI and add evidence; Next problem will advance to the next cluster.`,
    );
  };

  const nextProblem = () => {
    if (!project) return;
    if (workspaceMode === "legacy" && project.tracking?.problemRanges.length) {
      const next =
        project.tracking.problemRanges.find(
          (range) => range.worstFrame > currentFrame,
        ) ?? project.tracking.problemRanges[0];
      if (next) {
        setVideoFrame(next.worstFrame);
        setMessage(
          `Review frame ${next.worstFrame} (${next.startFrame}–${next.endFrame}).`,
        );
      }
      return;
    }

    // Best-quality calibration ranges are consumed by ROI evidence.  A range
    // is considered reviewed as soon as one evidence frame falls inside it;
    // the next click then advances to the next unreviewed motion cluster.
    const next = pendingCalibrationReviewRanges.find(
      (range) => range.startFrame > currentFrame,
    ) ?? pendingCalibrationReviewRanges[0];
    if (next) {
      openCalibrationReviewRange(next);
    } else {
      setMessage(language === "vi"
        ? "Đã duyệt hết các cụm ROI hiện có. Hãy chạy lại calibration để cập nhật quality gate."
        : "All current ROI clusters have been reviewed. Run calibration again to update the quality gate.");
    }
  };

  const interpolateCurrentRange = async () => {
    if (!project?.tracking || isBusy) return;
    const range =
      project.tracking.problemRanges.find(
        (item) =>
          currentFrame >= item.startFrame && currentFrame <= item.endFrame,
      ) ?? project.tracking.problemRanges[0];
    if (!range) return;
    setLoadingTask("saving");
    setError(null);
    setMessage(
      `Interpolating frames ${range.startFrame}–${range.endFrame} between locked anchors…`,
    );
    try {
      setProject(
        await interpolateTrackingRange(
          project.id,
          range.startFrame,
          range.endFrame,
        ),
      );
      setSelection(null);
      setMessage(
        "Range interpolated provisionally; review each affected range before rendering.",
      );
    } catch (interpolationError) {
      setError(getErrorMessage(interpolationError));
    } finally {
      setLoadingTask(null);
      setProgress(null);
    }
  };

  const acceptCurrentFrame = async () => {
    if (!project || !currentTracking || isBusy) return;
    setLoadingTask("saving");
    setError(null);
    try {
      setProject(await acceptTrackingFrame(project.id, currentFrame));
      setMessage(`Frame ${currentFrame} accepted and locked.`);
    } catch (acceptError) {
      setError(getErrorMessage(acceptError));
    } finally {
      setLoadingTask(null);
      setProgress(null);
    }
  };

  const markCurrentRangeOccluded = async () => {
    if (!project?.tracking || isBusy) return;
    const range = project.tracking.problemRanges.find(
      (item) =>
        currentFrame >= item.startFrame && currentFrame <= item.endFrame,
    );
    if (!range) return;
    setLoadingTask("saving");
    setError(null);
    try {
      setProject(
        await markOccludedRange(project.id, range.startFrame, range.endFrame),
      );
      setSelection(null);
      setMessage(
        `Frames ${range.startFrame}–${range.endFrame} marked occluded; rendering is a no-op for this range.`,
      );
    } catch (occludedError) {
      setError(getErrorMessage(occludedError));
    } finally {
      setLoadingTask(null);
      setProgress(null);
    }
  };

  const chooseReplacement = async () => {
    try {
      const path = await chooseReplacementPath();
      if (path)
        setRemoval((current) => ({
          ...current,
          mode: "REPLACEMENT",
          replacementPath: path,
        }));
    } catch (dialogError) {
      setError(getErrorMessage(dialogError));
    }
  };

  const chooseBestReplacement = async () => {
    try {
      const path = await chooseReplacementPath();
      if (path)
        setBestReplacement((current) => ({
          ...(current ?? defaultBestReplacement),
          kind: "image",
          imagePath: path,
        }));
    } catch (dialogError) {
      setError(getErrorMessage(dialogError));
    }
  };

  const render = async () => {
    if (!project || isBusy) return;
    setError(null);
    setLoadingTask("rendering");
    operationTaskRef.current = "rendering";
    setOperationDialog({ task: "rendering", status: "running", title: "Đang render Best-quality", detail: "Chuẩn bị profile, chạy ProPainter FP32 và kiểm tra QA…" });
    setMessage("Rendering full-resolution output…");
    try {
      const saved = await saveRemovalConfig(project.id, removal);
      setProject(saved);
      const result = await renderVideo(saved.id, removal);
      setMessage(`Render complete: ${result.outputPath}`);
      setOperationDialog({ task: "rendering", status: "success", title: "Render hoàn tất", detail: result.outputPath });
    } catch (renderError) {
      setError(getErrorMessage(renderError));
      setMessage("Render failed");
      setOperationDialog({ task: "rendering", status: "error", title: "Render không thành công", detail: getErrorMessage(renderError) });
    } finally {
      setLoadingTask(null);
      operationTaskRef.current = null;
      setProgress(null);
    }
  };

  const queueBestQuality = async () => {
    if (!project || isBusy || project.calibration?.quality.status !== "READY" || !project.calibration.scanRange) return;
    try {
      const job = await enqueueBestQualityJob(project.id, outputRoot || null, outputName || null, bestReplacement);
      setJobs((current) => [
        ...current.filter((item) => item.id !== job.id),
        job,
      ]);
      navigate("queue");
      setMessage(
        `Queued ${project.source.fileName} for sequential Best-quality rendering.`,
      );
    } catch (queueError) {
      setError(getErrorMessage(queueError));
    }
  };

  const runAdaptiveCalibration = async () => {
    if (!project || isBusy) return;
    const roi = roiFallbackArmed && selection && currentFrame >= scanRange.startFrame && currentFrame <= scanRange.endFrame && selection.width >= 32 && selection.height >= 16
      ? { ...selection, frame: currentFrame }
      : null;
    setError(null);
    setLoadingTask("calibrating");
    operationTaskRef.current = "calibrating";
    setOperationDialog({ task: "calibrating", status: "running", title: "Đang calibration Learna AI", detail: `Quét frame ${scanRange.startFrame}–${scanRange.endFrame}, fit quỹ đạo và tạo consensus mask…` });
    setProgress(null);
    setMessage(roi ? `Đang quét Learna AI trong ROI ở phạm vi ${scanRange.startFrame}–${scanRange.endFrame}…` : `Đang quét frame ${scanRange.startFrame}–${scanRange.endFrame} và khớp quỹ đạo Learna AI…`);
    try {
      const allEvidence = [
        ...roiEvidence,
        ...(roi ? [roi as RoiHint & { frame: number }] : []),
      ].filter((item, index, items) => items.findIndex((other) => other.frame === item.frame) === index);
      const evidence = allEvidence.filter((item) => item.frame >= scanRange.startFrame && item.frame <= scanRange.endFrame);
      if (evidence.length !== allEvidence.length) {
        setMessage(language === "vi" ? "Đã bỏ qua ROI evidence nằm ngoài phạm vi quét đã chọn." : "ROI evidence outside the selected scan range was ignored.");
      }
      const updatedProject = await autoCalibrateBestQuality(project.id, roi, null, evidence, scanRange);
      setProject(updatedProject);
      setProjects((current) => current.map((item) => item.id === updatedProject.id ? updatedProject : item));
      const status = updatedProject.calibration?.quality.status;
      if (status === "READY") {
        setRoiFallbackArmed(false);
        setSelection(null);
        setRoiEvidence([]);
        window.localStorage.removeItem(roiEvidenceStorageKey(updatedProject.id));
        setMessage(`CalibrationProfileV6 đã vượt quality gate trong phạm vi ${scanRange.startFrame}–${scanRange.endFrame}; có thể đưa job vào hàng đợi.`);
        setOperationDialog({ task: "calibrating", status: "success", title: "Calibration V6 đã đạt", detail: "Profile READY. Bạn có thể đưa video vào hàng đợi render." });
      } else {
        setMessage("Chưa tìm được quỹ đạo đủ tin cậy. Hãy khoanh ROI tương đối rồi chạy lại; Render vẫn bị khóa.");
        const reasons = updatedProject.calibration?.trajectoryGate?.failureReasons?.join(", ") || "TRAJECTORY_UNDERCONSTRAINED";
        const gate = updatedProject.calibration?.trajectoryGate;
        const measured = updatedProject.calibration?.quality.reliableFrames ?? 0;
        const evidenceFrames = new Set<number>([
          ...(updatedProject.calibration?.roiEvidenceFrames ?? []),
          ...evidence.map((item) => item.frame),
        ]);
        const actionableRanges = (gate?.reviewRanges ?? []).filter(
          (range) => !Array.from(evidenceFrames).some(
            (frame) => frame >= range.startFrame && frame <= range.endFrame,
          ),
        );
        const reviewRanges = formatCalibrationReviewRanges(actionableRanges);
        const roiCount = gate?.roiEvidenceFrames ?? roiEvidence.length;
        const refinementRequired = gate?.failureReasons?.includes("TRAJECTORY_REFINEMENT_REQUIRED")
          || Boolean(
            roiCount >= ROI_SATURATION_MIN_EVIDENCE
            && (gate?.confirmedCoverage ?? 0) >= ROI_SATURATION_MIN_CONFIRMED_COVERAGE
            && (gate?.measuredCoverage ?? 0) >= ROI_SATURATION_MIN_PATH_COVERAGE
            && (
              (gate?.maxObservationGap ?? gate?.maxInterpolationGap ?? 0) > 18
              || ((gate?.residualP95 ?? null) != null && (gate?.residualP95 ?? 0) > 3 * (project.video.width / 1080))
            ),
          );
        const guidance = refinementRequired
          ? `Đã đủ evidence đại diện (${roiCount} frame); không cần khoanh thêm ROI. Các cụm yếu còn lại được giữ trong diagnostics, còn profile cần refine quỹ đạo tự động trước khi được phép render.`
          : reviewRanges
          ? `Danh sách đã gom thành ${actionableRanges.length} cụm đại diện chưa có evidence. Mỗi cụm chỉ cần thêm 1 ROI ở frame ưu tiên (không cần chọn tất cả frame); chỉ thêm khi watermark còn xuất hiện, phần sau active interval không cần chọn.`
          : roiCount >= 12
            ? "Bạn đã cung cấp đủ ROI evidence đại diện; không cần khoanh thêm. Quỹ đạo còn cần bước refine tự động trước khi được phép render."
            : "Profile này chưa chứa reviewRanges (lần quét cũ hoặc chưa gửi ROI evidence). Hãy đóng dialog, thêm ROI ở vài đoạn watermark còn nhìn thấy rồi chạy lại để hệ thống tính gợi ý chính xác.";
        const hardMeasured = gate?.hardMeasuredFrames ?? 0;
        const confirmedCoverage = gate?.confirmedCoverage;
        const measuredCoverage = gate?.measuredCoverage;
        const coverageText = confirmedCoverage != null
          ? `${Math.round(confirmedCoverage * 100)}% evidence đã xác nhận`
          : gate?.directCoverage != null
            ? `${Math.round(gate.directCoverage * 100)}% hard-direct`
            : "—";
        setOperationDialog({
          task: "calibrating",
          status: "error",
          title: "Calibration cần Review",
          detail: `Quality gate chưa đạt trong phạm vi ${scanRange.startFrame}–${scanRange.endFrame}: ${reasons}. Candidate path: ${measured} frame; hard-direct: ${hardMeasured} frame; ${coverageText}; sampled path coverage: ${measuredCoverage != null ? `${Math.round(measuredCoverage * 100)}%` : "—"}; residual p95: ${gate?.residualP95 != null ? gate.residualP95.toFixed(2) : "—"} px. ${guidance} Ngoài phạm vi sẽ không được đưa vào review ROI.`,
          reviewRanges: actionableRanges.map((range) => ({
            startFrame: range.startFrame,
            endFrame: range.endFrame,
            suggestedFrames: range.suggestedFrames?.length
              ? range.suggestedFrames
              : [Math.round((range.startFrame + range.endFrame) / 2)],
            reason: range.reason,
          })),
        });
      }
    } catch (calibrationError) {
      setError(getErrorMessage(calibrationError));
      setMessage("Không thể hoàn tất adaptive calibration");
      setOperationDialog({ task: "calibrating", status: "error", title: "Calibration lỗi", detail: getErrorMessage(calibrationError) });
    } finally {
      setLoadingTask(null);
      operationTaskRef.current = null;
      setProgress(null);
    }
  };

  const updateScanRange = (field: "startFrame" | "endFrame", rawValue: string) => {
    if (!project) return;
    const lastFrame = Math.max(0, project.video.frameCount - 1);
    const parsed = Number(rawValue);
    if (!Number.isInteger(parsed)) return;
    const next = {
      startFrame: field === "startFrame" ? parsed : scanRange.startFrame,
      endFrame: field === "endFrame" ? parsed : scanRange.endFrame,
    };
    if (next.startFrame < 0 || next.startFrame > lastFrame || next.endFrame < 0 || next.endFrame > lastFrame) {
      return;
    }
    if (field === "startFrame" && next.startFrame > next.endFrame) next.endFrame = next.startFrame;
    if (field === "endFrame" && next.endFrame < next.startFrame) next.startFrame = next.endFrame;
    if (next.startFrame > lastFrame || next.endFrame > lastFrame) return;
    setScanRange(next);
    window.localStorage.setItem(scanRangeStorageKey(project.id), JSON.stringify(next));
  };

  const resetScanRange = () => {
    if (!project) return;
    const next = { startFrame: 0, endFrame: Math.max(0, project.video.frameCount - 1) };
    setScanRange(next);
    window.localStorage.setItem(scanRangeStorageKey(project.id), JSON.stringify(next));
  };

  const chooseAndSaveOutputRoot = async () => {
    try {
      const selected = await chooseOutputDirectory();
      if (selected) {
        setOutputRoot(selected);
        window.localStorage.setItem("watermark-studio:output-root", selected);
        setMessage(`Output folder set: ${selected}`);
      }
    } catch (dialogError) {
      setError(getErrorMessage(dialogError));
    }
  };

  const changeLanguage = (next: "vi" | "en") => {
    setLanguage(next);
    window.localStorage.setItem("watermark-studio:language", next);
  };

  const findBestQualitySamples = async (findAlternatives = false) => {
    if (!project || isBusy) return;
    const scanRound = findAlternatives ? bestSampleScanRound + 1 : 0;
    const excludeFrames = findAlternatives
      ? [
          ...new Set([
            ...rejectedBestSampleFrames,
            ...bestQualitySamples.map((sample) => sample.frame),
          ]),
        ]
      : [];
    const excludeSceneSignatures = findAlternatives
      ? [...new Set([...rejectedSceneSignatures, ...bestQualitySamples.map((sample) => sample.sceneSignature)])]
      : [];
    const roiHint = roiFallbackArmed && selection && selection.width >= 8 && selection.height >= 8
      ? selection
      : null;
    setError(null);
    setLoadingTask("sampling");
    operationTaskRef.current = "sampling";
    setOperationDialog({ task: "sampling", status: "running", title: "Đang tìm sample", detail: "Quét candidate, chấm điểm glyph và kiểm tra ổn định theo frame lân cận…" });
    setProgress(null);
    focusPreviewTokenRef.current += 1;
    setFocusPreview(null);
    setInspectionMode(false);
    lockedBestSampleFrameRef.current = null;
    setSelection(null);
    setMessage(
      findAlternatives
        ? `${t("scanningAllPhases")} (${language === "vi" ? "lượt thay thế" : "alternative pass"} ${scanRound})`
        : t("scanningAllPhases"),
    );
    try {
      const samples = await suggestBestQualitySamples(project.id, {
        scanRound,
        excludeFrames,
        excludeSceneSignatures,
        roi: roiHint,
        anchorFrame: roiHint ? currentFrame : undefined,
        scanRange,
      });
      setBestQualitySamples(samples);
      setSelectedBestQualitySample(null);
      setBestSampleScanRound(scanRound);
      setRejectedBestSampleFrames(excludeFrames);
      setRejectedSceneSignatures(excludeSceneSignatures);
      setMessage(
        samples.length > 0
          ? `Found ${samples.length} ${findAlternatives ? "alternative " : ""}samples. Click one to inspect and confirm it${roiHint ? " (ROI fallback)" : ""}.`
          : t("noValidSample"),
      );
      setOperationDialog({ task: "sampling", status: "success", title: samples.length ? "Đã tìm thấy sample" : "Không có sample đạt gate", detail: samples.length ? `Có ${samples.length} candidate để inspect.` : "Có thể dùng ROI evidence hoặc chạy phase khác; chưa được render khi chưa có profile READY." });
    } catch (sampleError) {
      setError(getErrorMessage(sampleError));
      setMessage("Could not find a usable Best-quality sample");
      setOperationDialog({ task: "sampling", status: "error", title: "Tìm sample lỗi", detail: getErrorMessage(sampleError) });
    } finally {
      setLoadingTask(null);
      operationTaskRef.current = null;
      setProgress(null);
    }
  };

  const beginRoiFallback = () => {
    videoRef.current?.pause();
    setPlaying(false);
    setInspectionMode(false);
    setFocusPreview(null);
    setSelection(null);
    selectionFrameRef.current = null;
    setRoiFallbackArmed(true);
    setSelectionMode(true);
    setMessage(
      roiEvidence.length > 0
        ? `Đã giữ ${roiEvidence.length} evidence frame. Chọn thêm một frame ở đoạn chuyển động khác rồi kéo ROI.`
        : "Chọn một frame watermark nhìn rõ, sau đó kéo một ROI tương đối bao quanh toàn bộ chữ Learna AI.",
    );
  };

  const addRoiEvidence = () => {
    if (!selection || !project || selectionFrameRef.current !== currentFrame || selection.width < 32 || selection.height < 16) {
      setMessage("ROI đã cũ hoặc chưa đủ rộng. Hãy dừng video và vẽ lại trên đúng frame đang xem.");
      return;
    }
    setRoiEvidence((current) => {
      const next = [
        ...current.filter((item) => item.frame !== currentFrame),
        { ...selection, frame: currentFrame },
      ];
      window.localStorage.setItem(roiEvidenceStorageKey(project.id), JSON.stringify(next));
      return next;
    });
    setMessage(`Đã thêm ROI evidence tại frame ${currentFrame}. Có thể thêm frame ở đoạn chuyển động khác.`);
  };

  const inspectBestQualitySample = (sample: BestQualitySample) => {
    setRoiFallbackArmed(false);
    setSelectedBestQualitySample(sample);
    lockedBestSampleFrameRef.current = sample.frame;
    setVideoFrame(sample.frame, true);
    setSelection(sample.bbox);
    setSelectionMode(false);
    void openFocusPreview(sample.frame, sample.bbox);
    setMessage(
      `Inspect frame ${sample.frame}. If the box covers the whole watermark, confirm this sample.`,
    );
  };

  const loadMaskEditor = useCallback((projectId: string, source: string) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const loadToken = maskLoadTokenRef.current + 1;
    maskLoadTokenRef.current = loadToken;
    setMaskEditorReady(false);
    canvas.width = 245;
    canvas.height = 75;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    void readProjectAssetBytes(projectId, source).then((bytes) => {
      if (maskLoadTokenRef.current !== loadToken) return;
      const image = new Image();
      image.onload = () => {
        if (maskLoadTokenRef.current !== loadToken) return;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d");
        if (!context) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0);
        setMaskUndo([]);
        setMaskRedo([]);
        setMaskEditorReady(true);
      };
      image.onerror = () => {
        if (maskLoadTokenRef.current !== loadToken) return;
        setMaskEditorReady(false);
        setError("Không thể tải mask vào Mask Editor.");
      };
      image.src = bytesToDataUrl(bytes);
    }).catch((loadError) => {
      if (maskLoadTokenRef.current !== loadToken) return;
      setMaskEditorReady(false);
      setError(getErrorMessage(loadError));
    });
  }, []);

  useEffect(() => {
    if (selectedBestQualitySample && project) {
      loadMaskEditor(project.id, selectedBestQualitySample.editorMaskPath);
    }
  }, [selectedBestQualitySample, project, loadMaskEditor]);

  const drawMaskBrush = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!maskDrawingRef.current) return;
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = brushMode === "add" ? "white" : "black";
    context.beginPath();
    context.arc(((event.clientX - rect.left) / rect.width) * canvas.width, ((event.clientY - rect.top) / rect.height) * canvas.height, brushSize, 0, Math.PI * 2);
    context.fill();
  };

  const maskSnapshot = (canvas: HTMLCanvasElement): string | null => {
    try {
      return canvas.toDataURL("image/png");
    } catch (snapshotError) {
      setError(`Mask Editor chưa sẵn sàng: ${snapshotError instanceof Error ? snapshotError.message : String(snapshotError)}`);
      return null;
    }
  };

  const startMaskBrush = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const snapshot = maskSnapshot(canvas);
    if (snapshot) setMaskUndo((current) => [...current.slice(-19), snapshot]);
    setMaskRedo([]); maskDrawingRef.current = true;
    canvas.setPointerCapture(event.pointerId); drawMaskBrush(event);
  };

  const restoreMaskSnapshot = (snapshot: string) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const image = new Image();
    image.onload = () => canvas.getContext("2d")?.drawImage(image, 0, 0);
    image.src = snapshot;
  };

  const undoMask = () => {
    const canvas = maskCanvasRef.current;
    const snapshot = maskUndo[maskUndo.length - 1];
    if (!canvas || !snapshot) return;
    const currentSnapshot = maskSnapshot(canvas);
    if (currentSnapshot) setMaskRedo((current) => [...current, currentSnapshot]);
    setMaskUndo((current) => current.slice(0, -1)); restoreMaskSnapshot(snapshot);
  };

  const redoMask = () => {
    const canvas = maskCanvasRef.current;
    const snapshot = maskRedo[maskRedo.length - 1];
    if (!canvas || !snapshot) return;
    const currentSnapshot = maskSnapshot(canvas);
    if (currentSnapshot) setMaskUndo((current) => [...current, currentSnapshot]);
    setMaskRedo((current) => current.slice(0, -1)); restoreMaskSnapshot(snapshot);
  };

  const openFocusPreview = async (frame: number, bbox: BoundingBox) => {
    if (!project) return;
    const previewToken = focusPreviewTokenRef.current + 1;
    focusPreviewTokenRef.current = previewToken;
    try {
      const preview = await extractFocusPreview(project.id, frame, bbox);
      if (focusPreviewTokenRef.current !== previewToken) return;
      setFocusPreview(preview);
      setInspectionZoom(1);
      setInspectionMode(true);
    } catch (previewError) {
      if (focusPreviewTokenRef.current !== previewToken) return;
      setError(getErrorMessage(previewError));
    }
  };

  const saveBestQualitySample = async () => {
    if (!project || !selectedBestQualitySample || isBusy) return;
    setError(null);
    setLoadingTask("saving");
    setMessage("Saving the confirmed Best-quality template and mask…");
    try {
      // This intentionally refreshes the primary template even when legacy
      // tracking data exists. Best-quality rendering does not use that queue.
      const canvas = maskCanvasRef.current;
      if (!canvas || !maskEditorReady) throw new Error("Mask Editor is not ready.");
      const blob = await new Promise<Blob>((resolve, reject) => {
        try {
          canvas.toBlob((value) => value ? resolve(value) : reject(new Error("Unable to encode edited mask.")), "image/png");
        } catch (blobError) {
          reject(new Error(`Mask canvas is not exportable: ${blobError instanceof Error ? blobError.message : String(blobError)}`));
        }
      });
      const editedMaskPath = await saveCalibrationMaskEdit(project.id, Array.from(new Uint8Array(await blob.arrayBuffer())));
      // The sample editor is only an evidence/descriptor step.  Always finish
      // through the same V6 adaptive calibration service used by the main
      // Best-quality button so a legacy V4 profile can never reach Queue.
      const updatedProject = await autoCalibrateBestQuality(
        project.id,
        { ...selectedBestQualitySample.bbox, frame: selectedBestQualitySample.frame },
        editedMaskPath,
        [],
        scanRange,
      );
      setProject(updatedProject);
      setSelection(updatedProject.watermark.anchor?.bbox ?? selection);
      setBestQualitySamples([]);
      setSelectedBestQualitySample(null);
      focusPreviewTokenRef.current += 1;
      setFocusPreview(null);
      setInspectionMode(false);
      setMessage(
        updatedProject.calibration?.quality.status === "READY"
          ? "CalibrationProfileV6 đã vượt quality gate; có thể đưa job vào hàng đợi."
          : "Mask đã lưu. Calibration V6 chưa vượt quality gate; hãy bổ sung ROI hoặc quét lại trước khi Queue.",
      );
    } catch (saveError) {
      setError(getErrorMessage(saveError));
      setMessage("Could not save the Best-quality template");
    } finally {
      setLoadingTask(null);
    }
  };

  const togglePlayback = () => {
    if (!videoRef.current || !project) return;
    if (playing) videoRef.current.pause();
    else void videoRef.current.play().catch((playError) => setError(getErrorMessage(playError)));
  };

  const clickTimeline = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!project) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    setVideoFrame(ratio * Math.max(0, project.video.frameCount - 1));
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <img src="/quanph-logo.png" alt="QuanPH" />
          </div>
          <div>
            <strong>Watermark Studio</strong>
            <span>by QuanPH · Desktop</span>
          </div>
        </div>
        <nav className="nav-stack">
          <button
            className={`nav-item${route === "projects" ? " active" : ""}`}
            onClick={() => navigate("projects")}
          >
            <Icon>◫</Icon> {t("projects")}
          </button>
          <button
            className={`nav-item${route === "review" ? " active" : ""}`}
            onClick={() => navigate("review", project?.id)}
            disabled={!project}
          >
            <Icon>⌕</Icon> {t("review")}
          </button>
          <button
            className={`nav-item${route === "queue" ? " active" : ""}`}
            onClick={() => navigate("queue")}
          >
            <Icon>≡</Icon> {t("queue")}
          </button>
          <button
            className={`nav-item${route === "history" ? " active" : ""}`}
            onClick={() => navigate("history")}
          >
            <Icon>☷</Icon> {t("history")}
          </button>
          <button
            className={`nav-item${route === "settings" ? " active" : ""}`}
            onClick={() => navigate("settings")}
          >
            <Icon>⚙</Icon> {t("settings")}
          </button>
        </nav>
        <div className="sidebar-card">
          <span className="eyebrow">{t("currentFile")}</span>
          <strong title={project?.source.fileName ?? t("noVideo")}>
            {project?.source.fileName ?? t("noVideo")}
          </strong>
          <div className="file-meta">
            <span>
              {project ? `${project.video.durationSeconds.toFixed(2)}s` : "—"}
            </span>
            <span>
              {project
                ? `${project.video.width} × ${project.video.height}`
                : "—"}
            </span>
            <span>{project ? `${project.video.fps.toFixed(3)} fps` : "—"}</span>
          </div>
        </div>
        <div className="sidebar-bottom">
          <div className="status-dot" />
          <span>
            {message}
            <small className="copyright">© QuanPH</small>
          </span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">{route.toUpperCase()}</span>
            <h1>
              {route === "projects" ? t("videoLibrary") : route === "review" ? t("reviewCalibration") : route === "queue" ? t("renderQueue") : route === "history" ? t("jobHistory") : t("settings")}
            </h1>
          </div>
          <div className="top-actions">
            {route === "projects" && (
              <button
                className="button secondary"
                onClick={() => void importVideos()}
                disabled={isBusy}
              >
                {loadingTask === "opening" ? "Importing…" : t("importVideos")}
              </button>
            )}
            {route === "review" && workspaceMode === "best" && (
              <>
                <button
                  className="button secondary"
                  onClick={() => void runAdaptiveCalibration()}
                  disabled={!project || isBusy}
                >
                    {loadingTask === "calibrating" ? t("calibrating") : t("autoCalibrate")}
                </button>
                <button
                  className="button secondary"
                  onClick={() => void queueBestQuality()}
                  disabled={
                    project?.calibration?.quality.status !== "READY" || !project?.calibration?.scanRange || isBusy
                  }
                >
                  {t("queueRender")}
                </button>
              </>
            )}
            {route === "review" && workspaceMode === "legacy" && (
              <button
                className="button secondary"
                onClick={() => setWorkspaceMode("best")}
                disabled={isBusy}
              >
                {language === "vi" ? "Quay lại Best-quality" : "Back to Best-quality"}
              </button>
            )}
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}
        {route === "projects" && (
          <section className="projects-page panel">
            <div className="timeline-header">
              <div>
                <span className="eyebrow">{t("projects")}</span>
                <h2>{t("importAndReview")}</h2>
              </div>
              <button className="button primary" onClick={() => void importVideos()} disabled={isBusy}>{t("importVideos")}</button>
            </div>
            <div className="project-list">
              {projects.length === 0 ? (
                <button className="dropzone" onClick={() => void importVideos()}>
                  <span className="drop-icon">＋</span><strong>{t("importBatch")}</strong><small>MP4, MOV, MKV, WEBM, M4V</small>
                </button>
              ) : projects.map((item) => (
                <article className="project-row" key={item.id}>
                  <div><strong>{item.source.fileName}</strong><small>{item.video.durationSeconds.toFixed(2)}s · {item.video.width}×{item.video.height} · {item.calibration?.quality.status ?? "AWAITING_REVIEW"}</small></div>
                  <div className="job-actions">
                    <button className="button secondary" onClick={() => void openLibraryProject(item.id)}>{t("review")}</button>
                    <button className="button secondary" onClick={() => void removeLibraryProject(item)}>{t("remove")}</button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}
        {route === "settings" && (
          <section className="settings-panel panel">
            <div>
              <span className="eyebrow">SETTINGS</span>
              <h2>
                {language === "vi"
                  ? "Cài đặt ứng dụng"
                  : "Application settings"}
              </h2>
            </div>
            <div className="settings-tabs" role="tablist" aria-label={t("settings")}>
              {(["general", "processing", "updates", "advanced", "about"] as SettingsTab[]).map((tab) => (
                <button key={tab} role="tab" aria-selected={settingsTab === tab} className={settingsTab === tab ? "selected" : ""} onClick={() => setSettingsTab(tab)}>{t(tab)}</button>
              ))}
            </div>
            <div className="settings-content">
              {settingsTab === "general" && <>
                <div className="settings-section-heading"><div><span className="eyebrow">GENERAL</span><h3>{language === "vi" ? "Cách ứng dụng hoạt động" : "How the app behaves"}</h3></div><span className="settings-status">{language === "vi" ? "Đã lưu tự động" : "Auto-saved"}</span></div>
                <div className="settings-grid">
                  <label className="field settings-card">
                    <span>{t("language")}</span>
                    <select value={language} onChange={(event) => changeLanguage(event.target.value as "vi" | "en")}>
                      <option value="vi">Tiếng Việt</option><option value="en">English</option>
                    </select>
                    <small>{language === "vi" ? "Ngôn ngữ được giữ lại cho lần mở sau." : "Your language is saved for the next launch."}</small>
                  </label>
                  <div className="settings-card output-setting">
                    <label className="field"><span>{t("outputFolder")}</span><input value={outputRoot} placeholder="Default: source\\output" onChange={(event) => { setOutputRoot(event.target.value); window.localStorage.setItem("watermark-studio:output-root", event.target.value); }} /></label>
                    <button className="button secondary" onClick={() => void chooseAndSaveOutputRoot()}>{language === "vi" ? "Chọn thư mục" : "Choose folder"}</button>
                    <small>{language === "vi" ? "Có thể đổi riêng cho từng job ở trang Review." : "You can override this per job on Review."}</small>
                  </div>
                </div>
                <div className="settings-info"><span>✓</span><small>{language === "vi" ? "Best-quality luôn kiểm tra calibration và QA trước khi ghi nhận hoàn tất. Legacy chỉ dùng để xem trước." : "Best-quality validates calibration and QA before completion. Legacy is preview-only."}</small></div>
              </>}
              {settingsTab === "processing" && <>
                <div className="settings-section-heading"><div><span className="eyebrow">PROCESSING</span><h3>{language === "vi" ? "Phần cứng và chất lượng" : "Hardware and quality"}</h3></div><button className="button secondary" onClick={() => void refreshHardware()}>{t("refreshGpu")}</button></div>
                <div className="hardware-card">
                  <div className="hardware-main"><span className="hardware-icon">GPU</span><div><strong>{hardware?.gpuName ?? (language === "vi" ? "Đang phát hiện…" : "Detecting…")}</strong><small>{hardware?.cudaAvailable ? t("cudaAvailable") : t("cudaUnavailable")}</small></div><span className={`hardware-pill ${hardware?.supported ? "supported" : "warning"}`}>{hardware?.tier ?? "—"}</span></div>
                  <div className="hardware-stats"><div><span>VRAM</span><strong>{hardware ? `${(hardware.vramMb / 1024).toFixed(1)} GB` : "—"}</strong></div><div><span>{t("aiProfile")}</span><strong>{hardware?.supported ? `${hardware.width}×${hardware.height}` : t("notSupported")}</strong></div><div><span>{t("context")}</span><strong>{hardware?.supported ? `${hardware.context} frames` : "—"}</strong></div><div><span>Queue</span><strong>{t("gpuQueue")}</strong></div></div>
                </div>
                <small className="settings-note">{language === "vi" ? "GPU mạnh hơn sẽ tăng resolution/context. Mỗi GPU vẫn chỉ chạy một job ProPainter để giữ chất lượng và tránh OOM." : "A stronger GPU increases resolution/context. Each GPU still runs one ProPainter job to preserve quality and avoid OOM."}</small>
              </>}
              {settingsTab === "updates" && <>
                <div className="settings-section-heading"><div><span className="eyebrow">UPDATES</span><h3>{language === "vi" ? "Phiên bản và phát hành" : "Version and releases"}</h3></div><span className="settings-status">Windows x64</span></div>
                <div className="update-card"><div><strong>Watermark Studio by QuanPH</strong><small>{language === "vi" ? "Kiểm tra khi mở app và mỗi 24 giờ." : "Checks at startup and every 24 hours."}</small></div><button className="button secondary" onClick={() => void checkForUpdatesNow()} disabled={isBusy}>{t("checkNow")}</button></div>
                <small className="settings-note">{language === "vi" ? "Updater chỉ cài đặt khi queue GPU không còn stage đang chạy." : "The updater installs only when no GPU stage is active."}</small>
              </>}
              {settingsTab === "advanced" && <>
                <div className="settings-section-heading"><div><span className="eyebrow">ADVANCED</span><h3>{language === "vi" ? "Công cụ tương thích" : "Compatibility tools"}</h3></div><span className="settings-status warning-text">Preview only</span></div>
                <div className="legacy-notice"><strong>Legacy / Preview</strong><small>{language === "vi" ? "Các công cụ cũ chỉ dùng chẩn đoán và không thể tạo output FINAL." : "Legacy tools are diagnostic only and cannot create a FINAL output."}</small><button className="button secondary" onClick={() => { setWorkspaceMode("legacy"); navigate("review", project?.id); }} disabled={!project}>{language === "vi" ? "Mở Legacy / Preview" : "Open Legacy / Preview"}</button></div>
              </>}
              {settingsTab === "about" && <div className="about-card"><img src="/quanph-logo.png" alt="QuanPH" /><div><span className="eyebrow">ABOUT</span><strong>Watermark Studio by QuanPH</strong><small>© 2026 QuanPH · Windows desktop</small><small>{language === "vi" ? "Xử lý video cục bộ, giữ nguyên source." : "Local video processing; source files remain untouched."}</small></div></div>}
            </div>
          </section>
        )}
        {(route === "queue" || route === "history") && (
          <section className="history-panel panel">
            <div className="timeline-header">
              <div>
                <span className="eyebrow">{route.toUpperCase()}</span>
                <h2>
                  {language === "vi"
                    ? route === "queue" ? "Đang chạy và chờ render" : "Lịch sử kết quả"
                    : route === "queue" ? "Running and queued" : "Render history"}
                </h2>
              </div>
              <button
                className="button secondary"
                onClick={() => void listJobs().then(setJobs).catch((refreshError) => setError(getErrorMessage(refreshError)))}
              >
                {t("refresh")}
              </button>
            </div>
            <div className="job-list">
              {visibleJobs.length === 0 ? (
                <small className="settings-note">
                  Chưa có job. Hãy dùng Queue render sau khi xác nhận sample.
                </small>
              ) : (
                orderedJobs.map((job) => (
                  <div className="job-row" key={job.id}>
                    <div className="job-main">
                      <strong>{job.sourceName}</strong>
                      <small>
                        {job.stage} · {Math.round(job.progress * 100)}%
                      </small>
                      {job.scanRange && (
                        <small>
                          {language === "vi" ? "Phạm vi quét" : "Scan range"}: {job.scanRange.startFrame}–{job.scanRange.endFrame}
                        </small>
                      )}
                      <progress max={1} value={job.progress} />
                    </div>
                    <span
                      className={`job-status job-${job.status.toLowerCase()}`}
                    >
                      {job.status}
                    </span>
                    <div className="job-actions">
                      {job.outputPath && (
                        <>
                          <button
                            className="button secondary"
                            onClick={() =>
                              void openArtifact(job.outputPath!).catch((error) =>
                                setMessage(getErrorMessage(error)),
                              )
                            }
                          >
                            {t("openOutput")}
                          </button>
                          <button
                            className="button secondary"
                            onClick={() =>
                              void revealItemInDir(job.outputPath!).catch(
                                (error) => setMessage(getErrorMessage(error)),
                              )
                            }
                          >
                            {t("openFolder")}
                          </button>
                        </>
                      )}
                      {job.qaReportPath && <button className="button secondary" onClick={() => { setError(null); void openArtifact(job.qaReportPath!).catch((openError) => setError(getErrorMessage(openError))); }}>{t("viewQa")}</button>}
                      {job.contactSheetPath && <button className="button secondary" onClick={() => { setError(null); void openArtifact(job.contactSheetPath!).catch((openError) => setError(getErrorMessage(openError))); }}>{t("contactSheet")}</button>}
                      {job.status === "NEEDS_REVIEW" && job.outputPath && (
                        <button
                          className="button primary"
                          onClick={() => {
                            setError(null);
                            setProgress(null);
                            setLoadingTask("rendering");
                            operationTaskRef.current = "rendering";
                            setOperationDialog({ task: "rendering", status: "running", title: language === "vi" ? "Đang kiểm tra lại QA" : "Re-validating QA", detail: language === "vi" ? "Đọc lại draft hiện có và chỉ chốt output khi toàn bộ frame vượt quality gate…" : "Rechecking the existing draft and promoting only after every frame passes the quality gate…" });
                            void revalidateReviewJob(job.id).then((updated) => {
                              setJobs((current) => current.map((item) => item.id === updated.id ? updated : item));
                              setMessage(language === "vi" ? "Draft đã vượt QA và được chuyển thành output final." : "The draft passed QA and was promoted to the final output.");
                              setOperationDialog({ task: "rendering", status: "success", title: language === "vi" ? "QA đạt – đã chốt output" : "QA passed – output promoted", detail: updated.outputPath ?? "" });
                            }).catch((reviewError) => {
                              const detail = getErrorMessage(reviewError);
                              setError(detail);
                              setOperationDialog({ task: "rendering", status: "error", title: language === "vi" ? "QA vẫn cần review" : "QA still needs review", detail });
                            }).finally(() => {
                              operationTaskRef.current = null;
                              setLoadingTask(null);
                            });
                          }}
                        >
                          {t("recheckPromote")}
                        </button>
                      )}
                      {[
                        "COMPLETED",
                        "FAILED",
                        "NEEDS_REVIEW",
                        "INTERRUPTED",
                      ].includes(job.status) && (
                        <button
                          className="button secondary"
                          onClick={() =>
                            void regenJob(job.id).then((updated) => {
                              setJobs((current) =>
                                current.map((item) =>
                                  item.id === updated.id ? updated : item,
                                ),
                              );
                              void openLibraryProject(job.projectId);
                            }).catch((regenError) => setError(getErrorMessage(regenError)))
                          }
                        >
                          Regen / rescan
                        </button>
                      )}
                      {[
                        "QUEUED",
                        "PREPARING",
                        "INFERENCING",
                        "ENCODING",
                        "VERIFYING",
                      ].includes(job.status) && (
                        <button
                          className="button secondary"
                          onClick={() =>
                            void cancelJob(job.id).then(
                              () => void listJobs().then(setJobs),
                            )
                          }
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                    {job.error && (
                      <small className="job-error">{formatJobError(job.error)}</small>
                    )}
                  </div>
                ))
              )}
            </div>
          </section>
        )}

        <div className="editor-grid" style={{ display: route === "review" ? undefined : "none" }}>
          <section className="viewer-panel panel">
            <div className="panel-toolbar viewer-toolbar">
              <div className="viewer-heading">
                <span className="eyebrow">{t("framePreview")}</span>
                <strong>{project ? project.source.fileName : t("noVideoSelected")}</strong>
                <small>{project ? `${currentFrame} / ${Math.max(0, project.video.frameCount - 1)} · ${formatTime(currentTime)}` : "—"}</small>
              </div>
              <div className="zoom-toolbar" role="group" aria-label={t("previewZoom")}>
                <button onClick={() => setInspectionZoom(1)} disabled={!project}>{t("fit")}</button>
                {[0.25, 0.5, 1, 2, 4].map((zoom) => <button key={zoom} className={inspectionZoom === zoom ? "selected" : ""} onClick={() => setInspectionZoom(zoom)} disabled={!project}>{Math.round(zoom * 100)}%</button>)}
                <label className="zoom-control">{t("previewZoom")} <input type="range" min="0.25" max="4" step="0.05" value={inspectionZoom} onChange={(event) => setInspectionZoom(Number(event.target.value))} /> {Math.round(inspectionZoom * 100)}%</label>
                <button onClick={() => { setInspectionZoom(1); setSelectionMode(false); }} disabled={!project}>{t("resetView")}</button>
              </div>
            </div>
            <div className="viewer-stage">
              {project && <div className="preview-hud"><span className="preview-live-dot" /> <span>{inspectionMode ? "INSPECT" : selectionMode ? "SELECT ROI" : "SOURCE PREVIEW"}</span><strong>Frame {currentFrame}</strong></div>}
              {inspectionMode &&
              focusPreview &&
              focusPreviewUrl &&
              inspectionTarget &&
              project ? (
                <div
                  className="focus-preview"
                  style={{
                    aspectRatio: `${focusPreview.crop.width}/${focusPreview.crop.height}`,
                  }}
                >
                  <div className="focus-viewport">
                    <div
                      className="focus-canvas"
                      style={{
                        transform: `scale(${inspectionZoom})`,
                        transformOrigin: `${((inspectionTarget.x + inspectionTarget.width / 2 - focusPreview.crop.x) / focusPreview.crop.width) * 100}% ${((inspectionTarget.y + inspectionTarget.height / 2 - focusPreview.crop.y) / focusPreview.crop.height) * 100}%`,
                      }}
                    >
                      <img
                        src={focusPreviewUrl}
                        alt={`Inspection crop for frame ${focusPreview.frame}`}
                        onError={() => setError(language === "vi" ? "Không thể tải ảnh inspect của sample này." : "Unable to load the sample inspection image.")}
                      />
                      <div
                        className="focus-target"
                        style={{
                          left: `${((inspectionTarget.x - focusPreview.crop.x) / focusPreview.crop.width) * 100}%`,
                          top: `${((inspectionTarget.y - focusPreview.crop.y) / focusPreview.crop.height) * 100}%`,
                          width: `${(inspectionTarget.width / focusPreview.crop.width) * 100}%`,
                          height: `${(inspectionTarget.height / focusPreview.crop.height) * 100}%`,
                        }}
                      >
                        <span>
                          VERIFY WATERMARK · frame {focusPreview.frame}
                        </span>
                      </div>
                      <ReplacementPreview
                        replacement={bestReplacement}
                        previewUrl={bestReplacementPreviewUrl}
                        crop={focusPreview.crop}
                        target={inspectionTarget}
                      />
                    </div>
                  </div>
                </div>
              ) : project && videoUrl ? (
                <div
                  className="video-frame"
                  ref={videoFrameRef}
                  style={{ aspectRatio: `${project.video.width}/${project.video.height}`, transform: `scale(${inspectionZoom})`, transformOrigin: "center center" }}
                >
                  <video
                    ref={videoRef}
                    key={videoUrl}
                    src={videoUrl}
                    controls={false}
                    playsInline
                    onLoadedMetadata={onVideoLoadedMetadata}
                    onTimeUpdate={onVideoTimeUpdate}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onError={() => {
                      const text = language === "vi" ? "Không thể tải video preview. Hãy kiểm tra file nguồn còn tồn tại và quyền truy cập." : "Unable to load video preview. Check that the source file still exists and is accessible.";
                      setVideoLoadError(text);
                      setError(text);
                    }}
                  />
                  {videoLoadError && <div className="preview-error" role="alert">{videoLoadError}</div>}
                  {contentRect && (
                    <div
                      ref={selectionSurfaceRef}
                      className={`selection-surface${selectionMode ? " editable" : ""}`}
                      style={{
                        left: contentRect.left,
                        top: contentRect.top,
                        width: contentRect.width,
                        height: contentRect.height,
                      }}
                      onPointerDown={onSelectionPointerDown}
                      onPointerMove={onSelectionPointerMove}
                      onPointerUp={onSelectionPointerUp}
                    >
                      {displaySelection && (
                        <div
                          className={`watermark-box${displayIsProvisional ? " provisional" : ""}`}
                          style={{
                            left: `${sourceToPercent(displaySelection.x, project.video.width)}%`,
                            top: `${sourceToPercent(displaySelection.y, project.video.height)}%`,
                            width: `${sourceToPercent(displaySelection.width, project.video.width)}%`,
                            height: `${sourceToPercent(displaySelection.height, project.video.height)}%`,
                          }}
                        >
                          <span>
                            {displayIsProvisional
                              ? `PROVISIONAL · ${currentTracking?.status}`
                              : (project.watermark.label ?? label)}
                          </span>
                          {!displayIsProvisional && (
                            <>
                              <i className="handle h1" />
                              <i className="handle h2" />
                              <i className="handle h3" />
                              <i className="handle h4" />
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <button
                  className="dropzone"
                  onClick={() => void chooseAndOpenVideo()}
                  disabled={isBusy}
                >
                  <span className="drop-icon">＋</span>
                  <strong>{t("dropVideo")}</strong>
                  <span>{t("clickChoose")}</span>
                  <small>MP4, MOV, MKV, WEBM, M4V</small>
                </button>
              )}
            </div>
            <div className="transport">
              <button onClick={() => setVideoFrame(0)} disabled={!project}>
                │◀
              </button>
              <button
                onClick={() => setVideoFrame(currentFrame - 1)}
                disabled={!project}
              >
                ◀
              </button>
              <button
                className="play"
                onClick={togglePlayback}
                disabled={!project}
              >
                {playing ? "Ⅱ" : "▶"}
              </button>
              <button
                onClick={() => setVideoFrame(currentFrame + 1)}
                disabled={!project}
              >
                ▶
              </button>
              <button
                onClick={() =>
                  setVideoFrame(project ? project.video.frameCount - 1 : 0)
                }
                disabled={!project}
              >
                ▶│
              </button>
              <label className="transport-input">
                Frame{" "}
                <input
                  type="number"
                  min="0"
                  max={Math.max(0, (project?.video.frameCount ?? 1) - 1)}
                  value={project ? currentFrame : ""}
                  onChange={(event) =>
                    setVideoFrame(Number(event.target.value))
                  }
                  disabled={!project}
                />
              </label>
              <label className="transport-input">
                Time{" "}
                <input
                  type="number"
                  min="0"
                  max={project?.video.durationSeconds ?? 0}
                  step="0.001"
                  value={project ? currentTime.toFixed(3) : ""}
                  onChange={(event) => setVideoTime(Number(event.target.value))}
                  disabled={!project}
                />
              </label>
            </div>
          </section>

          <aside className="inspector panel">
            <div className="inspector-title">
              <div>
                <span className="eyebrow">WATERMARK</span>
                <h2>
                  {workspaceMode === "best"
                    ? "Best-quality setup"
                    : project?.tracking
                      ? "Legacy track & preview"
                      : "Legacy enrollment"}
                </h2>
              </div>
              <span className="badge">
                {workspaceMode === "best" ? "FINAL" : "LEGACY"}
              </span>
            </div>
            <label className="field">
              <span>Label</span>
              <input
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                disabled={!project || isBusy}
              />
            </label>
            <div className="field-row">
              <label className="field">
                <span>X · source px</span>
                <input value={displaySelection?.x.toFixed(1) ?? "—"} readOnly />
              </label>
              <label className="field">
                <span>Y · source px</span>
                <input value={displaySelection?.y.toFixed(1) ?? "—"} readOnly />
              </label>
            </div>
            <div className="field-row">
              <label className="field">
                <span>Width · source px</span>
                <input
                  value={displaySelection?.width.toFixed(1) ?? "—"}
                  readOnly
                />
              </label>
              <label className="field">
                <span>Height · source px</span>
                <input
                  value={displaySelection?.height.toFixed(1) ?? "—"}
                  readOnly
                />
              </label>
            </div>
            <div className="debug-card">
              <span>Frame</span>
              <strong>{project ? currentFrame : "—"}</strong>
              <span>Time</span>
              <strong>{project ? formatTime(currentTime) : "—"}</strong>
              <span>Source</span>
              <strong>
                {project
                  ? `${project.video.width} × ${project.video.height}`
                  : "—"}
              </strong>
              <span>Frames</span>
              <strong>{project ? project.video.frameCount : "—"}</strong>
              <span>Padding</span>
              <strong>
                {project ? `${project.watermark.templatePadding} px` : "—"}
              </strong>
              {currentTracking && (
                <>
                  <span>Confidence</span>
                  <strong>
                    {(currentTracking.confidence * 100).toFixed(1)}%
                  </strong>
                  <span>Status</span>
                  <strong>{currentTracking.status}</strong>
                </>
              )}
            </div>
            {workspaceMode === "legacy" && (
              <>
                <div className="legacy-notice">
                  <strong>Legacy / Preview only</strong>
                  <small>
                    Auto Best, Temporal, Blur, Inpaint and legacy Replacement
                    are retained for comparison. Do not use these outputs as
                    final quality output.
                  </small>
                </div>
                <div className="divider" />
                <label className="field">
                  <span>Tracking</span>
                  <select
                    value={currentTracking?.status ?? "NOT_ANALYZED"}
                    disabled
                  >
                    <option value="NOT_ANALYZED">
                      {project?.tracking
                        ? "Select a frame to inspect"
                        : "Not analyzed"}
                    </option>
                    <option value="AUTO_GOOD">Auto good</option>
                    <option value="AUTO_WEAK">Auto weak</option>
                    <option value="NEED_REVIEW">Need review</option>
                    <option value="MANUAL">Manual locked</option>
                    <option value="INTERPOLATED">Interpolated</option>
                    <option value="OCCLUDED">Occluded (no-op)</option>
                  </select>
                </label>
                <label className="field">
                  <span>Removal</span>
                  <select
                    value={removal.mode}
                    onChange={(event) =>
                      setRemoval((current) => ({
                        ...current,
                        mode: event.target.value as RemovalConfig["mode"],
                      }))
                    }
                    disabled={!project?.tracking || isBusy}
                  >
                    <option value="AUTO_BEST">Auto Best</option>
                    <option value="TEMPORAL_RESTORE">Temporal Restore</option>
                    <option value="REPLACEMENT">Replacement PNG</option>
                    <option value="BLUR">Blur mask</option>
                    <option value="INPAINT">Spatial inpaint</option>
                  </select>
                </label>
                {removal.mode === "REPLACEMENT" && (
                  <>
                    <button
                      className="button secondary full"
                      onClick={() => void chooseReplacement()}
                      disabled={isBusy}
                    >
                      {removal.replacementPath
                        ? "Change replacement PNG"
                        : "Choose replacement PNG"}
                    </button>
                    <small className="path-hint">
                      {removal.replacementPath ?? "No PNG selected"}
                    </small>
                  </>
                )}
                {(removal.mode === "TEMPORAL_RESTORE" ||
                  removal.mode === "AUTO_BEST") && (
                  <div className="advanced-card">
                    <span className="eyebrow">RESTORATION SETTINGS</span>
                    <div className="field-row">
                      <label className="field">
                        <span>Before</span>
                        <input
                          type="number"
                          min="1"
                          max="32"
                          value={removal.temporalWindowBefore}
                          onChange={(event) =>
                            setRemoval((current) => ({
                              ...current,
                              temporalWindowBefore: Number(event.target.value),
                            }))
                          }
                          disabled={isBusy}
                        />
                      </label>
                      <label className="field">
                        <span>After</span>
                        <input
                          type="number"
                          min="1"
                          max="32"
                          value={removal.temporalWindowAfter}
                          onChange={(event) =>
                            setRemoval((current) => ({
                              ...current,
                              temporalWindowAfter: Number(event.target.value),
                            }))
                          }
                          disabled={isBusy}
                        />
                      </label>
                    </div>
                    <div className="field-row">
                      <label className="field">
                        <span>Max candidates</span>
                        <input
                          type="number"
                          min="2"
                          max="16"
                          value={removal.maxTemporalCandidates}
                          onChange={(event) =>
                            setRemoval((current) => ({
                              ...current,
                              maxTemporalCandidates: Number(event.target.value),
                            }))
                          }
                          disabled={isBusy}
                        />
                      </label>
                      <label className="field">
                        <span>ROI padding</span>
                        <input
                          type="number"
                          min="8"
                          max="96"
                          value={removal.restorationRoiPadding}
                          onChange={(event) =>
                            setRemoval((current) => ({
                              ...current,
                              restorationRoiPadding: Number(event.target.value),
                            }))
                          }
                          disabled={isBusy}
                        />
                      </label>
                    </div>
                    <label className="field">
                      <span>Artifact limit</span>
                      <input
                        type="number"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={removal.artifactThreshold}
                        onChange={(event) =>
                          setRemoval((current) => ({
                            ...current,
                            artifactThreshold: Number(event.target.value),
                          }))
                        }
                        disabled={isBusy}
                      />
                    </label>
                    <label className="field">
                      <span>Fallback</span>
                      <select
                        value={removal.fallbackPolicy}
                        onChange={(event) =>
                          setRemoval((current) => ({
                            ...current,
                            fallbackPolicy: event.target
                              .value as RemovalConfig["fallbackPolicy"],
                          }))
                        }
                        disabled={isBusy}
                      >
                        <option value="TEMPORAL_INPAINT_BLUR">
                          Inpaint → Blur
                        </option>
                        <option value="BLUR_ONLY">Blur only</option>
                      </select>
                    </label>
                  </div>
                )}
                <div className="confidence-card">
                  <div>
                    <span>
                      {project?.tracking
                        ? "Current tracking"
                        : "Phase 1 status"}
                    </span>
                    <strong>
                      {currentTracking?.status ??
                        (project?.watermark.templates
                          ? "Ready"
                          : project?.watermark.anchor
                            ? "Anchor saved"
                            : "Manual anchor")}
                    </strong>
                  </div>
                  <div className="confidence-bar">
                    <i
                      style={{
                        width: `${project?.tracking ? (currentTracking?.confidence ?? 0) * 100 : project?.watermark.templates ? 100 : project?.watermark.anchor ? 65 : 8}%`,
                      }}
                    />
                  </div>
                  <small>
                    {project?.tracking
                      ? trackingNeedsReview
                        ? "This bbox is provisional. Click Select watermark, draw the actual watermark, then save the manual correction."
                        : `${unresolvedCount} problem range(s); weak/review frames are blocked from rendering. Occluded frames are preserved unchanged.`
                      : project?.watermark.templates
                        ? "Templates and mask persisted in the project workspace."
                        : "Select a source-coordinate box to begin."}
                  </small>
                  {project?.tracking && unresolvedCount > 0 && (
                    <small className="review-queue">
                      Review frames: {problemRangeSummary}
                    </small>
                  )}
                </div>
              </>
            )}
            {workspaceMode === "best" && (
              <>
                <div className="best-quality-card">
                  <span className="eyebrow">FINAL PATH · VERIFIED LAYOUT</span>
                  <strong>Best-quality AI render</strong>
                  <small>
                    FP32 ProPainter theo chunk + context, chỉ blend glyph mask.
                    Profile:{" "}
                    {project?.calibration?.quality.status ??
                      "CALIBRATION PENDING"}{" "}
                    · V{project?.calibration?.version ?? 5} · source luôn được giữ nguyên.
                  </small>
                  {project?.calibration?.version && project.calibration.version >= 5 && (
                    <small className="review-queue">
                      Trajectory: {project.calibration.quality.reliableFrames} measured · {project.calibration.quality.lowConfidenceFrames} inferred
                    </small>
                  )}
                  {project?.calibration?.scanRange && (
                    <small className="review-queue">
                      {language === "vi"
                        ? `Phạm vi đã lưu: ${project.calibration.scanRange.startFrame}–${project.calibration.scanRange.endFrame}; ngoài phạm vi giữ nguyên.`
                        : `Saved range: ${project.calibration.scanRange.startFrame}–${project.calibration.scanRange.endFrame}; outside frames are passthrough.`}
                    </small>
                  )}
                  {project?.calibration && (project.calibration.excludedFrameCount ?? 0) > 0 && (
                    <small className="scan-range-caution">
                      {language === "vi"
                        ? `${project.calibration.excludedFrameCount} frame ngoài phạm vi không được kiểm tra/xử lý.`
                        : `${project.calibration.excludedFrameCount} frames outside the range were not checked or processed.`}
                    </small>
                  )}
                </div>
                {project && (
                  <div className="scan-range-card">
                    <div className="scan-range-header">
                      <div>
                        <span className="eyebrow">SCAN SCOPE</span>
                        <strong>{language === "vi" ? "Phạm vi quét watermark" : "Watermark scan range"}</strong>
                      </div>
                      <span className="scan-range-badge">{scanRange.startFrame}–{scanRange.endFrame}</span>
                    </div>
                    <div className="scan-range-fields">
                      <label className="field">
                        <span>{language === "vi" ? "Frame bắt đầu" : "Start frame"}</span>
                        <input
                          type="number"
                          min={0}
                          max={Math.max(0, project.video.frameCount - 1)}
                          value={scanRange.startFrame}
                          onChange={(event) => updateScanRange("startFrame", event.target.value)}
                          disabled={isBusy}
                        />
                        <small>{formatTime(scanRange.startFrame / project.video.fps)}</small>
                      </label>
                      <label className="field">
                        <span>{language === "vi" ? "Frame kết thúc" : "End frame"}</span>
                        <input
                          type="number"
                          min={0}
                          max={Math.max(0, project.video.frameCount - 1)}
                          value={scanRange.endFrame}
                          onChange={(event) => updateScanRange("endFrame", event.target.value)}
                          disabled={isBusy}
                        />
                        <small>{formatTime(scanRange.endFrame / project.video.fps)}</small>
                      </label>
                    </div>
                    <div className="scan-range-actions">
                      <button className="button secondary" onClick={() => updateScanRange("startFrame", String(currentFrame))} disabled={isBusy || currentFrame > scanRange.endFrame}>
                        {language === "vi" ? "Lấy frame hiện tại làm đầu" : "Use current as start"}
                      </button>
                      <button className="button secondary" onClick={() => updateScanRange("endFrame", String(currentFrame))} disabled={isBusy || currentFrame < scanRange.startFrame}>
                        {language === "vi" ? "Lấy frame hiện tại làm cuối" : "Use current as end"}
                      </button>
                      <button className="button secondary" onClick={resetScanRange} disabled={isBusy}>
                        {language === "vi" ? "Quét toàn bộ video" : "Scan full video"}
                      </button>
                    </div>
                    <small className="scan-range-warning">
                      {language === "vi"
                        ? `Đang quét frame ${scanRange.startFrame}–${scanRange.endFrame} / ${project.video.frameCount}. Frame ngoài phạm vi sẽ giữ nguyên và không yêu cầu ROI.`
                        : `Scanning frames ${scanRange.startFrame}–${scanRange.endFrame} / ${project.video.frameCount}. Frames outside the range stay unchanged and never require ROI.`}
                    </small>
                    {scanRange.startFrame > 0 || scanRange.endFrame < project.video.frameCount - 1 ? (
                      <small className="scan-range-caution">
                        {language === "vi"
                          ? "Nếu watermark xuất hiện ngoài phạm vi, output sẽ vẫn giữ watermark ở đoạn đó."
                          : "If the watermark appears outside this range, the output will keep it there."}
                      </small>
                    ) : null}
                  </div>
                )}
                <div className="best-quality-samples">
                  <span className="eyebrow">
                    1. FIND · 2. INSPECT · 3. CONFIRM
                  </span>
                  <small>
                    Chỉ các mẫu vượt hard gate glyph Learna AI mới xuất hiện.
                    Click một thẻ để kiểm tra crop và mask cố định.
                  </small>
                  {bestQualitySamples.length > 0 ? (
                    <>
                      <div className="sample-grid">
                        {bestQualitySamples.map((sample, index) => (
                          <button
                            key={sample.frame}
                            className={`sample-card${selectedBestQualitySample?.frame === sample.frame ? " selected" : ""}`}
                            onClick={() => inspectBestQualitySample(sample)}
                            disabled={isBusy}
                          >
                            <div className="sample-images">
                              <img
                                src={convertFileSrc(sample.previewPath)}
                                alt={`Sample frame ${sample.frame}`}
                              />
                              <img
                                className="sample-mask"
                                src={convertFileSrc(sample.maskPath)}
                                alt={`Glyph mask for frame ${sample.frame}`}
                              />
                            </div>
                            <strong>
                              #{index + 1} · Frame {sample.frame}
                            </strong>
                            <small>
                              corr {sample.glyphCorrelation.toFixed(3)} · IoU{" "}
                              {sample.glyphIou.toFixed(3)} · nhiễm{" "}
                              {(sample.contamination * 100).toFixed(1)}% · ổn
                              định {sample.temporalPassCount}/5
                            </small>
                          </button>
                        ))}
                      </div>
                      <small className="alternatives-hint">{t("alternativesHint")}</small>
                      <button
                        className="button secondary full"
                        onClick={() => void findBestQualitySamples(true)}
                        disabled={isBusy}
                      >
                        {t("findAlternatives")}
                      </button>
                    </>
                  ) : (
                    loadingTask !== "sampling" && loadingTask !== "calibrating" && (<>
                      {project?.calibration?.quality.status === "READY" && project.calibration.scanRange ? (
                        <small className="calibration-ready-note">Calibration V6 đã đạt. Không cần chọn sample thủ công; bạn có thể đưa job vào Queue.</small>
                      ) : project?.calibration?.quality.status === "READY" ? (
                        <small className="scan-range-caution">Profile V6 cũ chưa có phạm vi quét; hãy chạy lại Auto-find & calibrate trước khi Queue.</small>
                      ) : <small>{t("noValidSample")}</small>}
                      {roiFallbackArmed && <small className="review-queue">{t("roiFallbackActive")}</small>}
                      <button className="button secondary full" onClick={beginRoiFallback} disabled={isBusy}>{language === "vi" ? "Khoanh ROI tương đối" : "Draw relative ROI"}</button>
                      {roiFallbackArmed && <>
                        {selection && <button className="button secondary full" onClick={addRoiEvidence} disabled={isBusy}>{language === "vi" ? "Thêm ROI evidence frame này" : "Add ROI evidence for this frame"}</button>}
                        {(selection || roiEvidence.length > 0) && <small className="roi-evidence-count">{language === "vi" ? `Đã có ${roiEvidence.length} frame evidence. Nên thêm các đoạn chuyển động khác nhau.` : `${roiEvidence.length} evidence frame(s) saved. Add frames from different motion segments.`}</small>}
                        {roiEvidence.length > 0 && <button className="button secondary full" onClick={() => void runAdaptiveCalibration()} disabled={isBusy}>{t("roiFit")}</button>}
                      </>}
                      <button className="button secondary full" onClick={() => void findBestQualitySamples(true)} disabled={isBusy}>{t("scanAnotherPhase")}</button>
                    </>)
                  )}
                  {selectedBestQualitySample && (
                    <div className="mask-editor">
                      <div className="mask-editor-toolbar"><strong>{t("maskEditor")}</strong><button className={brushMode === "add" ? "selected" : ""} onClick={() => setBrushMode("add")} disabled={isBusy}>{t("add")}</button><button className={brushMode === "erase" ? "selected" : ""} onClick={() => setBrushMode("erase")} disabled={isBusy}>{t("erase")}</button><label>Brush <input type="range" min="1" max="40" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} disabled={isBusy} /> {brushSize}px</label><label>Opacity <input type="range" min="0.15" max="1" step="0.05" value={maskOpacity} onChange={(event) => setMaskOpacity(Number(event.target.value))} disabled={isBusy} /></label><button onClick={undoMask} disabled={isBusy || maskUndo.length === 0}>Undo</button><button onClick={redoMask} disabled={isBusy || maskRedo.length === 0}>Redo</button><button onClick={() => project && loadMaskEditor(project.id, selectedBestQualitySample.editorMaskPath)} disabled={isBusy}>{t("resetMask")}</button></div>
                      <div className="mask-editor-stage"><img src={convertFileSrc(selectedBestQualitySample.previewPath)} alt="Selected sample" onError={() => setError(language === "vi" ? "Không thể tải ảnh sample để chỉnh mask." : "Unable to load the selected sample image.")} /><canvas ref={maskCanvasRef} style={{ opacity: maskOpacity }} onPointerDown={startMaskBrush} onPointerMove={drawMaskBrush} onPointerUp={(event) => { maskDrawingRef.current = false; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }} onPointerCancel={(event) => { maskDrawingRef.current = false; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }} /></div>
                      <small>Gate khi Save: coverage ≥95%, contamination ≤5%, không thiếu mảng nét lớn. Mask được đối chiếu lại trên toàn bộ profile trước khi Queue.</small>
                    <button
                      className="button secondary full"
                      onClick={() => void saveBestQualitySample()}
                      disabled={isBusy || !maskEditorReady}
                    >
                      Save mask & run Calibration V6
                    </button>
                    </div>
                  )}
                </div>
              </>
            )}
            {workspaceMode === "best" && (
              <div className="best-replacement-card">
                <span className="eyebrow">OUTPUT</span>
                <small className="path-hint">
                  {outputRoot || "Mặc định: thư mục output cạnh source"}
                </small>
                <label className="field"><span>{t("outputFileName")}</span><input value={outputName} onChange={(event) => { setOutputName(event.target.value); if (project) window.localStorage.setItem(`watermark-studio:output-name:${project.id}`, event.target.value); }} placeholder="source_watermark_removed_best.mp4" disabled={isBusy} /></label>
                <button
                  className="button secondary full"
                  onClick={() => void chooseAndSaveOutputRoot()}
                  disabled={isBusy}
                >
                  {outputRoot ? t("changeOutput") : t("chooseOutput")}
                </button>
                <span className="eyebrow">OPTIONAL · REPLACEMENT LABEL</span>
                <label className="replacement-toggle">
                  <input
                    type="checkbox"
                    checked={Boolean(bestReplacement)}
                    onChange={(event) =>
                      setBestReplacement(
                        event.target.checked
                          ? { ...defaultBestReplacement }
                          : null,
                      )
                    }
                    disabled={isBusy}
                  />{" "}
                  {t("addLabel")}
                </label>
                {bestReplacement && (
                  <>
                    <label className="field">
                      <span>Kind</span>
                      <select
                        value={bestReplacement.kind}
                        onChange={(event) =>
                          setBestReplacement((current) =>
                            current
                              ? {
                                  ...current,
                                  kind: event.target
                                    .value as BestQualityReplacement["kind"],
                                }
                              : current,
                          )
                        }
                        disabled={isBusy}
                      >
                        <option value="text">Text</option>
                        <option value="image">Transparent PNG</option>
                      </select>
                    </label>
                    {bestReplacement.kind === "text" ? (
                      <label className="field">
                        <span>Text</span>
                        <input
                          value={bestReplacement.text}
                          onChange={(event) =>
                            setBestReplacement((current) =>
                              current
                                ? { ...current, text: event.target.value }
                                : current,
                            )
                          }
                          disabled={isBusy}
                          placeholder="Optional replacement label"
                        />
                      </label>
                    ) : (
                      <>
                        <button
                          className="button secondary full"
                          onClick={() => void chooseBestReplacement()}
                          disabled={isBusy}
                        >
                          {bestReplacement.imagePath
                            ? "Change replacement PNG"
                            : "Choose transparent PNG"}
                        </button>
                        <small className="path-hint">
                          {bestReplacement.imagePath ?? "No PNG selected"}
                        </small>
                      </>
                    )}
                    <label className="field">
                      <span>Placement</span>
                      <select
                        value={bestReplacement.placement}
                        onChange={(event) =>
                          setBestReplacement((current) =>
                            current
                              ? {
                                  ...current,
                                  placement: event.target
                                    .value as BestQualityReplacement["placement"],
                                }
                              : current,
                          )
                        }
                        disabled={isBusy}
                      >
                        <option value="follow">Follow watermark motion</option>
                        <option value="fixed">Fixed source position</option>
                      </select>
                    </label>
                    {bestReplacement.placement === "fixed" && (
                      <div className="field-row">
                        <label className="field">
                          <span>X · source px</span>
                          <input
                            type="number"
                            value={bestReplacement.fixedX}
                            onChange={(event) =>
                              setBestReplacement((current) =>
                                current
                                  ? {
                                      ...current,
                                      fixedX: Number(event.target.value),
                                    }
                                  : current,
                              )
                            }
                            disabled={isBusy}
                          />
                        </label>
                        <label className="field">
                          <span>Y · source px</span>
                          <input
                            type="number"
                            value={bestReplacement.fixedY}
                            onChange={(event) =>
                              setBestReplacement((current) =>
                                current
                                  ? {
                                      ...current,
                                      fixedY: Number(event.target.value),
                                    }
                                  : current,
                              )
                            }
                            disabled={isBusy}
                          />
                        </label>
                      </div>
                    )}
                    <div className="field-row">
                      <label className="field">
                        <span>Scale</span>
                        <input
                          type="number"
                          min="0.1"
                          max="4"
                          step="0.1"
                          value={bestReplacement.scale}
                          onChange={(event) =>
                            setBestReplacement((current) =>
                              current
                                ? {
                                    ...current,
                                    scale: Number(event.target.value),
                                  }
                                : current,
                            )
                          }
                          disabled={isBusy}
                        />
                      </label>
                      <label className="field">
                        <span>Opacity</span>
                        <input
                          type="number"
                          min="0"
                          max="1"
                          step="0.1"
                          value={bestReplacement.opacity}
                          onChange={(event) =>
                            setBestReplacement((current) =>
                              current
                                ? {
                                    ...current,
                                    opacity: Number(event.target.value),
                                  }
                                : current,
                            )
                          }
                          disabled={isBusy}
                        />
                      </label>
                    </div>
                    <small>
                      Preview vị trí ở panel Inspect watermark; render sẽ chèn
                      label sau khi glyph được AI xóa.
                    </small>
                  </>
                )}
              </div>
            )}
            {workspaceMode === "legacy" && (
              <>
                {project?.tracking && unresolvedCount > 0 && (
                  <div
                    className="review-range-actions"
                    aria-label="Review queue"
                  >
                    <span>Jump to range</span>
                    <div>
                      {project.tracking.problemRanges.map((range) => (
                        <button
                          key={`${range.startFrame}-${range.endFrame}`}
                          onClick={() => {
                            setVideoFrame(range.worstFrame);
                            setMessage(
                              `Review frame ${range.worstFrame} (${range.startFrame}–${range.endFrame}).`,
                            );
                          }}
                          disabled={isBusy}
                        >
                          {range.startFrame === range.endFrame
                            ? `Frame ${range.startFrame}`
                            : `${range.startFrame}–${range.endFrame}`}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <button
                  className="button secondary full"
                  onClick={() => void saveAnchor()}
                  disabled={
                    !project ||
                    (project.tracking ? !selection : !displaySelection) ||
                    isBusy
                  }
                >
                  {loadingTask === "saving"
                    ? "Saving…"
                    : project?.tracking
                      ? "Save manual correction"
                      : "Save anchor & templates"}
                </button>
                {project?.tracking && currentTracking && (
                  <>
                    <button
                      className="button secondary full"
                      onClick={() => void acceptCurrentFrame()}
                      disabled={isBusy || currentTracking.status === "MANUAL"}
                    >
                      Accept current frame
                    </button>
                    <button
                      className="button secondary full"
                      onClick={() => void markCurrentRangeOccluded()}
                      disabled={
                        isBusy ||
                        !project.tracking.problemRanges.some(
                          (item) =>
                            currentFrame >= item.startFrame &&
                            currentFrame <= item.endFrame,
                        )
                      }
                    >
                      Mark current range occluded (no-op)
                    </button>
                    <button
                      className="button secondary full"
                      onClick={() => void runRetrack()}
                      disabled={isBusy}
                    >
                      Re-track section
                    </button>
                    <button
                      className="button secondary full"
                      onClick={() => void interpolateCurrentRange()}
                      disabled={
                        isBusy || !project.tracking.problemRanges.length
                      }
                    >
                      Interpolate current range
                    </button>
                  </>
                )}
                <button
                  className="button primary full"
                  onClick={() => void render()}
                  disabled={!project?.tracking || isBusy || unresolvedCount > 0}
                >
                  {loadingTask === "rendering"
                    ? "Rendering…"
                    : "Render legacy preview"}
                </button>
                {isBusy &&
                  (loadingTask === "tracking" ||
                    loadingTask === "rendering" ||
                    loadingTask === "calibrating") && (
                    <button
                      className="button secondary full"
                      onClick={cancelBusy}
                    >
                      Cancel operation
                    </button>
                  )}
              </>
            )}
          </aside>
        </div>

        <section className="timeline-panel panel" style={{ display: route === "review" ? undefined : "none" }}>
          <div className="timeline-header">
            <div>
              <span className="eyebrow">NAVIGATION</span>
              <h2>Frame timeline</h2>
            </div>
            <div className="timeline-actions">
              <span>
                {project
                  ? `${project.video.frameCount} frames`
                  : "No video loaded"}
                {progress
                  ? ` · ${Math.round(progress.progress * 100)}% ${progress.phase}`
                  : ""}
                {workspaceMode === "best" && pendingCalibrationReviewRanges.length > 0
                  ? language === "vi"
                    ? ` · ${pendingCalibrationReviewRanges.length} cụm ROI chưa duyệt`
                    : ` · ${pendingCalibrationReviewRanges.length} ROI cluster(s) pending`
                  : workspaceMode === "best" && calibrationReviewRanges.length > 0
                    ? language === "vi" ? " · Đã duyệt hết cụm ROI" : " · All ROI clusters reviewed"
                    : workspaceMode === "best" && roiReviewSaturated
                      ? language === "vi" ? " · Đủ evidence; cần refine quỹ đạo" : " · Evidence saturated; trajectory refinement required"
                      : ""}
              </span>
              <button
                onClick={nextProblem}
                disabled={!hasNextReviewProblem || isBusy}
              >
                {workspaceMode === "best" && pendingCalibrationReviewRanges.length > 0
                  ? language === "vi" ? "Tiếp ROI →" : "Next ROI problem →"
                  : language === "vi" ? "Vấn đề tiếp theo →" : "Next problem →"}
              </button>
            </div>
          </div>
          <div className="timeline-ruler">
            <span>00:00</span>
            <span>
              {formatTime((project?.video.durationSeconds ?? 0) * 0.25)}
            </span>
            <span>
              {formatTime((project?.video.durationSeconds ?? 0) * 0.5)}
            </span>
            <span>
              {formatTime((project?.video.durationSeconds ?? 0) * 0.75)}
            </span>
            <span>{formatTime(project?.video.durationSeconds ?? 0)}</span>
          </div>
          <div
            className="timeline-track"
            onClick={clickTimeline}
            role="slider"
            aria-label="Video timeline"
            aria-valuemin={0}
            aria-valuemax={
              project ? Math.max(0, project.video.frameCount - 1) : 0
            }
            aria-valuenow={currentFrame}
            tabIndex={project ? 0 : -1}
          >
            <div className="track-good" />
            {project?.tracking &&
              trackingSegments(project.tracking.frames).map((segment) => (
                <i
                  className="track-segment"
                  key={`${segment.start}-${segment.end}`}
                  title={`${segment.status}: frames ${segment.start}–${segment.end}`}
                  style={{
                    left: `${(segment.start / project.video.frameCount) * 100}%`,
                    width: `${((segment.end - segment.start + 1) / project.video.frameCount) * 100}%`,
                    background: trackingColor(segment.status),
                  }}
                />
              ))}
            <i
              className="playhead"
              style={{
                left:
                  project && project.video.frameCount > 1
                    ? `${(currentFrame / (project.video.frameCount - 1)) * 100}%`
                    : "0%",
              }}
            />
          </div>
          <div className="timeline-hint">
            Click vào timeline hoặc dùng ◀ / ▶ để điều hướng từng frame.{" "}
            {project ? `Frame hiện tại: ${currentFrame}.` : ""}{" "}
            {project?.tracking
              ? `Problem ranges: ${unresolvedCount}. Manual frames are locked.`
              : ""}
          </div>
        </section>
      </section>
      {operationDialog && (
        <div className="operation-dialog-backdrop" role="presentation">
          <section className={`operation-dialog operation-${operationDialog.status}`} role="dialog" aria-modal="true" aria-label={operationDialog.title}>
            <div className="operation-dialog-header">
              <div>
                <span className="eyebrow">{operationDialog.task.toUpperCase()}</span>
                <h2>{operationDialog.title}</h2>
              </div>
              {operationDialog.status !== "running" && <button className="dialog-close" onClick={() => { setOperationDialog(null); setError(null); }}>×</button>}
            </div>
            <p className="operation-detail">{operationDialog.detail}</p>
            {operationDialog.status === "error" && operationDialog.task === "calibrating" && operationDialog.reviewRanges?.length ? (
              <div className="operation-review-ranges">
                <strong>Khoảng yếu cần bổ sung ROI</strong>
                <ul>
                  {operationDialog.reviewRanges.map((range) => (
                    <li key={`${range.startFrame}-${range.endFrame}`}>
                      <span>{range.startFrame}–{range.endFrame}</span>
                      <small>
                        ưu tiên frame {range.suggestedFrames[0]}
                        {range.suggestedFrames.length > 1 ? ` · dự phòng ${range.suggestedFrames.slice(1).join(", ")}` : ""}
                      </small>
                      <button
                        className="button secondary compact"
                        onClick={() => {
                          openCalibrationReviewRange(range);
                          setOperationDialog(null);
                        }}
                      >
                        Mở frame ưu tiên
                      </button>
                    </li>
                  ))}
                </ul>
                <small>Chọn ROI tương đối rộng tại một frame được gợi ý, chỉ khi watermark còn xuất hiện.</small>
              </div>
            ) : null}
            <div className="operation-progress-row"><strong>{progress || operationDialog.status !== "running" ? `${dialogProgress}%` : "…"}</strong><span>{progress ? `${progress.currentFrame}/${progress.totalFrames} frame · ${progress.phase}` : operationDialog.status === "running" ? "Đang xử lý; backend sẽ cập nhật kết quả khi hoàn tất…" : "Đã hoàn tất bước xử lý"}</span></div>
            <div className={`operation-progress-track${!progress && operationDialog.status === "running" ? " indeterminate" : ""}`}><i style={{ width: `${dialogProgress}%` }} /></div>
            <div className="operation-stepper">
              {dialogSteps.map((step, index) => <div className={`operation-step${index < dialogActiveStep ? " done" : index === dialogActiveStep ? " active" : ""}`} key={step}><span>{index < dialogActiveStep ? "✓" : index + 1}</span><small>{step}</small></div>)}
            </div>
            {operationDialog.status === "running" && (operationDialog.task === "calibrating" || operationDialog.task === "rendering") && <button className="button secondary full" onClick={cancelBusy}>Hủy tác vụ</button>}
            {operationDialog.status !== "running" && <button className="button primary full" onClick={() => { setOperationDialog(null); setError(null); }}>{operationDialog.status === "success" ? "Tiếp tục" : "Quay lại Review"}</button>}
          </section>
        </div>
      )}
    </main>
  );
}
