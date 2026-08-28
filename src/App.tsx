import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import { convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { acceptTrackingFrame, analyzeTrack, cancelRender, cancelTracking, chooseReplacementPath, chooseVideoPath, getErrorMessage, getProject, interpolateTrackingRange, markOccludedRange, openVideo, renderVideo, retrackTrack, saveManualAnchor, saveRemovalConfig, saveWatermarkAnchor } from './services/projectApi';
import type { BoundingBox, RemovalConfig, TrackingFrame, WatermarkProject } from './types/project';
import './styles.css';

type LoadingTask = 'opening' | 'saving' | 'tracking' | 'rendering' | null;
type Point = { x: number; y: number };
type ContentRect = { left: number; top: number; width: number; height: number };
type OperationProgress = { phase: string; currentFrame: number; totalFrames: number; progress: number };

const defaultRemoval: RemovalConfig = { mode: 'BLUR', maskPadding: 4, featherRadius: 3, replacementPath: null, replacementScale: 1, replacementOpacity: 1, replacementOffsetX: 0, replacementOffsetY: 0, temporalWindowBefore: 12, temporalWindowAfter: 12, maxTemporalCandidates: 10, restorationRoiPadding: 32, artifactThreshold: 0.25, fallbackPolicy: 'TEMPORAL_INPAINT_BLUR', inpaintVariant: 'ITERATIVE' };

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
  return `${String(minutes).padStart(2, '0')}:${remainingSeconds.toFixed(3).padStart(6, '0')}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function sourceToPercent(value: number, sourceSize: number): number {
  return sourceSize > 0 ? (value / sourceSize) * 100 : 0;
}

function trackingColor(status: TrackingFrame['status']): string {
  switch (status) {
    case 'AUTO_GOOD': return '#77d99a';
    case 'AUTO_WEAK': return '#e3bd70';
    case 'NEED_REVIEW': return '#d97676';
    case 'MANUAL': return '#8fb8ff';
    case 'INTERPOLATED': return '#b89be8';
    case 'OCCLUDED': return '#8b8f9c';
  }
}

function trackingSegments(frames: TrackingFrame[]): Array<{ start: number; end: number; status: TrackingFrame['status'] }> {
  if (!frames.length) return [];
  const segments: Array<{ start: number; end: number; status: TrackingFrame['status'] }> = [];
  let start = 0;
  for (let index = 1; index <= frames.length; index += 1) {
    if (index === frames.length || frames[index].status !== frames[start].status) {
      segments.push({ start, end: index - 1, status: frames[start].status });
      start = index;
    }
  }
  return segments;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoFrameRef = useRef<HTMLDivElement>(null);
  const selectionSurfaceRef = useRef<HTMLDivElement>(null);
  const selectionStartRef = useRef<Point | null>(null);
  const [project, setProject] = useState<WatermarkProject | null>(null);
  const [loadingTask, setLoadingTask] = useState<LoadingTask>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selection, setSelection] = useState<BoundingBox | null>(null);
  const [contentRect, setContentRect] = useState<ContentRect | null>(null);
  const [label, setLabel] = useState('Learna AI');
  const [playing, setPlaying] = useState(false);
  const [message, setMessage] = useState('Ready');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<OperationProgress | null>(null);
  const [removal, setRemoval] = useState<RemovalConfig>(defaultRemoval);

  const videoUrl = project ? convertFileSrc(project.source.path) : null;
  const currentTracking = project?.tracking?.frames.find((frame) => frame.frame === currentFrame) ?? null;
  const displaySelection = selection ?? currentTracking?.bbox ?? project?.watermark.anchor?.bbox ?? null;
  const unresolvedCount = project?.tracking?.problemRanges.length ?? 0;
  const isBusy = loadingTask !== null;

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void listen<OperationProgress>('operation-progress', (event) => {
      if (!disposed) setProgress(event.payload);
    }).then((stop) => {
      if (disposed) stop(); else unlisten = stop;
    });
    return () => { disposed = true; unlisten?.(); };
  }, []);

  useEffect(() => {
    const projectId = window.localStorage.getItem('watermark-studio:last-project-id');
    if (!projectId) return;
    let disposed = false;
    setLoadingTask('opening');
    setMessage('Loading saved project…');
    void getProject(projectId)
      .then((savedProject) => {
        if (disposed) return;
        setProject(savedProject);
        setSelection(savedProject.tracking ? null : savedProject.watermark.anchor?.bbox ?? null);
        setCurrentFrame(savedProject.watermark.anchor?.frame ?? 0);
        setCurrentTime(savedProject.watermark.anchor?.timestampSeconds ?? 0);
        setLabel(savedProject.watermark.label ?? 'Learna AI');
        setRemoval(normalizeRemoval(savedProject.removal));
        setMessage('Saved project loaded.');
      })
      .catch(() => {
        if (!disposed) window.localStorage.removeItem('watermark-studio:last-project-id');
      })
      .finally(() => {
        if (!disposed) setLoadingTask(null);
      });
    return () => { disposed = true; };
  }, []);

  const measureContentRect = useCallback(() => {
    const frame = videoFrameRef.current;
    const video = videoRef.current;
    if (!frame || !video || !project || video.clientWidth === 0 || video.clientHeight === 0) return;
    const frameBounds = frame.getBoundingClientRect();
    const videoBounds = video.getBoundingClientRect();
    const sourceRatio = project.video.width / project.video.height;
    const contentWidth = Math.min(video.clientWidth, video.clientHeight * sourceRatio);
    const contentHeight = contentWidth / sourceRatio;
    setContentRect({
      left: (video.clientWidth - contentWidth) / 2 + (videoBounds.left - frameBounds.left),
      top: (video.clientHeight - contentHeight) / 2 + (videoBounds.top - frameBounds.top),
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

  const setVideoFrame = (frame: number) => {
    if (!project || !videoRef.current) return;
    const lastFrame = Math.max(0, project.video.frameCount - 1);
    const nextFrame = clamp(Math.round(frame), 0, lastFrame);
    const nextTime = nextFrame / project.video.fps;
    videoRef.current.currentTime = nextTime;
    setSelection(null);
    setCurrentFrame(nextFrame);
    setCurrentTime(nextTime);
  };

  const setVideoTime = (time: number) => {
    if (!project || !videoRef.current || !Number.isFinite(time)) return;
    const nextTime = clamp(time, 0, project.video.durationSeconds);
    videoRef.current.currentTime = nextTime;
    setSelection(null);
    setCurrentTime(nextTime);
    setCurrentFrame(clamp(Math.round(nextTime * project.video.fps), 0, Math.max(0, project.video.frameCount - 1)));
  };

  const onVideoLoadedMetadata = () => {
    measureContentRect();
    if (!project || !videoRef.current) return;
    const initialFrame = project.watermark.anchor?.frame ?? 0;
    const initialTime = project.watermark.anchor?.timestampSeconds ?? initialFrame / project.video.fps;
    videoRef.current.currentTime = initialTime;
    setCurrentFrame(initialFrame);
    setCurrentTime(initialTime);
  };

  const onVideoTimeUpdate = () => {
    if (!project || !videoRef.current) return;
    const time = videoRef.current.currentTime;
    setCurrentTime(time);
    setCurrentFrame(clamp(Math.round(time * project.video.fps), 0, Math.max(0, project.video.frameCount - 1)));
    if (!selectionMode) setSelection(null);
  };

  const chooseAndOpenVideo = async () => {
    setError(null);
    try {
      const selectedPath = await chooseVideoPath();
      if (!selectedPath) return;
      setLoadingTask('opening');
      setMessage('Opening video and reading metadata…');
      const nextProject = await openVideo(selectedPath);
      window.localStorage.setItem('watermark-studio:last-project-id', nextProject.id);
      setProject(nextProject);
      setSelection(null);
      setContentRect(null);
      setCurrentFrame(0);
      setCurrentTime(0);
      setLabel(nextProject.watermark.label ?? 'Learna AI');
      setRemoval(normalizeRemoval(nextProject.removal));
      setSelectionMode(false);
      setPlaying(false);
      setMessage('Video loaded. Click Select watermark, then draw a box.');
    } catch (openError) {
      setError(getErrorMessage(openError));
      setMessage('Could not open video');
    } finally {
      setLoadingTask(null);
    }
  };

  const pointFromPointer = (event: ReactPointerEvent<HTMLDivElement>): Point | null => {
    const surface = selectionSurfaceRef.current;
    if (!surface) return null;
    const bounds = surface.getBoundingClientRect();
    return { x: clamp(event.clientX - bounds.left, 0, bounds.width), y: clamp(event.clientY - bounds.top, 0, bounds.height) };
  };

  const sourceBoxFromPoints = (start: Point, end: Point): BoundingBox | null => {
    if (!project || !contentRect) return null;
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    return {
      x: (left / contentRect.width) * project.video.width,
      y: (top / contentRect.height) * project.video.height,
      width: (Math.abs(end.x - start.x) / contentRect.width) * project.video.width,
      height: (Math.abs(end.y - start.y) / contentRect.height) * project.video.height,
    };
  };

  const onSelectionPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!selectionMode || isBusy) return;
    const point = pointFromPointer(event);
    if (!point) return;
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
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!start || !point) return;
    const nextSelection = sourceBoxFromPoints(start, point);
    if (nextSelection && nextSelection.width >= 8 && nextSelection.height >= 8) {
      setSelection(nextSelection);
      setMessage('Watermark selected. Review source coordinates, then save the anchor.');
    } else {
      setSelection(null);
      setMessage('Selection is too small. Minimum size is 8 × 8 source pixels.');
    }
  };

  const saveAnchor = async () => {
    if (!project || !displaySelection || isBusy) return;
    setError(null);
    setLoadingTask('saving');
    setMessage(project.tracking ? 'Saving manual correction…' : 'Saving anchor and extracting templates…');
    try {
      const updatedProject = project.tracking
        ? await saveManualAnchor(project.id, currentFrame, currentTime, displaySelection)
        : await saveWatermarkAnchor(project.id, currentFrame, currentTime, displaySelection, label);
      setProject(updatedProject);
      setSelection(project.tracking ? null : updatedProject.watermark.anchor?.bbox ?? null);
      setSelectionMode(false);
      setMessage(project.tracking ? 'Manual frame saved and locked. Click Re-track section.' : 'Anchor saved. Templates and mask are ready.');
    } catch (saveError) {
      setError(getErrorMessage(saveError));
      setMessage('Could not save anchor');
    } finally {
      setLoadingTask(null);
    }
  };

  const runAnalyzeTrack = async () => {
    if (!project || isBusy || !project.watermark.anchor) return;
    setError(null); setLoadingTask('tracking'); setProgress(null); setMessage('Analyzing forward and backward tracking…');
    try {
      const updatedProject = await analyzeTrack(project.id);
      setProject(updatedProject);
      setSelection(null);
      setMessage(`Tracking complete. ${updatedProject.tracking?.problemRanges.length ?? 0} problem range(s) need review.`);
    } catch (trackingError) { setError(getErrorMessage(trackingError)); setMessage('Tracking failed'); }
    finally { setLoadingTask(null); }
  };

  const runRetrack = async () => {
    if (!project || !project.tracking || isBusy) return;
    setError(null); setLoadingTask('tracking'); setProgress(null); setMessage(`Re-tracking around frame ${currentFrame}…`);
    try {
      const updatedProject = await retrackTrack(project.id, currentFrame);
      setProject(updatedProject);
      setSelection(null);
      setMessage(`Section re-tracked. ${updatedProject.tracking?.problemRanges.length ?? 0} problem range(s) remain.`);
    } catch (trackingError) { setError(getErrorMessage(trackingError)); setMessage('Re-track failed'); }
    finally { setLoadingTask(null); }
  };

  const cancelBusy = () => {
    if (loadingTask === 'tracking') void cancelTracking();
    if (loadingTask === 'rendering') void cancelRender();
  };

  const nextProblem = () => {
    if (!project?.tracking) return;
    const next = project.tracking.problemRanges.find((range) => range.worstFrame > currentFrame)
      ?? project.tracking.problemRanges[0];
    if (next) { setVideoFrame(next.worstFrame); setMessage(`Review frame ${next.worstFrame} (${next.startFrame}–${next.endFrame}).`); }
  };

  const interpolateCurrentRange = async () => {
    if (!project?.tracking || isBusy) return;
    const range = project.tracking.problemRanges.find((item) => currentFrame >= item.startFrame && currentFrame <= item.endFrame) ?? project.tracking.problemRanges[0];
    if (!range) return;
    setLoadingTask('saving'); setError(null); setMessage(`Interpolating frames ${range.startFrame}–${range.endFrame} between locked anchors…`);
    try { setProject(await interpolateTrackingRange(project.id, range.startFrame, range.endFrame)); setSelection(null); setMessage('Range interpolated and marked explicitly as INTERPOLATED.'); }
    catch (interpolationError) { setError(getErrorMessage(interpolationError)); }
    finally { setLoadingTask(null); }
  };

  const acceptCurrentFrame = async () => {
    if (!project || !currentTracking || isBusy) return;
    setLoadingTask('saving'); setError(null);
    try { setProject(await acceptTrackingFrame(project.id, currentFrame)); setMessage(`Frame ${currentFrame} accepted and locked.`); }
    catch (acceptError) { setError(getErrorMessage(acceptError)); }
    finally { setLoadingTask(null); }
  };

  const markCurrentRangeOccluded = async () => {
    if (!project?.tracking || isBusy) return;
    const range = project.tracking.problemRanges.find((item) => currentFrame >= item.startFrame && currentFrame <= item.endFrame);
    if (!range) return;
    setLoadingTask('saving'); setError(null);
    try {
      setProject(await markOccludedRange(project.id, range.startFrame, range.endFrame));
      setSelection(null);
      setMessage(`Frames ${range.startFrame}–${range.endFrame} marked occluded; rendering is a no-op for this range.`);
    } catch (occludedError) { setError(getErrorMessage(occludedError)); }
    finally { setLoadingTask(null); }
  };

  const chooseReplacement = async () => {
    const path = await chooseReplacementPath();
    if (path) setRemoval((current) => ({ ...current, mode: 'REPLACEMENT', replacementPath: path }));
  };

  const render = async () => {
    if (!project || isBusy) return;
    setError(null); setLoadingTask('rendering'); setMessage('Rendering full-resolution output…');
    try {
      const saved = await saveRemovalConfig(project.id, removal);
      setProject(saved);
      const result = await renderVideo(saved.id, removal);
      setMessage(`Render complete: ${result.outputPath}`);
    } catch (renderError) { setError(getErrorMessage(renderError)); setMessage('Render failed'); }
    finally { setLoadingTask(null); }
  };

  const togglePlayback = () => {
    if (!videoRef.current || !project) return;
    if (playing) videoRef.current.pause(); else void videoRef.current.play();
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
        <div className="brand"><div className="brand-mark">W</div><div><strong>Watermark Studio</strong><span>Desktop beta</span></div></div>
        <nav className="nav-stack">
          <button className="nav-item active"><Icon>◫</Icon> Project</button>
          <button className="nav-item" onClick={() => void runAnalyzeTrack()} disabled={!project || isBusy || !project.watermark.anchor}><Icon>⌁</Icon> Track</button>
          <button className="nav-item" onClick={() => setMessage('Choose a removal mode in the inspector.')} disabled={!project?.tracking}><Icon>◌</Icon> Remove</button>
          <button className="nav-item" disabled title="Available in a later phase"><Icon>⚙</Icon> Settings</button>
        </nav>
        <div className="sidebar-card">
          <span className="eyebrow">CURRENT FILE</span>
          <strong title={project?.source.fileName ?? 'No video selected'}>{project?.source.fileName ?? 'No video selected'}</strong>
          <div className="file-meta"><span>{project ? `${project.video.durationSeconds.toFixed(2)}s` : '—'}</span><span>{project ? `${project.video.width} × ${project.video.height}` : '—'}</span><span>{project ? `${project.video.fps.toFixed(3)} fps` : '—'}</span></div>
        </div>
        <div className="sidebar-bottom"><div className="status-dot" /><span>{message}</span></div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><span className="eyebrow">PROJECT</span><h1>Moving Watermark Removal</h1></div>
          <div className="top-actions">
            <button className="button secondary" onClick={() => void chooseAndOpenVideo()} disabled={isBusy}>{loadingTask === 'opening' ? 'Opening…' : 'Open video'}</button>
            <button className="button secondary" onClick={() => void runAnalyzeTrack()} disabled={!project || isBusy || !project.watermark.anchor}>{loadingTask === 'tracking' ? 'Tracking…' : 'Analyze track'}</button>
          </div>
        </header>
        {error && <div className="error-banner" role="alert">{error}</div>}

        <div className="editor-grid">
          <section className="viewer-panel panel">
            <div className="panel-toolbar"><div className="segmented"><button className={selectionMode ? 'selected' : ''} onClick={() => setSelectionMode(true)} disabled={!project || isBusy}>□ Select watermark</button><button disabled title="Paint Mask is planned for a later phase">✎ Paint mask</button></div><div className="zoom">Source resolution</div></div>
            <div className="viewer-stage">
              {project && videoUrl ? (
                <div className="video-frame" ref={videoFrameRef} style={{ aspectRatio: `${project.video.width}/${project.video.height}` }}>
                  <video ref={videoRef} key={videoUrl} src={videoUrl} controls={false} playsInline onLoadedMetadata={onVideoLoadedMetadata} onTimeUpdate={onVideoTimeUpdate} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
                  {contentRect && <div ref={selectionSurfaceRef} className={`selection-surface${selectionMode ? ' editable' : ''}`} style={{ left: contentRect.left, top: contentRect.top, width: contentRect.width, height: contentRect.height }} onPointerDown={onSelectionPointerDown} onPointerMove={onSelectionPointerMove} onPointerUp={onSelectionPointerUp}>
                    {displaySelection && <div className="watermark-box" style={{ left: `${sourceToPercent(displaySelection.x, project.video.width)}%`, top: `${sourceToPercent(displaySelection.y, project.video.height)}%`, width: `${sourceToPercent(displaySelection.width, project.video.width)}%`, height: `${sourceToPercent(displaySelection.height, project.video.height)}%` }}><span>{project.watermark.label ?? label}</span><i className="handle h1" /><i className="handle h2" /><i className="handle h3" /><i className="handle h4" /></div>}
                  </div>}
                </div>
              ) : <button className="dropzone" onClick={() => void chooseAndOpenVideo()} disabled={isBusy}><span className="drop-icon">＋</span><strong>Drop a video here</strong><span>or click to choose a file</span><small>MP4, MOV, MKV, WEBM, M4V</small></button>}
            </div>
            <div className="transport"><button onClick={() => setVideoFrame(0)} disabled={!project}>│◀</button><button onClick={() => setVideoFrame(currentFrame - 1)} disabled={!project}>◀</button><button className="play" onClick={togglePlayback} disabled={!project}>{playing ? 'Ⅱ' : '▶'}</button><button onClick={() => setVideoFrame(currentFrame + 1)} disabled={!project}>▶</button><button onClick={() => setVideoFrame(project ? project.video.frameCount - 1 : 0)} disabled={!project}>▶│</button><label className="transport-input">Frame <input type="number" min="0" max={Math.max(0, (project?.video.frameCount ?? 1) - 1)} value={project ? currentFrame : ''} onChange={(event) => setVideoFrame(Number(event.target.value))} disabled={!project} /></label><label className="transport-input">Time <input type="number" min="0" max={project?.video.durationSeconds ?? 0} step="0.001" value={project ? currentTime.toFixed(3) : ''} onChange={(event) => setVideoTime(Number(event.target.value))} disabled={!project} /></label></div>
          </section>

          <aside className="inspector panel">
            <div className="inspector-title"><div><span className="eyebrow">WATERMARK</span><h2>{project?.tracking ? 'Track & remove' : 'Manual enrollment'}</h2></div><span className="badge">{project?.tracking ? 'Phase 2' : 'Phase 1'}</span></div>
            <label className="field"><span>Label</span><input value={label} onChange={(event) => setLabel(event.target.value)} disabled={!project || isBusy} /></label>
            <div className="field-row"><label className="field"><span>X · source px</span><input value={displaySelection?.x.toFixed(1) ?? '—'} readOnly /></label><label className="field"><span>Y · source px</span><input value={displaySelection?.y.toFixed(1) ?? '—'} readOnly /></label></div>
            <div className="field-row"><label className="field"><span>Width · source px</span><input value={displaySelection?.width.toFixed(1) ?? '—'} readOnly /></label><label className="field"><span>Height · source px</span><input value={displaySelection?.height.toFixed(1) ?? '—'} readOnly /></label></div>
            <div className="debug-card"><span>Frame</span><strong>{project ? currentFrame : '—'}</strong><span>Time</span><strong>{project ? formatTime(currentTime) : '—'}</strong><span>Source</span><strong>{project ? `${project.video.width} × ${project.video.height}` : '—'}</strong><span>Frames</span><strong>{project ? project.video.frameCount : '—'}</strong><span>Padding</span><strong>{project ? `${project.watermark.templatePadding} px` : '—'}</strong>{currentTracking && <><span>Confidence</span><strong>{(currentTracking.confidence * 100).toFixed(1)}%</strong><span>Status</span><strong>{currentTracking.status}</strong></>}</div>
            <div className="divider" />
            <label className="field"><span>Tracking</span><select value={currentTracking?.status ?? 'NOT_ANALYZED'} disabled><option value="NOT_ANALYZED">{project?.tracking ? 'Select a frame to inspect' : 'Not analyzed'}</option><option value="AUTO_GOOD">Auto good</option><option value="AUTO_WEAK">Auto weak</option><option value="NEED_REVIEW">Need review</option><option value="MANUAL">Manual locked</option><option value="INTERPOLATED">Interpolated</option><option value="OCCLUDED">Occluded (no-op)</option></select></label>
            <label className="field"><span>Removal</span><select value={removal.mode} onChange={(event) => setRemoval((current) => ({ ...current, mode: event.target.value as RemovalConfig['mode'] }))} disabled={!project?.tracking || isBusy}><option value="AUTO_BEST">Auto Best</option><option value="TEMPORAL_RESTORE">Temporal Restore</option><option value="REPLACEMENT">Replacement PNG</option><option value="BLUR">Blur mask</option><option value="INPAINT">Spatial inpaint</option></select></label>
            {removal.mode === 'REPLACEMENT' && <><button className="button secondary full" onClick={() => void chooseReplacement()} disabled={isBusy}>{removal.replacementPath ? 'Change replacement PNG' : 'Choose replacement PNG'}</button><small className="path-hint">{removal.replacementPath ?? 'No PNG selected'}</small></>}
            {(removal.mode === 'TEMPORAL_RESTORE' || removal.mode === 'AUTO_BEST') && <div className="advanced-card"><span className="eyebrow">RESTORATION SETTINGS</span><div className="field-row"><label className="field"><span>Before</span><input type="number" min="1" max="32" value={removal.temporalWindowBefore} onChange={(event) => setRemoval((current) => ({ ...current, temporalWindowBefore: Number(event.target.value) }))} disabled={isBusy} /></label><label className="field"><span>After</span><input type="number" min="1" max="32" value={removal.temporalWindowAfter} onChange={(event) => setRemoval((current) => ({ ...current, temporalWindowAfter: Number(event.target.value) }))} disabled={isBusy} /></label></div><div className="field-row"><label className="field"><span>Max candidates</span><input type="number" min="2" max="16" value={removal.maxTemporalCandidates} onChange={(event) => setRemoval((current) => ({ ...current, maxTemporalCandidates: Number(event.target.value) }))} disabled={isBusy} /></label><label className="field"><span>ROI padding</span><input type="number" min="8" max="96" value={removal.restorationRoiPadding} onChange={(event) => setRemoval((current) => ({ ...current, restorationRoiPadding: Number(event.target.value) }))} disabled={isBusy} /></label></div><label className="field"><span>Artifact limit</span><input type="number" min="0.05" max="0.95" step="0.05" value={removal.artifactThreshold} onChange={(event) => setRemoval((current) => ({ ...current, artifactThreshold: Number(event.target.value) }))} disabled={isBusy} /></label><label className="field"><span>Fallback</span><select value={removal.fallbackPolicy} onChange={(event) => setRemoval((current) => ({ ...current, fallbackPolicy: event.target.value as RemovalConfig['fallbackPolicy'] }))} disabled={isBusy}><option value="TEMPORAL_INPAINT_BLUR">Inpaint → Blur</option><option value="BLUR_ONLY">Blur only</option></select></label></div>}
            <div className="confidence-card"><div><span>{project?.tracking ? 'Current tracking' : 'Phase 1 status'}</span><strong>{currentTracking?.status ?? (project?.watermark.templates ? 'Ready' : project?.watermark.anchor ? 'Anchor saved' : 'Manual anchor')}</strong></div><div className="confidence-bar"><i style={{ width: `${project?.tracking ? (currentTracking?.confidence ?? 0) * 100 : project?.watermark.templates ? 100 : project?.watermark.anchor ? 65 : 8}%` }} /></div><small>{project?.tracking ? `${unresolvedCount} problem range(s); weak/review frames are blocked from rendering. Occluded frames are preserved unchanged.` : project?.watermark.templates ? 'Templates and mask persisted in the project workspace.' : 'Select a source-coordinate box to begin.'}</small></div>
            <button className="button secondary full" onClick={() => void saveAnchor()} disabled={!project || !displaySelection || isBusy}>{loadingTask === 'saving' ? 'Saving…' : project?.tracking ? 'Save manual correction' : 'Save anchor & templates'}</button>
            {project?.tracking && currentTracking && <><button className="button secondary full" onClick={() => void acceptCurrentFrame()} disabled={isBusy || currentTracking.status === 'MANUAL'}>Accept current frame</button><button className="button secondary full" onClick={() => void markCurrentRangeOccluded()} disabled={isBusy || !project.tracking.problemRanges.some((item) => currentFrame >= item.startFrame && currentFrame <= item.endFrame)}>Mark current range occluded (no-op)</button><button className="button secondary full" onClick={() => void runRetrack()} disabled={isBusy}>Re-track section</button><button className="button secondary full" onClick={() => void interpolateCurrentRange()} disabled={isBusy || !project.tracking.problemRanges.length}>Interpolate current range</button></>}
            <button className="button primary full" onClick={() => void render()} disabled={!project?.tracking || isBusy || unresolvedCount > 0}>{loadingTask === 'rendering' ? 'Rendering…' : 'Render video'}</button>
            {isBusy && (loadingTask === 'tracking' || loadingTask === 'rendering') && <button className="button secondary full" onClick={cancelBusy}>Cancel operation</button>}
          </aside>
        </div>

        <section className="timeline-panel panel">
          <div className="timeline-header"><div><span className="eyebrow">NAVIGATION</span><h2>Frame timeline</h2></div><div className="timeline-actions"><span>{project ? `${project.video.frameCount} frames` : 'No video loaded'}{progress ? ` · ${Math.round(progress.progress * 100)}% ${progress.phase}` : ''}</span><button onClick={nextProblem} disabled={!project?.tracking?.problemRanges.length || isBusy}>Next problem →</button></div></div>
          <div className="timeline-ruler"><span>00:00</span><span>{formatTime((project?.video.durationSeconds ?? 0) * 0.25)}</span><span>{formatTime((project?.video.durationSeconds ?? 0) * 0.5)}</span><span>{formatTime((project?.video.durationSeconds ?? 0) * 0.75)}</span><span>{formatTime(project?.video.durationSeconds ?? 0)}</span></div>
          <div className="timeline-track" onClick={clickTimeline} role="slider" aria-label="Video timeline" aria-valuemin={0} aria-valuemax={project ? Math.max(0, project.video.frameCount - 1) : 0} aria-valuenow={currentFrame} tabIndex={project ? 0 : -1}><div className="track-good" />{project?.tracking && trackingSegments(project.tracking.frames).map((segment) => <i className="track-segment" key={`${segment.start}-${segment.end}`} title={`${segment.status}: frames ${segment.start}–${segment.end}`} style={{ left: `${(segment.start / project.video.frameCount) * 100}%`, width: `${((segment.end - segment.start + 1) / project.video.frameCount) * 100}%`, background: trackingColor(segment.status) }} />)}<i className="playhead" style={{ left: project && project.video.frameCount > 1 ? `${(currentFrame / (project.video.frameCount - 1)) * 100}%` : '0%' }} /></div>
          <div className="timeline-hint">Click vào timeline hoặc dùng ◀ / ▶ để điều hướng từng frame. {project ? `Frame hiện tại: ${currentFrame}.` : ''} {project?.tracking ? `Problem ranges: ${unresolvedCount}. Manual frames are locked.` : ''}</div>
        </section>
      </section>
    </main>
  );
}
