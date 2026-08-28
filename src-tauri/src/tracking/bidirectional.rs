use crate::project::model::{
    AnchorType, BoundingBox, ManualAnchor, TrackingConfig, TrackingFrame, TrackingScores,
    TrackingSource, TrackingStatus,
};
use crate::tracking::confidence::{fuse_scores, validated_status};

pub fn initial_anchor(frame: u64, timestamp_seconds: f64, bbox: BoundingBox) -> ManualAnchor {
    ManualAnchor {
        frame,
        timestamp_seconds,
        bbox,
        anchor_type: AnchorType::Initial,
        locked: true,
    }
}

pub fn make_seed(
    anchor: &ManualAnchor,
    source_width: u32,
    source_height: u32,
    analysis_width: u32,
    analysis_height: u32,
    template_padding: u32,
) -> TrackingFrame {
    let scale_x = f64::from(analysis_width) / f64::from(source_width.max(1));
    let scale_y = f64::from(analysis_height) / f64::from(source_height.max(1));
    let padded = expand_bbox(&anchor.bbox, f64::from(template_padding));
    TrackingFrame {
        frame: anchor.frame,
        timestamp_seconds: anchor.timestamp_seconds,
        bbox: scale_bbox(&padded, scale_x, scale_y),
        confidence: 1.0,
        status: TrackingStatus::Manual,
        source: TrackingSource::Manual,
        locked: true,
        scores: perfect_scores(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fuse_tracks(
    forward: &[Option<TrackingFrame>],
    backward: &[Option<TrackingFrame>],
    source_width: u32,
    source_height: u32,
    analysis_width: u32,
    analysis_height: u32,
    fps: f64,
    template_padding: u32,
    config: &TrackingConfig,
) -> Vec<TrackingFrame> {
    let scale_x = f64::from(source_width.max(1)) / f64::from(analysis_width.max(1));
    let scale_y = f64::from(source_height.max(1)) / f64::from(analysis_height.max(1));
    let padding_x = f64::from(template_padding) / scale_x;
    let padding_y = f64::from(template_padding) / scale_y;
    let length = forward.len().max(backward.len());
    (0..length)
        .map(|index| {
            match (
                forward.get(index).and_then(Clone::clone),
                backward.get(index).and_then(Clone::clone),
            ) {
                (Some(a), Some(b)) => {
                    let distance = center_distance(&a.bbox, &b.bbox);
                    let agreement = (1.0 - distance / 18.0).clamp(0.0, 1.0);
                    let mut scores = if a.confidence >= b.confidence {
                        a.scores.clone()
                    } else {
                        b.scores.clone()
                    };
                    scores.forward_backward = Some(agreement);
                    let confidence =
                        (fuse_scores(&scores) * 0.75 + agreement * 0.25).clamp(0.0, 1.0);
                    let use_fused = agreement >= 0.35;
                    let status = if agreement < 0.35 {
                        TrackingStatus::NeedReview
                    } else {
                        validated_status(&scores, confidence, config)
                    };
                    let bbox = if use_fused {
                        average_bbox(&a.bbox, &b.bbox)
                    } else if a.confidence >= b.confidence {
                        a.bbox.clone()
                    } else {
                        b.bbox.clone()
                    };
                    from_analysis(
                        index as u64,
                        bbox,
                        confidence,
                        status,
                        if use_fused {
                            TrackingSource::Fused
                        } else if a.confidence >= b.confidence {
                            TrackingSource::Forward
                        } else {
                            TrackingSource::Backward
                        },
                        a.locked || b.locked,
                        scores,
                        scale_x,
                        scale_y,
                        padding_x,
                        padding_y,
                        fps,
                    )
                }
                (Some(frame), None) => downgrade_one_way_match(convert_frame(
                    frame, scale_x, scale_y, padding_x, padding_y, fps,
                )),
                (None, Some(frame)) => downgrade_one_way_match(convert_frame(
                    frame, scale_x, scale_y, padding_x, padding_y, fps,
                )),
                (None, None) => fallback(index as u64, source_width, source_height, fps),
            }
        })
        .collect()
}

fn downgrade_one_way_match(mut frame: TrackingFrame) -> TrackingFrame {
    if frame.status == TrackingStatus::AutoGood {
        // Without an independent reverse track there is no trajectory
        // agreement. Keep the measured bbox for review but do not certify it.
        frame.status = TrackingStatus::AutoWeak;
    }
    frame
}

pub fn expand_bbox(bbox: &BoundingBox, padding: f64) -> BoundingBox {
    BoundingBox {
        x: bbox.x - padding,
        y: bbox.y - padding,
        width: bbox.width + padding * 2.0,
        height: bbox.height + padding * 2.0,
    }
}

fn convert_frame(
    frame: TrackingFrame,
    scale_x: f64,
    scale_y: f64,
    padding_x: f64,
    padding_y: f64,
    fps: f64,
) -> TrackingFrame {
    let mut converted = frame;
    converted.bbox = actual_bbox(&converted.bbox, scale_x, scale_y, padding_x, padding_y);
    converted.timestamp_seconds = converted.frame as f64 / fps.max(0.000_001);
    converted
}

#[allow(clippy::too_many_arguments)]
fn from_analysis(
    frame: u64,
    bbox: BoundingBox,
    confidence: f64,
    status: TrackingStatus,
    source: TrackingSource,
    locked: bool,
    scores: TrackingScores,
    scale_x: f64,
    scale_y: f64,
    padding_x: f64,
    padding_y: f64,
    fps: f64,
) -> TrackingFrame {
    TrackingFrame {
        frame,
        timestamp_seconds: frame as f64 / fps.max(0.000_001),
        bbox: actual_bbox(&bbox, scale_x, scale_y, padding_x, padding_y),
        confidence,
        status,
        source,
        locked,
        scores,
    }
}

fn actual_bbox(
    bbox: &BoundingBox,
    scale_x: f64,
    scale_y: f64,
    padding_x: f64,
    padding_y: f64,
) -> BoundingBox {
    BoundingBox {
        x: (bbox.x + padding_x) * scale_x,
        y: (bbox.y + padding_y) * scale_y,
        width: (bbox.width - padding_x * 2.0).max(8.0) * scale_x,
        height: (bbox.height - padding_y * 2.0).max(8.0) * scale_y,
    }
}

fn scale_bbox(bbox: &BoundingBox, scale_x: f64, scale_y: f64) -> BoundingBox {
    BoundingBox {
        x: bbox.x * scale_x,
        y: bbox.y * scale_y,
        width: bbox.width * scale_x,
        height: bbox.height * scale_y,
    }
}

fn average_bbox(a: &BoundingBox, b: &BoundingBox) -> BoundingBox {
    BoundingBox {
        x: (a.x + b.x) / 2.0,
        y: (a.y + b.y) / 2.0,
        width: (a.width + b.width) / 2.0,
        height: (a.height + b.height) / 2.0,
    }
}

fn center_distance(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let ax = a.x + a.width / 2.0;
    let ay = a.y + a.height / 2.0;
    let bx = b.x + b.width / 2.0;
    let by = b.y + b.height / 2.0;
    ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt()
}

fn perfect_scores() -> TrackingScores {
    TrackingScores {
        template: 1.0,
        highpass: 1.0,
        edge: 1.0,
        motion: 1.0,
        position: 1.0,
        size: 1.0,
        optical_flow: Some(1.0),
        forward_backward: Some(1.0),
        motion_smoothness: Some(1.0),
        match_margin: Some(1.0),
    }
}

fn fallback(frame: u64, width: u32, height: u32, fps: f64) -> TrackingFrame {
    TrackingFrame {
        frame,
        timestamp_seconds: frame as f64 / fps.max(0.000_001),
        bbox: BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 8.0_f64.min(f64::from(width)),
            height: 8.0_f64.min(f64::from(height)),
        },
        confidence: 0.0,
        status: TrackingStatus::NeedReview,
        source: TrackingSource::Interpolated,
        locked: false,
        scores: TrackingScores {
            template: 0.0,
            highpass: 0.0,
            edge: 0.0,
            motion: 0.0,
            position: 0.0,
            size: 0.0,
            optical_flow: None,
            forward_backward: None,
            motion_smoothness: None,
            match_margin: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::downgrade_one_way_match;
    use crate::project::model::{
        BoundingBox, TrackingFrame, TrackingScores, TrackingSource, TrackingStatus,
    };

    #[test]
    fn one_way_match_cannot_be_auto_good() {
        let frame = TrackingFrame {
            frame: 10,
            timestamp_seconds: 1.0,
            bbox: BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 20.0,
                height: 10.0,
            },
            confidence: 0.95,
            status: TrackingStatus::AutoGood,
            source: TrackingSource::Forward,
            locked: false,
            scores: TrackingScores {
                template: 0.9,
                highpass: 0.9,
                edge: 0.9,
                motion: 0.9,
                position: 0.9,
                size: 1.0,
                optical_flow: Some(0.9),
                forward_backward: None,
                motion_smoothness: Some(0.9),
                match_margin: Some(0.1),
            },
        };

        assert_eq!(
            downgrade_one_way_match(frame).status,
            TrackingStatus::AutoWeak
        );
    }
}
