use crate::media::ffmpeg::GrayFrame;
use crate::project::model::{BoundingBox, TrackingScores};
use image::imageops::{self, FilterType};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FeatureImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TemplateBank {
    pub gray: FeatureImage,
    pub highpass: FeatureImage,
    pub edge: FeatureImage,
    pub sample_points: Vec<(u32, u32)>,
    pub means: [f64; 3],
    pub deviations: [f64; 3],
}

impl TemplateBank {
    pub fn from_path(path: &Path, width: u32, height: u32) -> Result<Self, crate::error::AppError> {
        let source = image::open(path)
            .map_err(|error| {
                crate::error::AppError::Io(format!("Unable to read template: {error}"))
            })?
            .to_luma8();
        let resized = imageops::resize(&source, width.max(8), height.max(8), FilterType::Lanczos3);
        let gray = FeatureImage {
            width: resized.width(),
            height: resized.height(),
            pixels: resized.into_raw(),
        };
        let highpass = highpass(&gray);
        let edge = edge(&gray);
        let sample_points = sample_points(gray.width, gray.height);
        let means = [
            mean(&gray, &sample_points),
            mean(&highpass, &sample_points),
            mean(&edge, &sample_points),
        ];
        let deviations = [
            deviation(&gray, &sample_points, means[0]),
            deviation(&highpass, &sample_points, means[1]),
            deviation(&edge, &sample_points, means[2]),
        ];
        Ok(Self {
            gray,
            highpass,
            edge,
            sample_points,
            means,
            deviations,
        })
    }
}

pub fn frame_features(frame: &GrayFrame) -> (FeatureImage, FeatureImage) {
    let gray = FeatureImage {
        width: frame.width,
        height: frame.height,
        pixels: frame.pixels.clone(),
    };
    (highpass(&gray), edge(&gray))
}

pub fn find_best(
    frame: &GrayFrame,
    features: &(FeatureImage, FeatureImage),
    bank: &TemplateBank,
    predicted: &BoundingBox,
    radius: u32,
    global: bool,
) -> Option<(BoundingBox, TrackingScores)> {
    let template_width = bank.gray.width;
    let template_height = bank.gray.height;
    if template_width >= frame.width || template_height >= frame.height {
        return None;
    }

    let gray_frame = FeatureImage {
        width: frame.width,
        height: frame.height,
        pixels: frame.pixels.clone(),
    };
    let predicted_x = predicted.x + predicted.width / 2.0 - f64::from(template_width) / 2.0;
    let predicted_y = predicted.y + predicted.height / 2.0 - f64::from(template_height) / 2.0;
    let (min_x, min_y, max_x, max_y, step) = if global {
        (
            0,
            0,
            frame.width.saturating_sub(template_width),
            frame.height.saturating_sub(template_height),
            3,
        )
    } else {
        (
            (predicted_x - f64::from(radius)).max(0.0) as u32,
            (predicted_y - f64::from(radius)).max(0.0) as u32,
            (predicted_x + f64::from(radius)).min(f64::from(frame.width - template_width)) as u32,
            (predicted_y + f64::from(radius)).min(f64::from(frame.height - template_height)) as u32,
            2,
        )
    };
    if min_x > max_x || min_y > max_y {
        return None;
    }

    let mut best: Option<(f64, u32, u32, [f64; 3])> = None;
    let mut second_best = 0.0;
    for y in (min_y..=max_y).step_by(step as usize) {
        for x in (min_x..=max_x).step_by(step as usize) {
            let scores = [
                ncc(
                    &bank.gray,
                    &gray_frame,
                    x,
                    y,
                    &bank.sample_points,
                    bank.means[0],
                    bank.deviations[0],
                ),
                ncc(
                    &bank.highpass,
                    &features.0,
                    x,
                    y,
                    &bank.sample_points,
                    bank.means[1],
                    bank.deviations[1],
                ),
                ncc(
                    &bank.edge,
                    &features.1,
                    x,
                    y,
                    &bank.sample_points,
                    bank.means[2],
                    bank.deviations[2],
                ),
            ];
            let combined = scores[0] * 0.35 + scores[1] * 0.40 + scores[2] * 0.25;
            if best.as_ref().is_none_or(|candidate| combined > candidate.0) {
                second_best = best.as_ref().map(|candidate| candidate.0).unwrap_or(0.0);
                best = Some((combined, x, y, scores));
            } else if combined > second_best {
                second_best = combined;
            }
        }
    }
    let (best_score, x, y, scores) = best?;
    let found = BoundingBox {
        x: f64::from(x),
        y: f64::from(y),
        width: f64::from(template_width),
        height: f64::from(template_height),
    };
    let predicted_center = (
        predicted.x + predicted.width / 2.0,
        predicted.y + predicted.height / 2.0,
    );
    let found_center = (found.x + found.width / 2.0, found.y + found.height / 2.0);
    let distance = ((predicted_center.0 - found_center.0).powi(2)
        + (predicted_center.1 - found_center.1).powi(2))
    .sqrt();
    let motion = (1.0 - distance / (f64::from(radius.max(1)) * 2.0 + 1.0)).clamp(0.0, 1.0);
    Some((
        found,
        TrackingScores {
            template: scores[0],
            highpass: scores[1],
            edge: scores[2],
            motion,
            position: motion,
            size: 1.0,
            optical_flow: None,
            forward_backward: None,
            motion_smoothness: None,
            match_margin: Some((best_score - second_best).clamp(0.0, 1.0)),
        },
    ))
}

fn sample_points(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut points = Vec::with_capacity(96);
    for y in 0..8 {
        for x in 0..12 {
            points.push((
                (x * width.saturating_sub(1)) / 11,
                (y * height.saturating_sub(1)) / 7,
            ));
        }
    }
    points
}

fn ncc(
    template: &FeatureImage,
    frame: &FeatureImage,
    x: u32,
    y: u32,
    points: &[(u32, u32)],
    mean_t: f64,
    dev_t: f64,
) -> f64 {
    if dev_t < 1.0 || points.is_empty() {
        return 0.0;
    }
    let mut values = Vec::with_capacity(points.len());
    for &(px, py) in points {
        values.push(frame.pixels[((y + py) * frame.width + x + px) as usize] as f64);
    }
    let mean_f = values.iter().sum::<f64>() / values.len() as f64;
    let dev_f = (values
        .iter()
        .map(|value| (value - mean_f).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    if dev_f < 1.0 {
        return 0.0;
    }
    let numerator = points
        .iter()
        .enumerate()
        .map(|(index, &(px, py))| {
            (template.pixels[(py * template.width + px) as usize] as f64 - mean_t)
                * (values[index] - mean_f)
        })
        .sum::<f64>();
    ((numerator / points.len() as f64 / dev_t / dev_f + 1.0) / 2.0).clamp(0.0, 1.0)
}

fn mean(image: &FeatureImage, points: &[(u32, u32)]) -> f64 {
    points
        .iter()
        .map(|(x, y)| image.pixels[(y * image.width + x) as usize] as f64)
        .sum::<f64>()
        / points.len() as f64
}

fn deviation(image: &FeatureImage, points: &[(u32, u32)], mean: f64) -> f64 {
    (points
        .iter()
        .map(|(x, y)| (image.pixels[(y * image.width + x) as usize] as f64 - mean).powi(2))
        .sum::<f64>()
        / points.len() as f64)
        .sqrt()
}

fn highpass(image: &FeatureImage) -> FeatureImage {
    let mut pixels = vec![0u8; image.pixels.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let index = (y * image.width + x) as usize;
            let mut sum = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, image.width as i32 - 1) as u32;
                    let yy = (y as i32 + dy).clamp(0, image.height as i32 - 1) as u32;
                    sum += image.pixels[(yy * image.width + xx) as usize] as u32;
                }
            }
            pixels[index] = image.pixels[index].abs_diff((sum / 9) as u8);
        }
    }
    FeatureImage {
        width: image.width,
        height: image.height,
        pixels,
    }
}

fn edge(image: &FeatureImage) -> FeatureImage {
    let mut pixels = vec![0u8; image.pixels.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let left = image.pixels[(y * image.width + x.saturating_sub(1)) as usize] as i16;
            let right =
                image.pixels[(y * image.width + (x + 1).min(image.width - 1)) as usize] as i16;
            let up = image.pixels[(y.saturating_sub(1) * image.width + x) as usize] as i16;
            let down =
                image.pixels[((y + 1).min(image.height - 1) * image.width + x) as usize] as i16;
            pixels[(y * image.width + x) as usize] =
                ((right - left).abs() + (down - up).abs()).min(255) as u8;
        }
    }
    FeatureImage {
        width: image.width,
        height: image.height,
        pixels,
    }
}
