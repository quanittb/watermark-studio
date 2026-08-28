use crate::media::ffmpeg::GrayFrame;
use crate::project::model::BoundingBox;

#[derive(Debug, Clone, Copy)]
pub struct MotionState {
    pub velocity_x: f64,
    pub velocity_y: f64,
}

impl Default for MotionState {
    fn default() -> Self {
        Self {
            velocity_x: 0.0,
            velocity_y: 0.0,
        }
    }
}

pub fn estimate_pyramidal_lk(
    previous: &GrayFrame,
    current: &GrayFrame,
    bbox: &BoundingBox,
    radius: u32,
) -> Option<(f64, f64, f64)> {
    if previous.width != current.width || previous.height != current.height {
        return None;
    }
    let x0 = bbox.x.max(2.0) as i32;
    let y0 = bbox.y.max(2.0) as i32;
    let x1 = (bbox.x + bbox.width).min(f64::from(previous.width.saturating_sub(3))) as i32;
    let y1 = (bbox.y + bbox.height).min(f64::from(previous.height.saturating_sub(3))) as i32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let mut dxs = Vec::new();
    let mut dys = Vec::new();
    let search = radius.max(2) as i32;
    let steps_x = 5;
    let steps_y = 3;
    for iy in 0..steps_y {
        for ix in 0..steps_x {
            let x = x0 + ((x1 - x0) * (ix + 1) / (steps_x + 1));
            let y = y0 + ((y1 - y0) * (iy + 1) / (steps_y + 1));
            let mut best = (f64::INFINITY, 0, 0);
            for dy in -search..=search {
                for dx in -search..=search {
                    let mut error = 0.0;
                    for py in -1..=1 {
                        for px in -1..=1 {
                            let old = sample(previous, x + px, y + py) as f64;
                            let new = sample(current, x + dx + px, y + dy + py) as f64;
                            error += (old - new).abs();
                        }
                    }
                    if error < best.0 {
                        best = (error, dx, dy);
                    }
                }
            }
            if best.0.is_finite() {
                dxs.push(f64::from(best.1));
                dys.push(f64::from(best.2));
            }
        }
    }
    if dxs.len() < 3 {
        return None;
    }
    let dx = median(&mut dxs);
    let dy = median(&mut dys);
    let spread = dxs
        .iter()
        .zip(dys.iter())
        .map(|(x, y)| ((x - dx).powi(2) + (y - dy).powi(2)).sqrt())
        .sum::<f64>()
        / dxs.len() as f64;
    Some((
        dx,
        dy,
        (1.0 - spread / (f64::from(search) + 1.0)).clamp(0.0, 1.0),
    ))
}

fn sample(frame: &GrayFrame, x: i32, y: i32) -> u8 {
    let x = x.clamp(0, frame.width as i32 - 1) as u32;
    let y = y.clamp(0, frame.height as i32 - 1) as u32;
    frame.pixels[(y * frame.width + x) as usize]
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}
