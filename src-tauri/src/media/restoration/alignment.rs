use image::RgbImage;

use crate::project::model::BoundingBox;

#[derive(Debug, Clone, Copy)]
pub struct Translation {
    pub dx: i32,
    pub dy: i32,
    pub error: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AlignmentRegion<'a> {
    pub target_bbox: &'a BoundingBox,
    pub candidate_bbox: &'a BoundingBox,
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
    pub radius: i32,
}

/// Estimates a small local translation using scene pixels around the tracked
/// region. The watermark rectangle itself is excluded from the comparison.
pub fn estimate_translation(
    target: &RgbImage,
    candidate: &RgbImage,
    region: AlignmentRegion<'_>,
) -> Translation {
    let radius = region.radius.clamp(0, 12);
    let mut best = Translation {
        dx: 0,
        dy: 0,
        error: f64::MAX,
    };
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let mut total = 0.0;
            let mut count = 0u64;
            for y in (region.y0 - 12..region.y0 + region.height as i32 + 12).step_by(4) {
                for x in (region.x0 - 12..region.x0 + region.width as i32 + 12).step_by(4) {
                    if inside_bbox(x, y, region.target_bbox, 4.0)
                        || inside_bbox(x + dx, y + dy, region.candidate_bbox, 8.0)
                    {
                        continue;
                    }
                    let cx = x + dx;
                    let cy = y + dy;
                    if !in_bounds(target, x, y) || !in_bounds(candidate, cx, cy) {
                        continue;
                    }
                    let a = target.get_pixel(x as u32, y as u32);
                    let b = candidate.get_pixel(cx as u32, cy as u32);
                    total += (i32::from(a[0]) - i32::from(b[0])).unsigned_abs() as f64
                        + (i32::from(a[1]) - i32::from(b[1])).unsigned_abs() as f64
                        + (i32::from(a[2]) - i32::from(b[2])).unsigned_abs() as f64;
                    count += 3;
                }
            }
            if count > 0 {
                let error = (total / count as f64 / 255.0).clamp(0.0, 1.0);
                if error < best.error {
                    best = Translation { dx, dy, error };
                }
            }
        }
    }
    if best.error == f64::MAX {
        Translation {
            dx: 0,
            dy: 0,
            error: 1.0,
        }
    } else {
        best
    }
}

fn inside_bbox(x: i32, y: i32, bbox: &BoundingBox, padding: f64) -> bool {
    x as f64 >= bbox.x - padding
        && x as f64 <= bbox.x + bbox.width + padding
        && y as f64 >= bbox.y - padding
        && y as f64 <= bbox.y + bbox.height + padding
}

fn in_bounds(image: &RgbImage, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height()
}
