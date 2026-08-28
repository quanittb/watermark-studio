use image::RgbImage;

#[derive(Debug, Clone, Copy)]
pub struct Translation {
    pub dx: i32,
    pub dy: i32,
    pub error: f64,
}

/// Estimates a small local translation using scene pixels around the tracked
/// region. The watermark rectangle itself is excluded from the comparison.
pub fn estimate_translation(
    target: &RgbImage,
    candidate: &RgbImage,
    x0: i32,
    y0: i32,
    width: u32,
    height: u32,
    radius: i32,
) -> Translation {
    let radius = radius.clamp(0, 12);
    let mut best = Translation {
        dx: 0,
        dy: 0,
        error: f64::MAX,
    };
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let mut total = 0.0;
            let mut count = 0u64;
            for y in (y0 - 12..y0 + height as i32 + 12).step_by(4) {
                for x in (x0 - 12..x0 + width as i32 + 12).step_by(4) {
                    if x >= x0 && x < x0 + width as i32 && y >= y0 && y < y0 + height as i32 {
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

fn in_bounds(image: &RgbImage, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height()
}
