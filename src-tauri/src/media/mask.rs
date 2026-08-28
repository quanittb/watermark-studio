use crate::error::AppError;
use image::{imageops, GrayImage, Luma};
use std::path::Path;

pub fn generate_template_mask(input_path: &Path, output_path: &Path) -> Result<(), AppError> {
    let source = image::open(input_path)
        .map_err(|error| AppError::Io(format!("Unable to read template: {error}")))?
        .to_luma8();
    let blurred = imageops::blur(&source, 1.4);
    let mut mask = GrayImage::new(source.width(), source.height());
    for y in 0..source.height() {
        for x in 0..source.width() {
            let value = source.get_pixel(x, y)[0];
            let local = value.abs_diff(blurred.get_pixel(x, y)[0]);
            let left = source.get_pixel(x.saturating_sub(1), y)[0];
            let right = source.get_pixel((x + 1).min(source.width() - 1), y)[0];
            let up = source.get_pixel(x, y.saturating_sub(1))[0];
            let down = source.get_pixel(x, (y + 1).min(source.height() - 1))[0];
            let edge = ((i16::from(right) - i16::from(left)).abs()
                + (i16::from(down) - i16::from(up)).abs())
            .min(255) as u8;
            let strength = local.saturating_mul(3).max(edge);
            mask.put_pixel(
                x,
                y,
                Luma([if strength > 18 {
                    strength.saturating_add(30)
                } else {
                    0
                }]),
            );
        }
    }
    let mask = imageops::blur(&mask, 1.0);
    mask.save(output_path)
        .map_err(|error| AppError::Io(format!("Unable to write template mask: {error}")))
}

pub fn load_mask(path: &Path) -> Result<GrayImage, AppError> {
    image::open(path)
        .map(|image| image.to_luma8())
        .map_err(|error| AppError::Io(format!("Unable to read mask: {error}")))
}

/// Turns broken edge strokes into a soft, filled coverage mask. The template
/// mask is intentionally edge-driven, but applying it as-is can leave the
/// interiors of wide glyphs untouched and therefore still readable.
pub fn solidify_mask(mask: &GrayImage, threshold: u8, close_radius: u32) -> GrayImage {
    if mask.width() == 0 || mask.height() == 0 {
        return mask.clone();
    }
    let mut binary = GrayImage::new(mask.width(), mask.height());
    for (x, y, pixel) in mask.enumerate_pixels() {
        binary.put_pixel(x, y, Luma([if pixel[0] >= threshold { 255 } else { 0 }]));
    }
    let closed = if close_radius == 0 {
        binary
    } else {
        let expanded = morphology_dilate(&binary, close_radius);
        morphology_erode(&expanded, close_radius)
    };
    // The template mask is edge-derived. Filling enclosed holes turns closed
    // glyph outlines into coverage masks, so the glyph interior is restored as
    // well instead of leaving the watermark readable in its center.
    fill_enclosed_regions(&closed)
}

fn fill_enclosed_regions(mask: &GrayImage) -> GrayImage {
    let width = mask.width() as usize;
    let height = mask.height() as usize;
    if width == 0 || height == 0 {
        return mask.clone();
    }

    let mut outside = vec![false; width * height];
    let mut pending = std::collections::VecDeque::new();
    let enqueue = |x: usize,
                   y: usize,
                   outside: &mut [bool],
                   pending: &mut std::collections::VecDeque<(usize, usize)>| {
        let index = y * width + x;
        if mask.get_pixel(x as u32, y as u32)[0] == 0 && !outside[index] {
            outside[index] = true;
            pending.push_back((x, y));
        }
    };

    for x in 0..width {
        enqueue(x, 0, &mut outside, &mut pending);
        enqueue(x, height - 1, &mut outside, &mut pending);
    }
    for y in 0..height {
        enqueue(0, y, &mut outside, &mut pending);
        enqueue(width - 1, y, &mut outside, &mut pending);
    }

    while let Some((x, y)) = pending.pop_front() {
        for (nx, ny) in [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ] {
            if nx >= width || ny >= height {
                continue;
            }
            enqueue(nx, ny, &mut outside, &mut pending);
        }
    }

    let mut filled = mask.clone();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if mask.get_pixel(x as u32, y as u32)[0] == 0 && !outside[index] {
                filled.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    filled
}

fn morphology_dilate(mask: &GrayImage, radius: u32) -> GrayImage {
    let mut output = GrayImage::new(mask.width(), mask.height());
    let radius = radius as i32;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let mut value = 0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let xx = (x as i32 + dx).clamp(0, mask.width() as i32 - 1) as u32;
                    let yy = (y as i32 + dy).clamp(0, mask.height() as i32 - 1) as u32;
                    value = value.max(mask.get_pixel(xx, yy)[0]);
                }
            }
            output.put_pixel(x, y, Luma([value]));
        }
    }
    output
}

fn morphology_erode(mask: &GrayImage, radius: u32) -> GrayImage {
    let mut output = GrayImage::new(mask.width(), mask.height());
    let radius = radius as i32;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let mut value = u8::MAX;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let xx = (x as i32 + dx).clamp(0, mask.width() as i32 - 1) as u32;
                    let yy = (y as i32 + dy).clamp(0, mask.height() as i32 - 1) as u32;
                    value = value.min(mask.get_pixel(xx, yy)[0]);
                }
            }
            output.put_pixel(x, y, Luma([value]));
        }
    }
    output
}

pub fn dilate_mask(mask: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 || mask.width() == 0 || mask.height() == 0 {
        return mask.clone();
    }
    let mut dilated = GrayImage::new(mask.width(), mask.height());
    let radius = radius as i32;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let mut value = 0u8;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let distance = dx * dx + dy * dy;
                    if distance > radius * radius {
                        continue;
                    }
                    let xx = (x as i32 + dx).clamp(0, mask.width() as i32 - 1) as u32;
                    let yy = (y as i32 + dy).clamp(0, mask.height() as i32 - 1) as u32;
                    value = value.max(mask.get_pixel(xx, yy)[0]);
                }
            }
            dilated.put_pixel(x, y, Luma([value]));
        }
    }
    dilated
}

#[cfg(test)]
mod tests {
    use super::{dilate_mask, solidify_mask};
    use image::{GrayImage, Luma};

    #[test]
    fn dilation_expands_soft_mask_without_creating_a_full_rectangle() {
        let mut mask = GrayImage::new(9, 9);
        mask.put_pixel(4, 4, Luma([200]));

        let dilated = dilate_mask(&mask, 2);

        assert_eq!(dilated.get_pixel(4, 4)[0], 200);
        assert_eq!(dilated.get_pixel(4, 2)[0], 200);
        assert_eq!(dilated.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn solidify_mask_creates_a_binary_coverage_mask() {
        let mut mask = GrayImage::new(5, 5);
        for x in 0..5 {
            mask.put_pixel(x, 0, Luma([200]));
            mask.put_pixel(x, 4, Luma([200]));
        }
        for y in 0..5 {
            mask.put_pixel(0, y, Luma([200]));
            mask.put_pixel(4, y, Luma([200]));
        }

        let solid = solidify_mask(&mask, 18, 1);

        assert_eq!(solid.get_pixel(0, 0)[0], 255);
        assert_eq!(solid.get_pixel(2, 2)[0], 255);
    }
}
