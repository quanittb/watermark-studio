use image::{GrayImage, RgbImage};

pub fn artifact_score(alignment_error: f64, candidate_spread: f64) -> f64 {
    (alignment_error * 0.45 + candidate_spread * 0.55).clamp(0.0, 1.0)
}

/// Estimates residual structure and seam risk for a spatial fallback.
/// This is intentionally conservative and deterministic: it is used only to
/// decide whether Inpaint should be followed by the safer local Blur fallback.
pub fn spatial_artifact_score(
    original: &RgbImage,
    restored: &RgbImage,
    mask: &GrayImage,
    x0: i32,
    y0: i32,
) -> f64 {
    // Edge-derived masks are intentionally sparse so they blend cleanly, but
    // that same sparsity can hide readable glyph interiors from the quality
    // metric. Densify only the analysis mask; the render mask itself remains
    // unchanged, so this cannot expand the pixels being modified.
    let analysis_mask = densify_analysis_mask(mask);
    let mut original_detail = 0.0;
    let mut restored_detail = 0.0;
    let mut detail_count = 0usize;
    let mut boundary_error = 0.0;
    let mut boundary_count = 0usize;
    let mut original_glyph = 0.0;
    let mut original_context = 0.0;
    let mut restored_glyph = 0.0;
    let mut restored_context = 0.0;
    let mut glyph_count = 0usize;
    let mut context_count = 0usize;
    for y in 0..analysis_mask.height() {
        for x in 0..analysis_mask.width() {
            let px = x0 + x as i32;
            let py = y0 + y as i32;
            if !in_bounds(original, px, py) || px < 1 || py < 1 {
                continue;
            }
            let mask_value = analysis_mask.get_pixel(x, y)[0];
            if mask_value >= 180 {
                original_glyph += luma(original.get_pixel(px as u32, py as u32));
                restored_glyph += luma(restored.get_pixel(px as u32, py as u32));
                glyph_count += 1;
            } else if mask_value <= 64 {
                original_context += luma(original.get_pixel(px as u32, py as u32));
                restored_context += luma(restored.get_pixel(px as u32, py as u32));
                context_count += 1;
            }
            if mask_value < 180 {
                continue;
            }
            let Some(restored_energy) = local_detail(restored, px, py) else {
                continue;
            };
            let Some(original_energy) = local_detail(original, px, py) else {
                continue;
            };
            original_detail += original_energy;
            restored_detail += restored_energy;
            detail_count += 1;
            for (nx, ny) in [
                (x as i32 - 1, y as i32),
                (x as i32 + 1, y as i32),
                (x as i32, y as i32 - 1),
                (x as i32, y as i32 + 1),
            ] {
                if nx < 0
                    || ny < 0
                    || nx >= analysis_mask.width() as i32
                    || ny >= analysis_mask.height() as i32
                    || analysis_mask.get_pixel(nx as u32, ny as u32)[0] >= 64
                {
                    continue;
                }
                let bx = x0 + nx;
                let by = y0 + ny;
                if in_bounds(restored, bx, by) {
                    boundary_error += color_distance(
                        restored.get_pixel(px as u32, py as u32),
                        restored.get_pixel(bx as u32, by as u32),
                    );
                    boundary_count += 1;
                }
            }
        }
    }
    if detail_count == 0 {
        return 1.0;
    }
    let source_detail = (original_detail / detail_count as f64).max(0.04);
    let residual_detail = (restored_detail / detail_count as f64 / source_detail).clamp(0.0, 1.0);
    let seam = if boundary_count == 0 {
        0.0
    } else {
        (boundary_error / boundary_count as f64).clamp(0.0, 1.0)
    };
    let glyph_residual = if glyph_count == 0 || context_count == 0 {
        0.0
    } else {
        let original_contrast =
            (original_glyph / glyph_count as f64 - original_context / context_count as f64).abs();
        let restored_contrast =
            (restored_glyph / glyph_count as f64 - restored_context / context_count as f64).abs();
        (restored_contrast / original_contrast.max(8.0)).clamp(0.0, 1.0)
    };
    (residual_detail * 0.50 + seam * 0.25 + glyph_residual * 0.25).clamp(0.0, 1.0)
}

fn densify_analysis_mask(mask: &GrayImage) -> GrayImage {
    if mask.width() < 16 || mask.height() < 16 {
        return mask.clone();
    }

    let active = mask.pixels().filter(|pixel| pixel[0] >= 64).count() as f64;
    let total = f64::from(mask.width() * mask.height());
    if active / total >= 0.70 {
        return mask.clone();
    }

    let border_x = (mask.width() / 16).clamp(3, 12);
    let border_y = (mask.height() / 12).clamp(3, 10);
    let mut dense = mask.clone();
    for y in border_y..mask.height().saturating_sub(border_y) {
        for x in border_x..mask.width().saturating_sub(border_x) {
            dense.put_pixel(x, y, image::Luma([255]));
        }
    }
    dense
}

fn local_detail(image: &RgbImage, x: i32, y: i32) -> Option<f64> {
    if x < 1 || y < 1 || x + 1 >= image.width() as i32 || y + 1 >= image.height() as i32 {
        return None;
    }
    let center = luma(image.get_pixel(x as u32, y as u32));
    let neighbors = [
        luma(image.get_pixel((x - 1) as u32, y as u32)),
        luma(image.get_pixel((x + 1) as u32, y as u32)),
        luma(image.get_pixel(x as u32, (y - 1) as u32)),
        luma(image.get_pixel(x as u32, (y + 1) as u32)),
    ];
    Some((center - neighbors.iter().sum::<f64>() / neighbors.len() as f64).abs() / 255.0)
}

fn luma(pixel: &image::Rgb<u8>) -> f64 {
    f64::from(pixel[0]) * 0.299 + f64::from(pixel[1]) * 0.587 + f64::from(pixel[2]) * 0.114
}

fn color_distance(a: &image::Rgb<u8>, b: &image::Rgb<u8>) -> f64 {
    (f64::from(a[0].abs_diff(b[0]))
        + f64::from(a[1].abs_diff(b[1]))
        + f64::from(a[2].abs_diff(b[2])))
        / (3.0 * 255.0)
}

fn in_bounds(image: &RgbImage, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height()
}

pub fn temporal_consistency(valid_ratio: f64, artifact_score: f64) -> f64 {
    (valid_ratio * (1.0 - artifact_score)).clamp(0.0, 1.0)
}

pub fn accept_temporal_patch(valid_ratio: f64, artifact_score: f64, threshold: f64) -> bool {
    // A sparse patch can look statistically consistent while leaving a
    // readable watermark behind. Require broad pixel coverage before use.
    valid_ratio >= 0.50 && artifact_score <= threshold.clamp(0.05, 0.95)
}

#[cfg(test)]
mod tests {
    use super::{
        accept_temporal_patch, artifact_score, densify_analysis_mask, spatial_artifact_score,
        temporal_consistency,
    };
    use image::{GrayImage, Luma, Rgb, RgbImage};

    #[test]
    fn rejects_low_consensus_patch() {
        assert!(!accept_temporal_patch(0.49, 0.1, 0.25));
        assert!(!accept_temporal_patch(0.8, 0.4, 0.25));
        assert!(accept_temporal_patch(0.8, 0.1, 0.25));
    }

    #[test]
    fn combines_quality_signals_deterministically() {
        let score = artifact_score(0.2, 0.1);
        assert!((score - 0.145).abs() < f64::EPSILON);
        assert!((temporal_consistency(0.8, score) - 0.684).abs() < 0.001);
    }

    #[test]
    fn spatial_score_prefers_a_smoother_restoration() {
        let original = RgbImage::from_pixel(8, 8, Rgb([100, 100, 100]));
        let mut noisy = original.clone();
        noisy.put_pixel(4, 4, Rgb([0, 0, 0]));
        let mut mask = GrayImage::new(3, 3);
        for pixel in mask.pixels_mut() {
            *pixel = Luma([255]);
        }
        assert!(
            spatial_artifact_score(&original, &original, &mask, 3, 3)
                < spatial_artifact_score(&noisy, &noisy, &mask, 3, 3)
        );
    }

    #[test]
    fn sparse_analysis_mask_covers_glyph_interior_without_changing_small_masks() {
        let sparse = GrayImage::new(32, 24);
        let dense = densify_analysis_mask(&sparse);
        assert_eq!(dense.get_pixel(16, 12)[0], 255);
        assert_eq!(dense.get_pixel(0, 0)[0], 0);

        let small = GrayImage::new(8, 8);
        assert_eq!(densify_analysis_mask(&small), small);
    }
}
