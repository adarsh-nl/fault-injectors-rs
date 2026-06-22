/*!
Plotting utilities for the Griffin dataset.

Depends on the `plotters` crate for rendering to PNG files, and the `image`
crate for compositing multi-camera grids.
*/

use anyhow::{anyhow, Result};
use ndarray::Array2;
use plotters::prelude::*;
use std::path::Path;

// ── Category colours ──────────────────────────────────────────────────────

pub fn category_color(cat: &str) -> RGBColor {
    match cat {
        "car" => RGBColor(0, 114, 189),
        "pedestrian" => RGBColor(217, 83, 25),
        "truck" => RGBColor(126, 47, 142),
        "bus" => RGBColor(237, 177, 32),
        "motorcycle" => RGBColor(77, 190, 90),
        "bicycle" => RGBColor(0, 190, 213),
        _ => RGBColor(128, 128, 128),
    }
}

// ── Camera grid ───────────────────────────────────────────────────────────

/// Save up to four vehicle-camera images as a 2×2 PNG grid.
///
/// `images` should be a slice of `(H, W, 3)` u8 arrays in order
/// `[front, back, left, right]`.
pub fn plot_surround_cameras(
    images: &[ndarray::Array3<u8>],
    out_path: &Path,
) -> Result<()> {
    if images.is_empty() {
        return Err(anyhow!("No images to plot"));
    }

    let (tile_h, tile_w) = (images[0].shape()[0] as u32, images[0].shape()[1] as u32);
    let cols: u32 = 2;
    let rows: u32 = ((images.len() as u32) + 1) / 2;
    let total_w = tile_w * cols;
    let total_h = tile_h * rows;

    let root = BitMapBackend::new(out_path, (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE)?;

    for (idx, img) in images.iter().enumerate().take(4) {
        let col = (idx % 2) as u32;
        let row = (idx / 2) as u32;
        let x_off = (col * tile_w) as i32;
        let y_off = (row * tile_h) as i32;

        let (h, w) = (img.shape()[0], img.shape()[1]);
        for r in 0..h {
            for c in 0..w {
                let pixel = RGBColor(img[[r, c, 0]], img[[r, c, 1]], img[[r, c, 2]]);
                root.draw_pixel((x_off + c as i32, y_off + r as i32), &pixel)?;
            }
        }
    }

    root.present()?;
    Ok(())
}

// ── Bird's-eye view ───────────────────────────────────────────────────────

/// Render an ego-frame point cloud as a top-down (XY) heat map.
///
/// Points are coloured by height (Z value) from the `(N, 4)` array with
/// columns `[x, y, z, intensity]`.
pub fn plot_bev(
    points: &Array2<f32>,
    out_path: &Path,
    x_range: (f32, f32),
    y_range: (f32, f32),
    width_px: u32,
    height_px: u32,
) -> Result<()> {
    let root = BitMapBackend::new(out_path, (width_px, height_px)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    chart
        .configure_mesh()
        .x_desc("X (forward, m)")
        .y_desc("Y (left, m)")
        .draw()?;

    let z_min = points.column(2).fold(f32::INFINITY, f32::min);
    let z_max = points.column(2).fold(f32::NEG_INFINITY, f32::max);
    let z_range = (z_max - z_min).max(1e-6);

    let point_data: Vec<(f32, f32, u8, u8, u8)> = points
        .rows()
        .into_iter()
        .map(|row| {
            let t = ((row[2] - z_min) / z_range).clamp(0.0, 1.0);
            // Viridis-like: blue → green → yellow
            let r = (t * 255.0) as u8;
            let g = ((1.0 - (2.0 * t - 1.0).abs()) * 255.0) as u8;
            let b = ((1.0 - t) * 255.0) as u8;
            (row[0], row[1], r, g, b)
        })
        .collect();

    chart.draw_series(point_data.iter().map(|&(x, y, r, g, b)| {
        Circle::new((x, y), 1, RGBColor(r, g, b).filled())
    }))?;

    root.present()?;
    Ok(())
}

/// Render BEV with ego-frame LiDAR and 3D bounding box footprints.
pub fn plot_bev_with_boxes(
    points: &Array2<f32>,
    bev_corners: &[[[f64; 2]; 4]],
    categories: &[&str],
    out_path: &Path,
    x_range: (f32, f32),
    y_range: (f32, f32),
    width_px: u32,
    height_px: u32,
) -> Result<()> {
    let root = BitMapBackend::new(out_path, (width_px, height_px)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    chart
        .configure_mesh()
        .x_desc("X (forward, m)")
        .y_desc("Y (left, m)")
        .draw()?;

    // Points
    chart.draw_series(points.rows().into_iter().map(|row| {
        Circle::new((row[0], row[1]), 1, RGBColor(80, 80, 200).filled())
    }))?;

    // Box footprints
    for (corners, &cat) in bev_corners.iter().zip(categories.iter()) {
        let color = category_color(cat);
        let pts: Vec<(f32, f32)> = corners
            .iter()
            .map(|c| (c[0] as f32, c[1] as f32))
            .collect();
        for i in 0..4 {
            let j = (i + 1) % 4;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![pts[i], pts[j]],
                color.stroke_width(2),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

// ── Front / azimuth-elevation view ────────────────────────────────────────

/// Render a LiDAR front view (azimuth vs elevation) from ego-frame points.
pub fn plot_front_view(
    points: &Array2<f32>,
    out_path: &Path,
    width_px: u32,
    height_px: u32,
) -> Result<()> {
    let root = BitMapBackend::new(out_path, (width_px, height_px)).into_drawing_area();
    root.fill(&BLACK)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(-180.0_f32..180.0_f32, -30.0_f32..30.0_f32)?;

    chart
        .configure_mesh()
        .x_desc("Azimuth (°)")
        .y_desc("Elevation (°)")
        .draw()?;

    let azi_elev: Vec<(f32, f32)> = points
        .rows()
        .into_iter()
        .filter_map(|row| {
            let x = row[0];
            let y = row[1];
            let z = row[2];
            let r = (x * x + y * y + z * z).sqrt();
            if r < 1e-3 {
                return None;
            }
            let azi = y.atan2(x).to_degrees();
            let elev = (z / r).asin().to_degrees();
            Some((azi, elev))
        })
        .collect();

    chart.draw_series(azi_elev.iter().map(|&(a, e)| {
        Circle::new((a, e), 1, RGBColor(0, 200, 100).filled())
    }))?;

    root.present()?;
    Ok(())
}

// ── Sensor fusion view ────────────────────────────────────────────────────

/// Save a side-by-side comparison: raw RGB image next to RGB + projected LiDAR overlay.
///
/// `projected_pixels` is a list of `[u, v]` pixel coordinates with their depth.
pub fn plot_fusion(
    image: &ndarray::Array3<u8>,
    projected_pixels: &[[f32; 2]],
    depths: &[f32],
    out_path: &Path,
) -> Result<()> {
    let (h, w) = (image.shape()[0] as u32, image.shape()[1] as u32);
    let canvas_w = w * 2;
    let canvas_h = h;

    let root = BitMapBackend::new(out_path, (canvas_w, canvas_h)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(w);

    // Left panel: raw image
    for r in 0..h as usize {
        for c in 0..w as usize {
            let pixel = RGBColor(image[[r, c, 0]], image[[r, c, 1]], image[[r, c, 2]]);
            left.draw_pixel((c as i32, r as i32), &pixel)?;
        }
    }

    // Right panel: image with LiDAR overlay
    for r in 0..h as usize {
        for c in 0..w as usize {
            let pixel = RGBColor(image[[r, c, 0]], image[[r, c, 1]], image[[r, c, 2]]);
            right.draw_pixel((c as i32, r as i32), &pixel)?;
        }
    }

    if !depths.is_empty() {
        let d_min = depths.iter().cloned().fold(f32::INFINITY, f32::min);
        let d_max = depths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let d_range = (d_max - d_min).max(1e-6);

        for (&[u, v], &depth) in projected_pixels.iter().zip(depths.iter()) {
            let t = ((depth - d_min) / d_range).clamp(0.0, 1.0);
            let r_ch = ((1.0 - t) * 255.0) as u8;
            let g_ch = ((1.0 - (2.0 * t - 1.0).abs()) * 255.0) as u8;
            let b_ch = (t * 255.0) as u8;
            right.draw_pixel((u as i32, v as i32), &RGBColor(r_ch, g_ch, b_ch))?;
        }
    }

    root.present()?;
    Ok(())
}

// ── 3D boxes on image ─────────────────────────────────────────────────────

/// Draw projected 3D bounding box wire-frames onto an RGB image, saving to PNG.
///
/// `box_pixels` is a slice of 8-corner projections (one per box), with each
/// corner as `[u, v]` pixels.
pub fn plot_boxes_on_image(
    image: &ndarray::Array3<u8>,
    box_pixels: &[[[f32; 2]; 8]],
    categories: &[&str],
    out_path: &Path,
) -> Result<()> {
    let (h, w) = (image.shape()[0] as u32, image.shape()[1] as u32);
    let root = BitMapBackend::new(out_path, (w, h)).into_drawing_area();

    // Background: copy image
    for r in 0..h as usize {
        for c in 0..w as usize {
            let pixel = RGBColor(image[[r, c, 0]], image[[r, c, 1]], image[[r, c, 2]]);
            root.draw_pixel((c as i32, r as i32), &pixel)?;
        }
    }

    // Box edges (12 edges of a cuboid)
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), // top face
        (0, 4), (1, 5), (2, 6), (3, 7), // verticals
    ];

    for (corners, &cat) in box_pixels.iter().zip(categories.iter()) {
        let color = category_color(cat);
        for (a, b) in EDGES {
            let p1 = (corners[a][0] as i32, corners[a][1] as i32);
            let p2 = (corners[b][0] as i32, corners[b][1] as i32);
            root.draw(&PathElement::new(vec![p1, p2], color.stroke_width(2)))?;
        }
    }

    root.present()?;
    Ok(())
}
