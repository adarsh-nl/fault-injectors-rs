/*!
Coordinate-frame transforms for the Griffin dataset.

Frame conventions:
  * **ENU (world)**: East, North, Up.
  * **Ego (vehicle)**: X = forward, Y = left, Z = up.  LiDAR data and
    annotations live here.
  * **Camera (OpenCV)**: X = right, Y = down, Z = forward.

LiDAR points in Griffin are already in the ego frame, so projecting to the
camera only requires the sensor extrinsic — no pose transform is needed.
*/

use anyhow::{anyhow, Result};
use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector4};
use ndarray::Array2;

// ── LiDAR → image projection ──────────────────────────────────────────────

pub struct ProjectedPoints {
    /// `(M, 2)` pixel coordinates `[u, v]` for the M points that project
    /// within the image bounds.
    pub pixels: Vec<[f32; 2]>,
    /// Depth (Z in camera frame, metres) for each projected point.
    pub depths: Vec<f32>,
    /// Indices into the original point cloud for the surviving points.
    pub indices: Vec<usize>,
}

/// Project ego-frame LiDAR points to image coordinates.
///
/// Filters for points that are in front of the camera (`z_cam > 0`) and
/// whose pixel coordinates lie within `(0..width, 0..height)`.
pub fn project_lidar_to_image(
    points: &Array2<f32>,
    intrinsic: &Matrix3<f64>,
    extrinsic_ego_to_cam: &Matrix4<f64>,
    image_width: u32,
    image_height: u32,
    min_depth: f64,
) -> ProjectedPoints {
    let mut pixels = Vec::new();
    let mut depths = Vec::new();
    let mut indices = Vec::new();

    for (i, row) in points.rows().into_iter().enumerate() {
        let p_ego = Vector4::new(row[0] as f64, row[1] as f64, row[2] as f64, 1.0);
        let p_cam = extrinsic_ego_to_cam * p_ego;

        let z = p_cam[2];
        if z < min_depth {
            continue;
        }

        let u = (intrinsic[(0, 0)] * p_cam[0] / z + intrinsic[(0, 2)]) as f32;
        let v = (intrinsic[(1, 1)] * p_cam[1] / z + intrinsic[(1, 2)]) as f32;

        if u >= 0.0 && u < image_width as f32 && v >= 0.0 && v < image_height as f32 {
            pixels.push([u, v]);
            depths.push(z as f32);
            indices.push(i);
        }
    }

    ProjectedPoints { pixels, depths, indices }
}

/// Project ego-frame points to the image plane, retaining all points
/// (including those behind the camera).  Useful for drawing 3D bounding
/// box edges where occluded corners must still be connected.
pub fn project_ego_to_img(
    points: &[[f64; 3]],
    intrinsic: &Matrix3<f64>,
    extrinsic_ego_to_cam: &Matrix4<f64>,
) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|&[x, y, z]| {
            let p_ego = Vector4::new(x, y, z, 1.0);
            let p_cam = extrinsic_ego_to_cam * p_ego;
            let denom = if p_cam[2].abs() < 1e-9 { 1e-9 } else { p_cam[2] };
            let u = (intrinsic[(0, 0)] * p_cam[0] / denom + intrinsic[(0, 2)]) as f32;
            let v = (intrinsic[(1, 1)] * p_cam[1] / denom + intrinsic[(1, 2)]) as f32;
            [u, v]
        })
        .collect()
}

// ── 3D bounding box corners ───────────────────────────────────────────────

/// Compute the 8 corners of a 3D bounding box in the ego frame.
///
/// `position` = `(cx, cy, cz)` in metres (box centre), `dims` = `(l, w, h)`,
/// `yaw` = rotation about the Z axis in radians.
///
/// Corner ordering:
/// ```
///    4──5
///   /| /|
///  7──6 |        Z
///  | 0─|1        |  Y
///  |/  |/        | /
///  3──2           ──── X
/// ```
pub fn ego_box_corners_3d(
    position: [f64; 3],
    dims: [f64; 3],
    yaw: f64,
) -> [[f64; 3]; 8] {
    let (l, w, h) = (dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0);
    let (cy, sy) = (yaw.cos(), yaw.sin());

    // Local corners before rotation: (±l, ±w, ±h)
    let local: [[f64; 3]; 8] = [
        [-l, -w, -h],
        [ l, -w, -h],
        [ l,  w, -h],
        [-l,  w, -h],
        [-l, -w,  h],
        [ l, -w,  h],
        [ l,  w,  h],
        [-l,  w,  h],
    ];

    local.map(|[x, y, z]| {
        [
            cy * x - sy * y + position[0],
            sy * x + cy * y + position[1],
            z + position[2],
        ]
    })
}

/// Return the 4 BEV footprint corners of a 3D box (XY plane, ego frame).
pub fn ann_to_ego_corners_bev(position: [f64; 3], dims: [f64; 3], yaw: f64) -> [[f64; 2]; 4] {
    let corners = ego_box_corners_3d(position, dims, yaw);
    // Bottom face: corners 0-3
    [
        [corners[0][0], corners[0][1]],
        [corners[1][0], corners[1][1]],
        [corners[2][0], corners[2][1]],
        [corners[3][0], corners[3][1]],
    ]
}

// ── Ego → world transform ─────────────────────────────────────────────────

/// Transform ego-frame points into the ENU world frame.
///
/// `ego_to_world` is the 4×4 pose matrix returned by [`crate::data_loaders::load_pose`].
pub fn ego_points_to_world(
    points: &Array2<f32>,
    ego_to_world: &Matrix4<f64>,
) -> Array2<f32> {
    let n = points.nrows();
    let mut out = Array2::<f32>::zeros((n, 3));

    for (i, row) in points.rows().into_iter().enumerate() {
        let p = Vector4::new(row[0] as f64, row[1] as f64, row[2] as f64, 1.0);
        let pw = ego_to_world * p;
        out[[i, 0]] = pw[0] as f32;
        out[[i, 1]] = pw[1] as f32;
        out[[i, 2]] = pw[2] as f32;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn box_corners_count() {
        let corners = ego_box_corners_3d([0.0, 0.0, 0.0], [2.0, 1.0, 1.5], 0.0);
        assert_eq!(corners.len(), 8);
    }

    #[test]
    fn box_corners_zero_yaw() {
        let corners = ego_box_corners_3d([0.0, 0.0, 0.0], [4.0, 2.0, 2.0], 0.0);
        // At yaw=0, corner 1 should be at (+2, -1, -1)
        assert_abs_diff_eq!(corners[1][0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(corners[1][1], -1.0, epsilon = 1e-9);
    }

    #[test]
    fn bev_corners_count() {
        let bev = ann_to_ego_corners_bev([1.0, 2.0, 0.0], [4.0, 2.0, 1.5], 0.3);
        assert_eq!(bev.len(), 4);
    }
}
