/*!
Data loading utilities for the Griffin aerial-ground cooperative perception dataset.

Handles LiDAR (.ply), RGB images (.png), pose (.json), calibration (.json),
and annotation (.txt) files.
*/

use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, Array3};
use nalgebra::{Matrix4, Vector4};
use serde_json::Value;
use std::path::{Path, PathBuf};

// ── File discovery ────────────────────────────────────────────────────────

/// Sorted file inventories for all sensor modalities.
pub struct FileLists {
    pub cam_front: Vec<PathBuf>,
    pub cam_back: Vec<PathBuf>,
    pub cam_left: Vec<PathBuf>,
    pub cam_right: Vec<PathBuf>,
    pub lidar: Vec<PathBuf>,
    pub pose: Vec<PathBuf>,
    pub labels: Vec<PathBuf>,
    /// Five drone camera feeds (only populated when `include_drone = true`).
    pub drone_cams: Vec<Vec<PathBuf>>,
}

fn sorted_glob(dir: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let ext = pattern.trim_start_matches("*.");
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Cannot read directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e == ext)
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Build sorted file inventories from a Griffin scene directory.
pub fn get_file_lists(scene_dir: &str, include_drone: bool) -> Result<FileLists> {
    let root = Path::new(scene_dir);

    let cam_front = sorted_glob(&root.join("camera/front"), "*.png")?;
    let cam_back = sorted_glob(&root.join("camera/back"), "*.png")?;
    let cam_left = sorted_glob(&root.join("camera/left"), "*.png")?;
    let cam_right = sorted_glob(&root.join("camera/right"), "*.png")?;
    let lidar = sorted_glob(&root.join("lidar"), "*.ply")?;
    let pose = sorted_glob(&root.join("pose"), "*.json")?;
    let labels = sorted_glob(&root.join("labels"), "*.txt")?;

    let mut drone_cams = Vec::new();
    if include_drone {
        for i in 0..5 {
            let d = root.join(format!("drone/camera_{i}"));
            drone_cams.push(sorted_glob(&d, "*.png").unwrap_or_default());
        }
    }

    Ok(FileLists {
        cam_front,
        cam_back,
        cam_left,
        cam_right,
        lidar,
        pose,
        labels,
        drone_cams,
    })
}

// ── Image loading ─────────────────────────────────────────────────────────

/// Load a PNG as an `(H, W, 3)` u8 RGB array.
pub fn load_image(path: &Path) -> Result<Array3<u8>> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image {}", path.display()))?
        .to_rgb8();

    let (w, h) = img.dimensions();
    let raw = img.into_raw(); // length = h * w * 3

    let arr = Array3::from_shape_vec((h as usize, w as usize, 3), raw)
        .map_err(|e| anyhow!("Image shape error: {e}"))?;
    Ok(arr)
}

// ── LiDAR loading ─────────────────────────────────────────────────────────

/// Load a PLY point cloud and apply an extrinsic matrix to transform from
/// sensor frame into the ego (vehicle) frame.
///
/// Returns an `(N, 4)` array with columns `[x, y, z, intensity]` in ego frame.
pub fn load_lidar(path: &Path, extrinsic: &Matrix4<f64>) -> Result<Array2<f32>> {
    use ply_rs::{parser::Parser, ply::DefaultElement, ply::Property};

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open PLY file {}", path.display()))?;

    let parser = Parser::<DefaultElement>::new();
    let ply = parser
        .read_ply(&mut file)
        .with_context(|| format!("Failed to parse PLY {}", path.display()))?;

    let vertices = ply
        .payload
        .get("vertex")
        .ok_or_else(|| anyhow!("PLY file has no 'vertex' element"))?;

    let n = vertices.len();
    let mut out = Array2::<f32>::zeros((n, 4));

    for (i, v) in vertices.iter().enumerate() {
        let x = prop_to_f64(v.get("x"))? as f32;
        let y = prop_to_f64(v.get("y"))? as f32;
        let z = prop_to_f64(v.get("z"))? as f32;
        let intensity = v
            .get("intensity")
            .or_else(|| v.get("scalar_Intensity"))
            .map(|p| prop_to_f64(Some(p)).unwrap_or(0.0) as f32)
            .unwrap_or(0.0_f32);

        // Apply extrinsic: p_ego = extrinsic * [x, y, z, 1]^T
        let p_sensor = Vector4::new(x as f64, y as f64, z as f64, 1.0);
        let p_ego = extrinsic * p_sensor;

        out[[i, 0]] = p_ego[0] as f32;
        out[[i, 1]] = p_ego[1] as f32;
        out[[i, 2]] = p_ego[2] as f32;
        out[[i, 3]] = intensity;
    }

    Ok(out)
}

fn prop_to_f64(prop: Option<&ply_rs::ply::Property>) -> Result<f64> {
    use ply_rs::ply::Property;
    match prop {
        Some(Property::Float(v)) => Ok(*v as f64),
        Some(Property::Double(v)) => Ok(*v),
        Some(Property::Int(v)) => Ok(*v as f64),
        Some(Property::UInt(v)) => Ok(*v as f64),
        Some(Property::Short(v)) => Ok(*v as f64),
        Some(Property::UShort(v)) => Ok(*v as f64),
        Some(Property::Char(v)) => Ok(*v as f64),
        Some(Property::UChar(v)) => Ok(*v as f64),
        None => Err(anyhow!("Missing PLY property")),
        Some(other) => Err(anyhow!("Unsupported PLY property type: {:?}", other)),
    }
}

// ── Pose loading ──────────────────────────────────────────────────────────

pub struct PoseData {
    /// 4×4 ego-to-world (ENU) transformation matrix.
    pub transform: Matrix4<f64>,
    /// Raw JSON with Euler angles and position.
    pub raw: Value,
}

/// Load vehicle pose from a Griffin JSON file.
///
/// The JSON is expected to contain `position` (x, y, z in ENU metres) and
/// `orientation` (roll, pitch, yaw in radians).
pub fn load_pose(path: &Path) -> Result<PoseData> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read pose file {}", path.display()))?;
    let v: Value = serde_json::from_str(&text)?;

    let pos = &v["position"];
    let tx = pos["x"].as_f64().unwrap_or(0.0);
    let ty = pos["y"].as_f64().unwrap_or(0.0);
    let tz = pos["z"].as_f64().unwrap_or(0.0);

    let ori = &v["orientation"];
    let roll = ori["roll"].as_f64().unwrap_or(0.0);
    let pitch = ori["pitch"].as_f64().unwrap_or(0.0);
    let yaw = ori["yaw"].as_f64().unwrap_or(0.0);

    // ZYX Euler → rotation matrix
    let rot = euler_zyx_to_rotation(roll, pitch, yaw);

    let mut transform = Matrix4::<f64>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rot);
    transform[(0, 3)] = tx;
    transform[(1, 3)] = ty;
    transform[(2, 3)] = tz;

    Ok(PoseData { transform, raw: v })
}

/// Build a rotation matrix from ZYX Euler angles (roll, pitch, yaw).
fn euler_zyx_to_rotation(
    roll: f64,
    pitch: f64,
    yaw: f64,
) -> nalgebra::Matrix3<f64> {
    let (sr, cr) = (roll.sin(), roll.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cy) = (yaw.sin(), yaw.cos());

    nalgebra::Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

// ── Calibration loading ───────────────────────────────────────────────────

pub struct CameraCalibration {
    /// 3×3 intrinsic matrix K.
    pub intrinsic: nalgebra::Matrix3<f64>,
    /// 4×4 extrinsic (sensor → ego) matrix.
    pub extrinsic: Matrix4<f64>,
}

/// Load camera intrinsics and extrinsics from a Griffin calibration JSON.
pub fn load_camera_calibration(path: &Path) -> Result<CameraCalibration> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)?;

    let k = parse_matrix3(&v["intrinsic"])?;
    let ext = parse_matrix4(&v["extrinsic"])?;

    Ok(CameraCalibration { intrinsic: k, extrinsic: ext })
}

/// Load a general sensor extrinsic matrix from a calibration JSON.
pub fn load_extrinsic(path: &Path) -> Result<Matrix4<f64>> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)?;
    parse_matrix4(&v["extrinsic"])
}

fn parse_matrix3(v: &Value) -> Result<nalgebra::Matrix3<f64>> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone())
        .context("intrinsic matrix must be a 3x3 nested array")?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(anyhow!("intrinsic matrix must be 3×3"));
    }
    Ok(nalgebra::Matrix3::from_fn(|r, c| rows[r][c]))
}

fn parse_matrix4(v: &Value) -> Result<Matrix4<f64>> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone())
        .context("extrinsic matrix must be a 4x4 nested array")?;
    if rows.len() != 4 || rows.iter().any(|r| r.len() != 4) {
        return Err(anyhow!("extrinsic matrix must be 4×4"));
    }
    Ok(Matrix4::from_fn(|r, c| rows[r][c]))
}

// ── Annotation parsing ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Annotation {
    pub category: String,
    /// (length, width, height) in metres.
    pub dimensions: [f64; 3],
    /// 3D centre in ego frame (x, y, z) metres.
    pub position: [f64; 3],
    /// Yaw rotation in radians (ego frame).
    pub yaw: f64,
    pub track_id: String,
    pub visibility: f64,
}

/// Parse one Griffin annotation text file.
///
/// Expected columns per line (space-separated):
/// `category  l  w  h  x  y  z  yaw  track_id  visibility`
pub fn parse_label_txt(path: &Path) -> Result<Vec<Annotation>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read label file {}", path.display()))?;

    let mut annotations = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 10 {
            continue;
        }
        let ann = Annotation {
            category: parts[0].to_string(),
            dimensions: [
                parts[1].parse::<f64>()?,
                parts[2].parse::<f64>()?,
                parts[3].parse::<f64>()?,
            ],
            position: [
                parts[4].parse::<f64>()?,
                parts[5].parse::<f64>()?,
                parts[6].parse::<f64>()?,
            ],
            yaw: parts[7].parse::<f64>()?,
            track_id: parts[8].to_string(),
            visibility: parts[9].parse::<f64>()?,
        };
        annotations.push(ann);
    }

    Ok(annotations)
}
