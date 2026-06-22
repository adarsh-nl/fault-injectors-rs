/*!
Bernoulli sensor dropout simulator for the Griffin multimodal dataset.

Each frame independently drops the camera or LiDAR with a configurable
probability.  Dropped images become zero-filled (or mean-filled) tensors;
dropped point clouds become empty sets.
*/

use anyhow::{bail, Result};
use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand::rngs::StdRng;

// ── Low-level primitives ──────────────────────────────────────────────────

/// Draw one Bernoulli availability gate.
///
/// Returns `1` (sensor alive) with probability `1 - p_drop`, `0` (dropped)
/// with probability `p_drop`.
pub fn bernoulli_mask(p_drop: f64, rng: &mut impl Rng) -> u8 {
    if rng.gen::<f64>() >= p_drop { 1 } else { 0 }
}

/// Return an information-free (dropped) version of an RGB image.
///
/// `fill = "zero"` → black frame; `fill = "mean"` → per-channel constant.
/// The output has the same shape and dtype as the input.
pub fn drop_image(
    image: &Array3<u8>,
    fill: &str,
    mean_value: Option<[f32; 3]>,
) -> Result<Array3<u8>> {
    match fill {
        "zero" => Ok(Array3::zeros(image.raw_dim())),
        "mean" => {
            let mv = mean_value
                .ok_or_else(|| anyhow::anyhow!("fill='mean' requires mean_value"))?;
            let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
            let mut out = Array3::<u8>::zeros((h, w, c));
            for ch in 0..c {
                let val = mv[ch].round().clamp(0.0, 255.0) as u8;
                out.slice_mut(ndarray::s![.., .., ch]).fill(val);
            }
            Ok(out)
        }
        other => bail!("Unknown fill '{}'. Use 'zero' or 'mean'.", other),
    }
}

/// Return an empty point cloud with the same column count as `points`.
///
/// The returned array has shape `(0, C)`.
pub fn drop_points(points: &Array2<f32>) -> Array2<f32> {
    Array2::zeros((0, points.ncols()))
}

// ── Result container ──────────────────────────────────────────────────────

pub struct InjectionResult {
    /// Possibly corrupted image `(H, W, 3)`.
    pub image: Array3<u8>,
    /// Possibly empty point cloud `(N, C)` or `(0, C)`.
    pub points: Array2<f32>,
    /// `1` = camera alive, `0` = dropped.
    pub m_rgb: u8,
    /// `1` = LiDAR alive, `0` = dropped.
    pub m_lidar: u8,
}

/// Pre-drawn dropout schedule for a sequence of frames.
pub struct DropoutSchedule {
    /// Per-frame camera availability (`1` alive / `0` dropped).
    pub m_rgb: Vec<u8>,
    /// Per-frame LiDAR availability (`1` alive / `0` dropped).
    pub m_lidar: Vec<u8>,
}

// ── Stateful injector ─────────────────────────────────────────────────────

/// Apply Bernoulli sensor dropout to a stream of `(image, points)` samples.
///
/// ```rust
/// use fault_injectors_rs::fault_injectors::missing_modality::MissingModalityInjector;
///
/// let mut inj = MissingModalityInjector::new(0.0, 0.5, "zero", None, 0).unwrap();
/// // let result = inj.inject(&image, &points).unwrap();
/// // result.m_lidar == 1 iff the LiDAR survived this frame
/// ```
pub struct MissingModalityInjector {
    pub p_drop_rgb: f64,
    pub p_drop_lidar: f64,
    fill: String,
    mean_value: Option<[f32; 3]>,
    rng: StdRng,
}

impl MissingModalityInjector {
    pub fn new(
        p_drop_rgb: f64,
        p_drop_lidar: f64,
        fill: &str,
        mean_value: Option<[f32; 3]>,
        seed: u64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&p_drop_rgb) {
            bail!("p_drop_rgb must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&p_drop_lidar) {
            bail!("p_drop_lidar must be in [0, 1]");
        }
        Ok(Self {
            p_drop_rgb,
            p_drop_lidar,
            fill: fill.to_string(),
            mean_value,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Corrupt one `(image, points)` sample.
    pub fn inject(&mut self, image: &Array3<u8>, points: &Array2<f32>) -> Result<InjectionResult> {
        let m_rgb = bernoulli_mask(self.p_drop_rgb, &mut self.rng);
        let m_lidar = bernoulli_mask(self.p_drop_lidar, &mut self.rng);

        let out_image = if m_rgb == 1 {
            image.clone()
        } else {
            drop_image(image, &self.fill, self.mean_value)?
        };
        let out_points = if m_lidar == 1 {
            points.clone()
        } else {
            drop_points(points)
        };

        Ok(InjectionResult {
            image: out_image,
            points: out_points,
            m_rgb,
            m_lidar,
        })
    }

    /// Pre-draw the availability gates for `n_frames` without needing data.
    ///
    /// Useful for inspecting or plotting a dropout schedule before touching
    /// real samples.
    pub fn simulate_sequence(&mut self, n_frames: usize) -> DropoutSchedule {
        let m_rgb = (0..n_frames)
            .map(|_| bernoulli_mask(self.p_drop_rgb, &mut self.rng))
            .collect();
        let m_lidar = (0..n_frames)
            .map(|_| bernoulli_mask(self.p_drop_lidar, &mut self.rng))
            .collect();
        DropoutSchedule { m_rgb, m_lidar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_always_drop() {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            assert_eq!(bernoulli_mask(1.0, &mut rng), 0);
        }
    }

    #[test]
    fn bernoulli_never_drop() {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            assert_eq!(bernoulli_mask(0.0, &mut rng), 1);
        }
    }

    #[test]
    fn drop_image_zero_fill() {
        let img = Array3::<u8>::ones((4, 4, 3));
        let dropped = drop_image(&img, "zero", None).unwrap();
        assert!(dropped.iter().all(|&v| v == 0));
    }

    #[test]
    fn drop_image_mean_fill() {
        let img = Array3::<u8>::ones((4, 4, 3));
        let dropped = drop_image(&img, "mean", Some([128.0, 64.0, 32.0])).unwrap();
        assert_eq!(dropped[[0, 0, 0]], 128u8);
        assert_eq!(dropped[[0, 0, 1]], 64u8);
        assert_eq!(dropped[[0, 0, 2]], 32u8);
    }

    #[test]
    fn drop_points_empty() {
        let pts = Array2::<f32>::zeros((100, 4));
        let empty = drop_points(&pts);
        assert_eq!(empty.shape(), &[0, 4]);
    }

    #[test]
    fn inject_deterministic() {
        let img = Array3::<u8>::ones((2, 2, 3));
        let pts = Array2::<f32>::ones((10, 4));
        let mut inj = MissingModalityInjector::new(0.5, 0.5, "zero", None, 42).unwrap();
        let r1 = inj.inject(&img, &pts).unwrap();
        let mut inj2 = MissingModalityInjector::new(0.5, 0.5, "zero", None, 42).unwrap();
        let r2 = inj2.inject(&img, &pts).unwrap();
        assert_eq!(r1.m_rgb, r2.m_rgb);
        assert_eq!(r1.m_lidar, r2.m_lidar);
    }

    #[test]
    fn simulate_sequence_length() {
        let mut inj = MissingModalityInjector::new(0.5, 0.5, "zero", None, 0).unwrap();
        let sched = inj.simulate_sequence(10);
        assert_eq!(sched.m_rgb.len(), 10);
        assert_eq!(sched.m_lidar.len(), 10);
    }
}
