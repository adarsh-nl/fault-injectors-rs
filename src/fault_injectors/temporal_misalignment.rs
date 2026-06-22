/*!
Temporal-misalignment fault injector for the Griffin multimodal dataset.

Pairs the current LiDAR frame `P_k` with a stale image `I_{k - delta_k}`,
where `delta_k` is sampled from a Normal distribution and quantised to a
frame index.  At 50 km/h, a 200 ms delay displaces the image by ~2.8 m
relative to the LiDAR scan.
*/

use anyhow::{bail, Result};
use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

// ── Low-level primitives ──────────────────────────────────────────────────

/// Draw one discrete index shift `delta_k >= 0`.
///
/// Samples `delta_t ~ Normal(mu_delay, sigma_jitter)`, converts to frame
/// count, and clamps at 0 (a future frame cannot be delivered).
pub fn sample_index_shift(
    mu_delay: f64,
    sigma_jitter: f64,
    frame_period: f64,
    rng: &mut impl Rng,
) -> i64 {
    let dist = Normal::new(mu_delay, sigma_jitter.max(1e-9))
        .expect("Normal distribution parameters are valid");
    let delta_t: f64 = rng.sample(dist);
    let shift = (delta_t / frame_period).round() as i64;
    shift.max(0)
}

/// Physical displacement in metres: `d = v * delta_t`.
pub fn physical_displacement(velocity_m_s: f64, delay_s: f64) -> f64 {
    velocity_m_s * delay_s
}

// ── Result container ──────────────────────────────────────────────────────

pub struct TemporalInjectionResult {
    /// The stale image `I_{k_image}`.
    pub image: Array3<u8>,
    /// The current point cloud `P_k`.
    pub points: Array2<f32>,
    /// Clamped image index used (may be > `k - delta_k` if near sequence start).
    pub k_image: usize,
    /// Sampled frame shift before clamping to `k_min`.
    pub delta_k: i64,
}

// ── Stateful injector ─────────────────────────────────────────────────────

/// Pair LiDAR frame `k` with a stale image via index shifting.
///
/// ```rust
/// use fault_injectors_rs::fault_injectors::temporal_misalignment::TemporalMisalignmentInjector;
///
/// let mut inj = TemporalMisalignmentInjector::new(0.2, 0.05, 10.0, 0).unwrap();
/// let (k_img, dk) = inj.stale_index(5, 0);
/// // load images[k_img] paired with points[5]
/// ```
pub struct TemporalMisalignmentInjector {
    pub mu_delay: f64,
    pub sigma_jitter: f64,
    pub frame_period: f64,
    rng: StdRng,
}

impl TemporalMisalignmentInjector {
    pub fn new(mu_delay: f64, sigma_jitter: f64, fps: f64, seed: u64) -> Result<Self> {
        if mu_delay < 0.0 {
            bail!("mu_delay must be >= 0 seconds");
        }
        if sigma_jitter < 0.0 {
            bail!("sigma_jitter must be >= 0 seconds");
        }
        if fps <= 0.0 {
            bail!("fps must be positive");
        }
        Ok(Self {
            mu_delay,
            sigma_jitter,
            frame_period: 1.0 / fps,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Draw one `delta_k >= 0` from the configured delay distribution.
    pub fn sample_shift(&mut self) -> i64 {
        sample_index_shift(self.mu_delay, self.sigma_jitter, self.frame_period, &mut self.rng)
    }

    /// For LiDAR frame `k`, return `(k_image, delta_k)`.
    ///
    /// `k_image = max(k as i64 - delta_k, k_min as i64) as usize`
    pub fn stale_index(&mut self, k: usize, k_min: usize) -> (usize, i64) {
        let delta_k = self.sample_shift();
        let k_image = ((k as i64 - delta_k).max(k_min as i64)) as usize;
        (k_image, delta_k)
    }

    /// Build the corrupted pair `(I_{k_image}, P_k)` from in-memory sequences.
    pub fn inject(
        &mut self,
        k: usize,
        images: &[Array3<u8>],
        points_seq: &[Array2<f32>],
        k_min: usize,
    ) -> TemporalInjectionResult {
        let (k_image, delta_k) = self.stale_index(k, k_min);
        TemporalInjectionResult {
            image: images[k_image].clone(),
            points: points_seq[k].clone(),
            k_image,
            delta_k,
        }
    }

    /// Pre-draw `delta_k` values for `n_frames` without loading any data.
    ///
    /// Note: this consumes the same RNG stream as [`inject`][Self::inject], so
    /// use separate injector instances if you need both to be consistent.
    pub fn simulate_sequence(&mut self, n_frames: usize) -> Vec<i64> {
        (0..n_frames).map(|_| self.sample_shift()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_non_negative() {
        let mut inj = TemporalMisalignmentInjector::new(0.2, 0.05, 10.0, 42).unwrap();
        for _ in 0..1000 {
            assert!(inj.sample_shift() >= 0);
        }
    }

    #[test]
    fn stale_index_clamped_at_k_min() {
        let mut inj = TemporalMisalignmentInjector::new(100.0, 0.0, 10.0, 0).unwrap();
        let (k_img, _dk) = inj.stale_index(3, 0);
        // With mu_delay=100s and fps=10, delta_k=1000; clamped to 0
        assert_eq!(k_img, 0);
    }

    #[test]
    fn simulate_sequence_length() {
        let mut inj = TemporalMisalignmentInjector::new(0.2, 0.05, 10.0, 0).unwrap();
        let shifts = inj.simulate_sequence(50);
        assert_eq!(shifts.len(), 50);
    }

    #[test]
    fn physical_displacement_calc() {
        // 50 km/h = 13.889 m/s; 0.2 s delay => ~2.78 m
        let d = physical_displacement(50.0 / 3.6, 0.2);
        assert!((d - 2.778).abs() < 0.01, "got {d}");
    }
}
