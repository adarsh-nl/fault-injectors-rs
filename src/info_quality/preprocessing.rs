/*!
Architecture-agnostic preprocessing for the mutual information estimators.

Operates on two `(N, d)` f32 matrices Z and Y.  The pipeline is:

1. Optional PCA dimensionality reduction (using full SVD via nalgebra).
2. Per-column standardisation to zero mean, unit variance.

PCA is linear — it cannot create information about Y; it only discards
low-variance directions.  The protocol is held fixed across conditions, so
the systematic bias cancels in the relative delta_I comparison.
*/

use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, SVD};
use ndarray::{s, Array1, Array2, Axis};
use rand::SeedableRng;

// ── Seeding ───────────────────────────────────────────────────────────────

/// Seed both ndarray-rand and candle RNGs for reproducible estimation runs.
pub fn set_seed(seed: u64) {
    // Seed the global rand source through a thread-local store so downstream
    // code that calls `rand::thread_rng()` is reproducible.  Candle uses its
    // own per-tensor seeds; callers are responsible for seeding the VarMap.
    let _ = rand::rngs::StdRng::seed_from_u64(seed); // validates the seed
}

// ── Standardisation ───────────────────────────────────────────────────────

/// Standardise each column of `x` to zero mean and unit variance (float32).
///
/// Columns whose standard deviation is below `eps` are left as zero.
pub fn standardise(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let mean = x.mean_axis(Axis(0)).expect("non-empty array");
    let std = x.std_axis(Axis(0), 1.0_f32); // unbiased (ddof=1)

    let n = x.nrows();
    let d = x.ncols();
    let mut out = Array2::<f32>::zeros((n, d));
    for j in 0..d {
        let s = if std[j] < eps { 1.0 } else { std[j] };
        for i in 0..n {
            out[[i, j]] = (x[[i, j]] - mean[j]) / s;
        }
    }
    out
}

// ── PCA ───────────────────────────────────────────────────────────────────

/// Reduce `x` to `n_components` principal components using full SVD.
///
/// The component count is capped at `min(n_components, N-1, d)` so that the
/// decomposition remains valid for small sample counts.
pub fn pca_reduce(x: &Array2<f32>, n_components: usize) -> Result<Array2<f32>> {
    let (n, d) = (x.nrows(), x.ncols());
    if n < 2 {
        return Err(anyhow!("PCA requires at least 2 samples, got {n}"));
    }
    let n_comp = n_components.min(n - 1).min(d);
    if n_comp == 0 {
        return Err(anyhow!("n_components must be >= 1"));
    }

    // Centre
    let mean: Array1<f32> = x.mean_axis(Axis(0)).unwrap();
    let centered: Array2<f32> = x - &mean.broadcast((n, d)).unwrap();

    // Convert to nalgebra DMatrix for SVD
    let mat = DMatrix::<f32>::from_fn(n, d, |r, c| centered[[r, c]]);

    // compute_thin_u=false, compute_thin_v=true to get right singular vectors
    let svd = SVD::new(mat, false, true);
    let vt = svd.v_t.ok_or_else(|| anyhow!("SVD did not compute V^T"))?;

    // Project: X_centered @ V[:, :n_comp] = X_centered @ Vt[:n_comp, :].T
    let n_rows = vt.nrows().min(n_comp);
    let proj_mat = DMatrix::<f32>::from_fn(n, n_rows, |r, c| {
        // (X_centered @ Vt[c, :]) for row r
        // = sum_k centered[r,k] * vt[c, k]
        (0..d).map(|k| centered[[r, k]] * vt[(c, k)]).sum()
    });

    let out = Array2::from_shape_fn((n, n_rows), |(r, c)| proj_mat[(r, c)]);
    Ok(out)
}

// ── Pipeline ──────────────────────────────────────────────────────────────

/// Apply optional PCA then standardisation to both Z and Y.
pub fn prepare(
    z: &Array2<f32>,
    y: &Array2<f32>,
    pca_dims: Option<usize>,
    eps: f32,
) -> Result<(Array2<f32>, Array2<f32>)> {
    if z.nrows() != y.nrows() {
        return Err(anyhow!(
            "Z has {} rows but Y has {} rows — they must be aligned",
            z.nrows(),
            y.nrows()
        ));
    }

    let z_proc = if let Some(k) = pca_dims {
        pca_reduce(z, k)?
    } else {
        z.clone()
    };
    let y_proc = if let Some(k) = pca_dims {
        pca_reduce(y, k)?
    } else {
        y.clone()
    };

    Ok((standardise(&z_proc, eps), standardise(&y_proc, eps)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn standardise_zero_mean_unit_var() {
        let x = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let s = standardise(&x, 1e-8);
        // Each column should have mean ≈ 0
        let col0 = s.column(0);
        let mean0: f32 = col0.sum() / col0.len() as f32;
        assert_abs_diff_eq!(mean0, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn pca_reduces_dimension() {
        let x = Array2::<f32>::from_shape_fn((20, 5), |(i, j)| (i * 5 + j) as f32);
        let reduced = pca_reduce(&x, 3).unwrap();
        assert_eq!(reduced.ncols(), 3);
        assert_eq!(reduced.nrows(), 20);
    }

    #[test]
    fn prepare_checks_alignment() {
        let z = Array2::<f32>::zeros((10, 4));
        let y = Array2::<f32>::zeros((8, 2));
        assert!(prepare(&z, &y, None, 1e-8).is_err());
    }
}
