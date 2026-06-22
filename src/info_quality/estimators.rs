/*!
Architecture-agnostic mutual information estimators.

Information quality is operationalised as `I(Z; Y)` — how much task
information a learned representation carries.  Under fault injection we
compare clean vs corrupted runs and read off the fusion gain:

    delta_I = I(Z_fused; Y) - max( I(Z_a; Y), I(Z_b; Y) )

Two estimators with different biases are provided so that "fusion adds
information" means it holds under *both* InfoNCE and SMILE, which is far
harder to dismiss as an estimator artefact.

**InfoNCE** (van den Oord et al. 2018) — contrastive lower bound, capped
at `log(N)` nats.

**SMILE** (Song & Ermon 2020) — clipped variational lower bound.  Clips
the log-density ratio to `[-clip, +clip]` to prevent MINE's divergence
spiral.  No `log(N)` ceiling.

All estimators consume plain `(N, d)` f32 ndarray slices and operate on CPU
by default (candle: enable the `cuda` feature for GPU).
*/

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use ndarray::Array2;

use super::preprocessing::{prepare, set_seed};

// ── Helpers ───────────────────────────────────────────────────────────────

fn cpu() -> Device {
    Device::Cpu
}

/// Row-wise L2 normalisation: each row is divided by its Euclidean norm.
fn l2_normalize(t: &Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(t.broadcast_div(&norm)?)
}

/// Train / eval index split. `holdout = 0` ⇒ in-sample evaluation.
fn split(n: usize, holdout: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    if holdout <= 0.0 {
        let idx: Vec<usize> = (0..n).collect();
        return (idx.clone(), idx);
    }
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let n_ev = ((holdout * n as f64).round() as usize).max(2).min(n - 2);
    (perm[n_ev..].to_vec(), perm[..n_ev].to_vec())
}

fn gather_rows(arr: &Array2<f32>, idx: &[usize]) -> Array2<f32> {
    let d = arr.ncols();
    Array2::from_shape_fn((idx.len(), d), |(i, j)| arr[[idx[i], j]])
}

fn to_tensor(arr: &Array2<f32>, device: &Device) -> Result<Tensor> {
    let (n, d) = (arr.nrows(), arr.ncols());
    let flat: Vec<f32> = arr.iter().cloned().collect();
    Ok(Tensor::from_vec(flat, &[n, d], device)?)
}

/// Cosine-annealing factor: `0.5 * (1 + cos(π t / T))`.
fn cosine_lr(base_lr: f64, step: usize, total: usize) -> f64 {
    let factor =
        0.5 * (1.0 + (std::f64::consts::PI * step as f64 / total as f64).cos());
    base_lr * factor.max(0.0)
}

fn randperm(n: usize, rng: &mut impl rand::Rng) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(rng);
    v
}

fn idx_tensor(idx: &[usize], device: &Device) -> Result<Tensor> {
    let v: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
    Tensor::from_vec(v, &[idx.len()], device)
}

// ── Result container ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MIResult {
    /// Estimated `I(Z; Y)` in nats (lower bound).
    pub mi_nats: f64,
    pub estimator: String,
    pub n_samples: usize,
    /// Per-eval MI estimates after warmup (SMILE only).
    pub history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// InfoNCE
// ─────────────────────────────────────────────────────────────────────────────

/// Separable critic: cosine similarity of independently projected Z and Y,
/// scaled by `1 / tau`.
struct InfoNCECritic {
    pz1: Linear,
    pz2: Linear,
    py1: Linear,
    py2: Linear,
    tau: f64,
}

impl InfoNCECritic {
    fn new(
        z_dim: usize,
        y_dim: usize,
        proj_dim: usize,
        temperature: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            pz1: linear(z_dim, 128, vb.pp("pz1"))?,
            pz2: linear(128, proj_dim, vb.pp("pz2"))?,
            py1: linear(y_dim, 64, vb.pp("py1"))?,
            py2: linear(64, proj_dim, vb.pp("py2"))?,
            tau: temperature,
        })
    }

    fn project_z(&self, z: &Tensor) -> Result<Tensor> {
        l2_normalize(&self.pz2.forward(&self.pz1.forward(z)?.relu()?)?)
    }

    fn project_y(&self, y: &Tensor) -> Result<Tensor> {
        l2_normalize(&self.py2.forward(&self.py1.forward(y)?.relu()?)?)
    }

    /// `(N, N)` logit matrix = `(Z_proj @ Y_proj^T) / tau`.
    fn logits(&self, z: &Tensor, y: &Tensor) -> Result<Tensor> {
        let scores = self.project_z(z)?.matmul(&self.project_y(y)?.t()?)?;
        // affine(a, b) computes a*x + b element-wise
        scores.affine(1.0 / self.tau, 0.0)
    }

    fn loss(&self, z: &Tensor, y: &Tensor) -> Result<Tensor> {
        let n = z.dims()[0];
        let labels = Tensor::arange(0u32, n as u32, z.device())?;
        candle_nn::loss::cross_entropy(&self.logits(z, y)?, &labels)
    }

    /// `I >= log(K) - L_NCE`  evaluated over the supplied eval set.
    fn mi_bound(&self, z: &Tensor, y: &Tensor) -> Result<f64> {
        let loss: f32 = self.loss(z, y)?.to_scalar()?;
        Ok((z.dims()[0] as f64).ln() - loss as f64)
    }
}

/// InfoNCE contrastive lower bound on `I(Z; Y)`.
pub struct InfoNCEEstimator {
    pub proj_dim: usize,
    pub temperature: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub holdout: f64,
    pub seed: u64,
}

impl Default for InfoNCEEstimator {
    fn default() -> Self {
        Self {
            proj_dim: 64,
            temperature: 0.07,
            epochs: 100,
            batch_size: 32,
            lr: 1e-3,
            holdout: 0.0,
            seed: 0,
        }
    }
}

impl InfoNCEEstimator {
    /// Estimate `I(Z; Y)` in nats. `Z` and `Y` are `(N, d)` float32 arrays.
    pub fn estimate(
        &self,
        z_arr: &Array2<f32>,
        y_arr: &Array2<f32>,
        pca_dims: Option<usize>,
    ) -> Result<MIResult> {
        set_seed(self.seed);
        let (z, y) = prepare(z_arr, y_arr, pca_dims, 1e-8)?;
        let device = cpu();

        let (tr, ev) = split(z.nrows(), self.holdout, self.seed);
        let z_tr = to_tensor(&gather_rows(&z, &tr), &device)?;
        let y_tr = to_tensor(&gather_rows(&y, &tr), &device)?;
        let z_ev = to_tensor(&gather_rows(&z, &ev), &device)?;
        let y_ev = to_tensor(&gather_rows(&y, &ev), &device)?;

        let n_tr = z_tr.dims()[0];
        let bs = self.batch_size.min(n_tr);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let critic =
            InfoNCECritic::new(z.ncols(), y.ncols(), self.proj_dim, self.temperature, vb)?;

        let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: self.lr, ..Default::default() })?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        use rand::prelude::*;

        for epoch in 0..self.epochs {
            opt.set_learning_rate(cosine_lr(self.lr, epoch, self.epochs));
            let perm = randperm(n_tr, &mut rng);
            for chunk in perm.chunks(bs) {
                if chunk.len() < 2 {
                    continue;
                }
                let it = idx_tensor(chunk, &device)?;
                let zb = z_tr.index_select(&it, 0)?;
                let yb = y_tr.index_select(&it, 0)?;
                opt.backward_step(&critic.loss(&zb, &yb)?)?;
            }
        }

        let mi = critic.mi_bound(&z_ev, &y_ev)?;
        Ok(MIResult {
            mi_nats: mi,
            estimator: "infonce".to_string(),
            n_samples: z.nrows(),
            history: vec![],
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SMILE
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics network `T(z, y) = MLP([z; y]) → scalar`.
struct SMILENet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SMILENet {
    fn new(z_dim: usize, y_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(z_dim + y_dim, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, hidden / 2, vb.pp("fc2"))?,
            fc3: linear(hidden / 2, 1, vb.pp("fc3"))?,
        })
    }

    fn forward(&self, z: &Tensor, y: &Tensor) -> Result<Tensor> {
        let inp = Tensor::cat(&[z, y], 1)?;
        let h1 = self.fc1.forward(&inp)?.relu()?;
        let h2 = self.fc2.forward(&h1)?.relu()?;
        // (N, 1) → (N,)
        self.fc3.forward(&h2)?.squeeze(D::Minus1)
    }
}

/// SMILE lower bound: `E_joint[T] - log E_marg[exp(clamp(T, −clip, +clip))]`.
///
/// `t_joint: (B,)`, `t_marg: (B, M)`.
fn smile_bound(t_joint: &Tensor, t_marg: &Tensor, clip: f32) -> Result<Tensor> {
    let log_denom = t_marg.clamp(-clip, clip)?.exp()?.mean(D::Minus1)?.log()?;
    t_joint.sub(&log_denom)?.mean_all()
}

/// SMILE clipped variational lower bound on `I(Z; Y)`.
pub struct SMILEEstimator {
    pub hidden: usize,
    pub clip: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup: usize,
    pub avg_last: usize,
    pub eval_every: usize,
    pub max_negatives: usize,
    pub holdout: f64,
    pub floor_zero: bool,
    pub seed: u64,
}

impl Default for SMILEEstimator {
    fn default() -> Self {
        Self {
            hidden: 256,
            clip: 5.0,
            epochs: 500,
            batch_size: 64,
            lr: 2e-4,
            warmup: 100,
            avg_last: 50,
            eval_every: 5,
            max_negatives: 512,
            holdout: 0.0,
            floor_zero: false,
            seed: 0,
        }
    }
}

impl SMILEEstimator {
    pub fn estimate(
        &self,
        z_arr: &Array2<f32>,
        y_arr: &Array2<f32>,
        pca_dims: Option<usize>,
    ) -> Result<MIResult> {
        set_seed(self.seed);
        let (z, y) = prepare(z_arr, y_arr, pca_dims, 1e-8)?;
        let device = cpu();

        let (tr, ev) = split(z.nrows(), self.holdout, self.seed);
        let z_tr = to_tensor(&gather_rows(&z, &tr), &device)?;
        let y_tr = to_tensor(&gather_rows(&y, &tr), &device)?;
        let z_ev = to_tensor(&gather_rows(&z, &ev), &device)?;
        let y_ev = to_tensor(&gather_rows(&y, &ev), &device)?;

        let n_tr = z_tr.dims()[0];
        let n_ev = z_ev.dims()[0];
        let bs = self.batch_size.min(n_tr);
        let m = self.max_negatives.min(n_tr);
        let m_ev = self.max_negatives.min(n_ev);
        let z_dim = z.ncols();
        let y_dim = y.ncols();
        let clip = self.clip as f32;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = SMILENet::new(z_dim, y_dim, self.hidden, vb)?;

        let mut opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW { lr: self.lr, ..Default::default() },
        )?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let mut history: Vec<f64> = Vec::new();

        for step in 0..self.epochs {
            opt.set_learning_rate(cosine_lr(self.lr, step, self.epochs));

            // Anchor batch
            let anchor_idx = randperm(n_tr, &mut rng).into_iter().take(bs).collect::<Vec<_>>();
            let at = idx_tensor(&anchor_idx, &device)?;
            let zb = z_tr.index_select(&at, 0)?;
            let yb = y_tr.index_select(&at, 0)?;
            let b = zb.dims()[0];

            // Negative targets
            let neg_idx = randperm(n_tr, &mut rng).into_iter().take(m).collect::<Vec<_>>();
            let nt = idx_tensor(&neg_idx, &device)?;
            let y_neg = y_tr.index_select(&nt, 0)?;

            // t_joint: (B,)
            let t_joint = net.forward(&zb, &yb)?;

            // t_marg: (B, M) — cross all anchors with all negatives
            let zb_exp = zb.unsqueeze(1)?.expand(&[b, m, z_dim])?;
            let yn_exp = y_neg.unsqueeze(0)?.expand(&[b, m, y_dim])?;
            let t_marg = net
                .forward(&zb_exp.reshape(&[b * m, z_dim])?, &yn_exp.reshape(&[b * m, y_dim])?)?
                .reshape(&[b, m])?;

            let loss = smile_bound(&t_joint, &t_marg, clip)?.neg()?;
            opt.backward_step(&loss)?;

            if step >= self.warmup && (step - self.warmup) % self.eval_every == 0 {
                let mi = self.full_eval(&net, &z_ev, &y_ev, m_ev, z_dim, y_dim, &mut rng, &device)?;
                history.push(mi);
            }
        }

        let mi = if history.is_empty() {
            self.full_eval(&net, &z_ev, &y_ev, m_ev, z_dim, y_dim, &mut rng, &device)?
        } else {
            let tail_len = self.avg_last.min(history.len());
            let tail = &history[history.len() - tail_len..];
            tail.iter().sum::<f64>() / tail.len() as f64
        };

        let mi = if self.floor_zero { mi.max(0.0) } else { mi };
        Ok(MIResult {
            mi_nats: mi,
            estimator: "smile".to_string(),
            n_samples: z.nrows(),
            history,
        })
    }

    fn full_eval(
        &self,
        net: &SMILENet,
        z_ev: &Tensor,
        y_ev: &Tensor,
        m_ev: usize,
        z_dim: usize,
        y_dim: usize,
        rng: &mut impl rand::Rng,
        device: &Device,
    ) -> Result<f64> {
        let n_ev = z_ev.dims()[0];
        let clip = self.clip as f32;

        // Fixed negative set
        let neg_idx = randperm(n_ev, rng).into_iter().take(m_ev).collect::<Vec<_>>();
        let nt = idx_tensor(&neg_idx, device)?;
        let y_neg = y_ev.index_select(&nt, 0)?;

        // t_joint: scalar mean over all eval pairs
        let t_joint_mean: f32 = net.forward(z_ev, y_ev)?.mean_all()?.to_scalar()?;

        // log_denom per anchor, chunked to keep peak memory bounded
        let chunk_size = 256.min(n_ev);
        let mut logdenom_acc: f64 = 0.0;
        let mut n_acc: usize = 0;

        for start in (0..n_ev).step_by(chunk_size) {
            let end = (start + chunk_size).min(n_ev);
            let c = end - start;

            let chunk_idx: Vec<usize> = (start..end).collect();
            let ct = idx_tensor(&chunk_idx, device)?;
            let zc = z_ev.index_select(&ct, 0)?;

            let zc_exp = zc.unsqueeze(1)?.expand(&[c, m_ev, z_dim])?;
            let yn_exp = y_neg.unsqueeze(0)?.expand(&[c, m_ev, y_dim])?;
            let t_marg = net
                .forward(
                    &zc_exp.reshape(&[c * m_ev, z_dim])?,
                    &yn_exp.reshape(&[c * m_ev, y_dim])?,
                )?
                .reshape(&[c, m_ev])?;

            // log_denom: (c,) — then sum to a scalar
            let log_d = t_marg.clamp(-clip, clip)?.exp()?.mean(D::Minus1)?.log()?;
            let sum_val: f32 = log_d.sum_all()?.to_scalar()?;
            logdenom_acc += sum_val as f64;
            n_acc += c;
        }

        let mi = t_joint_mean as f64 - logdenom_acc / n_acc as f64;
        Ok(if self.floor_zero { mi.max(0.0) } else { mi })
    }
}

// ── Fusion gain ───────────────────────────────────────────────────────────

/// `delta_I = I(fused; Y) - max_i I(unimodal_i; Y)`.
pub fn delta_information(
    mi_by_name: &std::collections::HashMap<String, f64>,
    fused_key: &str,
    unimodal_keys: &[&str],
) -> f64 {
    let best_uni = unimodal_keys
        .iter()
        .filter_map(|k| mi_by_name.get(*k))
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    mi_by_name.get(fused_key).cloned().unwrap_or(0.0) - best_uni
}

// ── Validation: correlated Gaussians ─────────────────────────────────────

/// Sample `(Z, Y)` with known closed-form `I(Z; Y) = -(dim/2) log(1 - ρ²)`.
pub fn correlated_gaussians(
    n: usize,
    dim: usize,
    rho: f64,
    seed: u64,
) -> Result<(Array2<f32>, Array2<f32>, f64)> {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    if !(-1.0 < rho && rho < 1.0) {
        return Err(anyhow::anyhow!("rho must lie strictly in (-1, 1), got {rho}"));
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise_scale = (1.0 - rho * rho).sqrt() as f32;

    let z = Array2::from_shape_fn((n, dim), |_| rng.sample::<f64, _>(StandardNormal) as f32);
    let y = Array2::from_shape_fn((n, dim), |(i, j)| {
        let noise: f32 = rng.sample::<f64, _>(StandardNormal) as f32;
        rho as f32 * z[[i, j]] + noise_scale * noise
    });

    let true_mi = -0.5 * dim as f64 * (1.0 - rho * rho).ln();
    Ok((z, y, true_mi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlated_gaussians_shape() {
        let (z, y, mi) = correlated_gaussians(50, 4, 0.9, 0).unwrap();
        assert_eq!(z.shape(), &[50, 4]);
        assert_eq!(y.shape(), &[50, 4]);
        // True MI at rho=0.9, dim=4: -(4/2) * ln(1 - 0.81) ≈ 3.46 nats
        assert!(mi > 3.0);
    }

    #[test]
    fn delta_information_positive() {
        let mut map = std::collections::HashMap::new();
        map.insert("cam".to_string(), 1.0);
        map.insert("lid".to_string(), 1.5);
        map.insert("fused".to_string(), 2.5);
        let d = delta_information(&map, "fused", &["cam", "lid"]);
        assert!((d - 1.0).abs() < 1e-9);
    }
}
