/*!
Architecture-agnostic feature collection for MI estimation.

The MI estimators consume plain `(N, d)` arrays.  This module makes the
model-specific extraction step generic:

  * You provide a forward closure `forward_fn(batch) -> HashMap<name, Tensor>`.
  * The collector calls it over a data loader, pools each tapped output to a
    `(d,)` vector, and returns a [`FeatureSet`] with one `(N, d)` array per tap.

Because Rust generics work at compile time (unlike Python's dynamic hooks),
the taps are expressed as named closures that extract from the forward output,
which is equivalent to the Python `register_forward_hook` approach but without
runtime dynamic dispatch overhead.
*/

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;

// ── Pooling ───────────────────────────────────────────────────────────────

/// Global average pool a flat feature vector.
///
/// Input is already `(d,)` — returned as-is (no pooling needed).
pub fn global_pool_1d(v: &[f32]) -> Array1<f32> {
    Array1::from_vec(v.to_vec())
}

/// Global average pool a `(C, H, W)` feature map to `(C,)`.
pub fn global_pool_chw(data: &[f32], c: usize, h: usize, w: usize) -> Array1<f32> {
    let hw = h * w;
    let mut out = Array1::<f32>::zeros(c);
    for ch in 0..c {
        let sum: f32 = data[ch * hw..(ch + 1) * hw].iter().sum();
        out[ch] = sum / hw as f32;
    }
    out
}

/// Global average pool a `(B, C, H, W)` batch feature map to `(C,)`.
pub fn global_pool_bchw(data: &[f32], b: usize, c: usize, h: usize, w: usize) -> Array1<f32> {
    let chw = c * h * w;
    let mut acc = Array1::<f32>::zeros(c);
    for bi in 0..b {
        let slice = &data[bi * chw..(bi + 1) * chw];
        acc = acc + &global_pool_chw(slice, c, h, w);
    }
    acc / b as f32
}

// ── FeatureSet ────────────────────────────────────────────────────────────

/// Collected features and aligned targets from one extraction run.
pub struct FeatureSet {
    /// One `(N, d)` array per named tap.
    pub features: HashMap<String, Array2<f32>>,
    /// `(N, d_y)` target array.
    pub y: Array2<f32>,
}

impl FeatureSet {
    pub fn new(features: HashMap<String, Array2<f32>>, y: Array2<f32>) -> Result<Self> {
        let n = y.nrows();
        for (name, arr) in &features {
            if arr.nrows() != n {
                return Err(anyhow!(
                    "Tap '{}' has {} rows but Y has {} rows",
                    name, arr.nrows(), n
                ));
            }
        }
        Ok(Self { features, y })
    }

    pub fn n_samples(&self) -> usize {
        self.y.nrows()
    }

    /// Persist to a compressed `.npz`-style file (NumPy binary format via
    /// a simple custom writer).  Key `y` holds the target array; all other
    /// keys are tap names.
    pub fn save_npz(&self, path: &Path) -> Result<()> {
        // Write a minimal NPZ (ZIP of .npy files) compatible with numpy.load.
        use std::io::Write;

        let file = std::fs::File::create(path)?;
        let mut zip = zip::ZipWriter::new(file);
        let opts = zip::write::FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        // Writes a single array as .npy v1.0
        let write_npy = |writer: &mut dyn Write, arr: &Array2<f32>| -> Result<()> {
            let (rows, cols) = (arr.nrows(), arr.ncols());
            let header = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}"
            );
            // Pad header to multiple of 64 bytes
            let magic: &[u8] = b"\x93NUMPY\x01\x00";
            let header_len = header.len() + 1; // +1 for newline
            let padded = ((header_len + 10 + 63) / 64) * 64;
            let padding = padded - 10 - header_len;
            writer.write_all(magic)?;
            let hlen = (padded - 10) as u16;
            writer.write_all(&hlen.to_le_bytes())?;
            writer.write_all(header.as_bytes())?;
            for _ in 0..padding { writer.write_all(b" ")?; }
            writer.write_all(b"\n")?;
            for &v in arr.iter() {
                writer.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        };

        // Y
        zip.start_file("y.npy", opts)?;
        write_npy(&mut zip, &self.y)?;

        for (name, arr) in &self.features {
            zip.start_file(format!("{name}.npy"), opts)?;
            write_npy(&mut zip, arr)?;
        }

        zip.finish()?;
        Ok(())
    }

    /// Load a FeatureSet from an `.npz` file written by [`save_npz`].
    pub fn load_npz(path: &Path) -> Result<Self> {
        use std::io::Read;

        let file = std::fs::File::open(path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        let mut features: HashMap<String, Array2<f32>> = HashMap::new();
        let mut y_opt: Option<Array2<f32>> = None;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();
            let key = name.trim_end_matches(".npy");

            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            let arr = parse_npy_f32(&buf)?;

            if key == "y" {
                y_opt = Some(arr);
            } else {
                features.insert(key.to_string(), arr);
            }
        }

        let y = y_opt.ok_or_else(|| anyhow!("NPZ missing 'y.npy'"))?;
        FeatureSet::new(features, y)
    }
}

impl std::fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shapes: Vec<String> = self
            .features
            .iter()
            .map(|(k, v)| format!("{k}={:?}", v.shape()))
            .collect();
        write!(f, "FeatureSet({}, Y={:?})", shapes.join(", "), self.y.shape())
    }
}

// ── NPY parser (f32 only) ─────────────────────────────────────────────────

fn parse_npy_f32(buf: &[u8]) -> Result<Array2<f32>> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(anyhow!("Not a valid .npy file"));
    }
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header_start = 10;
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len])?;

    // Parse shape from header string (very minimal parser)
    let shape = parse_npy_shape(header)?;
    if shape.len() != 2 {
        return Err(anyhow!("Expected 2D array, got {}D", shape.len()));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let data_start = header_start + header_len;
    let floats: Vec<f32> = buf[data_start..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if floats.len() != rows * cols {
        return Err(anyhow!("Data length mismatch: {} vs {}×{}", floats.len(), rows, cols));
    }
    Ok(Array2::from_shape_vec((rows, cols), floats)?)
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let start = header.find("'shape'").ok_or_else(|| anyhow!("No 'shape' in npy header"))?;
    let after_colon = &header[start + 7..];
    let paren_start = after_colon
        .find('(')
        .ok_or_else(|| anyhow!("No '(' after shape"))?;
    let paren_end = after_colon
        .find(')')
        .ok_or_else(|| anyhow!("No ')' after shape"))?;
    let inner = &after_colon[paren_start + 1..paren_end];
    inner
        .split(',')
        .map(|s| {
            let s = s.trim();
            if s.is_empty() {
                Ok(0)
            } else {
                s.parse::<usize>().map_err(|e| anyhow!("{e}"))
            }
        })
        .filter(|r| r.as_ref().map(|&v| v > 0 || true).unwrap_or(true))
        .collect::<Result<Vec<_>>>()
        .map(|v| v.into_iter().filter(|&x| x > 0 || v.len() == 1).collect())
}

// ── FeatureCollector ──────────────────────────────────────────────────────

/// Collect features from a generic data loader using named forward-pass taps.
///
/// # Example
/// ```rust,ignore
/// let collector = FeatureCollector::new(tap_names);
/// let fs = collector.collect(&mut loader, |batch| {
///     let camera_feat = model.camera_encoder(&batch.images);
///     let lidar_feat = model.lidar_encoder(&batch.points);
///     let fused_feat = model.fuser(&camera_feat, &lidar_feat);
///     let label = batch.label.to_vec_f32();
///     (
///         HashMap::from([
///             ("camera".to_string(), camera_feat.to_vec_f32()),
///             ("lidar".to_string(), lidar_feat.to_vec_f32()),
///             ("fused".to_string(), fused_feat.to_vec_f32()),
///         ]),
///         label,
///     )
/// }, None).unwrap();
/// ```
pub struct FeatureCollector {
    tap_names: Vec<String>,
}

impl FeatureCollector {
    pub fn new(tap_names: Vec<String>) -> Self {
        Self { tap_names }
    }

    /// Run inference over the loader and collect features + targets.
    ///
    /// `forward_fn` receives each batch and must return a
    /// `(HashMap<name, Vec<f32>>, label_vec)` tuple where each feature vector
    /// has the same length across all batches and the label vector has
    /// consistent length.
    ///
    /// `n_samples` caps how many batches are processed.
    pub fn collect<B, F>(
        &self,
        loader: impl Iterator<Item = B>,
        mut forward_fn: F,
        n_samples: Option<usize>,
    ) -> Result<FeatureSet>
    where
        F: FnMut(B) -> (HashMap<String, Vec<f32>>, Vec<f32>),
    {
        let mut tap_bufs: HashMap<String, Vec<Vec<f32>>> = self
            .tap_names
            .iter()
            .map(|n| (n.clone(), Vec::new()))
            .collect();
        let mut label_rows: Vec<Vec<f32>> = Vec::new();

        for (idx, batch) in loader.enumerate() {
            if n_samples.map_or(false, |lim| idx >= lim) {
                break;
            }

            let (taps, label) = forward_fn(batch);

            for name in &self.tap_names {
                let feat = taps
                    .get(name)
                    .ok_or_else(|| anyhow!("forward_fn did not return tap '{name}'"))?;
                tap_bufs
                    .get_mut(name)
                    .unwrap()
                    .push(feat.clone());
            }

            label_rows.push(label);
        }

        if label_rows.is_empty() {
            return Err(anyhow!("No samples collected — loader was empty"));
        }

        let n = label_rows.len();
        let y_dim = label_rows[0].len();
        let y = Array2::from_shape_fn((n, y_dim), |(i, j)| label_rows[i][j]);

        let mut features = HashMap::new();
        for name in &self.tap_names {
            let rows = &tap_bufs[name];
            let d = rows[0].len();
            if rows.iter().any(|r| r.len() != d) {
                return Err(anyhow!("Tap '{name}' returned inconsistent feature lengths"));
            }
            let arr = Array2::from_shape_fn((n, d), |(i, j)| rows[i][j]);
            features.insert(name.clone(), arr);
        }

        FeatureSet::new(features, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_simple() {
        let taps = vec!["feat".to_string()];
        let collector = FeatureCollector::new(taps);

        let loader = (0..5_u32).map(|i| i);
        let fs = collector
            .collect(loader, |x| {
                let feat = vec![x as f32, x as f32 + 1.0];
                let label = vec![x as f32];
                (HashMap::from([("feat".to_string(), feat)]), label)
            }, None)
            .unwrap();

        assert_eq!(fs.n_samples(), 5);
        assert_eq!(fs.features["feat"].shape(), &[5, 2]);
    }
}
