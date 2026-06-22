/*!
Helpers shared between the `run-mi` binary and any programmatic callers.

The actual CLI entry point lives in `src/bin/run_mi.rs`.
*/

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use super::estimators::{InfoNCEEstimator, MIResult, SMILEEstimator};
use super::feature_extraction::FeatureSet;

/// Load a feature file and estimate MI for every representation with the
/// given estimator settings.
pub fn run_estimation(
    features_path: &Path,
    estimators: &[&str],
    pca_dims: Option<usize>,
    holdout: f64,
    seed: u64,
    infonce_epochs: usize,
    infonce_batch: usize,
    infonce_temperature: f64,
    infonce_lr: f64,
    smile_epochs: usize,
    smile_batch: usize,
    smile_clip: f64,
    smile_lr: f64,
) -> Result<HashMap<String, HashMap<String, f64>>> {
    let fs = FeatureSet::load_npz(features_path)?;

    let mut rep_names: Vec<&str> = fs.features.keys().map(|s| s.as_str()).collect();
    rep_names.sort();

    let mut results: HashMap<String, HashMap<String, f64>> = HashMap::new();

    for &est_name in estimators {
        let mut rep_mi: HashMap<String, f64> = HashMap::new();

        for &rep in &rep_names {
            let z_arr = &fs.features[rep];
            let y_arr = &fs.y;

            let result: MIResult = match est_name {
                "infonce" => InfoNCEEstimator {
                    epochs: infonce_epochs,
                    batch_size: infonce_batch,
                    temperature: infonce_temperature,
                    lr: infonce_lr,
                    holdout,
                    seed,
                    ..Default::default()
                }
                .estimate(z_arr, y_arr, pca_dims)?,

                "smile" => SMILEEstimator {
                    epochs: smile_epochs,
                    batch_size: smile_batch,
                    clip: smile_clip,
                    lr: smile_lr,
                    holdout,
                    seed,
                    ..Default::default()
                }
                .estimate(z_arr, y_arr, pca_dims)?,

                other => {
                    return Err(anyhow::anyhow!(
                        "Unknown estimator '{other}'. Valid: infonce, smile"
                    ))
                }
            };

            rep_mi.insert(rep.to_string(), result.mi_nats);
        }

        results.insert(est_name.to_string(), rep_mi);
    }

    Ok(results)
}
