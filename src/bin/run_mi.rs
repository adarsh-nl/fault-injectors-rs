/*!
CLI for estimating mutual information I(Z; Y) between representations and targets.

Loads a `.npz` feature file, estimates I(Z; Y) for every representation with
one or more estimators, prints a comparison table and optional fusion-gain
summary, then saves the MI values to a JSON file.

Usage:
    run-mi --features features.npz --estimators infonce smile
    run-mi --features features.npz --fused fused --unimodal camera lidar --plot mi.png
*/

use anyhow::{anyhow, Result};
use clap::Parser;
use fault_injectors_rs::info_quality::{
    feature_extraction::FeatureSet,
    reporting::{format_table, fusion_summary, plot_comparison},
    run_mi::run_estimation,
};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    name = "run-mi",
    about = "Estimate mutual information I(Z; Y) for one or more representations"
)]
struct Args {
    /// Path to a `.npz` feature file.
    #[arg(long)]
    features: String,

    /// Estimators to use.
    #[arg(long, num_args = 1.., default_values = ["infonce", "smile"])]
    estimators: Vec<String>,

    /// Fused representation key for fusion-gain summary.
    #[arg(long)]
    fused: Option<String>,

    /// Unimodal representation keys for fusion-gain summary.
    #[arg(long, num_args = 0..)]
    unimodal: Vec<String>,

    #[arg(long, default_value = "100")]
    infonce_epochs: usize,
    #[arg(long, default_value = "64")]
    infonce_batch: usize,
    #[arg(long, default_value = "0.07")]
    infonce_temperature: f64,
    #[arg(long, default_value = "1e-3")]
    infonce_lr: f64,

    #[arg(long, default_value = "500")]
    smile_epochs: usize,
    #[arg(long, default_value = "64")]
    smile_batch: usize,
    #[arg(long, default_value = "5.0")]
    smile_clip: f64,
    #[arg(long, default_value = "2e-4")]
    smile_lr: f64,

    #[arg(long)]
    pca_dims: Option<usize>,
    #[arg(long, default_value = "0.0")]
    holdout: f64,
    #[arg(long, default_value = "0")]
    seed: u64,

    /// Save a bar-chart PNG to this path.
    #[arg(long)]
    plot: Option<String>,

    /// Save MI values as JSON to this path.
    #[arg(long)]
    save: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let feat_path = Path::new(&args.features);
    let fs = FeatureSet::load_npz(feat_path)
        .map_err(|e| anyhow!("Failed to load '{}': {e}", args.features))?;

    println!("Loaded: {fs}");

    let est_refs: Vec<&str> = args.estimators.iter().map(|s| s.as_str()).collect();

    let results = run_estimation(
        feat_path,
        &est_refs,
        args.pca_dims,
        args.holdout,
        args.seed,
        args.infonce_epochs,
        args.infonce_batch,
        args.infonce_temperature,
        args.infonce_lr,
        args.smile_epochs,
        args.smile_batch,
        args.smile_clip,
        args.smile_lr,
    )?;

    // Print table
    let rep_names: Vec<String> = {
        let mut keys: Vec<String> = fs.features.keys().cloned().collect();
        keys.sort();
        keys
    };
    let rep_refs: Vec<&str> = rep_names.iter().map(|s| s.as_str()).collect();
    println!("\n{}", format_table(&results, Some(&rep_refs)));

    // Fusion gain
    if let Some(ref fused_key) = args.fused {
        if !args.unimodal.is_empty() {
            let uni_refs: Vec<&str> = args.unimodal.iter().map(|s| s.as_str()).collect();
            println!(
                "\nFusion gain ({fused_key} vs {}):",
                uni_refs.join(", ")
            );
            println!("{}", fusion_summary(&results, fused_key, &uni_refs, 0.01));
        }
    }

    // Plot
    if let Some(ref plot_path) = args.plot {
        plot_comparison(&results, plot_path, Some(&rep_refs), "Mutual information by representation")?;
        println!("\nPlot saved to {plot_path}");
    }

    // Save JSON
    if let Some(ref save_path) = args.save {
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(save_path, json)?;
        println!("Results saved to {save_path}");
    }

    Ok(())
}
