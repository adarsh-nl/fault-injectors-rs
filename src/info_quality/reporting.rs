/*!
Text and chart reporting for mutual information estimation results.

`format_table` — fixed-width comparison table (estimator columns × representation rows).
`fusion_summary` — per-estimator `delta_I` verdict with agreement check.
`plot_comparison` — grouped bar chart saved to PNG.
*/

use anyhow::{anyhow, Result};
use std::collections::HashMap;

// ── Verdict string ────────────────────────────────────────────────────────

fn verdict(delta: f64, tol: f64) -> String {
    if delta > tol {
        format!("fusion ADDS information (+{delta:.4f} nats)")
    } else if delta >= -tol {
        format!("fusion is REDUNDANT ({delta:+.4f} nats)")
    } else {
        format!("fusion LOSES information ({delta:+.4f} nats)")
    }
}

// ── Comparison table ──────────────────────────────────────────────────────

/// Build a fixed-width comparison table string.
///
/// `results` maps `estimator_name → (representation_name → mi_nats)`.
/// Columns are estimators; rows are representations.
pub fn format_table(
    results: &HashMap<String, HashMap<String, f64>>,
    representations: Option<&[&str]>,
) -> String {
    let estimators: Vec<&String> = {
        let mut keys: Vec<&String> = results.keys().collect();
        keys.sort();
        keys
    };

    let default_reps: Vec<String>;
    let reps: Vec<&str> = if let Some(r) = representations {
        r.to_vec()
    } else {
        default_reps = results
            .values()
            .next()
            .map(|m| {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                keys
            })
            .unwrap_or_default();
        default_reps.iter().map(|s| s.as_str()).collect()
    };

    let name_w = reps
        .iter()
        .map(|r| r.len())
        .chain(std::iter::once("representation".len()))
        .max()
        .unwrap_or(14);

    let col_w = estimators
        .iter()
        .map(|e| e.len())
        .chain(std::iter::once(12))
        .max()
        .unwrap_or(12);

    let header = format!(
        "{:<name_w$}  {}",
        "representation",
        estimators
            .iter()
            .map(|e| format!("{:>col_w$}", e))
            .collect::<Vec<_>>()
            .join("  ")
    );
    let sep = "-".repeat(header.len());

    let mut lines = vec![sep.clone(), header, sep.clone()];

    for rep in &reps {
        let cells: Vec<String> = estimators
            .iter()
            .map(|e| {
                results[*e]
                    .get(*rep)
                    .map(|&v| format!("{:>col_w$.4f}", v))
                    .unwrap_or_else(|| format!("{:>col_w$}", "nan"))
            })
            .collect();
        lines.push(format!("{:<name_w$}  {}", rep, cells.join("  ")));
    }
    lines.push(sep);
    lines.join("\n")
}

// ── Fusion summary ────────────────────────────────────────────────────────

/// Per-estimator `delta_I` verdict plus an agreement check.
pub fn fusion_summary(
    results: &HashMap<String, HashMap<String, f64>>,
    fused_key: &str,
    unimodal_keys: &[&str],
    tol: f64,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut deltas: HashMap<&str, f64> = HashMap::new();

    let mut est_names: Vec<&String> = results.keys().collect();
    est_names.sort();

    for est in &est_names {
        let mi = &results[*est];
        let best_uni = unimodal_keys
            .iter()
            .filter_map(|k| mi.get(*k))
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let delta = mi.get(fused_key).cloned().unwrap_or(0.0) - best_uni;
        deltas.insert(est.as_str(), delta);
        lines.push(format!(
            "  {:<10} delta_I = {:+.4f}  ->  {}",
            est, delta, verdict(delta, tol)
        ));
    }

    let sign_set: std::collections::HashSet<&'static str> = deltas
        .values()
        .map(|&d| {
            if d > tol { "+" } else if d < -tol { "-" } else { "0" }
        })
        .collect();

    lines.push(String::new());
    if sign_set.len() == 1 {
        lines.push(
            "All estimators AGREE on the direction of fusion gain \
             (conclusion is estimator-agnostic)."
                .to_string(),
        );
    } else {
        lines.push(
            "Estimators DISAGREE on the direction of fusion gain \
             (treat with caution; consider more epochs / samples)."
                .to_string(),
        );
    }

    lines.join("\n")
}

// ── Bar chart ─────────────────────────────────────────────────────────────

/// Grouped bar chart: one group per representation, one bar per estimator.
/// Saved as a PNG at `out_path`.
pub fn plot_comparison(
    results: &HashMap<String, HashMap<String, f64>>,
    out_path: &str,
    representations: Option<&[&str]>,
    title: &str,
) -> Result<String> {
    use plotters::prelude::*;

    let estimators: Vec<&String> = {
        let mut keys: Vec<&String> = results.keys().collect();
        keys.sort();
        keys
    };

    let default_reps: Vec<String>;
    let reps: Vec<&str> = if let Some(r) = representations {
        r.to_vec()
    } else {
        default_reps = results
            .values()
            .next()
            .map(|m| {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                keys
            })
            .unwrap_or_default();
        default_reps.iter().map(|s| s.as_str()).collect()
    };

    if reps.is_empty() {
        return Err(anyhow!("No representations to plot"));
    }

    let width = (1.6 * reps.len() as f64 * 100.0 + 300.0) as u32;
    let height = 500u32;

    let root = BitMapBackend::new(out_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    // Gather all MI values to find y range
    let all_vals: Vec<f64> = results
        .values()
        .flat_map(|m| m.values())
        .cloned()
        .filter(|v| v.is_finite())
        .collect();

    let y_max = all_vals.iter().cloned().fold(0.0_f64, f64::max) * 1.15;
    let y_min = all_vals.iter().cloned().fold(0.0_f64, f64::min).min(0.0) * 1.1;

    let n_groups = reps.len();
    let n_est = estimators.len();
    let bar_w = 0.8 / n_est.max(1) as f64;

    // Use x-axis as group index (0..n_groups)
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (-0.5_f64)..(n_groups as f64 - 0.5),
            y_min..y_max,
        )?;

    chart
        .configure_mesh()
        .x_labels(n_groups)
        .x_label_formatter(&|v: &f64| {
            let idx = *v as usize;
            reps.get(idx).unwrap_or(&"").to_string()
        })
        .y_desc("MI lower bound (nats)")
        .draw()?;

    let palette: Vec<RGBColor> = vec![
        RGBColor(0, 114, 189),
        RGBColor(217, 83, 25),
        RGBColor(126, 47, 142),
        RGBColor(119, 172, 48),
    ];

    for (ei, est) in estimators.iter().enumerate() {
        let color = palette[ei % palette.len()];
        let offset = (ei as f64 - (n_est as f64 - 1.0) / 2.0) * bar_w;

        let bar_data: Vec<(f64, f64)> = reps
            .iter()
            .enumerate()
            .filter_map(|(gi, rep)| {
                results[*est].get(*rep).map(|&v| (gi as f64 + offset, v))
            })
            .collect();

        chart.draw_series(bar_data.iter().map(|&(x, h)| {
            let x0 = x - bar_w / 2.0;
            let x1 = x + bar_w / 2.0;
            Rectangle::new([(x0, 0.0_f64.min(h)), (x1, 0.0_f64.max(h))], color.filled())
        }))?
        .label(est.as_str())
        .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled()));
    }

    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;

    Ok(out_path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_results() -> HashMap<String, HashMap<String, f64>> {
        let mut m = HashMap::new();
        let mut infonce = HashMap::new();
        infonce.insert("camera".to_string(), 0.8);
        infonce.insert("lidar".to_string(), 1.2);
        infonce.insert("fused".to_string(), 1.9);
        m.insert("infonce".to_string(), infonce);
        let mut smile = HashMap::new();
        smile.insert("camera".to_string(), 0.9);
        smile.insert("lidar".to_string(), 1.3);
        smile.insert("fused".to_string(), 2.1);
        m.insert("smile".to_string(), smile);
        m
    }

    #[test]
    fn table_contains_headers() {
        let results = sample_results();
        let table = format_table(&results, None);
        assert!(table.contains("representation"));
        assert!(table.contains("infonce"));
        assert!(table.contains("smile"));
        assert!(table.contains("camera"));
    }

    #[test]
    fn summary_agreement() {
        let results = sample_results();
        let summary = fusion_summary(&results, "fused", &["camera", "lidar"], 0.01);
        assert!(summary.contains("AGREE"));
    }
}
