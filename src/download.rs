/*!
Griffin dataset downloader — fetches subsets from Hugging Face.

The full dataset is ~967 GB.  Four predefined subsets are provided with
altitude-stratified scene counts.  The downloader supports a "minimal"
mode (metadata + LiDAR + front camera only) and optionally auto-extracts
downloaded ZIPs.
*/

use anyhow::{anyhow, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

const HF_REPO: &str = "wjh-svm/Griffin";

#[derive(Debug, Clone)]
pub struct SubsetInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub approx_gb: f64,
    pub n_scenes: usize,
}

const SUBSETS: &[SubsetInfo] = &[
    SubsetInfo {
        name: "griffin_50scenes_25m",
        description: "50 scenes at 25 m drone altitude",
        approx_gb: 167.0,
        n_scenes: 47,
    },
    SubsetInfo {
        name: "griffin_50scenes_40m",
        description: "50 scenes at 40 m drone altitude",
        approx_gb: 190.0,
        n_scenes: 54,
    },
    SubsetInfo {
        name: "griffin_50scenes_55m",
        description: "50 scenes at 55 m drone altitude",
        approx_gb: 175.0,
        n_scenes: 50,
    },
    SubsetInfo {
        name: "griffin_100scenes_random",
        description: "100 scenes at random altitudes",
        approx_gb: 435.0,
        n_scenes: 104,
    },
];

/// Print a table of available subsets with sizes.
pub fn list_available_subsets() {
    println!("{:<30} {:>8}  {:<10}  {}", "Subset", "~GB", "Scenes", "Description");
    println!("{}", "-".repeat(80));
    for s in SUBSETS {
        println!(
            "{:<30} {:>8.0}  {:<10}  {}",
            s.name, s.approx_gb, s.n_scenes, s.description
        );
    }
}

/// Resolve a subset name to its info struct.
pub fn find_subset(name: &str) -> Option<&'static SubsetInfo> {
    SUBSETS.iter().find(|s| s.name == name)
}

/// Files included in the minimal download (metadata + LiDAR + front camera).
fn minimal_file_patterns() -> Vec<&'static str> {
    vec!["metadata/", "calib/", "pose/", "labels/", "lidar/", "camera/front/"]
}

/// Download a Griffin subset from Hugging Face.
///
/// # Arguments
/// * `subset`   – one of the four preset names.
/// * `minimal`  – if `true`, only metadata, LiDAR, and front-camera files.
/// * `extract`  – auto-extract downloaded ZIPs.
pub fn download_griffin(subset: &str, minimal: bool, extract: bool) -> Result<()> {
    let info = find_subset(subset)
        .ok_or_else(|| anyhow!("Unknown subset '{}'. Run with --list to see options.", subset))?;

    println!("Downloading subset '{}' (~{:.0} GB)", info.name, info.approx_gb);
    if minimal {
        println!("Minimal mode: only metadata, LiDAR, and front-camera files.");
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let api_url = format!(
        "https://huggingface.co/api/datasets/{}/tree/main/{}",
        HF_REPO, info.name
    );

    let resp = client
        .get(&api_url)
        .header("User-Agent", "fault-injectors-rs/0.1")
        .send()
        .context("Failed to contact Hugging Face API")?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Hugging Face API returned status {}",
            resp.status()
        ));
    }

    let file_list: serde_json::Value = resp.json()?;
    let files: Vec<String> = file_list
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|f| f["path"].as_str().map(String::from))
        .filter(|path| {
            if minimal {
                minimal_file_patterns()
                    .iter()
                    .any(|pat| path.contains(pat))
            } else {
                true
            }
        })
        .collect();

    if files.is_empty() {
        return Err(anyhow!("No files found for subset '{}'.", subset));
    }

    println!("Found {} file(s) to download.", files.len());

    let out_dir = PathBuf::from("griffin-release").join(info.name);
    std::fs::create_dir_all(&out_dir)?;

    let bar = ProgressBar::new(files.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
            .unwrap(),
    );

    for file_path in &files {
        let dl_url = format!(
            "https://huggingface.co/datasets/{}/resolve/main/{}",
            HF_REPO, file_path
        );

        let local_path = out_dir.join(
            file_path
                .trim_start_matches(&format!("{}/", info.name)),
        );
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        bar.set_message(
            file_path
                .split('/')
                .last()
                .unwrap_or(file_path)
                .to_string(),
        );

        download_file(&client, &dl_url, &local_path)?;

        if extract && local_path.extension().and_then(|e| e.to_str()) == Some("zip") {
            extract_zip(&local_path, local_path.parent().unwrap_or(&out_dir))?;
        }

        bar.inc(1);
    }

    bar.finish_with_message("Done.");
    println!(
        "Downloaded to {}",
        out_dir.display()
    );

    Ok(())
}

fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
) -> Result<()> {
    use std::io::Write;

    if dest.exists() {
        return Ok(());
    }

    let mut resp = client
        .get(url)
        .header("User-Agent", "fault-injectors-rs/0.1")
        .send()
        .with_context(|| format!("GET {url}"))?;

    if !resp.status().is_success() {
        return Err(anyhow!("HTTP {} for {}", resp.status(), url));
    }

    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("Cannot create {}", dest.display()))?;

    resp.copy_to(&mut file)
        .with_context(|| format!("Writing {}", dest.display()))?;

    Ok(())
}

fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("Not a valid ZIP: {}", zip_path.display()))?;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let out_path = dest_dir.join(entry.name());
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)?;
        } else {
            if let Some(p) = out_path.parent() {
                std::fs::create_dir_all(p)?;
            }
            let mut out_file = std::fs::File::create(&out_path)?;
            std::io::copy(&mut entry, &mut out_file)?;
        }
    }

    Ok(())
}
