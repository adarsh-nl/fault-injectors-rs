use anyhow::Result;
use clap::{Parser, Subcommand};

mod data_loaders;
mod download;
mod fault_injectors;
mod info_quality;
mod transforms;
mod visualisation;

#[derive(Parser)]
#[command(name = "fault-injectors", about = "Griffin fault-injection toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download the Griffin dataset from Hugging Face
    Download {
        #[arg(long, default_value = "griffin_50scenes_25m")]
        subset: String,
        #[arg(long)]
        minimal: bool,
        #[arg(long)]
        no_extract: bool,
        #[arg(long)]
        list: bool,
    },
    /// Run missing-modality injection on a dataset directory
    InjectMissing {
        #[arg(long)]
        data_dir: String,
        #[arg(long, default_value = "0.0")]
        p_drop_rgb: f64,
        #[arg(long, default_value = "0.0")]
        p_drop_lidar: f64,
        #[arg(long, default_value = "zero")]
        fill: String,
        #[arg(long, default_value = "0")]
        seed: u64,
    },
    /// Run temporal-misalignment injection on a dataset directory
    InjectTemporal {
        #[arg(long)]
        data_dir: String,
        #[arg(long, default_value = "0.2")]
        mu_delay: f64,
        #[arg(long, default_value = "0.05")]
        sigma_jitter: f64,
        #[arg(long, default_value = "10.0")]
        fps: f64,
        #[arg(long, default_value = "0")]
        seed: u64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download { subset, minimal, no_extract, list } => {
            if list {
                download::list_available_subsets();
            } else {
                download::download_griffin(&subset, minimal, !no_extract)?;
            }
        }
        Commands::InjectMissing {
            data_dir, p_drop_rgb, p_drop_lidar, fill, seed,
        } => {
            use fault_injectors::missing_modality::MissingModalityInjector;

            let file_lists = data_loaders::get_file_lists(&data_dir, false)?;
            let mut injector =
                MissingModalityInjector::new(p_drop_rgb, p_drop_lidar, &fill, None, seed)?;

            println!("Running missing-modality injection over {} frames ...", file_lists.lidar.len());
            let schedule = injector.simulate_sequence(file_lists.lidar.len());
            let alive_rgb: usize = schedule.m_rgb.iter().filter(|&&v| v == 1).count();
            let alive_lidar: usize = schedule.m_lidar.iter().filter(|&&v| v == 1).count();
            let n = file_lists.lidar.len();
            println!(
                "Camera alive: {}/{n}  LiDAR alive: {}/{n}",
                alive_rgb, alive_lidar
            );
        }
        Commands::InjectTemporal {
            data_dir, mu_delay, sigma_jitter, fps, seed,
        } => {
            use fault_injectors::temporal_misalignment::TemporalMisalignmentInjector;

            let file_lists = data_loaders::get_file_lists(&data_dir, false)?;
            let mut injector = TemporalMisalignmentInjector::new(mu_delay, sigma_jitter, fps, seed)?;

            let n = file_lists.lidar.len();
            println!("Running temporal-misalignment injection over {n} frames ...");
            let shifts = injector.simulate_sequence(n);
            let mean_shift: f64 = shifts.iter().sum::<i64>() as f64 / n as f64;
            let max_shift = shifts.iter().copied().max().unwrap_or(0);
            println!("mean delta_k = {mean_shift:.2}  max delta_k = {max_shift}");
        }
    }

    Ok(())
}
