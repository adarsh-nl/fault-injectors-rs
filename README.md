# fault-injectors-rs

A **Rust port** of the [Griffin fault-injection and information-quality toolkit](https://github.com/adarsh-nl/Fault-Injectors).

The original codebase is written in Python and targets the [Griffin aerial-ground cooperative perception dataset](https://huggingface.co/datasets/wjh-svm/Griffin). This port translates every module to idiomatic Rust while preserving the same interfaces, algorithms, and mathematical semantics.

---

## Repository structure

```
fault-injectors-rs/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── main.rs                          # fault-injectors CLI
    ├── bin/
    │   └── run_mi.rs                    # run-mi CLI
    ├── data_loaders.rs                  # PLY / PNG / JSON / annotation loading
    ├── transforms.rs                    # Coordinate-frame transforms & 3-D boxes
    ├── visualisation.rs                 # BEV, surround-camera, fusion plots
    ├── download.rs                      # Hugging Face dataset downloader
    ├── fault_injectors/
    │   ├── missing_modality.rs          # Bernoulli sensor dropout
    │   └── temporal_misalignment.rs     # Index-shift stale-image pairing
    └── info_quality/
        ├── preprocessing.rs             # PCA (nalgebra SVD) + standardisation
        ├── estimators.rs                # InfoNCE & SMILE MI estimators (candle)
        ├── feature_extraction.rs        # FeatureSet, FeatureCollector, NPZ I/O
        ├── reporting.rs                 # Comparison table, fusion summary, bar chart
        └── run_mi.rs                    # Shared estimation logic
```

---

## Python → Rust dependency mapping

| Python | Rust crate |
|--------|------------|
| `numpy` | `ndarray` |
| `torch` / `torch.nn` | `candle-core` + `candle-nn` |
| `sklearn.decomposition.PCA` | `nalgebra` (full SVD) |
| `matplotlib` | `plotters` |
| `plyfile` | `ply-rs` |
| `huggingface_hub` | `reqwest` (blocking HTTP) |
| `np.random.default_rng` | `rand` + `rand_distr` |
| `opencv-python` / `Pillow` | `image` |
| `argparse` | `clap` |

---

## Modules

### Fault injectors

#### `fault_injectors::missing_modality`

Bernoulli sensor dropout — independently drops the camera or LiDAR each frame with a configurable probability.

```rust
use fault_injectors_rs::fault_injectors::missing_modality::MissingModalityInjector;

let mut inj = MissingModalityInjector::new(
    0.0,    // p_drop_rgb
    0.5,    // p_drop_lidar
    "zero", // fill style for dropped images
    None,   // mean_value (required only for fill = "mean")
    0,      // seed
)?;

let result = inj.inject(&image, &points)?;
// result.image    — (H, W, 3) u8, zeroed if dropped
// result.points   — (N, 4) f32, empty if dropped
// result.m_rgb    — 1 alive / 0 dropped
// result.m_lidar  — 1 alive / 0 dropped
```

#### `fault_injectors::temporal_misalignment`

Pairs the current LiDAR scan `P_k` with a stale image `I_{k - delta_k}`.  
`delta_k ~ round(Normal(mu_delay, sigma_jitter) / frame_period)`, clamped ≥ 0.

```rust
use fault_injectors_rs::fault_injectors::temporal_misalignment::TemporalMisalignmentInjector;

let mut inj = TemporalMisalignmentInjector::new(
    0.2,  // mu_delay  (seconds)
    0.05, // sigma_jitter (seconds)
    10.0, // fps
    0,    // seed
)?;

let (k_image, delta_k) = inj.stale_index(k, /*k_min=*/ 0);
// Load images[k_image] paired with points[k]
```

### Data loaders

`data_loaders::get_file_lists` discovers all sensor files for a Griffin scene directory.  
`load_image` → `(H, W, 3)` u8 RGB array.  
`load_lidar` → `(N, 4)` f32 `[x, y, z, intensity]` in ego frame (extrinsic applied).  
`load_pose` → 4×4 ego-to-world matrix + raw JSON.  
`load_camera_calibration` → intrinsic K and extrinsic matrices.  
`parse_label_txt` → `Vec<Annotation>` with category, dimensions, position, yaw, track ID, visibility.

### Coordinate transforms

All transforms operate in the Griffin frame conventions:

- **Ego**: X forward, Y left, Z up — where LiDAR data and annotations live.
- **Camera** (OpenCV): X right, Y down, Z forward.
- **ENU (world)**: East, North, Up.

```rust
use fault_injectors_rs::transforms;

// Project ego-frame LiDAR to image pixels (filters behind camera / out of bounds)
let proj = transforms::project_lidar_to_image(&points, &K, &T_ego_to_cam, w, h, 0.1);

// 8 corners of a 3-D bounding box in ego frame
let corners = transforms::ego_box_corners_3d(position, dims, yaw);

// 4 BEV footprint corners
let bev = transforms::ann_to_ego_corners_bev(position, dims, yaw);

// Transform ego-frame points to ENU world frame
let world_pts = transforms::ego_points_to_world(&points, &T_ego_to_world);
```

### Information quality

Two MI estimators with different biases, so "fusion adds information" is hard to dismiss as an estimator artefact.

#### InfoNCE (van den Oord et al. 2018)
Contrastive lower bound. Bound is capped at `log(N)` nats.

#### SMILE (Song & Ermon 2020)
Clipped variational lower bound. Clips the log-density ratio to `[-clip, +clip]` to prevent MINE's divergence spiral. No `log(N)` ceiling.

```rust
use fault_injectors_rs::info_quality::estimators::{InfoNCEEstimator, SMILEEstimator};

let result = InfoNCEEstimator::default().estimate(&Z, &Y, None)?;
println!("I(Z;Y) ≈ {:.4} nats  [InfoNCE]", result.mi_nats);

let result = SMILEEstimator::default().estimate(&Z, &Y, None)?;
println!("I(Z;Y) ≈ {:.4} nats  [SMILE]", result.mi_nats);
```

Fusion gain:
```rust
use fault_injectors_rs::info_quality::estimators::delta_information;

let delta = delta_information(&mi_map, "fused", &["camera", "lidar"]);
// delta > 0 → fusion is synergistic
```

---

## CLI binaries

### `fault-injectors`

```
fault-injectors download --subset griffin_50scenes_25m --minimal
fault-injectors inject-missing --data-dir ./scene_01 --p-drop-lidar 0.5
fault-injectors inject-temporal --data-dir ./scene_01 --mu-delay 0.2
```

### `run-mi`

```
run-mi --features features.npz --estimators infonce smile
run-mi --features features.npz --fused fused --unimodal camera lidar --plot mi.png --save mi.json
```

---

## Building

Install Rust via [rustup](https://rustup.rs), then:

```bash
cargo build --release

# Run tests
cargo test

# Run MI estimation CLI
./target/release/run-mi --features features.npz --estimators infonce smile
```

### Optional: GPU acceleration

The MI estimators use [candle](https://github.com/huggingface/candle) and default to CPU. Enable CUDA in `Cargo.toml`:

```toml
candle-core = { version = "0.8", features = ["cuda"] }
candle-nn  = { version = "0.8", features = ["cuda"] }
```

---

## Relation to the Python original

This repository is a direct translation of [`adarsh-nl/Fault-Injectors`](https://github.com/adarsh-nl/Fault-Injectors). The Python original is the reference implementation and is under active development. Algorithmic changes made there should be ported here.
