## Supernova PDS Trigger Pipeline (VD/HD)

This repository contains a configurable pipeline to study and estimate the trigger efficiency for supernova (SN) neutrino events in liquid argon TPCs. It supports both DUNE VD (vertical drift) and HD (horizontal drift) geometries.

At a high level, the pipeline:
- loads simulated SN signal and detector background hits from ROOT files,
- clusters hits to form candidate optical activity bursts,
- extracts physics-motivated features from clusters,
- optionally trains a BDT to separate signal-like from background-like clusters,
- and evaluates the expected trigger efficiency at a target fake-trigger rate using statistical tests.


### Key capabilities
- **Modular stages** controlled by YAML: `load_events` → `clustering` → `cluster_feature_extraction` → `bdt_training` → `trigger_efficiency`.
- **Configurable data sources** for SN channels and background types, with file name patterns and limits.
- **Rich console UI** for progress and summary tables.
- **Optionally save and reload intermediate outputs** per stage.
- **Physics knobs**: SN spectra, cross sections, energy cuts, error model, and test statistic.


## Repository structure

- **`scripts/`**: main code modules
  - `hit_stat.py`: CLI entry point orchestrating the full pipeline
  - `stage_manager.py`: runs the configured stage list with IO and error behavior
  - `stages.py`: stage implementations (`LoadEvents`, `Clustering`, `ClusterFeatureExtraction`, `BDTTraining`, `TriggerEfficiency`)
  - `data_loader.py`: reads ROOT files via `uproot`, builds per-event hit lists and truth info
  - `config.py`: YAML loader and access helpers; applies defaults; loads coordinate/aux arrays
  - `argument_parser.py`: CLI (`-c`, `-i`, `-o`, `--info_output`)
  - `saver.py`: wrapper to persist/load stage data with embedded config
  - `clustering.py`, `classifier.py`, `trigger_efficiency_computer.py`, `enhanced_histogram.py`, etc.: core algorithms
  - `gui.py`: rich console helpers for progress and tables
  - `marley_reaction_files/`: reaction input files; used only if reweighting is enabled
- **`configs/`**: YAML configuration files
  - `default_config_vd.yaml`, `default_config_hd.yaml`: main defaults for VD/HD
  - many task-specific or optimized configs under subfolders
- **`cross-sections/`**: cross section inputs used by the trigger-efficiency stage
- **`eval_and_testing/`**: test inputs/plots (not needed to run the pipeline)


## How the pipeline works

Stages are configured via the YAML key `Stages`, in order. The default VD pipeline is:

```yaml
Stages:
  - load_events
  - clustering
  - cluster_feature_extraction
  - bdt_training
  - trigger_efficiency
```

- **`load_events`** (`LoadEvents`):
  - Loads SN and background hits using `data_loader.py` with file patterns provided in the YAML.
  - Splits SN and BG into BDT-training and efficiency subsets according to `train_split` and `classifier_energy_limit`.
  - Outputs per-channel arrays: e.g. `sn_eff_hit_list_per_event`, `bg_eff_hit_list_per_event`, and corresponding truth info.

- **`clustering`** (`Clustering`):
  - Groups hits into clusters given time/space thresholds (`max_cluster_time`, `max_hit_time_diff`, `max_hit_distance`, etc.).
  - Produces final SN/BG clusters for both efficiency and training streams.

- **`cluster_feature_extraction`** (`ClusterFeatureExtraction`):
  - Computes features (e.g., multiplicities, shapes) for SN and BG clusters.
  - Builds arrays and per-event groupings; prepares targets for training.

- **`bdt_training`** (`BDTTraining`, optional):
  - Trains a scikit-learn histogram gradient-boosted tree (HGBT) on SN vs BG feature arrays.
  - Can fix hyperparameters or run an optimization.

- **`trigger_efficiency`** (`TriggerEfficiency`):
  - Constructs the expected BG histogram over the configured burst window and evaluates the statistic (e.g. log-likelihood ratio).
  - Optionally weighs SN clusters by a chosen SN spectrum and cross section.
  - Scans over distances or number of interactions and estimates trigger efficiency at the requested fake-trigger rate.

The orchestrator (`StageManager`) enforces stage order and can save outputs per stage according to `Output.stages_with_saved_output_data`.


## Configuration

Use a config in `configs/`, or write your own. The defaults are:
- VD: `configs/default_config_vd.yaml`
- HD: `configs/default_config_hd.yaml`

Important sections and keys:

- **Top-level**
  - `DEFAULT_CONFIG_FILE`: if set, values from that file act as defaults (the active file overrides them). Many provided configs inherit from a default by pointing this to a base YAML.
  - `Stages`: the ordered list of stages to run.

- **`Input` / `Output`**
  - `Input.input_data_file`: path to a previously saved `DataWithConfig` to start from (only valid if the first stage is not `load_events`).
  - `Output.verbosity`: `DEBUG|INFO|WARNING|ERROR`.
  - `Output.stages_with_saved_output_data`: `all`, `last`, a list of stage names, or `null` to disable saving.
  - `Output.output_data_file`: if saving a single stage (`last`), a string; otherwise a mapping `{stage_name: path}`.
  - `Output.output_info_file`: JSON with run metadata and per-stage summaries.
  - `Output.error_behavior`: `kill` or `graceful` (continue after logging the error and write what’s available).

- **`DataFormat`**
  - Expected counts and lengths for SN/BG samples.
  - `aux_coordinate_dir`, `aux_data_dir`: folders containing detector coordinate `.dat` and auxiliary pickle arrays (optical distance arrays). These must be available locally; update the paths to your environment.

- **`Simulation.load_events`**
  - `sn_channels`: e.g. `["cc", "es"]` (VD) or `["cc"]` (HD).
  - `sn_data_dir`: list of directories (one per SN channel).
  - `sn_hit_file_start_pattern`, `sn_hit_file_end_pattern`, `sn_info_file_end_pattern`: filename patterns for SN hit and truth files.
  - `bg_data_dir`, `bg_hit_file_start_pattern`, `bg_hit_file_end_pattern`: background location and patterns.
  - `split_for_classifier`, `classifier_energy_limit`, `train_split`: how SN/BG are split between training and efficiency streams.
  - `modified_bgs`: optional per-type scaling factors to thin certain backgrounds.

- **`Simulation.clustering`**
  - `parameters`: time/space thresholds and multiplicity/neighbour cuts.

- **`Simulation.cluster_feature_extraction`**
  - `cluster_features`: `all` or explicit list.

- **`Simulation.bdt_training`**
  - `optimize_hyperparameters`, `optimize_hyperparameters_random_state`, or `bdt_hyperparameters` to fix values.

- **`Simulation.trigger_efficiency`**
  - `use_classifier`: whether to use the trained BDT or not.
  - `fake_trigger_rate`: target false positive probability per test window.
  - `distance_to_evaluate` or `number_of_interactions_to_evaluate`: choose one mode.
  - `energy_lower_limit`: SN energy cut (relevant for the interactions mode).
  - `burst_time_window`: microseconds covered per test window.
  - `physics`: SN spectra and cross sections. For VD, CC is dominant; ES is supported.
  - `statistical_info`: `histogram_variable` (`bdt_output` or `hit_multiplicity`), `statistical_method` (`log_likelihood_ratio`, `pearson_chi_squared`, or `cash`), `histogram_bunch_threshold`, `use_bg_variations`.
  - `error_info`: `bayesian` or `frequentist` with parameters.


## Installation

Python 3.8+ recommended.

Minimal dependencies (install with pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib seaborn scikit-learn uproot rich plotille mergedeep pyyaml
```

Optional (advanced CC reweighting): requires MARLEY and `py_marley` available on your system. This is disabled by default (`do_reweighting: False`). If enabling, ensure the corresponding environment variables and shared library paths are configured for your platform.

Note: `scripts/requirements.txt` lists a very broad environment; prefer the minimal set above unless you specifically need extras.


## Data requirements

You must have local access to the SN and BG ROOT files referenced in your YAML:
- Update `Simulation.load_events.sn_data_dir` (one path per `sn_channel`).
- Update `Simulation.load_events.bg_data_dir`.
- Ensure the start/end filename patterns match your files.
- Ensure `DataFormat.aux_coordinate_dir` and `DataFormat.aux_data_dir` point to valid local folders containing the detector coordinates and auxiliary arrays.

If you just want to test the pipeline wiring (without large datasets), reduce `sn_file_limit`/`bg_file_limit` and run with very small samples.


## Quick start

1) Create and activate a Python environment, then install dependencies (see Installation).

2) Copy a config and adjust paths. For VD, start from `configs/default_config_vd.yaml`; for HD, use `configs/default_config_hd.yaml`.

3) Run the pipeline. The simplest invocation only needs a config:

```bash
python scripts/hit_stat.py -c configs/default_config_vd.yaml
```

Optional arguments:
- `-i, --input`: start from a previously saved `DataWithConfig` pickle (only if the first stage is not `load_events`).
- `-o, --output`: override `Output.output_data_file`.
- `--info_output`: override `Output.output_info_file`.

Examples:

```bash
# VD end-to-end run with defaults
python scripts/hit_stat.py -c configs/default_config_vd.yaml

# HD run
python scripts/hit_stat.py -c configs/default_config_hd.yaml

# Save stage outputs to a custom location (when saving is enabled in YAML)
python scripts/hit_stat.py -c configs/default_config_vd.yaml -o /path/to/output.pkl --info_output /path/to/run_info.json
```

Outputs:
- Stage data (if enabled) are pickled via `saver.DataWithConfig`, either per-stage or only for the last stage, according to `Output.stages_with_saved_output_data`.
- A run-info JSON is written to `Output.output_info_file`, aggregating per-stage summaries and the final parameters used.


## Tips and troubleshooting

- If you see missing-file errors, double-check directory paths and filename patterns in YAML. The loader verifies ROOT tree presence and will skip corrupted/mismatched files.
- `Output.error_behavior: graceful` helps exploratory runs continue past a failing stage while still saving what’s available.
- For quick tests, reduce `sn_file_limit`, `bg_file_limit`, and `number_of_tests`.
- Reweighting (MARLEY-based) is experimental and disabled by default; enable only if your environment is set up for it.


## Utilities

- `scripts/input_file_counter.py`: counts available input files and basic stats for your config. Invoke with `-c <config>`.


