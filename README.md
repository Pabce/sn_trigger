# hit_stat.py

hit_stat.py is the main driver script for performing trigger efficiency analysis.
## Overview

This script handles:
- **Configuration and Setup:** Reads configuration parameters (detector size, simulation time window, background sample information, etc.) from a configuration file.
- **Data Loading:** Loads raw hits *or* pre-processed data (such as efficiency clusters, cluster features, and pre-trained boosted decision tree objects) from saved files.
- **Histogram Generation:** Computes multiple histograms like the background hit multiplicity histogram and the BDT output histogram. The histograms are marked with error bars and plotted using matplotlib.
- **Filtering with Classifier:** Applies a BDT filter to assign a BDT score to each cluster, separating signal and background clusters.
- **Trigger Efficiency Evaluation:** Computes the expected number of supernova (SN) events in a given time window and evaluates the trigger efficiency by comparing the signal and background histograms.
- **Visualization:** Generates multiple figures showing background histograms (both before and after BDT filtering), as well as an expected background histogram. There is additional (commented) code for advanced statistical tests and profiling.
- **Stage Management:** In its `main()` function, the script sets up a stage manager that runs different processing stages (such as data loading, clustering, etc.) and generates summary statistics.
## Usage

The script is designed to be run from the command line. It accepts command-line arguments for:
- The configuration file.
- Optional input file paths.
- Optional output file paths and output info file paths.

A typical invocation might look like:

```bash
python hit_stat.py --config path/to/config_file.yaml --input path/to/input_file --output path/to/output_file --output-info path/to/info_file
```

> **Note:** The command-line arguments are parsed using the `parse_arguments()` function, and any provided file paths will override the settings in the configuration file.

## Configuring a Run Using `default_config_vd.yaml`

The configuration file `configs/default_config_vd.yaml` provides a full set of parametersfor running the trigger algorithm for the Vertical Drift (VD) detector.

### Overview of the configuration file

- **Stages:**
  - Lists the processing stages to execute in order:
    - `load_events`
    - `clustering`
    - `cluster_feature_extraction`
    - `bdt_training`
    - `trigger_efficiency`

- **Input:**
  - `input_data_file`: Specifies the path to the input hit data file. Can be overridden via the command line.

- **Output:**
  - `verbosity`: Sets the logging level (e.g., DEBUG, INFO, WARNING, ERROR).
  - `stages_with_saved_output_data`: Determines if output data should be saved from all stages, just the last stage, or none.
  - `output_data_file`: The file path where the trigger output is stored.
  - `output_info_file`: The file path for additional run information.
  - `error_behavior`: Defines how errors should be handled ("graceful" or "kill").

- **DataFormat:**
  - Specifies data-related settings such as the number of SN events per file (`sn_event_number_per_file`), background sample number and length, and directories for auxiliary data.

- **Simulation:**
  - **load_events:** Configuration for loading signal and background event data, including file limits, directory paths, and file name patterns.
  - **clustering:** Parameters such as `max_cluster_time`, `max_hit_time_diff`, `max_hit_distance`, `min_hit_multiplicity`, and `min_neighbours` used for grouping hits into clusters.
  - **cluster_feature_extraction:** Determines which features to extract for each cluster (`all` or a specified list).
  - **bdt_training:** Settings for training the Boosted Decision Tree, including hyperparameter optimization and threshold settings.
  - **trigger_efficiency:** Contains parameters for the trigger efficiency stage:
    - `use_classifier`: Whether to apply the classifier.
    - `number_of_tests`: The number of tests to run for efficiency evaluation.
    - Physical and statistical parameters such as `fake_trigger_rate`, `distance_to_evaluate`, `burst_time_window`, and details for supernova spectra and statistical comparisons.

- **Detector:**
  - Contains detector-specific settings:
    - `type`: The detector type (VD or HD).
    - `true_tpc_size` and `used_tpc_size`: Define the full and simulated volumes of the detector.
    - `tpc_size_correction_factor`: Correction factor to adjust the active volume.
    - `sn_event_multiplier`: Scaling factor to account for light flashes outside the active volume.
    - `optical_channel_number`: The number of optical channels in the detector.

- **Backgrounds:**
  - A list of background types to simulate (e.g., Ar39GenInLAr, Kr85GenInLAr, etc.).

## Dependencies

- **External Libraries:** `numpy`, `matplotlib`, `scipy`, and `rich` (for enhanced console output and table formatting).
- **Custom Modules:**  
  - `gui` and `plot_hits` for visualization.
  - `classifier` and `clustering` for data processing.
  - `data_loader` for I/O operations.
  - `config` and `argument_parser` for configuration management.
  - `statistical_comparison` for statistical tests.
  - Additional utility modules like `aux`, `stages`, and `stage_manager`.
