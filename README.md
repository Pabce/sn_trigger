# hit_stat.py

hit_stat.py is the main driver script for performing hit statistics analysis and trigger efficiency evaluation in the simulation and reconstruction pipeline. It is designed to work within a larger framework that processes detector hit data, clusters the hits, applies a pre-trained classifier (BDT), and evaluates background and signal statistics.

## Overview

This script performs several key tasks:
- **Configuration and Setup:** Reads configuration parameters (detector size, simulation time window, background sample information, etc.) from a configuration file.
- **Data Loading:** Loads pre-processed data (such as efficiency clusters, cluster features, and pre-trained boosted decision tree objects) from saved files.
- **Histogram Generation:** Computes multiple histograms like the background hit multiplicity histogram and the BDT output histogram. The histograms are marked with error bars and plotted using matplotlib.
- **Filtering with Classifier:** Applies a BDT filter to assign a BDT score to each cluster, separating signal and background clusters.
- **Trigger Efficiency Evaluation:** Computes the expected number of supernova (SN) events in a given time window and evaluates the trigger efficiency by comparing the signal and background histograms.
- **Visualization:** Generates multiple figures showing background histograms (both before and after BDT filtering), as well as an expected background histogram. There is additional (commented) code for advanced statistical tests and profiling.
- **Stage Management:** In its `main()` function, the script sets up a stage manager that runs different processing stages (such as data loading, clustering, etc.) and generates summary statistics along with custom tables for the loaded simulation parameters.

## Usage

The script is designed to be run from the command line. It accepts command-line arguments for:
- The configuration file
- Optional input file paths
- Optional output file paths and output info file paths

A typical invocation might look like:

```bash
python hit_stat.py --config path/to/config_file.yaml --input path/to/input_file --output path/to/output_file --output-info path/to/info_file
```

> **Note:** The command-line arguments are parsed using the `parse_arguments()` function, and any provided file paths will override the settings in the configuration file.

## `main()`
This is the primary entry point when the script is executed:
- Prints startup messages with fun visual cues.
- Reads command-line arguments and updates the configuration accordingly.
- Instantiates a `StageManager` with the list of stages to run (the stages include event loading, clustering, etc.).
- Runs the stages and collects cumulative output and info data.
- Logs the simulation run information (including any exceptions) into an output JSON file.
- Contains commented sections that hint at additional capabilities (such as time clustering and advanced spatial processing) for further analysis and debugging.

## Configuration

The script expects a configuration file with several sections such as:
- **Detector:** Parameters like `true_tpc_size`, `used_tpc_size`, and `tpc_size_correction_factor`.
- **Simulation:** Parameters such as `burst_time_window`, `distance_to_evaluate`, and `fake_trigger_rate`.
- **Input/Output:** File paths for input data, output data, and additional information.
- **Physics:** Parameters related to the interaction number (e.g., at 10 kpc) and other physics-based scaling factors.
- **Stages:** A list of stages that define the processing pipeline (e.g., `load_events`, `clustering`, etc.).

These parameters are accessed using methods like `config.get(section, parameter)` and are used throughout the analysis.

## Configuring a Run Using `default_config_vd.yaml`

The configuration file `configs/default_config_vd.yaml` provides a comprehensive set of parameters tailored for running the trigger algorithm on a Virtual Detector (VD) setup. Here are the key sections and steps to configure a run:

### Overview of Key Sections

- **Stages:**
  - Lists the processing stages to execute in order:
    - `load_events`
    - `clustering`
    - `cluster_feature_extraction`
    - `bdt_training`
    - `trigger_efficiency`

- **Input:**
  - `input_data_file`: Specifies the path to the input hit data file. This can be overridden via the command line.

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
    - `type`: The detector type (e.g., VD).
    - `true_tpc_size` and `used_tpc_size`: Define the full and active volumes of the detector.
    - `tpc_size_correction_factor`: A correction factor to reconcile simulation dimensions with physical reality.
    - `sn_event_multiplier`: A scaling factor applied to the number of SN events.
    - `optical_channel_number`: The number of optical channels in the detector.

- **Backgrounds:**
  - A list of background types to simulate (e.g., Ar39GenInLAr, Kr85GenInLAr, etc.).

### Customizing the Run

1. **File Paths and Patterns:**
   - Update `sn_data_dir` and `bg_data_dir` under the `Simulation: load_events` section to point to the directories containing your simulation data.
   - Adjust the file name patterns (`sn_hit_file_start_pattern`, `sn_hit_file_end_pattern`, `bg_hit_file_start_pattern`, and `bg_hit_file_end_pattern`) as needed.

2. **Clustering and Feature Extraction:**
   - Tune the clustering parameters such as `max_cluster_time`, `max_hit_time_diff`, and `max_hit_distance` in the `clustering` section to optimize the grouping of hit data.
   - Specify which cluster features should be extracted (e.g., `all` features or a selected subset) in the `cluster_feature_extraction` section.

3. **BDT Training:**
   - Adjust `optimize_hyperparameters`, `threshold`, and optionally provide specific `bdt_hyperparameters` in the `bdt_training` section to control the classifier's performance.

4. **Trigger Efficiency Evaluation:**
   - Set `use_classifier` to `True` (or `False` if not using the classifier).
   - Define the number of tests (`number_of_tests`), the `fake_trigger_rate`, and the `distance_to_evaluate`.
   - Configure the `burst_time_window` to match your simulation requirements.
   - Modify the `physics` sub-section to select the correct supernova spectra for your analysis.
   - Configure the statistical parameters such as `histogram_variable`, `statistical_method`, and `classifier_threshold`.

5. **Detector Settings and Background Simulation:**
   - Ensure the `Detector` section contains accurate values for your setup, including volume corrections and the number of optical channels.
   - Review the list of backgrounds in the `Backgrounds` section and adjust it if necessary for your particular simulation scenario.

### Running the Script

To run hit_stat.py with the default VD configuration, use the following command:
```bash
python hit_stat.py --config configs/custom_config.yaml --input path/to/input_file --output path/to/output_file --output-info path/to/info_file
```
This command will launch the full simulation pipeline based on the parameters defined in `custom_config.yaml`.

By customizing this configuration file, you can adjust the simulation and analysis parameters to suit various experimental setups and research requirements.

## Dependencies

hit_stat.py relies on several Python packages and modules:
- **Standard Libraries:** `logging`, `datetime`, `json`, `sys`, `itertools`, `multiprocessing`, and `random`.
- **External Libraries:** `numpy`, `matplotlib`, `scipy`, and `rich` (for enhanced console output and table formatting).
- **Custom Modules:**  
  - `gui` and `plot_hits` for visualization  
  - `classifier` and `clustering` for data processing  
  - `data_loader`, `trigger_efficiency_computer`, and `saver` for I/O operations  
  - `config` and `argument_parser` for configuration management  
  - `statistical_comparison` for statistical tests  
  - Additional utility modules like `aux`, `stages`, and `stage_manager`
