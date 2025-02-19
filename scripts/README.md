# hit_stat.py

hit_stat.py is the main driver script for performing hit statistics analysis and trigger efficiency evaluation in the simulation and reconstruction pipeline. It is designed to work within a larger framework that processes detector hit data, clusters the hits, applies a pre-trained classifier (BDT), and evaluates background and signal statistics.

## Overview

This script performs several key tasks:
- **Configuration and Setup:** Reads configuration parameters (detector size, simulation time window, background sample information, etc.) from a configuration file.
- **Data Loading:** Loads pre-processed data (such as efficiency clusters, cluster features, and pre-trained boosted decision tree objects) from saved files.
- **Histogram Generation:** Computes multiple histograms like the background hit multiplicity histogram and the BDT output histogram. The histograms are marked with error bars and plotted using matplotlib.
- **Filtering with Classifier:** Applies a BDT filter (using a fixed threshold, currently hardcoded at 0.9) to select background events before histogramming.
- **Trigger Efficiency Evaluation:** Computes the expected number of supernova (SN) events in a given time window and evaluates the trigger efficiency by comparing the signal and background histograms.
- **Visualization:** Generates multiple figures showing background histograms (both before and after BDT filtering), as well as an expected background histogram. There is additional (commented) code for advanced statistical tests and profiling.
- **Stage Management:** In its `main()` function, the script also sets up a stage manager that runs different processing stages (such as data loading and clustering) and generates summary statistics and custom tables for the loaded simulation parameters.

## Usage

The script is designed to be run from the command line. It accepts command-line arguments for:
- The configuration file
- Optional input file paths
- Optional output file paths and output info file paths

A typical invocation might look like:
```bash
python hit_stat.py --config path/to/config_file.ini --input path/to/input_file --output path/to/output_file --output-info path/to/info_file
```

> **Note:** The command-line arguments are parsed using the `parse_arguments()` function, and any provided file paths will override the settings in the configuration file.

## Main Functions

### `start_from_bdt()`
This function demonstrates using a pre-trained BDT for hit cluster filtering:
- Reads the configuration and calculates scaling factors (e.g., TPC sizes, burst time window, etc.).
- Loads saved data (clusters, features, BDT model, and SN info).
- Computes two types of histograms:
  - The background hit multiplicity histogram (without the BDT cut)
  - The BDT output histogram (after applying the classifier)
- Applies additional filtering to generate a "filtered" background histogram and plots these histograms with error bars.
- Determines the expected number of SN events in the simulation time window and evaluates the trigger efficiency.

### `main()`
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
- **Input/Output:** File paths for input data, output data, and additional info.
- **Physics:** Parameters related to the interaction number (e.g., at 10 kpc) and other physics-based scaling factors.

These parameters are accessed using methods like `config.get(section, parameter)` and are used throughout the analysis.

## Dependencies

hit_stat.py relies on several Python packages and modules:
- Standard libraries: `logging`, `datetime`, `json`, `sys`, `itertools`, `multiprocessing`, and `random`.
- External libraries: `numpy`, `matplotlib`, `scipy`, and `rich` (for console output and table formatting).
- Custom modules:  
  - `gui` and `plot_hits` for visualization  
  - `classifier` and `clustering` for data processing  
  - `data_loader`, `trigger_efficiency_computer`, and `saver` for I/O operations  
  - `config` and `argument_parser` for configuration management  
  - `statistical_comparison` for statistical tests  
  - Additional utility modules like `aux`, `stages`, and `stage_manager`

## Running and Debugging

- The script can be run in profiling mode as the commented sections contain code for running extensive statistics over multiple toy simulations and profiling using Python's `cProfile`.
- Pseudocode and comments throughout the file indicate sections that are considered temporary or subject for further development (e.g., "TODO: This is a temporary function" or "fix the global variables, they should be passed as arguments").

## Conclusion

hit_stat.py plays a central role in the simulation pipeline by coordinating data loading, histogram creation, event filtering, and statistical evaluation, all while providing visual outputs and logging the run parameters and outcomes. It serves both as a practical tool for trigger efficiency evaluation and as a test-bed for further refinement and debugging of the hit processing algorithms.

For further details or updates, refer to the inline comments within the code and the documentation provided alongside the configuration and utility modules.
