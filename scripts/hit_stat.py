'''
hit_stat.py

'''

import gui
from gui import console

import sys
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import stats
import itertools
from itertools import repeat
import multiprocessing as mp
import random
from rich.table import Table
from rich.progress import track

import classifier
import aux 
import data_loader as dl
import clustering as cl
import config as cf
import plot_hits as ph
import saver as sv
import trigger_efficiency_computer as tec
from enhanced_histogram import EnhancedHistogram as ehistogram
from statistical_comparison import StatisticalComparison as statcomp
import stages
from stage_manager import StageManager
from utils import NpEncoder
from argument_parser import parse_arguments

import cProfile
import pstats


def start_from_bdt():
    # TODO: This is a temporary function to start from a pre-trained BDT

    # Read configuration file from the command line
    # Create the configuration object, including the "loaded" parameters
    config = cf.Configurator.file_from_command_line() # Instance of the config class

    # Config reads ----
    true_tpc_size = config.get('Detector', 'true_tpc_size') * config.get('Detector', 'tpc_size_correction_factor')
    used_tpc_size = config.get('Detector', 'used_tpc_size') * config.get('Detector', 'tpc_size_correction_factor')
    tpc_size_correction_factor = config.get('Detector', 'tpc_size_correction_factor')
    burst_time_window = config.get('Simulation', 'burst_time_window')
    bg_sample_length = config.get("Input", "bg_sample_length")
    bg_sample_number_per_file = config.get("Input", "bg_sample_number_per_file")
    bdt_threshold = 0.9#config.get("Simulation", "classifier_threshold")
    # -----------------
    
    # Load the eff clusters
    final_eff_sn_clusters, final_eff_bg_clusters, cluster_feature_names = sv.DataWithConfig.load_data_from_file(f"{save_dir}eff_clusters.pkl").data
    final_eff_sn_clusters_per_event, final_eff_bg_clusters_per_event, _ = sv.DataWithConfig.load_data_from_file(f"{save_dir}eff_clusters_per_event.pkl").data
    # Load the cluster features
    sn_eff_features_array, sn_eff_features_per_event, sn_eff_targets, _ = sv.DataWithConfig.load_data_from_file(f"{save_dir}sn_eff_features_targets.pkl").data
    bg_eff_features_array, bg_eff_features_per_event, bg_eff_targets, _ = sv.DataWithConfig.load_data_from_file(f"{save_dir}bg_eff_features_targets.pkl").data

    # Load the BDT
    hist_boosted_tree = sv.DataWithConfig.load_data_from_file(f"{save_dir}hist_boosted_tree.pkl").data
    # Load the SN info per event
    sn_eff_info_per_event = sv.DataWithConfig.load_data_from_file(f"{save_dir}sn_info_per_event.pkl").data[0]

    # The total time window spanned by the loaded background events
    total_bg_time_window = bg_sample_length * len(bg_eff_features_per_event)
    bg_histogram_multiplier = 1/total_bg_time_window * burst_time_window/1000 * true_tpc_size/used_tpc_size * tpc_size_correction_factor
    console.log(f"BG histogram multiplier: {bg_histogram_multiplier}")
    console.log(f"Total BG time window: {total_bg_time_window} ms")
    console.log(f"Number of BG clusters: {len(final_eff_bg_clusters)}")
    console.log(f"Number of BG events: {len(bg_eff_features_per_event)}, {len(final_eff_bg_clusters_per_event)}")

    # Create TriggerEfficiencyComputer object
    # TODO: Statistical info should be read from the config file
    statistical_info = {
        "classification": {"classifier": hist_boosted_tree, "threshold": bdt_threshold},
        "histogram_variable": "hit_multiplicity",
        "statistical_method": "pearson_chi_squared"
    }
    statistical_info_no_classifier = {
        "classification": False,
        "histogram_variable": "hit_multiplicity",
        "statistical_method": "pearson_chi_squared"
    }
    error_info_f = {"type": "frequentist"}
    error_info_f = {"type": "bayesian", 
                    "params": 
                        {"prior": "jeffreys",
                          "confidence_level": 0.68}
                    }
    log_level = logging.INFO
    trigger_eff_computer = tec.TriggerEfficiencyComputer(config, logging_level=log_level, statistical_info=statistical_info, error_info=error_info_f)
    trigger_eff_computer_no_classifier = tec.TriggerEfficiencyComputer(config, logging_level=log_level, 
                                                                       statistical_info=statistical_info_no_classifier, error_info=error_info_f)

    # Get the background histogram (remember you still haven't applied the BDT cut)
    bg_hit_multiplicity_histogram = trigger_eff_computer_no_classifier.compute_hit_multiplicity_filtered_histogram(
                                    final_eff_bg_clusters, bg_eff_features_array, bins=None, bunch_limit=4, multiplier=bg_histogram_multiplier)

    bdt_output_histogram = trigger_eff_computer.compute_bdt_output_histogram(
                                    final_eff_bg_clusters, bg_eff_features_array, bins=None, bunch_limit=4,
                                    multiplier=bg_histogram_multiplier)
    
    # console.log(bg_hit_multiplicity_histogram.lower_bunched_errors)
    # console.log(bg_hit_multiplicity_histogram.upper_bunched_errors)


    # Plot the background histograms, make the bars with solid edges and translucent fill
    plt.figure(1)
    # plt.bar(bg_hit_multiplicity_histogram.bunched_bins[:-1] + np.diff(bg_hit_multiplicity_histogram.bunched_bins)[0]/2, bg_hit_multiplicity_histogram.bunched_upper_limit,
    #         width=np.diff(bg_hit_multiplicity_histogram.bunched_bins)[0], edgecolor='none', linewidth=2, alpha=0.5, color='pink')
    plt.bar(bg_hit_multiplicity_histogram.bunched_bins[:-1] + np.diff(bg_hit_multiplicity_histogram.bunched_bins)[0]/2, bg_hit_multiplicity_histogram.bunched_values,
            width=np.diff(bg_hit_multiplicity_histogram.bunched_bins)[0], edgecolor='black', linewidth=2, alpha=0.5)
    plt.axhline(y=4, color='darkred', linestyle='--', linewidth=2)
    # Plot error
    plt.errorbar(bg_hit_multiplicity_histogram.bunched_bins[:-1] + np.diff(bg_hit_multiplicity_histogram.bunched_bins)[0]/2, bg_hit_multiplicity_histogram.bunched_values,
                 yerr=[bg_hit_multiplicity_histogram.lower_bunched_errors, bg_hit_multiplicity_histogram.upper_bunched_errors], 
                 fmt='none', capsize=5, capthick=2, ecolor='black')
    plt.yscale('log')

    plt.figure(2)
    plt.bar(bdt_output_histogram.bunched_bins[:-1] + np.diff(bdt_output_histogram.bunched_bins)[0]/2, bdt_output_histogram.bunched_values,
            width=np.diff(bdt_output_histogram.bunched_bins)[0], edgecolor='black', linewidth=2, alpha=0.5)
    plt.axhline(y=4, color='darkred', linestyle='--', linewidth=2)
    # Add Poisson error (sqrt of the number of events)
    plt.errorbar(bdt_output_histogram.bunched_bins[:-1] + np.diff(bdt_output_histogram.bunched_bins)[0]/2, bdt_output_histogram.bunched_values,
                 yerr=[bdt_output_histogram.lower_bunched_errors, bdt_output_histogram.upper_bunched_errors], fmt='none', capsize=3, capthick=2, ecolor='black')
    plt.yscale('log')
    console.log(f"Num bins for bdt_output_histogram: {len(bdt_output_histogram.bunched_values)}")

    # Filter the BG events with the BDT
    filtered_bg_clusters, filtered_bg_features_array = classifier.cluster_filter(
        hist_boosted_tree, final_eff_bg_clusters, bg_eff_features_array, threshold=bdt_threshold)
    # Build the hit multiplicity histogram
    bg_filtered_hit_multiplicity_histogram = trigger_eff_computer.compute_hit_multiplicity_filtered_histogram(
                                                                        filtered_bg_clusters, filtered_bg_features_array, bins=None,
                                                                        bunch_limit=4, multiplier=bg_histogram_multiplier)
    # Plot the filtered background histograms
    plt.figure(1)
    plt.bar(bg_filtered_hit_multiplicity_histogram.bunched_bins[:-1] + np.diff(bg_filtered_hit_multiplicity_histogram.bunched_bins)[0]/2, bg_filtered_hit_multiplicity_histogram.bunched_values,
            width=np.diff(bg_filtered_hit_multiplicity_histogram.bunched_bins)[0], edgecolor='black', linewidth=2, alpha=0.5)
    # Add Poisson error (sqrt of the number of events)
    plt.errorbar(bg_filtered_hit_multiplicity_histogram.bunched_bins[:-1] + np.diff(bg_filtered_hit_multiplicity_histogram.bunched_bins)[0]/2, bg_filtered_hit_multiplicity_histogram.bunched_values,
                yerr=[bg_filtered_hit_multiplicity_histogram.lower_bunched_errors, bg_filtered_hit_multiplicity_histogram.upper_bunched_errors],
                fmt='none', capsize=5, capthick=2, ecolor='black')
    plt.yscale('log')



    # Compute he expected number of SN events in the time window
    # The model is irrelevant here, we can adjust later
    # CONFIG READS ----
    interaction_number_10kpc = config.get('Physics', 'interaction_number_10kpc', 'GKVM')
    distance_to_evaluate = config.get('Simulation', 'distance_to_evaluate')
    fake_trigger_rate = config.get('Simulation', 'fake_trigger_rate')
    sn_event_multiplier = config.get('Detector', 'sn_event_multiplier')
    # -----------------

    event_number_full_burst = aux.distance_to_event_number(distance_to_evaluate, interaction_number_10kpc, 
                                                            true_tpc_size * tpc_size_correction_factor) * sn_event_multiplier
    # Load the time profile
    data_loader = dl.DataLoader(config, logging_level=logging.INFO)
    time_profile_x, time_profile_y = data_loader.load_time_profile()
    expected_event_number_in_time_window = aux.expected_event_number_in_time_window(time_profile_x, time_profile_y, event_number_full_burst, burst_time_window)
    console.log(f"Expected event number in full burst: {event_number_full_burst}")
    console.log(f"Expected SN event number in time window: {expected_event_number_in_time_window}")

    # Evaluate the trigger efficiency

    # This is redundant but create a new histogram utilizing the function mapping
    expected_bg_histogram = trigger_eff_computer.compute_histogram(final_eff_bg_clusters, bg_eff_features_array, bins=None,
                                                                    bunch_limit=0.0, multiplier=bg_histogram_multiplier)
    
    plt.figure(3)
    plt.bar(expected_bg_histogram.bunched_bins[:-1] + np.diff(expected_bg_histogram.bunched_bins)[0]/2, expected_bg_histogram.bunched_values,
            width=np.diff(expected_bg_histogram.bunched_bins)[0], edgecolor='black', linewidth=2, alpha=0.5)
    plt.axhline(y=4, color='darkred', linestyle='--', linewidth=2)
    plt.yscale('log')
    
    # ---- STATISTICS TEST ----------------  

    # Run the Cash statistic distribution...
    # dof = len(expected_bg_histogram.bunched_values)
    # fake_trigger_rate_chi_sq = statcomp.chi_squared_statistic_from_p_value(fake_trigger_rate, dof=dof)
    # console.log(f"Fake trigger rate chi squared: {fake_trigger_rate_chi_sq}")

    # dof = len(expected_bg_histogram.bunched_values)
    # chi2_distribution = stats.chi2(dof)
    # bin_left = 1
    # bin_right = fake_trigger_rate_chi_sq * 1.5
    # bin_num = 100
    # bins = np.linspace(bin_left, bin_right, bin_num)
    # chi2_x = bins[:-1] + np.diff(bins)[0]/2
    # chi2_y = chi2_distribution.pdf(chi2_x)
    # chi2_area = np.trapz(chi2_y, chi2_x)

    # cash_hist_values = np.zeros(len(bins) - 1)
    # chisq_hist_values = np.zeros(len(bins) - 1)
    # min_stat_value = 10000
    # max_stat_value = 0

    # # We have to do this in several iterations and chunk sizes to avoid memory issues
    # n_iter = 5
    # n_toys = 200_000_000
    # chunk_size = 5_000_000

    # total_expected_fake_triggers = fake_trigger_rate * n_toys * n_iter
    # total_cash_fake_triggers = 0
    # total_chi_sq_fake_triggers = 0

    # # Profile this
    # profiler = cProfile.Profile()
    # profiler.enable()
    # for i in range(n_iter):
    #     console.log(f"Iteration {i+1}")
    #     random_seed = np.random.randint(0, 1000000)
    #     # cash_stat_distribution, _ = statcomp.generate_statistic_distribution(expected_bg_histogram.bunched_values, 
    #     #                                                                     type="cash", n_toys=n_toys, random_seed=random_seed,
    #     #                                                                     chunk_size=chunk_size)
    #     # _, chisq_stat_distribution = statcomp.generate_statistic_distribution(expected_bg_histogram.bunched_values,
    #     #                                                                         type="chi_squared", n_toys=n_toys, random_seed=random_seed,
    #     #                                                                         chunk_size=chunk_size)

    #     cash_stat_distribution, chisq_stat_distribution = statcomp.generate_statistic_distribution(expected_bg_histogram.bunched_values,
    #                                                                     type="both", n_toys=n_toys, random_seed=random_seed,
    #                                                                     chunk_size=chunk_size)

    #     min_stat_value = min(min_stat_value, min(np.nanmin(cash_stat_distribution), np.nanmin(chisq_stat_distribution)))
    #     max_stat_value = max(max_stat_value, max(np.nanmax(cash_stat_distribution), np.nanmax(chisq_stat_distribution)))

    #     expected_fake_triggers = fake_trigger_rate * n_toys
    #     cash_fake_triggers = np.sum(cash_stat_distribution > fake_trigger_rate_chi_sq)
    #     chi_sq_fake_triggers = np.sum(chisq_stat_distribution > fake_trigger_rate_chi_sq)
    #     total_cash_fake_triggers += cash_fake_triggers
    #     total_chi_sq_fake_triggers += chi_sq_fake_triggers
    #     console.log(f"Expected {expected_fake_triggers} fake triggers")
    #     console.log(f"Got {cash_fake_triggers} fake triggers from Cash")
    #     console.log(f"Got {chi_sq_fake_triggers} fake triggers from Chi2")

    #     cash_hist_values += np.histogram(cash_stat_distribution, bins=bins)[0]
    #     chisq_hist_values += np.histogram(chisq_stat_distribution, bins=bins)[0]

    # profiler.disable()
    # ps = pstats.Stats(profiler).sort_stats('cumtime')
    # ps.print_stats(15)

    # console.log(f"Expected TOTAL {total_expected_fake_triggers} fake triggers")
    # console.log(f"Got TOTAL {total_cash_fake_triggers} fake triggers from Cash")
    # console.log(f"Got TOTAL {total_chi_sq_fake_triggers} fake triggers from Chi2")

    # # Normalize the histograms to the area under the chi2 distribution spanned by the min and max values
    # console.log(f"Min stat value: {min_stat_value}, Max stat value: {max_stat_value}")
    # cash_hist_values /= np.sum(cash_hist_values) * np.diff(bins)[0] / chi2_area
    # chisq_hist_values /= np.sum(chisq_hist_values) * np.diff(bins)[0] / chi2_area
    # console.log()

    # plt.figure(4)
    # console.log(f"Bin left: {bin_left}, Bin right: {bin_right}")
    # # Plot the Cash and Chisq statistic distribution histogram
    # plt.bar(bins[:-1] + np.diff(bins)[0]/2, cash_hist_values, width=np.diff(bins)[0], linewidth=2, alpha=0.5, edgecolor='none')
    # plt.bar(bins[:-1] + np.diff(bins)[0]/2, chisq_hist_values, width=np.diff(bins)[0], linewidth=2, alpha=0.5, edgecolor='none')
    # # Plot the chi squared distribution for the given degrees of freedom
    # plt.plot(chi2_x, chi2_y, label=f"Chi2, dof={dof}")
    # plt.axvline(x=fake_trigger_rate_chi_sq, color='red', linestyle='--')
    # plt.yscale('log')

    # # Plot the difference between the two distributions and the chi2 pdf
    # plt.figure(5)
    # plt.step(chi2_x, cash_hist_values - chi2_y, label="Cash - Chi2 PDF")
    # plt.step(chi2_x, chisq_hist_values - chi2_y, label="Chi2 - Chi2 PDF")
    # plt.axhline(y=0, color='black', linestyle='--')

    # # Plot the ratio between the two distributions and the chi2 pdf
    # plt.figure(6)
    # plt.step(chi2_x, cash_hist_values/chi2_y, label="Cash/Chi2 PDF")
    # plt.step(chi2_x, chisq_hist_values/chi2_y, label="Chi2/Chi2 PDF")
    # plt.axhline(y=1, color='black', linestyle='--')
    # plt.axvline(x=fake_trigger_rate_chi_sq, color='red', linestyle='--')
    # plt.yscale('log')

    # -----------------

    plt.show()

    trigger_efficiency = trigger_eff_computer.evaluate_trigger_efficiency(expected_event_number_in_time_window, 
                                                    final_eff_sn_clusters_per_event, sn_eff_features_per_event,
                                                    sn_eff_info_per_event, 
                                                    expected_bg_histogram,
                                                    number_of_tests=1000,
                                                    in_parallel=False)

    console.log(f"Trigger efficiency: {trigger_efficiency}")



def main():
    console.log('--- START ---> :fire: :fire: :fire:', style="bold green")
    console.log(":chipmunk:  Let's get started! :chipmunk:", style="bold yellow")
    for i in range(3):
        console.print("\t:chipmunk:🫡  Marching Chipmunks! :chipmunk:🫡", style="bold yellow")

    # Read the command line for the configuration file, and possibly the input and output files
    config_path, input_path, output_path, output_info_path = parse_arguments()

    print(output_path, output_info_path, "ÑFDÑLAKDJFÑLS")

    # Create the configuration object, including the "loaded" parameters
    # If an input/output file is provided from the command line, it will overwrite the configuration file here
    config = cf.Configurator.from_file(config_path) # Instance of the config class
    if input_path:
        config.set_value("Input", "input_data_file", value=input_path)
        logging.info(f"Input file read from command line: {input_path}")
    if output_path:
        config.set_value("Output", "output_data_file", value=output_path)
        logging.info(f"Output file read from command line: {output_path}")
    if output_info_path:
        config.set_value("Output", "output_info_file", value=output_info_path)
        logging.info(f"Output info file read from command line: {output_info_path}")

    # Get the list of stages to run and create the stage manager
    stage_names_list = config.get("Stages")
    verbosity_level = config.get("Output", "verbosity")
    stage_manager = StageManager(config, stage_names_list, logging_level=verbosity_level)
    console.log(f"[bold yellow]Runing the following stages:")
    console.log(f"{stage_names_list}\n")
    
    # Create a table to display the simulation parameters
    table = gui.get_custom_table(config, "sim_parameters")
    console.log(table)

    # Run this shit
    cumulative_output_data, cumulative_info_output_data, exception = stage_manager.run_stages()

    # Log some info into the "output_info" file
    output_info_file = config.get("Output", "output_info_file")
    output_info_dict = {}
    output_info_dict['exception'] = exception
    output_info_dict['timestamp'] = f'{datetime.now()}'
    output_info_dict['config_file_path'] = config_path
    output_info_dict.update(cumulative_info_output_data)
    output_info_dict['config'] = config.get_dict()

    with open(output_info_file, 'w') as f:
        json.dump(output_info_dict, f, indent=2, cls=NpEncoder)

    console.log(f"[bold yellow] Written run info to: {output_info_file}")

    exit()


    #print(final_sn_clusters[0])
    # sn_time_candidate_clusters = []
    # for i in range(100):
    #     sn_time_candidate_clusters.extend(
    #         clustering.time_clustering(sn_eff_hit_list_per_event[i], max_cluster_time=0.25, max_hit_time_diff=0.2, min_hit_multiplicity=5)[0]
    #     )
    # logger.info(f"SN time candidate clusters: {len(sn_time_candidate_clusters)}")
    # bg_time_candidate_clusters, bg_time_candidate_hit_multiplicities =\
    #                     clustering.time_clustering(bg_eff_hit_list_per_event[0], max_cluster_time=0.25, max_hit_time_diff=0.2, min_hit_multiplicity=5)
    # logger.info(f"BG time candidate clusters: {len(bg_time_candidate_clusters)}")
    
    # hitplotter = ph.HitPlotter(config, logging_level=logging.INFO)
    # for tcc in sn_time_candidate_clusters:
    #     sc1 = clustering.algorithm_spatial_neighbour_grouping(tcc, max_hit_distance=400)
    #     sc2 = clustering.algorithm_spatial_naive(tcc, max_hit_distance=400, min_neighbours=1)
    #     hitplotter.plot_3d_clusters(tcc, sc1, plot_complement_cluster=True)
    #     hitplotter.plot_3d_clusters(tcc, sc2)

    # for tcc in bg_time_candidate_clusters:
    #     clustering.algorithm_spatial_neighbour_grouping(tcc, max_hit_distance=350, min_hit_multiplicity=8)

    # for tcc in bg_time_candidate_clusters:
    #     space_candidate_cluster = cluster.enforce_spatial_correlation(tcc, max_hit_distance=220, min_neighbours=3)
    #     console.print(tcc.shape)
    #     console.print(space_candidate_cluster.shape)
    #     console.print('........')
    
    # console.log("Starting clustering SN...")
    # sn_clusters, sn_hit_multiplicities = clustering.full_clustering(sn_eff_hit_list_per_event[0], max_cluster_time=0.25, max_hit_time_diff=0.2, 
    #     max_hit_distance=220, max_x_hit_distance=1e5, max_y_hit_distance=1e5, max_z_hit_distance=1e5, min_neighbours=2, min_hit_multiplicity=8,
    #     spatial_filter=True)

    # console.log("Starting clustering BG...")
    # bg_clusters, bg_hit_multiplicities = clustering.full_clustering(bg_eff_hit_list_per_event[0], max_cluster_time=0.25, max_hit_time_diff=0.2, 
    #     max_hit_distance=220, max_x_hit_distance=1e5, max_y_hit_distance=1e5, max_z_hit_distance=1e5, min_neighbours=2, min_hit_multiplicity=8,
    #     spatial_filter=True)
    
    # for sc in sn_clusters:
    #     print(sc.shape)
    # for bc in bg_clusters:
    #     print(bc.shape)

    exit("STOP")



if __name__ == '__main__':
    # TODO: fix the global variables, they should be passed as arguments!
    # TODO: fix weird bug where sometimes the BDT training crashes
    main()
    #start_from_bdt()





    


