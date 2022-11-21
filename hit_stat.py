'''
hit_stat.py

This is the main script for computing the trigger efficiency. It contains functions to compute the optimal trigger efficiency for a given set of clustering parameters.
Also to compute the efficiency curve, once the optimal parameters and BDT are found for a given distance.
To run the script, use the following command:

python hit_stat.py -e <average energy> -a <alpha> -m <mode> -d <distance to optimize for> -o <output name>

All the arguments are optional. If not specified, the default values in parameters.py will be used. To get more information run
python hit_stat.py --help
'''

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

import sys
from scipy import stats
import itertools
from itertools import repeat
import multiprocessing as mp
import random

import classifier
import aux
import save_n_load as sl
import clustering
from parameters import *
#from numba import jit



def stat_cluster_parameter_check_parallel(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                                            bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, params_list, verbose, detector,
                                            true_tpc_size, used_tpc_size, min_mcm, max_mcm, classify, loaded_tree, fake_trigger_rate,
                                            number_of_tests):

    # Random seed for each parallel process
    random.seed()
    # ---------------------------

    to_plot = False
    print("STARTING NEW CPU PROCESS", mp.current_process())

    # Arguments to add
    burst_time_window = BURST_TIME_WINDOW # microseconds

    ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold = params_list
    # Does this even make sense?
    if ct < ht:
        return None

    
    optimal_mcm = 0
    optimal_trigger_efficiency = 0
    optimal_tree = None

    # Iterate over the minimum cluster multiplicities
    for mcm in range(min_mcm, max_mcm + 1):
        # Run the clustering algorithm for all SN and BG samples, both regular and those for training

        tot_start = time.time()

        bg_clusters, bg_hit_multiplicities, sn_clusters, sn_hit_multiplicities, sn_energies = clustering.full_clustering(sn_hit_list_per_event, sn_info_per_event,
                                                    bg_hit_list_per_event, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)
        
        if classify and not loaded_tree:
            bg_clusters_train, _, sn_clusters_train, _, _ = \
                                                    clustering.full_clustering(sn_train_hit_list_per_event, sn_train_info_per_event,
                                                        bg_train_hit_list_per_event, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                        max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                        min_hit_mult=mcm, detector=detector, verbose=verbose-1)

        if len(sn_hit_multiplicities) < 3 or len(bg_hit_multiplicities) < 3:
            continue
        
        # hbins = np.arange(min(sn_hit_multiplicities)-0.5, max(sn_hit_multiplicities) + 1.5, 2)
        # plt.hist(sn_hit_multiplicities, bins=hbins, alpha=0.5, density=True, label="SN")
        # plt.hist(bg_hit_multiplicities, bins=hbins, alpha=0.5, density=True, label='BG')
        # plt.xlabel("Hit multiplicity")
        # plt.yscale('log')
        # plt.legend()
        # plt.show()


        # Train and apply the classifier to the background clusters to get the new expected background
        # (If required)
        sn_features = None
        if classify:
            print(loaded_tree)
            if not loaded_tree:
                train_features, train_targets = clustering.cluster_comparison(sn_clusters_train, bg_clusters_train)
                tree = classifier.gradient_boosted_tree(train_features, train_targets, n_estimators=200)
            else:
                tree = loaded_tree

            # Now, we filter the regular clusters with the newly trained classifier
            features, targets = clustering.cluster_comparison(sn_clusters, bg_clusters)
            new_bg_clusters, new_bg_hit_multiplicities, _ = classifier.tree_filter(tree, bg_clusters, features[targets==0, :], threshold=classifier_threshold)
            sn_features = features[targets==1, :]
        else:
            tree = None
        
        # ---------------------------

        # We create the bg min hit mult histogram
        hbins = np.arange(min(sn_hit_multiplicities), max(sn_hit_multiplicities) + 2, 1)
        #sn_hist, _ = np.histogram(sn_hit_multiplicities, bins=hbins, density=False)
        bg_hist, bg_bins = np.histogram(bg_hit_multiplicities, bins=hbins, density=False)

        if classify:
            # We create the new bg min hit mult histogram for the new bg clusters, if classifier is used, and compute the ratios
            new_bg_hist, _ = np.histogram(new_bg_hit_multiplicities, bins=hbins, density=False)

            zeros = np.where(bg_hist == 0)[0]
            bg_hist[zeros] = 1
            new_bg_hist[zeros] = 1

            filter_bg_ratios = new_bg_hist / bg_hist
            #print(filter_bg_ratios)

        else:
            filter_bg_ratios = 1


        # -------------------
        # Compute the efficiency for these clustering and global parameters

        time_profile_x, time_profile_y = sl.load_time_profile()
        true_fake_trigger_rate = fake_trigger_rate * burst_time_window / 1e6
        sn_model = "LIVERMORE" # Don't worry, we can correct this later! (no need to change to GKVM or GARCHING, we can normalize the distances a posteriori)

        sn_event_num = len(sn_hit_list_per_event)
        total_event_num = aux.distance_to_event_number(distance_to_optimize, sn_model, true_tpc_size) * SN_EVENT_MULTIPLIER
        event_num_per_time = aux.event_number_per_time(time_profile_x, time_profile_y, total_event_num, burst_time_window)
        
        expected_bg_hist = bg_hist * 1/bg_length * burst_time_window/1000 * true_tpc_size/used_tpc_size

        start = time.time()

        trigger_efficiency = stat_trigger_efficiency(event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, sn_energies,
                                                    sn_features, sn_info_per_event, expected_bg_hist, filter_bg_ratios,
                                                    hbins, fake_trigger_rate=true_fake_trigger_rate, number_of_tests=number_of_tests, 
                                                    classify=classify, tree=tree, threshold=classifier_threshold)
        
        end = time.time()
        print("Time:", end - start)

        if verbose > 0:
            print("\n")
            print("Min multiplicity: {}, Trigger efficiency: {}, (params: {}, {}, {}, {}, {}, {}, (DTO: {}, Thresh: {}))".format(mcm, 
                            trigger_efficiency, *params_list))

        if trigger_efficiency > optimal_trigger_efficiency:
            optimal_trigger_efficiency = trigger_efficiency
            optimal_mcm = mcm
            optimal_tree = tree
        
        tot_end = time.time()
        print("Total time:", tot_end - tot_start)

    print("----------------- Tested parameters ->")
    print("Max cluster time: {}, Max hit time diff: {}, Max hit distance: {}".format(ct, ht, hd))
    print("Max X distance {}, Max Y distance {}, Max Z distance: {}".format(xhd, yhd, zhd))
    print("Min hit multiplicity from bg:", optimal_mcm)
    print("DTO, Threshold", distance_to_optimize, classifier_threshold)
    print("Trigger efficiency:", optimal_trigger_efficiency)
    print("--------------------------------------")


    return optimal_trigger_efficiency, optimal_mcm, optimal_tree, params_list


def stat_cluster_parameter_scan_parallel(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event, bg_hit_list_per_event,
                             bg_train_hit_list_per_event, bg_length, max_cluster_times, max_hit_time_diffs, max_hit_distances, detector="VD", verbose=0,
                             max_x_hit_distances=None, max_y_hit_distances=None, max_z_hit_distances=None, true_tpc_size=10,
                             used_tpc_size=2.6, distance_to_optimize=[30], min_mcm=15, max_mcm=30, classify=False, tree=None, 
                             classifier_threshold=[0.5], fake_trigger_rate=1/(60 * 60 * 24 * 30), number_of_tests=500):

    opt_parameters = ()
    max_trigger_efficiency = 0
    opt_mcm = 0
    opt_tree = None
    
    params_list = itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances, 
                                    max_x_hit_distances, max_y_hit_distances, max_z_hit_distances, distance_to_optimize, classifier_threshold)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(stat_cluster_parameter_check_parallel, zip(repeat(sn_hit_list_per_event), repeat(sn_train_hit_list_per_event),repeat(sn_info_per_event),
                                                    repeat(sn_train_info_per_event), repeat(bg_hit_list_per_event), repeat(bg_train_hit_list_per_event),
                                                    repeat(bg_length), params_list, repeat(verbose), repeat(detector),
                                                    repeat(true_tpc_size), repeat(used_tpc_size),
                                                    repeat(min_mcm), repeat(max_mcm), repeat(classify), repeat(tree),
                                                    repeat(fake_trigger_rate), repeat(number_of_tests)))
    
    all_efficiencies = []
    all_parameters = []
    for result in results:
        if result:
            trigger_efficiency, mcm, o_tree, params = result
            ct, ht, hd, xhd, yhd, zhd, dto, _ = params
            a = max_cluster_times.index(ct)
            b = max_hit_time_diffs.index(ht)
            c = max_hit_distances.index(hd)
            d = max_x_hit_distances.index(xhd)
            e = max_y_hit_distances.index(yhd)
            f = max_z_hit_distances.index(zhd)
            
            all_efficiencies.append(trigger_efficiency)
            all_parameters.append(params)

            if trigger_efficiency > max_trigger_efficiency:
                max_trigger_efficiency = trigger_efficiency
                opt_parameters = params
                opt_mcm = mcm
                opt_tree = o_tree

    return max_trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_efficiencies, all_parameters



def stat_trigger_efficiency(event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, sn_energies, sn_features, sn_info_per_event,
                            expected_bg_hist, filter_bg_ratios, hbins, fake_trigger_rate, to_plot=False, number_of_tests=1000, classify=False, 
                            tree=None, threshold=0.5):
    
    print("TE GOING ON")
    energies = np.linspace(4, 100, 100)
    energies_histogram = aux.pinched_spectrum(energies, average_e=AVERAGE_ENERGY, alpha=ALPHA)

    detected_events = 0
    for i in range(number_of_tests):
        
        # Draw a true event number from the expected event number. REMEMBER, NOT ALL SN EVENTS HAVE CLUSTERS.
        r = int(stats.poisson.rvs(event_num_per_time, size=1))

        if r == 0:
            continue

        # Draw randomly from the SN event pool, while following a given energy distribution!
        sample_ind_e = aux.get_energy_indices(energies, energies_histogram, sn_energies, size=r)

        # selected_energies = sn_energies[sample_ind_e]
        # plt.figure()
        # plt.hist(selected_energies, bins=20, density=True)
        # plt.plot(energies, energies_histogram)
        # plt.show()

        sn_hit_multiplicities = np.array(sn_hit_multiplicities)
        sn_clusters = np.array(sn_clusters, dtype=object)
        
        # Remove clusterless samples
        sample_ind_with_clusters = []
        for ind in sample_ind_e:
            if sn_hit_multiplicities[ind] != -1:
                sample_ind_with_clusters.append(ind)

        sample_ind = np.array(sample_ind_with_clusters)
        if len(sample_ind) == 0:
            continue

        sample = sn_hit_multiplicities[sample_ind]

        # Filter the sample if we chose to apply the classifier
        if classify:
            class_sn_features, th = clustering.cluster_comparison(sn_clusters[sample_ind], [])
            _, sample, _ = classifier.tree_filter(tree, sn_clusters[sample_ind], 
                                                    class_sn_features[th==1], threshold=threshold)
            #print(sample)
            #print("Sample after classifier ratio", len(sample)/l1)

        
        # Generate a random poisson background from the expected backgrounds
        fake_bg_hist = np.round(stats.poisson.rvs(expected_bg_hist, size=len(hbins)-1) * filter_bg_ratios)

        # Generate histogram and compute statistic
        # --- CHI SQUARED ---
        # Bunch together where the number of entries is < 3 or 4 (so the chi_squared statistic is reliable)
        
        sn_hist, _ = np.histogram(sample, bins=hbins, density=False)

        expected_bg_chi, chi_last_index = aux.bunch_histogram(expected_bg_hist * filter_bg_ratios, limit=3)
        observed_chi, chi_last_index = aux.bunch_histogram(sn_hist + fake_bg_hist, last_index=chi_last_index)

        # fake_bg_chi, chi_last_index = aux.bunch_histogram(fake_bg_hist, limit=4)# last_index=chi_last_index)
        # sn_chi, chi_last_index = aux.bunch_histogram(sn_hist, last_index=chi_last_index)
        # print(expected_bg_chi, "BG-CH")
        # print(observed_chi, "CHICH")


        # print("CHI SQUARED", aux.chi_squared(sn_chi + fake_bg_chi, expected_bg_chi, dof=1))
        # print(once_a_month_rate)

        cut = 0
        chi_sq, dof, chi_squared_significance = aux.chi_squared(observed_chi[cut:], 
                                                    expected_bg_chi[cut:], dof=len(expected_bg_chi[cut:]) - 1)
        
        if chi_squared_significance < fake_trigger_rate:
            detected_events += 1

        #to_plot = True
        if to_plot:
            plt.figure(2)
            plt.bar(hbins[:-1], fake_bg_hist + sn_hist, width=1, alpha=0.8)
            plt.bar(hbins[:-1], expected_bg_hist * filter_bg_ratios, width=1, alpha=0.5)
            plt.yscale('log')
            plt.title("{} - {}".format(chi_squared_significance, fake_trigger_rate))
            #plt.legend()
            # plt.ylim(top=10)
            # plt.xlim(left=15)

            # plt.figure(3)
            # plt.bar(hbins[:chi_last_index + 1], observed_chi - expected_bg_chi, width=1)

            
            # plt.figure(4)
            # plt.bar(hbins[:-1], sn_hist, width=1)

            plt.show() 
    
    trigger_efficiency = detected_events/number_of_tests
    #print("OOOOOOOO", sn_event_num)

    return trigger_efficiency



def get_efficiency_curve(opt_parameters, sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, bg_length, detector, true_tpc_size, 
                            used_tpc_size, distances, opt_mcm, opt_tree, fake_trigger_rate, number_of_tests, save=True, plot=True):

    mct, mht, mhd, mxd, myd, mzd, _, cthresh = opt_parameters
    distance_te, _, _, _, all_te, all_params = stat_cluster_parameter_scan_parallel(
                                    sn_hit_list_per_event, None, 
                                    sn_info_per_event, None,
                                    bg_hit_list_per_event, None, bg_length,
                                    max_cluster_times=[mct],
                                    max_hit_time_diffs=[mht], 
                                    max_hit_distances=[mhd],
                                    max_x_hit_distances=[mxd],
                                    max_y_hit_distances=[myd],
                                    max_z_hit_distances=[mzd],
                                    detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                    used_tpc_size=used_tpc_size, distance_to_optimize=distances,
                                    min_mcm=opt_mcm, max_mcm=opt_mcm, classify=CLASSIFY,
                                    classifier_threshold=[cthresh], tree=opt_tree,
                                    fake_trigger_rate=fake_trigger_rate, number_of_tests=number_of_tests)
    

    print(AVERAGE_ENERGY, ALPHA, "OE")
    sim_parameters = [FAKE_TRIGGER_RATE, BURST_TIME_WINDOW, DISTANCE_TO_OPTIMIZE, SIM_MODE, ADC_MODE, DETECTOR, CLASSIFY, AVERAGE_ENERGY, ALPHA]
    eff_curve_data=[distances, all_te, sim_parameters]
    if save:
        sl.save_efficiency_data(eff_data=eff_curve_data, sim_parameters=sim_parameters, file_name=OUTPUT_NAME_CURVE, data_type="curve")
    
    if plot:
        plt.plot(distances, all_te)
        plt.show()

    #return trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_params
    return eff_curve_data


def optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event, 
                        bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, fake_trigger_rate, save=True):
    
    # Add background to our SN events (this is basically useless as the SN events are so short in time)
    # for i in range(len(sn_hit_list_per_event)):
    #     #display_hits(sn_hit_list_per_event[i], time=True)
    #     if len(sn_hit_list_per_event[i]) > 0:
    #         sn_hit_list_per_event[i] = aux.spice_sn_event(sn_hit_list_per_event[i], bg_hit_list_per_event,
    #                                                 bg_length_to_add=0.8, bg_length=bg_sample_length * 1000)
    #     #display_hits(sn_hit_list_per_event[i], time=True)


    # Find cluster parameters that maximize trigger efficiency
    max_cluster_times = MAX_CLUSTER_TIMES
    max_hit_time_diffs = MAX_HIT_TIME_DIFFS
    max_hit_distances = MAX_HIT_DISTANCES
    # New cluster parameters! (wooo) (These have been discovered to be useless!)
    max_x_hit_distances = [25000]
    max_y_hit_distances = [20000]
    max_z_hit_distances = [30000]

    trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_effs, _ = stat_cluster_parameter_scan_parallel(
                                sn_hit_list_per_event, sn_train_hit_list_per_event, 
                                sn_info_per_event, sn_train_info_per_event,
                                bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length,
                                max_cluster_times=max_cluster_times,
                                max_hit_time_diffs=max_hit_time_diffs, 
                                max_hit_distances=max_hit_distances,
                                max_x_hit_distances=max_x_hit_distances,
                                max_y_hit_distances=max_y_hit_distances,
                                max_z_hit_distances=max_z_hit_distances,
                                detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                used_tpc_size=used_tpc_size, distance_to_optimize=[DISTANCE_TO_OPTIMIZE],
                                min_mcm=11, max_mcm=13, classify=CLASSIFY, tree=None, classifier_threshold=CLASSIFIER_THRESHOLDS,
                                fake_trigger_rate=fake_trigger_rate, number_of_tests=300)

    ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold = opt_parameters
    print(all_effs, "EFF LIST")

    print("\n")
    print("------------------------------------------------------------------------------------------------")
    print("--------- FOUND OPTIMAL PARAMETERS -----------")
    print("Max cluster time: {} µs, Max hit time diff: {} µs, Max hit distance: {} cm".format(ct, ht, hd))
    print("Min hit multiplicity:", opt_mcm)
    print("BDT threshold", classifier_threshold)
    print("Trigger efficiency:", trigger_efficiency)
    print("------------------------------------------------------------------------------------------------")
    print("Calculated for d={} kpc, <E>={} MeV, alpha={}, BTW={} s, SIM_MODE={}".format(distance_to_optimize, AVERAGE_ENERGY, ALPHA, BURST_TIME_WINDOW/1e6, SIM_MODE))
    print("------------------------------------------------------------------------------------------------")
    print("\n")

    sim_parameters = [FAKE_TRIGGER_RATE, BURST_TIME_WINDOW, DISTANCE_TO_OPTIMIZE, SIM_MODE, ADC_MODE, DETECTOR, CLASSIFY, AVERAGE_ENERGY, ALPHA]
    eff_data = [trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_effs, sim_parameters]
    if save:
        file_name = sl.save_efficiency_data(eff_data=eff_data, sim_parameters=sim_parameters, file_name=OUTPUT_NAME_DATA, data_type="data")

    return eff_data


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--energy", type=float, help="Average energy of the SN neutrino spectrum")
    parser.add_argument("-a", "--alpha", type=float, help="Pinching parameter of the SN neutrino spectrum")
    parser.add_argument("-m", "--mode", type=str, help="Simulation mode ('aronly' or 'xe')")
    parser.add_argument("-o", "--output", type=str, help="Name of the output file for the efficiency data. If not specified, the name will be generated randomly.")
    parser.add_argument("-d", "--dto", type=float, help="Distance to optimize for (in kpc)")

    # TODO change this shit
    #parser.add_argument("--params", type=str, help="Parameters file to use for the simulation (default is parameters.py)")
    
    parser.add_argument("--eff-curve", action="store_true", help="Attemp to compute the efficiency curve vs. distance for the current parameters, if the efficiency data file exists.\
                                                                    Else, we will run the whole algorithm first.")
    parser.add_argument("-i", "--input", type=str, help="Name of the input file for the efficiency data, when computing the efficiency curve. If not specified, \
                                                                    the file matching the current parameters will be used.")
    parser.add_argument("--eff-curve-output", type=str, help="Name of the output file for the efficiency curve. If not specified, the name will be generated randomly.")

    args = parser.parse_args()

    if args.energy:
        global AVERAGE_ENERGY
        AVERAGE_ENERGY = args.energy
    if args.alpha:
       global ALPHA
       ALPHA = args.alpha
    if args.mode:
        global SIM_MODE
        SIM_MODE = args.mode
    if args.output:
        global OUTPUT_NAME_DATA
        OUTPUT_NAME_DATA = args.output
    if args.dto:
        global DISTANCE_TO_OPTIMIZE
        DISTANCE_TO_OPTIMIZE = args.dto

    if args.input and args.eff_curve:
        global INPUT_NAME
        INPUT_NAME = args.input
    if args.eff_curve_output:
        global OUTPUT_NAME_CURVE
        OUTPUT_NAME_CURVE = args.eff_curve_output
    calculate_curve = args.eff_curve

    return calculate_curve


def main():

    calculate_eff_curve = parse_arguments()

    print("START ---> ")

    # Load things
    detector = DETECTOR

    true_tpc_size = TRUE_TPC_SIZES[detector]
    used_tpc_size = USED_TPC_SIZES[detector]
    once_a_month_rate = FAKE_TRIGGER_RATE

    # Load SN and BG hits
    sn_limit = SN_FILE_LIMIT
    sn_total_hits, sn_hit_list_per_event, sn_info_per_event, _ = sl.load_all_sn_events_chunky(limit=sn_limit, event_num=1000, detector=detector)

    bg_limit = BG_FILE_LIMIT
    bg_length = bg_limit * BG_SAMPLE_LENGTHS[detector] # in miliseconds
    bg_total_hits, bg_hit_list_per_event, _, _, _ = sl.load_all_backgrounds_chunky_type_separated(limit=bg_limit, detector=detector)

    # Do this to free much needed memory... (actually this is probably useless)
    del bg_total_hits
    del sn_total_hits

    # We need to spit the SN and BG events for training the BDT and efficiency evaluation 
    # (this is not a train/test split, that is done in the BDT training later)
    # (we don't really need to shuffle this as the loading is paralelised anyways)
    if CLASSIFY:
        sn_train_hit_list_per_event = sn_hit_list_per_event[:int(len(sn_hit_list_per_event)*0.5)] 
        sn_hit_list_per_event = sn_hit_list_per_event[int(len(sn_hit_list_per_event)*0.5):]
        sn_train_info_per_event = sn_info_per_event[:int(len(sn_info_per_event)*0.5)]
        sn_info_per_event = sn_info_per_event[int(len(sn_info_per_event)*0.5):]
        bg_train_hit_list_per_event = bg_hit_list_per_event[:int(len(bg_hit_list_per_event)*0.5)]
        bg_hit_list_per_event = bg_hit_list_per_event[int(len(bg_hit_list_per_event)*0.5):]

    
    # The actual computation
    if calculate_eff_curve:
        # This will attemp to load the efficiendu data for a set of parameters, and then calculate it for a set of distances.
        # If the data file does not exist, it will run the whole algorithm.
        try:
            if INPUT_NAME is None:
                eff_data, _ = sl.load_efficiency_data(
                        sim_parameters=[FAKE_TRIGGER_RATE, BURST_TIME_WINDOW, DISTANCE_TO_OPTIMIZE, SIM_MODE, ADC_MODE, DETECTOR, CLASSIFY, AVERAGE_ENERGY, ALPHA], data_type="data")
            else:
                eff_data, _ = sl.load_efficiency_data(file_name=INPUT_NAME, data_type="data")
        except KeyError:
            eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate, save=True)

        distances = DISTANCES
        print(eff_data)
        opt_parameters = eff_data[1]
        opt_mcm = eff_data[2]
        opt_tree = eff_data[3]

        eff_curve_data = get_efficiency_curve(opt_parameters, sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, bg_length, detector, 
                                            true_tpc_size, used_tpc_size, distances, opt_mcm, opt_tree, once_a_month_rate, number_of_tests=400, save=True, plot=True)
        
    else:
        eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate, save=True)

    



if __name__ == '__main__':
    # TODO: fix the global variables, they should be passed as arguments!
    # TODO: fix weird bug where sometimes the BDT training crashes
    main()





    


