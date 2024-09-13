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
from rich.console import Console
from rich.table import Table

import classifier
import aux 
import data_loader as dl
import clustering
import config as cf
#from numba import jit



def stat_cluster_parameter_check_parallel(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                                            bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, params_list, verbose, detector,
                                            true_tpc_size, used_tpc_size, classify, loaded_tree, fake_trigger_rate,
                                            number_of_tests, classifier_hist_type, optimize_hyperparameters):

    # Random seed for each parallel process
    random.seed()
    # ---------------------------

    to_plot = False
    print("STARTING NEW CPU PROCESS", mp.current_process())

    # Arguments to add
    burst_time_window = BURST_TIME_WINDOW # microseconds

    ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold, mcm = params_list
    # Does this even make sense?
    if ct < ht:
        return None

    # Run the clustering algorithm for all SN and BG samples, both regular and those for training
    tot_start = time.time()

    bg_clusters, bg_hit_multiplicities, sn_clusters, sn_hit_multiplicities, sn_energies = clustering.full_clustering(sn_hit_list_per_event, sn_info_per_event,
                                                bg_hit_list_per_event, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                min_hit_mult=mcm, detector=detector, verbose=verbose-1)
    
    if classify and not loaded_tree:
        bg_clusters_train, _, sn_clusters_train, sn_hit_multiplicities_train, _ = \
                                                clustering.full_clustering(sn_train_hit_list_per_event, sn_train_info_per_event,
                                                    bg_train_hit_list_per_event, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)

    if len(sn_hit_multiplicities) < 3 or len(bg_hit_multiplicities) < 3:
        return None
    
    # Convert to np arrays
    sn_hit_multiplicities = np.array(sn_hit_multiplicities)
    sn_clusters = np.array(sn_clusters, dtype=object)
    if classify and not loaded_tree:
        sn_hit_multiplicities_train = np.array(sn_hit_multiplicities_train)
        sn_clusters_train = np.array(sn_clusters_train, dtype=object)

    # hbins = np.arange(min(sn_hit_multiplicities[sn_hit_multiplicities>=0])-0.5, max(sn_hit_multiplicities) + 1.5, 2)
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
        #print(loaded_tree)
        if not loaded_tree:
            # We remove the clusterless SN samples for training!
            sn_clusters_train = sn_clusters_train[sn_hit_multiplicities_train >= 0]
            train_features, train_targets = clustering.cluster_comparison(sn_clusters_train, bg_clusters_train)
            train_time = time.time()
            tree, _, _, _, _ = classifier.hist_gradient_boosted_tree(train_features, train_targets, n_estimators=200, optimize_hyperparameters=optimize_hyperparameters)
            print("Tree training time:", time.time() - train_time)
        else:
            tree = loaded_tree

        # Now, we filter the regular clusters with the newly trained classifier
        features, targets = clustering.cluster_comparison(sn_clusters, bg_clusters)
        _ , new_bg_hit_multiplicities, bg_predictions_prob = classifier.tree_filter(tree, bg_clusters, features[targets==0, :], threshold=classifier_threshold)
        sn_features = features[targets==1, :]
    else:
        tree = None
    
    # ---------------------------

    # We can use two types of histograms: one with hit multiplicity, one with the BDT output
    if classifier_hist_type == "hit_multiplicity":
        hbins = np.arange(min(sn_hit_multiplicities[sn_hit_multiplicities >= 0]), max(sn_hit_multiplicities) + 2, 1)
        bg_hist, _ = np.histogram(bg_hit_multiplicities, bins=hbins, density=False)
        # _ , _, sn_predictions_prob = classifier.tree_filter(tree, sn_clusters, sn_features, threshold=classifier_threshold)
        # plt.hist(bg_hit_multiplicities, bins=hbins, density=True)
        # plt.hist(sn_hit_multiplicities, bins=hbins, density=True, alpha=0.5)
        # plt.show()
    elif classifier_hist_type == "bdt":
        hbins = np.arange(0, 1.06, 0.05)
        bg_hist, _ = np.histogram(bg_predictions_prob, bins=hbins, density=False)
        
        # _ , _, sn_predictions_prob = classifier.tree_filter(tree, sn_clusters, sn_features, threshold=classifier_threshold)
        # sn_hist, _ = np.histogram(sn_predictions_prob, bins=hbins, density=False)
        # plt.hist(bg_predictions_prob, bins=hbins, density=True)
        # plt.hist(sn_predictions_prob, bins=hbins, density=True, alpha=0.5)
        # plt.show()

    if classify and classifier_hist_type == "hit_multiplicity":
        # We create the new bg min hit mult histogram for the new bg clusters, if classifier is used, and compute the ratios
        new_bg_hist, _ = np.histogram(new_bg_hit_multiplicities, bins=hbins, density=False)
        placeholder_bg_hist = deepcopy(bg_hist)

        zeros = np.where(bg_hist == 0)[0]
        placeholder_bg_hist[zeros] = 1
        new_bg_hist[zeros] = 1

        filter_bg_ratios = new_bg_hist / placeholder_bg_hist
        #print(filter_bg_ratios)
    else:
        filter_bg_ratios = 1
    
    #print(filter_bg_ratios)
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
                                                classify=classify, tree=tree, threshold=classifier_threshold, classifier_hist_type=classifier_hist_type)
    
    end = time.time()
    print("Trigger efficiency Time:", end - start)
    tot_end = time.time()
    print("Total time:", tot_end - tot_start)

    if verbose > 0:
        print("\n")
        print("Min multiplicity: {}, Trigger efficiency: {}, (params: {}, {}, {}, {}, {}, {}, (DTO: {}, Thresh: {}))".format(mcm, 
                        trigger_efficiency, *params_list))
        
        print("Number of SN clusters: {}, Number of BG clusters: {}".format(len(sn_clusters), len(bg_clusters)))
        print("\n")


    print("----------------- Tested parameters ->")
    print("Max cluster time: {}, Max hit time diff: {}, Max hit distance: {}".format(ct, ht, hd))
    print("Max X distance {}, Max Y distance {}, Max Z distance: {}".format(xhd, yhd, zhd))
    print("Min hit multiplicity:", mcm)
    print("DTO, Threshold", distance_to_optimize, classifier_threshold)
    print("Trigger efficiency:", trigger_efficiency)
    print("--------------------------------------")


    return trigger_efficiency, mcm, tree, params_list


def stat_cluster_parameter_scan_parallel(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event, bg_hit_list_per_event,
                             bg_train_hit_list_per_event, bg_length, max_cluster_times, max_hit_time_diffs, max_hit_distances, detector="VD", verbose=0,
                             max_x_hit_distances=None, max_y_hit_distances=None, max_z_hit_distances=None, true_tpc_size=10,
                             used_tpc_size=2.6, distance_to_optimize=[30], mcms=[10], classify=False, tree=None, 
                             classifier_threshold=[0.5], fake_trigger_rate=1/(60 * 60 * 24 * 30), number_of_tests=500, classifier_hist_type="hit_multiplicity",
                             optimize_hyperparameters=False):

    opt_parameters = ()
    max_trigger_efficiency = 0
    opt_mcm = 0
    opt_tree = None
    
    params_list = itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances, 
                                    max_x_hit_distances, max_y_hit_distances, max_z_hit_distances, distance_to_optimize, classifier_threshold, mcms)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(stat_cluster_parameter_check_parallel, zip(repeat(sn_hit_list_per_event), repeat(sn_train_hit_list_per_event),repeat(sn_info_per_event),
                                                    repeat(sn_train_info_per_event), repeat(bg_hit_list_per_event), repeat(bg_train_hit_list_per_event),
                                                    repeat(bg_length), params_list, repeat(verbose), repeat(detector),
                                                    repeat(true_tpc_size), repeat(used_tpc_size), repeat(classify), repeat(tree),
                                                    repeat(fake_trigger_rate), repeat(number_of_tests), repeat(classifier_hist_type), repeat(optimize_hyperparameters)))
    
    all_efficiencies = []
    all_parameters = []
    for result in results:
        if result:
            trigger_efficiency, mcm, o_tree, params = result
            ct, ht, hd, xhd, yhd, zhd, dto, mcm, _ = params
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
                            tree=None, threshold=0.5, classifier_hist_type="hit_multiplicity"):
    
    #print("TE GOING ON")
    energies = np.linspace(4, 100, 100)
    energies_histogram = aux.pinched_spectrum(energies, average_e=AVERAGE_ENERGY, alpha=ALPHA)

    # Bunch together BG histogram where the number of entries is < 3 or 4 (so the chi_squared statistic is reliable)      
    expected_bg_chi, chi_bunched_indices, chi_bins = aux.bunch_histogram(expected_bg_hist * filter_bg_ratios, hbins, limit=3)
        
    # plt.bar(hbins[:-1], expected_bg_hist, width=1, alpha=1, label="Expected BG")
    # plt.bar(hbins[:-1], expected_bg_hist * filter_bg_ratios, width=1, alpha=0.5, label="")
    # plt.yscale("log")
    # plt.title("{}, {}".format(np.sum(expected_bg_hist * filter_bg_ratios), event_num_per_time))
    # plt.show()

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
        
        # Remove clusterless samples (clusterless samples have been labeled with a hit multiplicity of -1)
        # Why the fuck is this so convoluted? I seem to remember there was a reason
        sample_ind_with_clusters = []
        for ind in sample_ind_e:
            if sn_hit_multiplicities[ind] != -1:
                sample_ind_with_clusters.append(ind)

        sample_ind = np.array(sample_ind_with_clusters)
        if len(sample_ind) == 0:
            continue
        sample = sn_hit_multiplicities[sample_ind]

        #print(r, len(sample_ind_e), len(sample))

        # Filter the sample if we chose to apply the classifier
        # FIXME
        if classify:
            sample = sample
            class_sn_features, targ = clustering.cluster_comparison(sn_clusters[sample_ind], [])
            _, sample, sn_predictions_prob = classifier.tree_filter(tree, sn_clusters[sample_ind], 
                                                    class_sn_features[targ==1], threshold=threshold)
        #     #print(sample)
        #     #print("Sample after classifier ratio", len(sample)/l1)
        
        # print(len(sample))

        # Generate a random poisson background from the expected backgrounds
        fake_bg_hist = np.round(stats.poisson.rvs(expected_bg_hist, size=len(hbins)-1) * filter_bg_ratios)
        # Put this into an array with entries correspoinding to the histogram values
        fake_bg = np.repeat(hbins[:-1], fake_bg_hist.astype(int))

        # Generate histogram and compute statistic
        # --- CHI SQUARED ---
        # Bunch together where the number of entries is < 3 or 4 (so the chi_squared statistic is reliable)
        
        if classifier_hist_type == "hit_multiplicity":
            sn_hist, _ = np.histogram(sample, bins=hbins, density=False)
        elif classifier_hist_type == "bdt":
            sn_hist, _ = np.histogram(sn_predictions_prob, bins=hbins, density=False)
            # plt.hist(sn_predictions_prob, bins=hbins, density=False, alpha=0.6)
            # plt.show()

        observed_chi, _, _ = aux.bunch_histogram(sn_hist + fake_bg_hist, hbins, bunched_indices=chi_bunched_indices)

        # Get the chi squared statistic
        cut = 0
        chi_sq, dof, chi_squared_significance = aux.chi_squared(observed_chi[cut:], 
                                                    expected_bg_chi[cut:], dof=len(expected_bg_chi[cut:]) - 1)
        
        # If any of the bins has a smaller Poisson likelihood, we use that instead
        # poisson_sig = 1 - stats.poisson.cdf(sn_hist + fake_bg_hist + 0.01, expected_bg_hist * filter_bg_ratios + 0.01)
        # poisson_sig = np.min(poisson_sig[poisson_sig > 0])

        # if poisson_sig < chi_squared_significance:
        #     chi_squared_significance = poisson_sig
        
        # --- KS ---
        # The "observed" here shouldn't be the histograms, but rather the arrays
        if STATISITCAL_METHOD == "ks":
            sn_arr = sn_predictions_prob if classifier_hist_type == "bdt" else sample
            ks, pval = aux.kolmogorov_smirnov(np.concatenate((sn_arr, fake_bg)), expected_bg_hist, filter_bg_ratios, hbins)
            chi_squared_significance = pval                                        
        # --- KS ---


        if chi_squared_significance < fake_trigger_rate: # or np.isnan(chi_squared_significance):
            detected_events += 1
        elif chi_squared_significance is np.nan:
            print("NAN")
            print(len(sample))
            print(observed_chi)
        # elif classifier_hist_type == "hit_multiplicity" and 1 < 0:
        #     if len(sample[sample > 21]) >= 2:
        #         detected_events += 1


        #to_plot = True
        if to_plot:
            plt.figure(5)
            plt.bar(hbins[:-1], fake_bg_hist + sn_hist, width=np.diff(hbins), alpha=0.8)
            plt.bar(hbins[:-1], expected_bg_hist * filter_bg_ratios, width=np.diff(hbins), alpha=0.5)
            plt.yscale('log')
            plt.title("{} - {}".format(chi_squared_significance, fake_trigger_rate))

            plt.figure(7)
            plt.bar(chi_bins[:-1] + np.diff(chi_bins)*0.5, observed_chi, width=np.diff(chi_bins), alpha=0.8)
            plt.bar(chi_bins[:-1] + np.diff(chi_bins)*0.5, expected_bg_chi, width=np.diff(chi_bins), alpha=0.5)
            plt.yscale('log')
            plt.title("{} - {}".format(chi_squared_significance, fake_trigger_rate))

            plt.show()
    
    trigger_efficiency = detected_events/number_of_tests
    #print("OOOOOOOO", sn_event_num)

    return trigger_efficiency



def get_efficiency_curve(opt_parameters, sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, bg_length, detector, true_tpc_size, 
                            used_tpc_size, distances, opt_mcm, opt_tree, fake_trigger_rate, number_of_tests, save=True, plot=True):

    mct, mht, mhd, mxd, myd, mzd, _, cthresh, mcm = opt_parameters
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
                                    mcms=[opt_mcm], classify=CLASSIFY,
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
                        bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, fake_trigger_rate,
                        max_cluster_times, max_hit_time_diffs, max_hit_distances, mcms, optimize_hyperparameters, verbose=1, save=True):
    
    # Add background to our SN events (this is basically useless as the SN events are so short in time)
    # for i in range(len(sn_hit_list_per_event)):
    #     #display_hits(sn_hit_list_per_event[i], time=True)
    #     if len(sn_hit_list_per_event[i]) > 0:
    #         sn_hit_list_per_event[i] = aux.spice_sn_event(sn_hit_list_per_event[i], bg_hit_list_per_event,
    #                                                 bg_length_to_add=0.8, bg_length=bg_sample_length * 1000)
    #     #display_hits(sn_hit_list_per_event[i], time=True)


    # Find cluster parameters that maximize trigger efficiency
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
                                mcms=mcms,
                                detector=detector, verbose=verbose, true_tpc_size=true_tpc_size, 
                                used_tpc_size=used_tpc_size, distance_to_optimize=[DISTANCE_TO_OPTIMIZE],
                                classify=CLASSIFY, tree=None, classifier_threshold=CLASSIFIER_THRESHOLDS,
                                fake_trigger_rate=fake_trigger_rate, number_of_tests=300,
                                classifier_hist_type=CLASSIFIER_HIST_TYPE,
                                optimize_hyperparameters=optimize_hyperparameters)

    # We attempt to extract the optimal parameters. If all efficiencies are equal to 0, this will throw an error (not exactly...).
    # We select the first set of parameters by default
    try:
        ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold, opt_mcm = opt_parameters
    except ValueError:
        opt_parameters = max_cluster_times[0], max_hit_time_diffs[0], max_hit_distances[0], max_x_hit_distances[0], max_y_hit_distances[0], max_z_hit_distances[0],\
                            DISTANCE_TO_OPTIMIZE, CLASSIFIER_THRESHOLDS[0], mcms[0]
        ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold, opt_mcm = opt_parameters
        print("----- WARNING: No optimal parameters found, using default parameters -----")
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

    parser.add_argument("--eff-curve-from-scratch", action="store_true", help="Run the optimization algorithm from scratch, even if a previous file already exists. Then compute the\
                                                                                efficiency curve vs. distance.")

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
    calculate_curve_from_scratch = args.eff_curve_from_scratch

    return calculate_curve, calculate_curve_from_scratch


def main():

    # Read configuration file from the command line
    # And create the configuration class, including the "loaded" parameters
    config = cf.Configurator.file_from_command_line() # Instance of the config class

    console = Console()

    console.log('--- START --->', style="bold green")
    console.print(":chipmunk:  Let's get started! :chipmunk:", style="bold yellow")
    
    # Create a table to display the parameters
    table = Table(title="Parameters", title_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Average energy", f"{config.get('Physics', 'neutrino_flux', 'average_energy')} MeV")
    table.add_row("Alpha", f"{config.get('Physics', 'neutrino_flux', 'alpha')}")
    table.add_row("Simulation mode", f"{config.get('Simulation', 'sim_mode')}")
    table.add_row("Detector", f"{config.get('Detector', 'type')}")
    table.add_row("Distance to optimize", f"{config.get('Simulation', 'distance_to_optimize')} kpc")
    console.print(table)

    # Load parameters from the configuration file
    true_tpc_size = config.get('Detector', 'true_tpc_size') * config.get('Detector', 'tpc_size_correction_factor')
    used_tpc_size = config.get('Detector', 'used_tpc_size') * config.get('Detector', 'tpc_size_correction_factor')
    once_a_month_rate = config.get('Simulation', 'fake_trigger_rate')

    # Load SN and BG hits for the parameter search
    sn_file_limit_parameter_search = config.get('Simulation', 'sn_file_limit_parameter_search')
    bg_file_limit_parameter_search = config.get('Simulation', 'bg_file_limit_parameter_search')

    # Create a dataloader object
    loader = dl.DataLoader(config)

    sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event,\
    sn_train_info_per_event, bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length =\
        loader.load_and_split(sn_file_limit_parameter_search, bg_file_limit_parameter_search)
    
    exit("STOP")
    # The actual computation
    # TODO: fix and tidy up the ifs below
    if calculate_eff_curve or calculate_eff_curve_from_scratch:
        # This will attemp to load the efficiency data for a set of parameters, and then calculate it for a set of distances.
        # If the data file does not exist, it will run the whole algorithm.

        if calculate_eff_curve:
            try:
                if INPUT_NAME is None:
                    eff_data, _ = sl.load_efficiency_data(
                            sim_parameters=[FAKE_TRIGGER_RATE, BURST_TIME_WINDOW, DISTANCE_TO_OPTIMIZE, SIM_MODE, ADC_MODE, DETECTOR, CLASSIFY, AVERAGE_ENERGY, ALPHA], data_type="data")
                else:
                    eff_data, _ = sl.load_efficiency_data(file_name=INPUT_NAME, data_type="data")
            except KeyError:
                eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                    bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate, save=True)
        
        elif calculate_eff_curve_from_scratch:
            eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                    bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate,
                    max_cluster_times=MAX_CLUSTER_TIMES, max_hit_time_diffs=MAX_HIT_TIME_DIFFS, max_hit_distances=MAX_HIT_DISTANCES,
                    mcms=np.arange(LOWER_MIN_HIT_MULTUPLICITY, UPPER_MIN_HIT_MULTUPLICITY + 1), optimize_hyperparameters=False, save=True, verbose=1)

            trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_effs, sim_parameters = eff_data
            ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold, mcm = opt_parameters

            print("----- Found optimal clustering parameters ----- RERUNNING WITH FULL STATISTICS -----")
            
            # Now, rerun this with full statistics for the optimal parameters
            sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event, bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length =\
                load_and_split(SN_FILE_LIMIT, BG_FILE_LIMIT, detector)
            
            eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                    bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate,
                    max_cluster_times=[ct], max_hit_time_diffs=[ht], max_hit_distances=[hd],
                    mcms=[mcm], optimize_hyperparameters=False, save=True)

        distances = DISTANCES
        print(eff_data)
        opt_parameters = eff_data[1]
        opt_mcm = eff_data[2]
        opt_tree = eff_data[3]

        eff_curve_data = get_efficiency_curve(opt_parameters, sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, bg_length, detector, 
                                            true_tpc_size, used_tpc_size, distances, opt_mcm, opt_tree, once_a_month_rate, number_of_tests=400, save=True, plot=True)
        
    else:
        eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate,
                max_cluster_times=MAX_CLUSTER_TIMES, max_hit_time_diffs=MAX_HIT_TIME_DIFFS, max_hit_distances=MAX_HIT_DISTANCES,
                mcms=np.arange(LOWER_MIN_HIT_MULTUPLICITY, UPPER_MIN_HIT_MULTUPLICITY + 1), optimize_hyperparameters=False, save=True)

        trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_effs, sim_parameters = eff_data
        ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold, mcm = opt_parameters

        print("----- Found optimal clustering parameters ----- RERUNNING WITH FULL STATISTICS -----")
        
        # Now, rerun this with full statistics for the optimal parameters
        sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event, bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length =\
            load_and_split(SN_FILE_LIMIT, BG_FILE_LIMIT, detector)
        
        eff_data = optimize_efficiency(sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event, sn_train_info_per_event,
                bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length, detector, true_tpc_size, used_tpc_size, once_a_month_rate,
                max_cluster_times=[ct], max_hit_time_diffs=[ht], max_hit_distances=[hd],
                mcms=[mcm], optimize_hyperparameters=True, save=True)


if __name__ == '__main__':
    # TODO: fix the global variables, they should be passed as arguments!
    # TODO: fix weird bug where sometimes the BDT training crashes
    main()





    


