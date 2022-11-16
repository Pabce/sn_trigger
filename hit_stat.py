from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time

import uproot
#import ROOT
import pickle
import os
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



def stat_cluster_parameter_check_parallel(sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, 
                                            bg_length, params_list, verbose, detector,
                                            true_tpc_size, used_tpc_size,
                                            min_mcm, max_mcm, classify, loaded_tree, fake_trigger_rate,
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

    for mcm in range(min_mcm, max_mcm + 1):
        tot_start = time.time()
        bg_clusters = []
        bg_hit_multiplicities = []
        for i, hit_list in enumerate(bg_hit_list_per_event):
            if hit_list.size == 0:
                continue 

            bg_clusters_i, bg_hit_multiplicities_i = clustering.clustering_continuous(hit_list, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)
            
            # if bg_clusters_i:
            #     tools.display_hits(bg_clusters_i[0], three_d=True)

            bg_clusters.extend(bg_clusters_i)
            bg_hit_multiplicities.extend(bg_hit_multiplicities_i)
        
        # Find SN clusters for current parameters
        sn_clusters = []
        sn_hit_multiplicities = []
        sn_event_num = len(sn_hit_list_per_event)
        sn_energies = []

        for i, hit_list in enumerate(sn_hit_list_per_event):
            if hit_list.size == 0:
                continue 

            sn_clusters_i, sn_hit_multiplicities_i = clustering.clustering_continuous(hit_list, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)
            
            # if sn_clusters_i:
            #     tools.display_hits(sn_clusters_i[0], three_d=True)

            num_clusters = len(sn_clusters_i)

            if num_clusters == 0:
                sn_clusters_i = [None]
                sn_hit_multiplicities_i = [-1]
                num_clusters = 1

            #print(i, "sn_info_per_event", len(sn_info_per_event), "sn_hit_list_per_event", len(sn_hit_list_per_event))
            sn_energies.extend([sn_info_per_event[i, 0]] * num_clusters)

            sn_clusters.extend(sn_clusters_i)
            sn_hit_multiplicities.extend(sn_hit_multiplicities_i)

        sn_energies = np.array(sn_energies)

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
            features, targets = clustering.cluster_comparison(sn_clusters, bg_clusters)

            if not loaded_tree:
                tree = classifier.gradient_boosted_tree(features, targets, n_estimators=200)
            else:
                tree = loaded_tree

            new_bg_clusters, new_bg_hit_multiplicities, _ = classifier.tree_filter(tree, bg_clusters, features[targets==0, :], threshold=classifier_threshold)
            sn_features = features[targets==1, :]
        else:
            tree = None

        hbins = np.arange(min(sn_hit_multiplicities), max(sn_hit_multiplicities) + 2, 1)
        sn_hist, _ = np.histogram(sn_hit_multiplicities, bins=hbins, density=False)
        bg_hist, bg_bins = np.histogram(bg_hit_multiplicities, bins=hbins, density=False)

        if classify:
            new_bg_hist, _ = np.histogram(new_bg_hit_multiplicities, bins=hbins, density=False)

            zeros = np.where(bg_hist == 0)[0]
            bg_hist[zeros] = 1
            new_bg_hist[zeros] = 1

            filter_bg_ratios = new_bg_hist / bg_hist

            #print(filter_bg_ratios)

        else:
            filter_bg_ratios = 1


        # -------------------
        # Compute for 1 second 

        time_profile_x, time_profile_y = sl.load_time_profile()
        true_fake_trigger_rate = fake_trigger_rate * burst_time_window / 1e6
        sn_model = "LIVERMORE"

        total_event_num = aux.distance_to_event_number(distance_to_optimize, sn_model, true_tpc_size) * SN_EVENT_MULTIPLIER
        event_num_per_time = aux.event_number_per_time(time_profile_x, time_profile_y, total_event_num, burst_time_window)
        #print(total_event_num, event_num_per_time)
        
        expected_bg_hist = bg_hist * 1/bg_length * burst_time_window/1000 * true_tpc_size/used_tpc_size

        # print(aux.get_acceptable_likelihood(expected_bg_hist, once_a_month_rate))
        # aux.cluster_comparison(sn_clusters, bg_clusters)
        # exit()

        start = time.time()
        trigger_efficiency = stat_trigger_efficiency(event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, sn_energies,
                                                    sn_features, sn_info_per_event, expected_bg_hist, filter_bg_ratios,
                                                    hbins, fake_trigger_rate=true_fake_trigger_rate, number_of_tests=number_of_tests, 
                                                    classify=classify, tree=tree, threshold=classifier_threshold)
        
        end = time.time()
        print("Time:", end - start)

        if verbose > 0:
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


def stat_cluster_parameter_scan_parallel(sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event,
                             bg_length, max_cluster_times, max_hit_time_diffs, max_hit_distances, detector="VD", verbose=0,
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
        results = pool.starmap(stat_cluster_parameter_check_parallel, zip(repeat(sn_hit_list_per_event), repeat(sn_info_per_event),
                                                    repeat(bg_hit_list_per_event),
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
    
    print("DALJHFLKJDHFLKAHDJSLSDJFLÑKSHFJLÑAKSDJÑLAJKSD")
    energies = np.linspace(4, 100, 100)
    energies_histogram = aux.pinched_spectrum(energies, average_e=AVERAGE_ENERGY, alpha=ALPHA)

    detected_events = 0
    for i in range(number_of_tests):
        
        # Draw a true event number from the expected event number. REMEMBER, NOT ALL SN EVENTS HAVE CLUSTERS.
        # with_cluster_rate = len(sn_clusters)/sn_event_num
        # r = int(stats.poisson.rvs(event_num_per_time, size=1) * with_cluster_rate)
        r = int(stats.poisson.rvs(event_num_per_time, size=1))

        if r == 0:
            continue

        # Draw randomly from the SN event pool, while following a given energy distribution!
        # sample_ind = np.random.choice(np.arange(len(sn_clusters)), size=r, replace=False)
        sample_ind_e = aux.get_energy_indices(energies, energies_histogram, sn_energies, size=r)


        # selected_energies = sn_energies[sample_ind_e]
        # plt.figure()
        # plt.hist(selected_energies, bins=20, density=True)
        # plt.plot(energies, energies_histogram)
        # plt.show()

        sn_hit_multiplicities = np.array(sn_hit_multiplicities)
        sn_clusters = np.array(sn_clusters)
        
        # Remove clusterless samples
        sample_ind_with_clusters = []
        for ind in sample_ind_e:
            if sn_hit_multiplicities[ind] != -1:
                sample_ind_with_clusters.append(ind)

        sample_ind = np.array(sample_ind_with_clusters)

        #print(sn_clusters[0].shape)
        #print(sample_ind)
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
        # Bunch together where the number of entries is < 3 or 4
        
        sn_hist, _ = np.histogram(sample, bins=hbins, density=False)

        expected_bg_chi, chi_last_index = aux.bunch_histogram(expected_bg_hist * filter_bg_ratios, limit=3)
        observed_chi, chi_last_index = aux.bunch_histogram(sn_hist + fake_bg_hist, last_index=chi_last_index)
        # fake_bg_chi, chi_last_index = aux.bunch_histogram(fake_bg_hist, limit=4)# last_index=chi_last_index)
        # sn_chi, chi_last_index = aux.bunch_histogram(sn_hist, last_index=chi_last_index)
        # print(expected_bg_chi, "BG-CH")
        # print(observed_chi, "CHICH")

        # ------------------
        # Remove terms where we have no SN hits
        #to_keep = np.where(sn_hist != 0)[0]

        # print("CHI SQUARED", aux.chi_squared(sn_chi + fake_bg_chi, expected_bg_chi, dof=1))
        # print(once_a_month_rate)

        cut = 0

        chi_sq, dof, chi_squared_significance = aux.chi_squared(observed_chi[cut:], 
                                                                expected_bg_chi[cut:], 
                                                                dof=len(expected_bg_chi[cut:]) - 1)
        
        # ------------------------
        # time_profile_x, time_profile_y = sl.load_time_profile()

        # csqs = []
        # for fact in np.arange(0.1, 10, 0.1):
        #     ev1= event_number_per_time(time_profile_x, time_profile_y, 100, BURST_TIME_WINDOW*fact)
        #     ev2 = event_number_per_time(time_profile_x, time_profile_y, 100, BURST_TIME_WINDOW)
        #     prop = ev1/ev2
        #     observed_chi, chi_last_index = aux.bunch_histogram(sn_hist * prop + fake_bg_hist * fact, limit=3)
        #     expected_bg_chi, chi_last_index = aux.bunch_histogram(expected_bg_hist * filter_bg_ratios * fact, last_index=chi_last_index)
        #     chi_sqf, doff, chi_squared_significancef = aux.chi_squared(observed_chi[cut:], 
        #                                                         expected_bg_chi[cut:], 
        #                                                         dof=len(expected_bg_chi[cut:]) - 1)
        #     csqs.append(chi_squared_significancef / fact)

        #     #print("FACTOR", fact, chi_squared_significancef)
        # min_csqs = np.min(csqs)
        # minfact = np.arange(0.1, 10, 0.1)[np.argmin(csqs)]

        # if min_csqs < fake_trigger_rate:
        #     print("MIN", min_csqs, fake_trigger_rate, minfact)

        # ------------------------

        # corrected_expected_bg_hist = deepcopy(expected_bg_hist)
        # corrected_expected_bg_hist[expected_bg_hist == 0] = 1 / (bg_length/1000)
        # likelihood = aux.poisson_likelihood(sn_hist + fake_bg_hist, corrected_expected_bg_hist)
        # print(likelihood)
        # if likelihood > 140:
        #     detected_events_lk += 1

        #print(chi_sq, chi_squared_significance, dof, "CHI")
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
    print("OOOOOOOO", sn_event_num)

    return trigger_efficiency



def get_efficiency_curve(opt_parameters, sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, bg_length, detector, true_tpc_size, 
                            used_tpc_size, distances, opt_mcm, opt_tree, fake_trigger_rate, number_of_tests):
                            
    mct, mht, mhd, mxd, myd, mzd, _, cthresh = opt_parameters
    distance_te, _, _, _, all_te, all_params = stat_cluster_parameter_scan_parallel(
                                    sn_hit_list_per_event,  sn_info_per_event,
                                    bg_hit_list_per_event, bg_length,
                                    max_cluster_times=[mct],
                                    max_hit_time_diffs=[mht], 
                                    max_hit_distances=[mhd],
                                    max_x_hit_distances=[mxd],
                                    max_y_hit_distances=[myd],
                                    max_z_hit_distances=[mzd],
                                    detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                    used_tpc_size=used_tpc_size, distance_to_optimize=distances,
                                    min_mcm=opt_mcm, max_mcm=opt_mcm, classify=True,
                                    classifier_threshold=[cthresh], tree=opt_tree,
                                    fake_trigger_rate=fake_trigger_rate, number_of_tests=number_of_tests)
    
    # distance_te, _, _, _, all_te_2, all_params_2 = stat_cluster_parameter_scan_parallel(
    #                                 sn_hit_list_per_event, 
    #                                 bg_hit_list_per_event, bg_length,
    #                                 max_cluster_times=[mct],
    #                                 max_hit_time_diffs=[mht], 
    #                                 max_hit_distances=[mhd],
    #                                 max_x_hit_distances=[mxd],
    #                                 max_y_hit_distances=[myd],
    #                                 max_z_hit_distances=[mzd],
    #                                 detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
    #                                 used_tpc_size=used_tpc_size, distance_to_optimize=distances,
    #                                 min_mcm=opt_mcm, max_mcm=opt_mcm, classify=True,
    #                                 classifier_threshold=[cthresh], tree=opt_tree,
    #                                 fake_trigger_rate=once_a_month_rate * 10)

    dists = [p[-2] for p in all_params]
    dists_2 = [p[-2] for p in all_params]
    # plt.plot(dists, all_te)
    # plt.plot(dists_2, all_te)
    # plt.show()
    print(AVERAGE_ENERGY, ALPHA, "OE")
    pickle.dump([dists, dists_2, all_te, all_te], open("../saved_pickles/LAn{}_b{}_kys_{}_{}_{}".format(DISTANCE_TO_OPTIMIZE,
                                                                                         BURST_TIME_WINDOW, AVERAGE_ENERGY, ALPHA, SIM_MODE), "wb"))
    pickle.dump( (trigger_efficiency, opt_parameters, opt_mcm) , open("../saved_pickles/LAn{}_b{}_pys_{}_{}_{}".format(DISTANCE_TO_OPTIMIZE,
                                                                                         BURST_TIME_WINDOW ,AVERAGE_ENERGY, ALPHA, SIM_MODE), "wb"))

    return trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_params



def stat_main(save=True, load_detected_clusters=False):
    detector = DETECTOR

    true_tpc_size = TRUE_TPC_SIZES[detector]
    used_tpc_size = USED_TPC_SIZES[detector]

    # Load SN and BG hits
    sn_limit = 10 # SN_FILE_LIMIT
    sn_total_hits, sn_hit_list_per_event, sn_info_per_event, _ = sl.load_all_sn_events_chunky(limit=sn_limit, event_num=1000, detector=detector)


    bg_limit = 30 # BG_FILE_LIMIT
    #bg_sample_length = 8.5 # in ms
    bg_length = bg_limit * BG_SAMPLE_LENGTHS[detector] # in miliseconds
    bg_total_hits, bg_hit_list_per_event, _, _, _ = sl.load_all_backgrounds_chunky_type_separated(limit=bg_limit, detector=detector)

    # Do this to free much needed memory... (actually this is probably useless)
    del bg_total_hits
    del sn_total_hits

    # for i, l in enumerate(sn_hit_list_per_event):
    #     plt.scatter(len(l), sn_info_per_event[i][0])
    #     plt.show()

    # plt.scatter(len(sn_total_hits), sn_info_t[0])
    # plt.show()
    
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

    once_a_month_rate = FAKE_TRIGGER_RATE

    # Find the optimal parameters without cluster filtering
    if load_detected_clusters:
        # trigger_efficiency, opt_params =\
        #         pickle.load(open("../saved_pickles/detected_clusters_{}".format(bg_rate), "rb"))
        pass
    else:
        trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_effs, _ = stat_cluster_parameter_scan_parallel(
                                        sn_hit_list_per_event, sn_info_per_event,
                                        bg_hit_list_per_event, bg_length,
                                        max_cluster_times=max_cluster_times,
                                        max_hit_time_diffs=max_hit_time_diffs, 
                                        max_hit_distances=max_hit_distances,
                                        max_x_hit_distances=max_x_hit_distances,
                                        max_y_hit_distances=max_y_hit_distances,
                                        max_z_hit_distances=max_z_hit_distances,
                                        detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                        used_tpc_size=used_tpc_size, distance_to_optimize=[DISTANCE_TO_OPTIMIZE],
                                        min_mcm=11, max_mcm=13, classify=True, tree=None, classifier_threshold=CLASSIFIER_THRESHOLD,
                                        fake_trigger_rate=once_a_month_rate, number_of_tests=300)

        print(all_effs, "EFF LIST")

    # Once we have the trigger efficiency optimized for the desired distance, we compute it for all distances
    efficiencies_list = []
    distances = np.arange(4, 50, 1.5)

    return trigger_efficiency, opt_parameters, opt_mcm, opt_tree, None
    
    mct, mht, mhd, mxd, myd, mzd, _, cthresh = opt_parameters
    distance_te, _, _, _, all_te, all_params = stat_cluster_parameter_scan_parallel(
                                    sn_hit_list_per_event,  sn_info_per_event,
                                    bg_hit_list_per_event, bg_length,
                                    max_cluster_times=[mct],
                                    max_hit_time_diffs=[mht], 
                                    max_hit_distances=[mhd],
                                    max_x_hit_distances=[mxd],
                                    max_y_hit_distances=[myd],
                                    max_z_hit_distances=[mzd],
                                    detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                    used_tpc_size=used_tpc_size, distance_to_optimize=distances,
                                    min_mcm=opt_mcm, max_mcm=opt_mcm, classify=True,
                                    classifier_threshold=[cthresh], tree=opt_tree,
                                    fake_trigger_rate=once_a_month_rate, number_of_tests=400)
    
    # distance_te, _, _, _, all_te_2, all_params_2 = stat_cluster_parameter_scan_parallel(
    #                                 sn_hit_list_per_event, 
    #                                 bg_hit_list_per_event, bg_length,
    #                                 max_cluster_times=[mct],
    #                                 max_hit_time_diffs=[mht], 
    #                                 max_hit_distances=[mhd],
    #                                 max_x_hit_distances=[mxd],
    #                                 max_y_hit_distances=[myd],
    #                                 max_z_hit_distances=[mzd],
    #                                 detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
    #                                 used_tpc_size=used_tpc_size, distance_to_optimize=distances,
    #                                 min_mcm=opt_mcm, max_mcm=opt_mcm, classify=True,
    #                                 classifier_threshold=[cthresh], tree=opt_tree,
    #                                 fake_trigger_rate=once_a_month_rate * 10)

    dists = [p[-2] for p in all_params]
    dists_2 = [p[-2] for p in all_params]
    # plt.plot(dists, all_te)
    # plt.plot(dists_2, all_te)
    # plt.show()
    print(AVERAGE_ENERGY, ALPHA, "OE")
    pickle.dump([dists, dists_2, all_te, all_te], open("../saved_pickles/LAn{}_b{}_kys_{}_{}_{}".format(DISTANCE_TO_OPTIMIZE,
                                                                                         BURST_TIME_WINDOW, AVERAGE_ENERGY, ALPHA, SIM_MODE), "wb"))
    pickle.dump( (trigger_efficiency, opt_parameters, opt_mcm) , open("../saved_pickles/LAn{}_b{}_pys_{}_{}_{}".format(DISTANCE_TO_OPTIMIZE,
                                                                                         BURST_TIME_WINDOW ,AVERAGE_ENERGY, ALPHA, SIM_MODE), "wb"))

    return trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_params





if __name__ == '__main__':
    if len(sys.argv) == 3:
        AVERAGE_ENERGY = float(sys.argv[1])
        ALPHA = float(sys.argv[2])
    
    if len(sys.argv) == 4:
        AVERAGE_ENERGY = float(sys.argv[1])
        ALPHA = float(sys.argv[2])
        SIM_MODE = sys.argv[3]


    print("START ---> ")

    print(stat_main())




    


