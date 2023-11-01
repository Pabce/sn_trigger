from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import uproot
#import ROOT
import pickle
import os
from scipy import stats
import itertools
from itertools import repeat
import multiprocessing as mp
import random

import classifier
import aux
#from numba import jit


# We also need to read the geometry info for the FDVD detector
# Distance is in cm
# There is only 168 "real" optical channels that get hits
OP_DISTANCE_ARRAY_VD = pickle.load(open("../saved_pickles/op_distance_array_VD", "rb"))[0:168, 0:168]
OP_X_DISTANCE_ARRAY_VD = pickle.load(open("../saved_pickles/op_x_distance_array_VD", "rb"))[0:168, 0:168]
OP_Y_DISTANCE_ARRAY_VD = pickle.load(open("../saved_pickles/op_y_distance_array_VD", "rb"))[0:168, 0:168]
OP_Z_DISTANCE_ARRAY_VD = pickle.load(open("../saved_pickles/op_z_distance_array_VD", "rb"))[0:168, 0:168]

OP_DISTANCE_ARRAY_HD = pickle.load(open("../saved_pickles/op_distance_array_HD", "rb"))
OP_DISTANCE_ARRAY = {"VD": OP_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_X_DISTANCE_ARRAY = {"VD": OP_X_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_Y_DISTANCE_ARRAY = {"VD": OP_Y_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_Z_DISTANCE_ARRAY = {"VD": OP_Z_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}

TRUE_TPC_SIZES = {"VD": 10, "HD": 10}
USED_TPC_SIZES = {"VD": 2.6, "HD": 1}
BG_SAMPLE_LENGTHS = {"VD": 1000, "HD": 4.492 * 200} 


# Values of v_e CC interactions at 10 kpc for a 40 kton LArTPC
INTERACTION_NUMBER_10KPC = {"LIVERMORE": 2684, "GKVM": 3295, "GARCHING": 882}

def load_all_backgrounds(limit=1, detector="VD"):
    bg_total_hits = []
    bg_hit_list_per_event = []
    directory = os.fsencode('../')

    if detector == 'VD':
        dir = '../'
        #dir = '../vd_backgrounds/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'pbg'
    elif detector == 'HD':
        dir = '../horizontaldrift/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'hd_pbg'

    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(endswith) and filename.startswith(startswith):
            i += 1
            if i > limit:
                break
            print(filename)
            
            filename = dir + filename 
            bg_total_hits_i, bg_hit_list_per_event_i, _ = load_hit_data(file_name=filename)
            bg_total_hits.append(bg_total_hits_i)
            bg_hit_list_per_event.extend(bg_hit_list_per_event_i)

    bg_total_hits = np.concatenate(bg_total_hits, axis=0)

    return bg_total_hits, bg_hit_list_per_event, None


def load_all_backgrounds_chunky(limit=1, detector="VD"):
    bg_total_hits = []
    bg_hit_list_per_event = []

    if detector == 'VD':
        dir = '../'
        dir = '../vd_backgrounds/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'pbg'
    elif detector == 'HD':
        dir = '../horizontaldrift/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'hd_pbg'
    
    file_names = []
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(endswith) and filename.startswith(startswith):
            i += 1
            if i > limit:
                break
            print(filename)
            
            filename = dir + filename 
            file_names.append(filename)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(load_hit_data, zip(file_names, repeat(-1)))
    
    for i, result in enumerate(results):
        bg_total_hits_i, bg_hit_list_per_event_i, _ = result
        bg_total_hits.append(bg_total_hits_i)
        bg_hit_list_per_event.extend(bg_hit_list_per_event_i)

    bg_total_hits = np.concatenate(bg_total_hits, axis=0)

    return bg_total_hits, bg_hit_list_per_event, None


def load_hit_data(file_name="pbg_g4_digi_reco_hist.root", event_num=-1):
    # The hit info is imported from a .root file.
    # From the hit info, we find the clusters

    # Keep it simple for now (shape: X*7)
    # X and Y coordinates are switched!!!

    print(file_name, mp.current_process())


    uproot_ophit = uproot.open(file_name)["opflashana/PerOpHitTree"]
    total_hits_dict = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime', 'X_OpDet_hit', 'Y_OpDet_hit', 'Z_OpDet_hit',
                                            'Width', 'Area', 'Amplitude', 'PE'], library="np")
    total_hits = [total_hits_dict[key] for key in total_hits_dict]
    total_hits = np.array(total_hits).T
    #print(total_hits.shape)    

    # Remove the signal data for the HD "backgrounds"
    # if is_bg:
    #     time_sort = np.argsort(total_hits[:,3])
    #     ordered_hits = total_hits[time_sort, :]
    
    #     zero_remove = np.where(np.abs(ordered_hits[:,3]) > 10)[0]
    #     ordered_hits = ordered_hits[zero_remove, :]

    #     total_hits = ordered_hits

    hit_list_per_event, hit_num_per_channel = process_hit_data(total_hits, event_num)

    return total_hits, hit_list_per_event, hit_num_per_channel


def process_hit_data(total_hits, event_num=-1):
    if event_num == -1:
        event_num = int(np.max(total_hits[:, 0]))

    hit_number_per_event = np.zeros(event_num)
    hit_list_per_event = [[] for _ in range(event_num)]
    detection_per_event = np.zeros(event_num)

    for i in range(len(total_hits)):
        event_id = total_hits[i, 0]
        ev = int(event_id) - 1

        if ev >= event_num:
            continue

        hit_number_per_event[ev] += 1

        hit_list_per_event[ev].append(total_hits[i, :])

    for i in range(event_num):
        hit_list_per_event[i] = np.array(hit_list_per_event[i])

        # We order the hits in time
        if hit_list_per_event[i].shape != (0,):
            time_sort = np.argsort(hit_list_per_event[i][:,3])
            hit_list_per_event[i] = hit_list_per_event[i][time_sort, :]


    unique, counts = np.unique(total_hits[:, 2], return_counts=True)
    hit_num_per_channel = dict(zip(unique, counts))

    return hit_list_per_event, hit_num_per_channel    


# TIME PROFILE IS IN MS!!!!
def load_time_profile():
    uproot_time_profile = uproot.open("../TimeProfile.root")["h_MarlTime"]
    time_profile = uproot_time_profile.to_numpy()

    # print(time_profile[0].shape)
    # print(time_profile[1].shape)
    # plt.scatter(time_profile[1][:-1], time_profile[0][:])
    # plt.yscale('log')
    # plt.show()

    return time_profile[1][:-1], time_profile[0][:]


def display_hits(hits, three_d=False, time=False):
    # Weird notation to accomodate for list of individual hits
    x_coords, y_coords, z_coords = hits[:, 5], hits[:, 4], hits[:, 6]
    time_coords = hits[:, 3]

    fig = plt.figure()

    if three_d:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z_coords, x_coords, y_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        #ax.set_box_aspect((np.ptp(hits[:][6]), np.ptp(hits[:][5]), np.ptp(hits[:][4])))

    elif time:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z_coords, x_coords, time_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')
        ax.set_zlabel('time')
        #ax.set_box_aspect((np.ptp(hits[:,6]), np.ptp(hits[:,5]), np.ptp(hits[:,4])))
    
    else:
        ax = fig.add_subplot()
        ax.scatter(z_coords, x_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')

    plt.show()



#@jit(nopython=True)
def time_clustering(hits, max_cluster_time, max_hit_time_diff, min_hit_mult, verbose=0):
    hit_num = len(hits)
    if verbose > 1:
        print("Number of hits:", hit_num)

    candidate_clusters = []

    # First get the candidate clusters timewise. We're using a 'greedy approach' where we always start clustering
    # from earlier to later times.
    i = 0
    while i < hit_num:
        #print(i)
        hit = hits[i, :]
        time = hit[3]
        #op_channel = hit[2]

        # Find the furthest hit still within the maximum cluster time
        max_i = hit_num
        for j in range(i + 1, hit_num):
            if hits[j, 3] - time > max_cluster_time:
                max_i = j
                break
        
        current_cluster = hits[i:max_i, :]

        if max_i - i < min_hit_mult:
            if verbose > 0:
                print("Cluster too small after max_cluster_time filter, size:", (max_i - i))
            i += 1
            continue
        else:
            i = max_i

        # Find which are within the allowed time
        subclusters = []
        subcluster = []
        prev_time = 1e10
        for k in range(len(current_cluster)):
            ktime = current_cluster[k, 3]

            if ktime - prev_time <= max_hit_time_diff:
                #print(time-prev_time)
                subcluster.append(current_cluster[k, :])
            else:
                subclusters.append(subcluster)
                subcluster = []
            
            if k == len(current_cluster) - 1:
                subclusters.append(subcluster)
                subcluster = []
            
            prev_time = ktime
        
        candidate_clusters.extend(subclusters)

    # And now we get the length of each cluster
    candidate_clusters_2 = []
    candidate_hit_multiplicities = []
    for cluster in candidate_clusters:
        if len(cluster) < min_hit_mult:
            if verbose > 0:
                print("Cluster too small after max_time_dif filter, size:", len(cluster))
        else:
            candidate_clusters_2.append(cluster)
            candidate_hit_multiplicities.append(len(cluster))
    
    
    return candidate_clusters_2, candidate_hit_multiplicities



def spatial_correlation(cluster, max_hit_distance, max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, detector="VD", verbose=0):
    # This algorithm is extremely stupid...
    space_cluster = []

    for j, hit1 in enumerate(cluster):
        min_distance = 1e5
        min_x_distance, min_y_distance, min_z_distance = 1e5, 1e5, 1e5
        for k, hit2 in enumerate(cluster):
            
            op_ch1, op_ch2 = int(hit1[2]), int(hit2[2])

            # We don't want two hits in the same place to count as 0 distance!
            if op_ch1 == op_ch2:
                continue

            distance = OP_DISTANCE_ARRAY[detector][op_ch1, op_ch2]
            x_distance = OP_X_DISTANCE_ARRAY[detector][op_ch1, op_ch2]
            y_distance = OP_Y_DISTANCE_ARRAY[detector][op_ch1, op_ch2]
            z_distance = OP_Z_DISTANCE_ARRAY[detector][op_ch1, op_ch2]
            
            if distance < min_distance:
                min_distance = distance
            if x_distance < min_x_distance:
                min_x_distance = x_distance
            if y_distance < min_y_distance:
                min_y_distance = y_distance
            if z_distance < min_z_distance:
                min_z_distance = z_distance

        acceptable_distance = True
        if min_distance > max_hit_distance:
            acceptable_distance = False
        if min_x_distance > max_x_hit_distance or min_y_distance > max_y_hit_distance or min_z_distance > max_z_hit_distance:
            #print("ROSI", min_x_distance, min_y_distance, min_z_distance)
            acceptable_distance = False
        
        if acceptable_distance:
            space_cluster.append(hit1)

    return space_cluster


# Do a continuous clustering respecting the time distribution (for bg)
# I guess this can also be used for "discrete"? Just feeding events one by one...
def clustering_continuous(hits, max_cluster_time, max_hit_time_diff, max_hit_distance, max_x_hit_distance, max_y_hit_distance, max_z_hit_distance,
                            min_hit_mult, verbose=0, spatial_filter=True, detector="VD"):
    
    # If min hit multiplicity is -1, we collect all clusters and store their individual hit multiplicity 
    
    candidate_clusters_2, candidate_hit_multiplicities = time_clustering(hits, max_cluster_time, max_hit_time_diff,
                                                    min_hit_mult, verbose=0)

    # Which of these candidate clusters satisfy spatial requirements?
    final_clusters = []
    final_hit_multiplicities = []
    if spatial_filter:
        for cluster in candidate_clusters_2:
            space_cluster = spatial_correlation(cluster, max_hit_distance, max_x_hit_distance, max_y_hit_distance, 
                                                max_z_hit_distance, verbose=verbose, detector=detector)
            if len(space_cluster) < min_hit_mult:
                if verbose > 0:
                    print("Cluster too small after spatial filter, size:", len(space_cluster))
            else:
                final_clusters.append(np.array(space_cluster))
                final_hit_multiplicities.append(len(space_cluster))
    else:
        final_clusters = candidate_clusters_2
        final_hit_multiplicities = candidate_hit_multiplicities

    # cand_len = len(candidate_clusters_2) 
    # final_len = len(final_clusters)
    # if cand_len == 0:
    #     space_frac = -1
    # else:
    #     space_frac = (cand_len - final_len)/cand_len

    return final_clusters, final_hit_multiplicities


# It only makes sense to call this with BG hits!!!
# This is unnecessarily complicated, we can reuse the clusters. Or can we? You have to think about this. 
# Even if you can, it may not be possible with the more sophisticated clustering you are planning
# (actually it might be the other way round!)
def hit_multiplicity_scan(hit_list_per_event, threshold, min_mult, max_mult, max_cluster_time=100e-3, 
                            max_hit_time_diff=70e-3, max_hit_distance=250, max_x_hit_distance=-1, max_y_hit_distance=-1,
                            max_z_hit_distance=-1, verbose=0, spatial_filter=True, detector="VD"):
    
    multiplicities_tested = []
    clusters_found = []

    allow_skips = True
    i = min_mult
    while i <= max_mult:

        clusters_per_event = []
        s_frac_list = []
        t_frac_list = []

        for j, hit_list in enumerate(hit_list_per_event):
            #print("Event:", j)
            clusters, _ = clustering_continuous(hit_list, max_cluster_time, max_hit_time_diff, max_hit_distance,
                                                    max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, 
                                                    min_hit_mult=i, spatial_filter=spatial_filter, detector=detector)
            clusters_per_event.append(clusters)
            # if space_frac > -1:
            #     s_frac_list.append(space_frac)
            # if time_frac > -1:
            #     t_frac_list.append(time_frac)
        
        # s_frac_total = np.mean(s_frac_list)
        # t_frac_total = np.mean(t_frac_list)
        # print("TIME REMOVED FRACTION:", t_frac_total, "MULT:", i)
        # print("SPATIAL REMOVED FRACTION:", s_frac_total, "MULT:", i)
        #print("Hit multiplicity:", i)

        total_cl = 0
        for k, cl in enumerate(clusters_per_event):
            if verbose > 1:
                print("Event: {}, Clusters: {}".format(k, len(cl)))
            total_cl += len(cl)
        
        multiplicities_tested.append(i)
        clusters_found.append(total_cl)

        if verbose > 0:
            print("Multiplicity: {}, BG clusters found: {}, (params: {}, {}, {}, {}, {}, {})".format(i, 
                            total_cl, max_cluster_time, max_hit_time_diff, max_hit_distance,
                            max_x_hit_distance, max_y_hit_distance, max_z_hit_distance))

        if total_cl <= threshold:
            # If we have got here by skipping, we need to go back!!!
            if skipped:
                i -= 3
                allow_skips = False
                print("--------------------dfsad--")
                continue
            if verbose > 0:
                print("Threshold met", i)

            return i, multiplicities_tested, clusters_found
        
        # If the cluster multiplicity is much higher than the threshold, we can skip a few!
        skipped = False
        if total_cl > threshold * 100 and allow_skips:
            print(threshold, "DINONONONONONO")
            i += 6
            skipped = True

        elif total_cl > threshold * 10 and allow_skips:
            print(threshold, "AJKJKJKJKJKJKJ")
            i += 3
            skipped = True

        else:
            i += 1

    print("No hit multiplicity was found satisfying >= threshold within the specified limits")

    return -1, -1, -1


def spice_sn_event(hit_list, bg_hit_list_per_event, bg_length_to_add, bg_length):
    # Select one random BG sample and time order it (it should already be time ordered, but to be safe and future-proof)
    bg_sample = random.choice(bg_hit_list_per_event)
    time_sort = np.argsort(bg_sample[:, 3])
    bg_sample = bg_sample[time_sort, :]

    # Select a random time interval within the bg sample
    #bg_hit_number = len(bg_sample[:, 3])
    starting_point = np.random.rand() * bg_length - bg_length / 2 - bg_length_to_add
    end_point = starting_point + bg_length_to_add

    bg_sample = bg_sample[bg_sample[:, 3] > starting_point, :]
    bg_sample = bg_sample[bg_sample[:, 3] < end_point, :]
    
    #print(bg_sample.shape, starting_point, end_point)

    # Shift the center of the interval to t = the centre of the SN event
    bg_sample[:, 3] -= (starting_point + end_point)/2 # to 0
    bg_sample[:, 3] += (hit_list[0, 3] + hit_list[-1, 3])/2

    #print(bg_sample[0, 3], bg_sample[-1, 3])

    # Now combine with the sn hit sample (and sort in time)
    spiced_hit_list = np.vstack((bg_sample, hit_list))

    time_sort = np.argsort(spiced_hit_list[:, 3])
    spiced_hit_list = spiced_hit_list[time_sort, :]

    return spiced_hit_list


def cluster_parameter_check_parallel(sn_hit_list_per_event, bg_hit_list_per_event,
                                    bg_threshold_rate, bg_length, cluster_params, verbose, detector, each_dim_distance):            

    print("AKFDSJHFLKADSHJFLSDFFASDKJLHJLKJ", mp.current_process())

    if each_dim_distance:
        ct, ht, hd, xhd, yhd, zhd  = cluster_params
    else:
        ct, ht, hd = cluster_params
        xhd, yhd, zhd = -1, -1, -1
    if ct < ht:
        return None

    # Convert threshold from Hz to event number
    threshold = bg_threshold_rate * bg_length / 1000

    # Find the minimum hit multiplicity imposed by the background for these parameters
    min_hit_mult, _, _ = hit_multiplicity_scan(bg_hit_list_per_event, threshold, 14, 85, ct, ht, hd, xhd, yhd, zhd,
                                                verbose=verbose, detector=detector)

    if min_hit_mult == -1:
        return None

    # Then, find the SN event reconstructed clusters
    clusters_per_event = []
    for i, hit_list in enumerate(sn_hit_list_per_event):
        if hit_list.size == 0:
            continue 

        clusters, _ = clustering_continuous(hit_list, max_cluster_time=ct, max_hit_time_diff=ht, 
                                            max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                            min_hit_mult=min_hit_mult, detector=detector, verbose=verbose-1)
        
        # if clusters:
        #     display_hits(clusters[0], three_d=True)
        clusters_per_event.append(clusters)

    total_cl = 0
    for i, cl in enumerate(clusters_per_event):
        #print("Event: {}, Clusters: {}".format(i, len(cl)))

        total_cl += len(cl)

    print("----------------- Tested parameters ->")
    print("Max cluster time: {}, Max hit time diff: {}, Max hit distance: {}".format(ct, ht, hd))
    print("Max X distance {}, Max Y distance {}, Max Z distance: {}".format(xhd, yhd, zhd))
    print("Min hit multiplicity from bg:", min_hit_mult)
    print("Total clusters:", total_cl)
    print("--------------------------------------")

    return clusters_per_event, total_cl, cluster_params, min_hit_mult


def stat_cluster_parameter_check_parallel(sn_hit_list_per_event, bg_hit_list_per_event, 
                                            bg_length, params_list, verbose, detector,
                                            true_tpc_size, used_tpc_size,
                                            min_mcm, max_mcm, classify, tree, fake_trigger_rate):

    # Random seed for each parallel process
    random.seed()
    # ---------------------------

    to_plot = False
    print("BLOBERYEYEY", mp.current_process())

    # Arguments to add
    burst_time_window = 1e6

    ct, ht, hd, xhd, yhd, zhd, distance_to_optimize, classifier_threshold = params_list
    # Does this even make sense?
    if ct < ht:
        return None

    
    optimal_mcm = 0
    optimal_trigger_efficiency = 0
    optimal_tree = None
    # Find background clusters for current parameters
    for mcm in range(min_mcm, max_mcm + 1):
        bg_clusters = []
        bg_hit_multiplicities = []
        for i, hit_list in enumerate(bg_hit_list_per_event):
            if hit_list.size == 0:
                continue 

            bg_clusters_i, bg_hit_multiplicities_i = clustering_continuous(hit_list, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)
            
            # if clusters:
            #     display_hits(clusters[0], three_d=True)

            bg_clusters.extend(bg_clusters_i)
            bg_hit_multiplicities.extend(bg_hit_multiplicities_i)
        
        # Find SN clusters for current parameters
        sn_clusters = []
        sn_hit_multiplicities = []
        sn_event_num = len(sn_hit_list_per_event)

        for i, hit_list in enumerate(sn_hit_list_per_event):
            if hit_list.size == 0:
                continue 

            sn_clusters_i, sn_hit_multiplicities_i = clustering_continuous(hit_list, max_cluster_time=ct, max_hit_time_diff=ht, 
                                                    max_hit_distance=hd, max_x_hit_distance=xhd, max_y_hit_distance=yhd, max_z_hit_distance=zhd,
                                                    min_hit_mult=mcm, detector=detector, verbose=verbose-1)
            
            # if clusters:
            #     display_hits(clusters[0], three_d=True)

            sn_clusters.extend(sn_clusters_i)
            sn_hit_multiplicities.extend(sn_hit_multiplicities_i)

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
            features, targets = aux.cluster_comparison(sn_clusters, bg_clusters)

            if not tree:
                tree = classifier.gradient_boosted_tree(features, targets, n_estimators=200)

            new_bg_clusters, new_bg_hit_multiplicities, _ = classifier.tree_filter(tree, bg_clusters, features[targets==0, :], threshold=classifier_threshold)
            sn_features = features[targets==1, :]

        hbins = np.arange(min(sn_hit_multiplicities), max(sn_hit_multiplicities) + 2, 1)
        sn_hist, _ = np.histogram(sn_hit_multiplicities, bins=hbins, density=False)
        bg_hist, bg_bins = np.histogram(bg_hit_multiplicities, bins=hbins, density=False)

        if classify:
            new_bg_hist, _ = np.histogram(new_bg_hit_multiplicities, bins=hbins, density=False)

            filter_bg_ratios = new_bg_hist / bg_hist
            filter_bg_ratios[bg_hist == 0] = 1
            print(filter_bg_ratios)

        else:
            filter_bg_ratios = 1

        # --- LIKELIHOOD ---
        # To compensate for low background statistics, we assume a conservative value of 2 where bg is found to be 0
        # (is this maybe not needed?)
        #bg_hist[bg_hist < 1] = 0.1 # WTF TO DO WITH THIS??????

        # -------------------
        # Compute for 1 second 
        time_profile_x, time_profile_y = load_time_profile()
        true_fake_trigger_rate = fake_trigger_rate * burst_time_window / 1e6
        sn_model = "LIVERMORE"

        total_event_num = distance_to_event_number(distance_to_optimize, sn_model, true_tpc_size)
        event_num_per_time = event_number_per_time(time_profile_x, time_profile_y, total_event_num, burst_time_window)
        #print(total_event_num, event_num_per_time)
        
        expected_bg_hist = bg_hist * 1/bg_length * burst_time_window/1000 * true_tpc_size/used_tpc_size

        # print(aux.get_acceptable_likelihood(expected_bg_hist, once_a_month_rate))
        # aux.cluster_comparison(sn_clusters, bg_clusters)
        # exit()

        
        trigger_efficiency = stat_trigger_efficiency(event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, 
                                                    sn_features, expected_bg_hist, filter_bg_ratios,
                                                    hbins, fake_trigger_rate=true_fake_trigger_rate, number_of_tests=1000, 
                                                    classify=classify, tree=tree, threshold=classifier_threshold)
            
        if verbose > 0:
            print("Min multiplicity: {}, Trigger efficiency: {}, (params: {}, {}, {}, {}, {}, {}, (DTO: {}, Thresh: {}))".format(mcm, 
                            trigger_efficiency, *params_list))

        if trigger_efficiency > optimal_trigger_efficiency:
            optimal_trigger_efficiency = trigger_efficiency
            optimal_mcm = mcm
            optimal_tree = tree

    print("----------------- Tested parameters ->")
    print("Max cluster time: {}, Max hit time diff: {}, Max hit distance: {}".format(ct, ht, hd))
    print("Max X distance {}, Max Y distance {}, Max Z distance: {}".format(xhd, yhd, zhd))
    print("Min hit multiplicity from bg:", optimal_mcm)
    print("DTO, Threshold", distance_to_optimize, classifier_threshold)
    print("Trigger efficiency:", optimal_trigger_efficiency)
    print("--------------------------------------")

    return optimal_trigger_efficiency, optimal_mcm, optimal_tree, params_list


def stat_cluster_parameter_scan_parallel(sn_hit_list_per_event, bg_hit_list_per_event,
                             bg_length, max_cluster_times, max_hit_time_diffs, max_hit_distances, detector="VD", verbose=0,
                             max_x_hit_distances=None, max_y_hit_distances=None, max_z_hit_distances=None, true_tpc_size=10,
                             used_tpc_size=2.6, distance_to_optimize=[30], min_mcm=15, max_mcm=30, classify=False, tree=None, 
                             classifier_threshold=[0.5], fake_trigger_rate=1/(60 * 60 * 24 * 30)):

    opt_parameters = ()
    max_trigger_efficiency = 0
    opt_mcm = 0
    opt_tree = None
    
    params_list = itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances, 
                                    max_x_hit_distances, max_y_hit_distances, max_z_hit_distances, distance_to_optimize, classifier_threshold)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(stat_cluster_parameter_check_parallel, zip(repeat(sn_hit_list_per_event), repeat(bg_hit_list_per_event),
                                                    repeat(bg_length), params_list, repeat(verbose), repeat(detector),
                                                    repeat(true_tpc_size), repeat(used_tpc_size),
                                                    repeat(min_mcm), repeat(max_mcm), repeat(classify), repeat(tree),
                                                    repeat(fake_trigger_rate)))
    
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



def stat_trigger_efficiency(event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, sn_features, expected_bg_hist, 
                            filter_bg_ratios, hbins, fake_trigger_rate, to_plot=False, number_of_tests=1000, classify=False, 
                            tree=None, threshold=0.5):
    
    detected_events = 0
    for i in range(number_of_tests):
        
        # Draw a true event number from the expected event number. REMEMBER, NOT ALL SN EVENTS HAVE CLUSTERS.
        with_cluster_rate = len(sn_clusters)/sn_event_num
        r = int(stats.poisson.rvs(event_num_per_time, size=1) * with_cluster_rate)

        if r == 0:
            continue

        # Draw randomly from the SN event pool. 
        sample_ind = np.random.choice(np.arange(len(sn_clusters)), size=r, replace=False)


        sn_hit_multiplicities = np.array(sn_hit_multiplicities)
        sn_clusters = np.array(sn_clusters, dtype=object)
        
        #print(sn_clusters[0].shape)
        sample = sn_hit_multiplicities[sample_ind]
        #print(sample)
        l1 = len(sample)

        # Filter the sample if we chose to apply the classifier
        if classify:
            class_sn_features, th = aux.cluster_comparison(sn_clusters[sample_ind], [])
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

        observed_chi, chi_last_index = aux.bunch_histogram(sn_hist + fake_bg_hist, limit=3)
        expected_bg_chi, chi_last_index = aux.bunch_histogram(expected_bg_hist * filter_bg_ratios, last_index=chi_last_index)
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
            plt.bar(hbins[:-1], fake_bg_hist + sn_hist, width=1)
            plt.bar(hbins[:-1], expected_bg_hist, width=1)
            plt.yscale('log')
            # plt.ylim(top=10)
            # plt.xlim(left=15)

            plt.figure(3)
            plt.bar(hbins[:chi_last_index + 1], observed_chi - expected_bg_chi, width=1)

            
            plt.figure(4)
            plt.bar(hbins[:-1], sn_hist, width=1)

            plt.show() 
    
    trigger_efficiency = detected_events/number_of_tests

    return trigger_efficiency


def cluster_parameter_scan_parallel(sn_hit_list_per_event, bg_hit_list_per_event,
                             bg_threshold_rate, bg_length, max_cluster_times, max_hit_time_diffs, max_hit_distances, detector="VD", verbose=0,
                             max_x_hit_distances=None, max_y_hit_distances=None, max_z_hit_distances=None, each_dim_distance=False):
    detected_clusters = {}
    detected_clusters_count = np.zeros((len(max_cluster_times), len(max_hit_time_diffs), len(max_hit_distances)))
    if each_dim_distance:
        detected_clusters_count = np.zeros((len(max_cluster_times), len(max_hit_time_diffs), len(max_hit_distances), 
                                            len(max_x_hit_distances), len(max_y_hit_distances), len(max_z_hit_distances)))
    max_detected_clusters = 0
    opt_parameters = ()
    
    params_list = itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances)
    if each_dim_distance:
        params_list = itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances, 
                                        max_x_hit_distances, max_y_hit_distances, max_z_hit_distances)

    #param_index_list = list(itertools.product(max_cluster_times, max_hit_time_diffs, max_hit_distances))
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(cluster_parameter_check_parallel, zip(repeat(sn_hit_list_per_event), repeat(bg_hit_list_per_event),
                                                    repeat(bg_threshold_rate), repeat(bg_length), params_list, repeat(verbose), 
                                                    repeat(detector), repeat(each_dim_distance)))
    
    for result in results:
        if result:
            if not each_dim_distance:
                clusters_per_event, total_cl, cluster_params, min_hit_mult = result
                ct, ht, hd = cluster_params
                a = max_cluster_times.index(ct)
                b = max_hit_time_diffs.index(ht)
                c = max_hit_distances.index(hd)
                detected_clusters[(ct, ht, hd)] = clusters_per_event
                detected_clusters_count[a, b, c] = total_cl

            else:
                clusters_per_event, total_cl, cluster_params, min_hit_mult = result
                ct, ht, hd, xhd, yhd, zhd = cluster_params
                a = max_cluster_times.index(ct)
                b = max_hit_time_diffs.index(ht)
                c = max_hit_distances.index(hd)
                d = max_x_hit_distances.index(xhd)
                e = max_y_hit_distances.index(yhd)
                f = max_z_hit_distances.index(zhd)
                detected_clusters[(ct, ht, hd, xhd, yhd, zhd)] = clusters_per_event
                detected_clusters_count[a, b, c, d, e, f] = total_cl
            

            if total_cl > max_detected_clusters:
                max_detected_clusters = total_cl
                opt_parameters = cluster_params + (min_hit_mult,)

    return detected_clusters, detected_clusters_count, max_detected_clusters, opt_parameters



# Bg rate in Hz, window in mus
def fake_trigger_rate(bg_rate, maximum_bg_rate, burst_time_window, min_cluster_mult, tpc_size=10):
    # Poisson parameter
    mu = bg_rate * burst_time_window/1e6

    ft_rate = bg_rate * (1 - stats.poisson.cdf(min_cluster_mult - 1, mu))
    #print(ft_rate, maximum_bg_rate)
    if ft_rate <= maximum_bg_rate:
        print("FOUND MIN CLUST MULT:", min_cluster_mult)

    return ft_rate, ft_rate <= maximum_bg_rate


def snb_trigger_efficiency(bg_rate, burst_time_window, min_cluster_mult, event_number, time_profile_x,
                            time_profile_y, sn_event_detection_efficiency, true_tpc_size=10, used_tpc_size=2.6):

    signal = sn_event_detection_efficiency * event_number_per_time(time_profile_x, time_profile_y, 
                                                                    event_number, burst_time_window)
    # Correction for TPC size
    # COMMENTING THIS OUT MAY MAKE MUCH MORE SENSE THAN IT SEEMS!!!
    #signal *= true_tpc_size/used_tpc_size 
                                                
    mu = bg_rate * burst_time_window/1e6 + signal

    snb_te = 1 - stats.poisson.cdf(min_cluster_mult - 1, mu)

    return snb_te


def event_number_per_time(time_profile_x, time_profile_y, total_event_number, burst_time_window):
    # We integrate the time profile in (0, time_window)
    # TIME PROFILE IS IN MS!
    spacing = time_profile_x[1] - time_profile_x[0]
    total_integral = np.sum(time_profile_y) * spacing/1000 * 10 # From ms to s, times 9.9 seconds duration
    # The total integral is normalized to 1 so no need for ratios

    # We compute the ratio with the desired burst window and multiply by the total number of events
    # WE NEED TO CONVERT THE BURST TIME WINDOW TO MS!!!
    ms_burst_time_window = burst_time_window/1000
    b_index = 0
    interp = 0
    for i, t in enumerate(time_profile_x):
        if t > ms_burst_time_window:
            b_index = i - 1
            interp = (t - ms_burst_time_window) / spacing
            break

    #print(b_index, interp)
    # Do some interpolation
    int_1 = np.sum(time_profile_y[0: b_index - 1]) * spacing/1000 * 10
    int_2 = np.sum(time_profile_y[b_index - 1: b_index]) * spacing/1000 * 10
    window_integral = int_1 + interp * int_2

    event_number = window_integral * total_event_number
    #print(event_number)
    #print(window_integral, total_integral, b_index, interp)

    return event_number


def event_number_to_distance(event_number, model="LIVERMORE", tpc_size=10):

    distance = np.sqrt(INTERACTION_NUMBER_10KPC[model] * 10**2 * (tpc_size/40) * 1/event_number)

    return distance


def distance_to_event_number(distance, model="LIVERMORE", tpc_size=10):

    event_num = INTERACTION_NUMBER_10KPC[model] * 10**2 * (tpc_size/40) * 1/distance**2

    return event_num


def calculate_optimal_burst_time_window(bg_rate, threshold_rate, time_profile_x, time_profile_y, sn_event_eff, true_tpc_size=10, used_tpc_size=2.6):
    # Correction to bg_rate due to using a subsection of the TPC
    bg_rate *= true_tpc_size/used_tpc_size

    max_trigger_efficiency = 0 # np.zeros(5)
    optimal_btw = 0
    optimal_mcm = 0

    for burst_time_window in range(int(1e5), int(5e6), 10000):
        minimum_cluster_mult = 0
        for mcm in range(1000):
            ftr, acceptable = fake_trigger_rate(bg_rate, threshold_rate, burst_time_window, mcm)
            #print("Fake trigger rate:", ftr, "mcm:", mcm)
            if acceptable:
                minimum_cluster_mult = mcm
                break
            if mcm == 999:
                minimum_cluster_mult = 4000
                break
        
        # We now can find the trigger efficiency for a particular value of event number, it is enough
        #detected_evnum = event_number_per_time(time_profile_x, time_profile_y, 10, burst_time_window)
        # test_numbers = np.array([1, 5, 10, 20, 50]) * true_tpc_size/used_tpc_size
        # trigger_efficiency = snb_trigger_efficiency(bg_rate, burst_time_window, minimum_cluster_mult, test_numbers,
        #                             time_profile_x, time_profile_y, sn_event_eff, true_tpc_size=true_tpc_size, used_tpc_size=used_tpc_size)

        # print("TE", trigger_efficiency, "BTW", burst_time_window, mcm)       

        # update = True
        # for i, n in enumerate(test_numbers):
        #     if trigger_efficiency[i] < max_trigger_efficiency[i]:
        #         update = False
        #         break
        
        # Your previous assumption was wrong: a higher MCM will make a steeper slope, so you actually have to choose
        # at what event number of distance to optimize!
        update = False
        test_number = distance_to_event_number(20, tpc_size=true_tpc_size)
        trigger_efficiency = snb_trigger_efficiency(bg_rate, burst_time_window, minimum_cluster_mult, test_number,
                                time_profile_x, time_profile_y, sn_event_eff, true_tpc_size=true_tpc_size, used_tpc_size=used_tpc_size)

        
        if trigger_efficiency > max_trigger_efficiency:
            update = True

        if update:
            max_trigger_efficiency = trigger_efficiency
            optimal_btw = burst_time_window
            optimal_mcm = minimum_cluster_mult

    return max_trigger_efficiency, optimal_btw, optimal_mcm


def stat_main(save=True, load_detected_clusters=False):
    detector = "VD"

    true_tpc_size = 10
    used_tpc_size = 2.6

    # Load hits
    sn_sim_number = 10000
    sn_total_hits, sn_hit_list_per_event, hit_num_per_channel = load_hit_data(
                                                                file_name="../psn-specll_g4_digi_reco_hist.root", event_num=sn_sim_number)

     # Do this to free much needed memory...
    del sn_total_hits

    bg_limit = 200
    bg_sample_length = 100 # in ms
    bg_total_hits, bg_hit_list_per_event, hit_num_per_channel = load_all_backgrounds_chunky(limit=bg_limit, detector=detector)

    # Do this to free much needed memory...
    del bg_total_hits

    # # Add background to our SN events
    # for i in range(len(sn_hit_list_per_event)):
    #     #display_hits(sn_hit_list_per_event[i], time=True)
    #     if len(sn_hit_list_per_event[i]) > 0:
    #         sn_hit_list_per_event[i] = spice_sn_event(sn_hit_list_per_event[i], bg_hit_list_per_event,
    #                                                 bg_length_to_add=0.8, bg_length=bg_sample_length * 1000)
    #     #display_hits(sn_hit_list_per_event[i], time=True)

    bg_length = bg_limit * bg_sample_length # in miliseconds
    #bg_length = 4.492 * 200 * 8

    # Find cluster parameters that maximize trigger efficiency
    max_cluster_times = [0.25]
    max_hit_time_diffs = [0.14]
    max_hit_distances = [150]
    # New cluster parameters! (wooo)
    max_x_hit_distances = [2500]
    max_y_hit_distances = [2000]
    max_z_hit_distances = [3000]

    once_a_month_rate = 1/(60 * 60 * 24 * 30)

    # Find the optimal parameters without cluster filtering
    if load_detected_clusters:
        # trigger_efficiency, opt_params =\
        #         pickle.load(open("../saved_pickles/detected_clusters_{}".format(bg_rate), "rb"))
        pass
    else:
        trigger_efficiency, opt_parameters, opt_mcm, opt_tree,  _, _ = stat_cluster_parameter_scan_parallel(
                                        sn_hit_list_per_event, 
                                        bg_hit_list_per_event, bg_length,
                                        max_cluster_times=max_cluster_times,
                                        max_hit_time_diffs=max_hit_time_diffs, 
                                        max_hit_distances=max_hit_distances,
                                        max_x_hit_distances=max_x_hit_distances,
                                        max_y_hit_distances=max_y_hit_distances,
                                        max_z_hit_distances=max_z_hit_distances,
                                        detector=detector, verbose=1, true_tpc_size=true_tpc_size, 
                                        used_tpc_size=used_tpc_size, distance_to_optimize=[20],
                                        min_mcm=10, max_mcm=17, classify=True, tree=None, classifier_threshold=[0.4],
                                        fake_trigger_rate=once_a_month_rate)
    
    # Apply the classifier to these optimal parameters, and recalculate the efficiency


    # Once we have the trigger efficiency optimized for the desired distance, we compute it for all distances
    efficiencies_list = []
    distances = np.arange(5, 50, 1)

    return trigger_efficiency, opt_parameters, opt_mcm, opt_tree
    
    mct, mht, mhd, mxd, myd, mzd, _, cthresh = opt_parameters
    distance_te, _, _, _, all_te, all_params = stat_cluster_parameter_scan_parallel(
                                    sn_hit_list_per_event, 
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
                                    fake_trigger_rate=once_a_month_rate)
    
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
    plt.plot(dists, all_te)
    plt.plot(dists_2, all_te)

    pickle.dump([dists, dists_2, all_te, all_te], open("../saved_pickles/kys_3", "wb"))

    return trigger_efficiency, opt_parameters, opt_mcm, opt_tree, all_params


def main(save=True, load_detected_clusters=False):
    detector = "VD"

    true_tpc_size = 10
    used_tpc_size = 2.6

    # Load hits
    sn_sim_number = 10000
    sn_total_hits, sn_hit_list_per_event, hit_asnum_per_channel = load_hit_data(
                                                                file_name="../psn-specll_g4_digi_reco_hist.root", event_num=sn_sim_number)
    #sn_total_hits, sn_hit_list_per_event, hit_num_per_channel = load_hit_data(
    #                                                            file_name="../horizontaldrift/hdprod_g4_digi_reco_hist.root", event_num=sn_sim_number)

    bg_limit = 200
    bg_total_hits, bg_hit_list_per_event, hit_num_per_channel = load_all_backgrounds_chunky(limit=bg_limit, detector=detector)

    for i in range(len(sn_hit_list_per_event)):
        #display_hits(sn_hit_list_per_event[i], time=True)
        if len(sn_hit_list_per_event[i]) > 0:
            sn_hit_list_per_event[i] = spice_sn_event(sn_hit_list_per_event[i], bg_hit_list_per_event, bg_length_to_add=1, bg_length=100*1000)
        #display_hits(sn_hit_list_per_event[i], time=True)

    bg_rate = 0.1
    bg_length = bg_limit * 100 # in miliseconds
    #bg_length = 4.492 * 200 * 8

    # For a fixed bg rate, find cluster parameters that maximize SN event detection efficiency
    max_cluster_times = [0.15]
    max_hit_time_diffs = [0.05]
    max_hit_distances = [230]
    # New cluster parameters! (wooo)
    max_x_hit_distances = [2500]
    max_y_hit_distances = [2000]
    max_z_hit_distances = [3000]

    if load_detected_clusters:
        detected_clusters, detected_clusters_count, max_detected_clusters, opt_params =\
                pickle.load(open("../saved_pickles/detected_clusters_{}".format(bg_rate), "rb"))
    else:
        detected_clusters, detected_clusters_count, max_detected_clusters, opt_params = cluster_parameter_scan_parallel(
                                        sn_hit_list_per_event, 
                                        bg_hit_list_per_event, bg_rate, bg_length,
                                        max_cluster_times=max_cluster_times,
                                        max_hit_time_diffs=max_hit_time_diffs, 
                                        max_hit_distances=max_hit_distances,
                                        max_x_hit_distances=max_x_hit_distances,
                                        max_y_hit_distances=max_y_hit_distances,
                                        max_z_hit_distances=max_z_hit_distances,
                                        detector=detector, verbose=1, each_dim_distance=True)
    
    sn_event_detection_efficiency = max_detected_clusters/sn_sim_number

    if save:
        detected_clusters_data = {"detected_clusters": detected_clusters, "detected_clusters_count": detected_clusters_count, 
                                    "max_detected_clusters": max_detected_clusters, "opt_params": opt_params}
        pickle.dump(detected_clusters_data, open(
                    "../saved_pickles/detected_clusters_{}".format(bg_rate), "wb"))
    

    # ----------------------------------------- #
 
    # ---- From here, you can start with saved data! (kinda, you have to change a couple of things first) -------

    # Find minimum cluster multiplicity that respects the 1/month rate fake trigger, while optimizing for the burst time window
    time_profile_x, time_profile_y = load_time_profile()
    once_a_month_rate = 1/(60 * 60 * 24 * 30)
    sn_model = "LIVERMORE"
    
    _, burst_time_window, minimum_cluster_mult = calculate_optimal_burst_time_window(bg_rate, once_a_month_rate, time_profile_x,
                                     time_profile_y, sn_event_detection_efficiency, true_tpc_size=true_tpc_size, used_tpc_size=used_tpc_size)
    
    
    # Get the detected SN event number for several supernova "true" event numbers
    rn = np.arange(0, 3.5, 0.005)
    total_sn_numbers = 10**rn #np.array([1, 2, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300])
    detected_sn_numbers = event_number_per_time(time_profile_x, time_profile_y, total_sn_numbers, burst_time_window)


    trigger_efficiency = snb_trigger_efficiency(bg_rate * true_tpc_size/used_tpc_size, burst_time_window, minimum_cluster_mult, total_sn_numbers,
                                    time_profile_x, time_profile_y, sn_event_detection_efficiency, true_tpc_size=true_tpc_size, used_tpc_size=used_tpc_size)
    
    distances = event_number_to_distance(total_sn_numbers, model=sn_model, tpc_size=true_tpc_size)
    
    print("-------------- DONE -------------------")
    print("SN detection efficiency:", sn_event_detection_efficiency)
    print("OPTIMAL CLUSTERING PARAMETERS:", opt_params)
    print("OPTIMAL BTR", burst_time_window)
    print("TRIGGER EFF:", trigger_efficiency[:-10], "MCM:", minimum_cluster_mult)

    if save:
        sim_parameters = {"max_cluster_times": max_cluster_times, "max_hit_time_diffs": max_hit_time_diffs,
                            "max_hit_distances": max_hit_distances, "bg_rate":bg_rate, "sn_model":sn_model, 
                            "sn_sim_number": sn_sim_number}
        pickle.dump(sim_parameters, open("../saved_pickles/sim_parameters_{}".format(bg_rate), "wb"))

        sim_results = {"sn_event_eff":sn_event_detection_efficiency, "sn_trigger_eff":trigger_efficiency,
                        "minimum_cluster_mult": minimum_cluster_mult, "total_sn_numbers":total_sn_numbers, 
                        "detected_sn_numbers": detected_sn_numbers, "distances": distances,
                        "burst_time_window": burst_time_window, "opt_params": opt_params}

        pickle.dump(sim_results, open("../saved_pickles/sim_results_{}".format(bg_rate), "wb"))
    
    # plt.figure(1)
    # plt.scatter(total_sn_numbers, trigger_efficiency)
    # plt.xscale('log')

    # plt.figure(2)
    # plt.scatter(distances, trigger_efficiency)
    # plt.xscale('log')
    # plt.show()
    
    
    #plotters.plot_sn_event_efficiencies(detected_clusters_count, sn_event_number=sn_sim_number)
    
    return trigger_efficiency, detected_clusters_count

    



if __name__ == '__main__':

    print("START ---> ")

    #main()
    print(stat_main())
    exit()

    time_profile_x, time_profile_y = load_time_profile()
    print(time_profile_x[-1])
    print(event_number_per_time(time_profile_x, time_profile_y, total_event_number=10, burst_time_window=10e6))
    plt.scatter(time_profile_x, time_profile_y)
    plt.show()


    sn_total_hits, sn_hit_list_per_event, hit_num_per_channel = load_hit_data(file_name="psn_g4_digi_reco_hist.root")
    bg_total_hits, bg_hit_list_per_event, hit_num_per_channel = load_all_backgrounds(limit=10)

    #cluster_parameter_scan_parallel(sn_hit_list_per_event, bg_hit_list_per_event, 1, 2000, [0.1, 0.2], [0.05], [250], 0)

    #bg_total_hits, bg_hit_list_per_event, hit_num_per_channel = load_hit_data(file_name="pbg_g4_digi_condor_1647520565_reco_hist.root")# "background/pbg_g4_digi_condor_reco_hist.root")
    
    # bg_total_hits = pickle.load(open("saved_pickles/fake_bg", "rb"))
    # bg_hit_list_per_event = [bg_total_hits]

    plt.hist(bg_total_hits[:, 2], bins=168)
    plt.title("BG hits received per Optical Channel")
    plt.show()

    print("BG Shape", bg_total_hits.shape)

    #display_hits(bg_total_hits, three_d=True)
    # bg_lenght = 1000 # in mus
    # detected_clusters, dc_count = cluster_parameter_scan(sn_hit_list_per_event, bg_hit_list_per_event, 1, bg_lenght,
    #                                     max_cluster_times=[63e-3, 125e-3, 0.25, 0.5, 1, 2],
    #                                     max_hit_time_diffs=[50e-3, 0.1, 0.2, 0.4], 
    #                                     max_hit_distances=[150, 250, 350, 450], verbose=0)
    
    #print(dc_count)
    
    _, mults, clusters = hit_multiplicity_scan(bg_hit_list_per_event, threshold=1, min_mult=5, max_mult=50, max_cluster_time=1, 
                            max_hit_time_diff=0.5, max_hit_distance=450, verbose=1, spatial_filter=False)




    


