'''
clustering.py

This module contains the functions that implement the clustering algorithm.
'''

import numpy as np

def time_clustering(hits, max_cluster_time, max_hit_time_diff, min_hit_mult, verbose=0):
    hit_num = len(hits)
    if verbose > 1:
        print("Number of hits:", hit_num)

    pre_candidate_clusters = []

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
        subcluster_groups = []
        subcluster = []
        prev_time = 1e10
        for k in range(len(current_cluster)):
            ktime = current_cluster[k, 3]
            khit = current_cluster[k, :]

            if ktime - prev_time <= max_hit_time_diff:
                #print(time-prev_time)
                subcluster.append(khit)
            else:
                subcluster_groups.append(subcluster)
                subcluster = [khit]
            
            if k == len(current_cluster) - 1:
                subcluster_groups.append(subcluster)
            
            prev_time = ktime
        
        pre_candidate_clusters.extend(subcluster_groups)

    # And now we get the length of each cluster
    candidate_clusters = []
    candidate_hit_multiplicities = []
    for cluster in pre_candidate_clusters:
        if len(cluster) < min_hit_mult:
            if verbose > 0:
                print("Cluster too small after max_time_dif filter, size:", len(cluster))
        else:
            candidate_clusters.append(cluster)
            candidate_hit_multiplicities.append(len(cluster))
    
    
    return candidate_clusters, candidate_hit_multiplicities



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


# Do a continuous clustering respecting the time distribution
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


def cluster_comparison(sn_clusters, bg_clusters):#, sn_hit_multiplicities, bg_hit_multiplicites):
    # Average time between hits (get rid of the huge amount)
    #print("CLUSTER FEATURE EXTRACTION")

    # Remove empty clusters
    #sn_clusters = [cluster for cluster in sn_clusters if cluster is not None]
   
    hit_multiplicity = [[],[]]
    average_pe = [[],[]]
    average_width = [[],[]]
    average_amplitude = [[],[]]

    max_amplitude = [[],[]]
    max_width = [[],[]]
    max_pe = [[],[]]

    total_pe = [[],[]]
    total_width = [[],[]]

    average_time_diffs = [[],[]]
    max_time_diff = [[],[]]
    max_time_extention = [[],[]]

    average_x_diffs = [[],[]] # between consecutive (in this axis) hits
    average_y_diffs = [[],[]]
    average_z_diffs = [[],[]]
    max_x_diff = [[],[]]
    max_y_diff = [[],[]]
    max_z_diff = [[],[]]
    average_distance_diffs = [[],[]] # between consecutive (in time) hits
    max_x_extention = [[],[]]
    max_y_extention = [[],[]]
    max_z_extention = [[],[]]

    wall_hit_fraction = [[],[]]

    # THIS IS THE NUMBER OF FEATURES OF A HIT, NOT OF THE CLUSTER!!!
    number_of_features = 11 # bg_clusters[0].shape[1] 

    for i, cluster_list in enumerate([sn_clusters, bg_clusters]):

        for j, cluster in enumerate(cluster_list):

            if cluster is None or len(cluster[:, 0]) == 0:
                # Fill a dummy cluster with -1s (it will never be used for anything, but we need to keep track of the position of hitless events)
                cluster = np.zeros((10, number_of_features)) - 1

            hit_multiplicity[i].append(len(cluster[:,0]))

            # ------------------

            times = cluster[:,3]
            time_diffs = np.diff(times)
            average_time = np.sum(time_diffs)/len(times)
            
            average_time_diffs[i].append(average_time)
            max_time_diff[i].append(np.max(time_diffs))
            max_time_extention[i].append(times[-1] - times[0])

            # -----------------

            x, y, z = cluster[:,4], cluster[:,5], cluster[:,6]
            averaged2 = np.sum(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)/len(x)
            average_distance_diffs[i].append(averaged2)

            max_x_extention[i].append(np.max(x)-np.min(x))
            max_y_extention[i].append(np.max(y)-np.min(y))
            max_z_extention[i].append(np.max(z)-np.min(z))

            # Order the x, y, z arrays to compute the average diffs
            ordered_x = np.sort(x)
            average_x = np.sum(np.diff(ordered_x))/len(ordered_x)
            average_x_diffs[i].append(average_x)
            ordered_y = np.sort(y)
            average_y = np.sum(np.diff(ordered_y))/len(ordered_y)
            average_y_diffs[i].append(average_y)
            ordered_z = np.sort(z)
            average_z = np.sum(np.diff(ordered_z))/len(ordered_z)
            average_z_diffs[i].append(average_z)

            max_x_diff[i].append(np.max(np.diff(ordered_x)))
            max_y_diff[i].append(np.max(np.diff(ordered_y)))
            max_z_diff[i].append(np.max(np.diff(ordered_z)))

            # ------------------

            average_pe[i].append(np.average(cluster[:,10]))
            average_width[i].append(np.average(cluster[:,7]))
            average_amplitude[i].append(np.average(cluster[:,9]))
            max_pe[i].append(np.max(cluster[:,10]))
            max_width[i].append(np.max(cluster[:,7]))
            max_amplitude[i].append(np.max(cluster[:,9]))

            total_pe[i].append(np.sum(cluster[:,10]))
            total_width[i].append(np.sum(cluster[:,7]))

            # Get the fraction of hits on walls
            # Count number of x values larger than 5
            wall_hits = len(cluster[cluster[:,4] > -325, 0])
            wall_hit_fraction[i].append(wall_hits / len(cluster[:,0]))

            # TODO: add max PE and max amplitude as cluster features
            # TODO: remove redundant area features


    sn_average_time_diff, bg_average_time_diff = average_time_diffs
    sn_average_distance_diff, bg_average_distance_diff = average_distance_diffs
    sn_max_time_diff, bg_max_time_diff = max_time_diff
    sn_max_x_extention, bg_max_x_extention = max_x_extention
    
    # plt.figure(1)
    # plt.hist(sn_average_time_diff, bins=15, density=True, alpha=0.5)
    # plt.hist(bg_average_time_diff, bins=15, density=True, alpha=0.5)

    # plt.figure(2)
    # plt.hist(sn_average_distance_diff, bins=20, density=True, alpha=0.5)
    # plt.hist(bg_average_distance_diff, bins=40, density=True, alpha=0.5)

    # plt.figure(3)
    # plt.hist(sn_max_time_diff, bins=15, density=True, alpha=0.5)
    # plt.hist(bg_max_time_diff, bins=15, density=True, alpha=0.5)

    # plt.figure(4)
    # plt.hist(sn_max_x_extention, bins=15, density=True, alpha=0.5)
    # plt.hist(bg_max_x_extention, bins=15, density=True, alpha=0.5)
    # plt.show()

    sn_max_y_extention, bg_max_y_extention = max_y_extention
    sn_max_z_extention, bg_max_z_extention = max_z_extention

    sn_max_time_extention, bg_max_time_extention = max_time_extention

    sn_average_x_diff, bg_average_x_diff = average_x_diffs
    sn_average_y_diff, bg_average_y_diff = average_y_diffs
    sn_average_z_diff, bg_average_z_diff = average_z_diffs

    sn_max_x_diff, bg_max_x_diff = max_x_diff
    sn_max_y_diff, bg_max_y_diff = max_y_diff
    sn_max_z_diff, bg_max_z_diff = max_z_diff

    sn_hit_multiplicity, bg_hit_multiplicity = hit_multiplicity
    sn_average_pe, bg_average_pe = average_pe

    

    # Save the features for the classifier
    # Array of size (n_samples, n_features)
    features_sn = np.vstack((sn_average_time_diff, sn_average_distance_diff,
                                sn_max_time_extention, 
                                sn_max_time_diff, sn_max_x_extention,
                                sn_max_y_extention, sn_max_z_extention,
                                sn_average_x_diff, sn_average_y_diff,
                                sn_average_z_diff, sn_max_x_diff, sn_max_y_diff,
                                sn_max_z_diff, sn_average_pe,
                                average_width[0], average_amplitude[0], max_pe[0], max_width[0], max_amplitude[0],
                                total_pe[0], total_width[0], wall_hit_fraction[0], sn_hit_multiplicity)).T
    targets_sn = np.ones((features_sn.shape[0], ))

    features_bg = np.vstack((bg_average_time_diff, bg_average_distance_diff,
                                bg_max_time_extention, 
                                bg_max_time_diff, bg_max_x_extention, 
                                bg_max_y_extention, bg_max_z_extention,
                                bg_average_x_diff, bg_average_y_diff,
                                bg_average_z_diff, bg_max_x_diff, bg_max_y_diff,
                                bg_max_z_diff, bg_average_pe,
                                average_width[1], average_amplitude[1], max_pe[1], max_width[1], max_amplitude[1],
                                total_pe[1], total_width[1], wall_hit_fraction[1], bg_hit_multiplicity)).T

    targets_bg = np.zeros((features_bg.shape[0], ))

    features = np.concatenate((features_sn, features_bg))
    targets = np.concatenate((targets_sn, targets_bg))
    
    #pickle.dump([features, targets], open("../saved_pickles/classifier/points", "wb"))


    return features, targets


def full_clustering(sn_hit_list_per_event, sn_info_per_event, bg_hit_list_per_event, max_cluster_time, max_hit_time_diff, 
                                                    max_hit_distance, max_x_hit_distance, max_y_hit_distance, max_z_hit_distance,
                                                    min_hit_mult, detector, verbose):
    bg_clusters = []
    bg_hit_multiplicities = []

    # Find clusters in BG
    for i, hit_list in enumerate(bg_hit_list_per_event):
        if hit_list.size == 0:
            continue 
        
        bg_clusters_i, bg_hit_multiplicities_i = clustering_continuous(hit_list, max_cluster_time=max_cluster_time, max_hit_time_diff=max_hit_time_diff, 
                                                max_hit_distance=max_hit_distance, max_x_hit_distance=max_x_hit_distance, max_y_hit_distance=max_y_hit_distance, 
                                                max_z_hit_distance=max_z_hit_distance, min_hit_mult=min_hit_mult, detector=detector, verbose=verbose)

        bg_clusters.extend(bg_clusters_i)
        bg_hit_multiplicities.extend(bg_hit_multiplicities_i)
    

    # Find SN clusters for current parameters
    sn_clusters = []
    sn_hit_multiplicities = []
    sn_energies = []

    for i, hit_list in enumerate(sn_hit_list_per_event):
        if hit_list.size == 0:
            continue 

        sn_clusters_i, sn_hit_multiplicities_i = clustering_continuous(hit_list, max_cluster_time=max_cluster_time, max_hit_time_diff=max_hit_time_diff, 
                                                max_hit_distance=max_hit_distance, max_x_hit_distance=max_x_hit_distance, max_y_hit_distance=max_y_hit_distance, 
                                                max_z_hit_distance=max_z_hit_distance, min_hit_mult=min_hit_mult, detector=detector, verbose=verbose)

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

    return bg_clusters, bg_hit_multiplicities, sn_clusters, sn_hit_multiplicities, sn_energies