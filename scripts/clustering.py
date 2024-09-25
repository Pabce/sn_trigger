import logging
import copy
import multiprocessing as mp
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
# Suppress debug messages from matplotlib
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
from rich.console import Group

from gui import console
import gui
import plot_hits

class Clustering:

    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)
    

    def group_clustering(self, hit_list_per_event, max_cluster_time, max_hit_time_diff, max_hit_distance, 
                        max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, min_neighbours, 
                        min_hit_multiplicity, spatial_filter=True, data_type_str="Chipmonc", in_parallel=False):

        final_clusters = []
        final_clusters_per_event = []
        final_hit_multiplicities = []
        final_hit_multiplicities_per_event = []

        results = []

        status_fstring = f'[bold green] Clustering {data_type_str} events'
        with gui.live_progress(console=console, status_fstring=status_fstring) as (progress, live, group):
            
            if not in_parallel:
                task = progress.add_task(f'[cyan]Clustering {data_type_str} events', total=len(hit_list_per_event))
                for i, hit_list in enumerate(hit_list_per_event):
                    if hit_list.size == 0:
                        progress.update(task, advance=1)
                        continue
                    
                    result = self.full_clustering(hit_list, max_cluster_time, max_hit_time_diff, max_hit_distance,
                            max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, min_neighbours,
                            min_hit_multiplicity, spatial_filter=spatial_filter)
                    
                    results.append(result)
                    progress.update(task, advance=1)

            else:
                task = progress.add_task(f'[cyan]Clustering {data_type_str} events (in parallel)', total=len(hit_list_per_event))
                num_processes = mp.cpu_count()
                #console.log(f"Number of processes: {num_processes}")

                with mp.Pool(num_processes) as pool:
                    # Unfortunately, imap_unordered is incredibly slower than map. So we use map in chunks to retain 
                    # some progress bar updates (we loose a tiny bit of performance by chunking, but it's fine)
                    chunk_size = int(0.05 * len(hit_list_per_event))
                    chunk_size = chunk_size if chunk_size > num_processes else num_processes
                    chunks = [hit_list_per_event[i:i+chunk_size] for i in range(0, len(hit_list_per_event), chunk_size)]

                    for i, chunk in enumerate(chunks):
                        args = zip(chunk, repeat(max_cluster_time), repeat(max_hit_time_diff), repeat(max_hit_distance),
                                repeat(max_x_hit_distance), repeat(max_y_hit_distance), repeat(max_z_hit_distance), repeat(min_neighbours),
                                repeat(min_hit_multiplicity), repeat(spatial_filter))

                        results_chunk = pool.map(self.wrapped_full_clustering, args)
                        results.extend(results_chunk)

                        progress.update(task, advance=len(chunk))    
                

            for result in results:
                clusters_i, hit_multiplicities_i = result
                num_clusters = len(clusters_i)
                if num_clusters != 0:
                    final_clusters.extend(clusters_i)
                    final_hit_multiplicities.extend(hit_multiplicities_i)

                final_clusters_per_event.append(clusters_i)
                final_hit_multiplicities_per_event.append([-1])

            # Create a new group with just the progress bar
            group = Group(progress)
            live.update(group)
        


        return final_clusters, final_clusters_per_event, final_hit_multiplicities, final_hit_multiplicities_per_event
        


    # TODO: Add the algorithm to use in the config, and take the neccesary parameters as *kwargs
    # TODO: Maybe make static method?
    def full_clustering(self, hits, max_cluster_time, max_hit_time_diff, max_hit_distance,
                        max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, min_neighbours,
                        min_hit_multiplicity, spatial_filter=True):
        
        final_clusters = []
        final_hit_multiplicities = []

        time_candidate_clusters, time_candidate_hit_multiplicities = self.time_clustering(
            hits, max_cluster_time, max_hit_time_diff, min_hit_multiplicity)

        if spatial_filter:
            for cluster in time_candidate_clusters:
                space_cluster = self.algorithm_spatial_naive(
                    cluster, max_hit_distance, max_x_hit_distance, max_y_hit_distance, max_z_hit_distance, min_neighbours)

                if space_cluster.shape[0] < min_hit_multiplicity:
                    #self.log.debug(f"\tCluster too small after spatial correlation filter, size: {len(spacecluster)}")
                    continue
                
                # # Plot the cluster in 3D
                # plotter = plot_hits.HitPlotter(self.config)
                # plotter.plot_3d_clusters(cluster, space_cluster)
                # -----------------
                
                final_clusters.append(space_cluster)
                final_hit_multiplicities.append(space_cluster.shape[0])
            
        else:
            final_clusters = time_candidate_clusters
            final_hit_multiplicities = time_candidate_hit_multiplicities
        
        return final_clusters, final_hit_multiplicities

    def wrapped_full_clustering(self, arg_list):
        return self.full_clustering(*arg_list)

    
    def time_clustering(self, hits, max_cluster_time, max_hit_time_diff, min_hit_multiplicity):
        hit_num = len(hits)
        self.log.debug(f"Number of hits: {hit_num}")

        # TODO: Check that the hits are sorted by time and raise an error if not

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
            
            current_candidate_cluster = hits[i:max_i, :]

            if max_i - i < min_hit_multiplicity:
                #self.log.debug(f"\tCluster too small after max_cluster_time filter, size: {(max_i - i)}")
                i += 1
                continue
            else:
                i = max_i

            # Find which are within the allowed time
            subcluster_groups = []
            subcluster = []
            prev_time = 1e10
            for k in range(len(current_candidate_cluster)):
                ktime = current_candidate_cluster[k, 3]
                khit = current_candidate_cluster[k, :]

                if ktime - prev_time <= max_hit_time_diff:
                    #print(time-prev_time)
                    subcluster.append(khit)
                else:
                    subcluster = np.array(subcluster) # Make the subcluster a numpy array
                    subcluster_groups.append(subcluster)
                    subcluster = [khit]
                
                if k == len(current_candidate_cluster) - 1:
                    subcluster = np.array(subcluster) # Make the subcluster a numpy array
                    subcluster_groups.append(subcluster)
                
                prev_time = ktime
            
            pre_candidate_clusters.extend(subcluster_groups)

        # And now we filter each pre-candidate cluster by min_hit_multiplicity
        candidate_clusters = []
        candidate_hit_multiplicities = []
        for cluster in pre_candidate_clusters:
            if cluster.shape[0] < min_hit_multiplicity:
                #self.log.debug(f"\tCluster too small after max_hit_time_dif filter, size: {len(cluster)}")
                pass
            else:
                # TODO: Move this into gui.py
                # self.log.debug(f"\tTime candidate cluster, size: {len(cluster)}")
                # Start Generation Here
                # first_hit_time = cluster[0][3]
                # # absolute_times = [hit[3] for hit in cluster]
                # # console.print(absolute_times)
                # times = [hit[3] - first_hit_time for hit in cluster]
                # start_time = 0
                # end_time = times[-1]
                # total_time = end_time - start_time if end_time > start_time else 1  # Prevent division by zero
                
                # timeline_length = console.width - 20
                # timeline = [' '] * timeline_length
                
                # for t in times:
                #     pos = int((t - start_time) / total_time * (timeline_length - 1))
                #     timeline[pos] = '●'  # Using a filled circle for each hit
                
                # timeline_str = f"{start_time:.1f} |" + ''.join(timeline) + f"| {end_time:.4f}"
                # console.print(timeline_str)

                candidate_clusters.append(cluster)
                candidate_hit_multiplicities.append(cluster.shape[0])
        
        #self.log.debug(type(candidate_clusters[0]))

        return candidate_clusters, candidate_hit_multiplicities
    
    
    # TODO: Add max hit distance for x, y and z too
    def algorithm_spatial_neighbour_grouping(self, cluster, max_hit_distance):
        """
        Take a time-candidate cluster and group the hits into subclusters based on the maximum hit distance.
        """
        # Config reads ----
        op_distance_array = self.config.loaded["op_distance_array"]
        op_x_distance_array = self.config.loaded["op_x_distance_array"]
        op_y_distance_array = self.config.loaded["op_y_distance_array"]
        op_z_distance_array = self.config.loaded["op_z_distance_array"]
        # -----------------

        candidate_clusters = []
               
        opchannels = cluster[:, 2].astype(int)

        unsearched_indices = list(np.arange(cluster.shape[0]))
        console.log(f"Length of cluster: {len(unsearched_indices)}")
        neighbour_index_groups = []
    
        while len(unsearched_indices) > 0:
            # 1. Set a random hit as the starting point
            starting_idx = unsearched_indices[0]

            #console.log(f"Starting index: {starting_idx}")

            previous_neighbour_indices = [starting_idx]
            neighbour_indices = [starting_idx]
            new_neighbour_indices = [starting_idx]
            unsearched_indices.remove(starting_idx)

            # 2. Find all the neighbours of the starting hit (i.e. the hits within the maximum hit distance)
            # Repeat until no new neighbours are found
            while True:
                temp_new_neighbour_indices = []
                for i in new_neighbour_indices:
                    for j in unsearched_indices:
                        if op_distance_array[opchannels[i], opchannels[j]] < max_hit_distance:
                            temp_new_neighbour_indices.append(j)
                            unsearched_indices.remove(j)

                neighbour_indices.extend(temp_new_neighbour_indices)
                new_neighbour_indices = copy.deepcopy(temp_new_neighbour_indices)
                # unsearched_indices = [i for i in unsearched_indices if i not in neighbour_indices]

                # console.log(f"\tTemp new neighbour indices: {temp_new_neighbour_indices}")
                # console.log(f"\tNeighbour indices: {neighbour_indices}")
                # console.log(f"\tPrevious neighbour indices: {previous_neighbour_indices}")
                # console.log(f"\tUnsearched indices: {unsearched_indices}")

                if len(neighbour_indices) == len(previous_neighbour_indices):
                    # console.log(f"\tNo new neighbours found, breaking")
                    # console.log(f"\tNeighbour indices: {neighbour_indices}, length:{len(neighbour_indices)}")
                    break
                
                previous_neighbour_indices = copy.deepcopy(neighbour_indices)

            # if len(neighbour_indices) >= min_hit_multiplicity:
            #     neighbour_index_groups.append(neighbour_indices)
            neighbour_index_groups.append(neighbour_indices)
        
        console.log(neighbour_index_groups)
        console.log(f"Length of neighbour index groups: {len(neighbour_index_groups)}")

        # Build the candidate clusters from the neighbour index groups
        for group in neighbour_index_groups:
            candidate_clusters.append(cluster[group, :])

        # Plot the clusters in 3D
        # plotter = plot_hits.HitPlotter(self.config)
        # plotter.plot_3d_clusters(cluster, candidate_clusters)
        
        return candidate_clusters
       

    def algorithm_spatial_naive(self, cluster, max_hit_distance, 
                                    max_x_hit_distance=1e5, max_y_hit_distance=1e5, max_z_hit_distance=1e5,
                                    min_neighbours=1):
        """
        Check if a time-candidate cluster satisfies the spatial correlation requirements.
        My old self thought that this algorithm was stupid, but I'm not sure why... 
        """
        # Config reads ----
        op_distance_array = self.config.loaded["op_distance_array"]
        op_x_distance_array = self.config.loaded["op_x_distance_array"]
        op_y_distance_array = self.config.loaded["op_y_distance_array"]
        op_z_distance_array = self.config.loaded["op_z_distance_array"]
        # -----------------

        candidate_cluster = []

        # TODO: Make the opchannels an int by default 
        # --> This seems problematic as they're part of a larger np array which is float64
        cluster_opchannels = cluster[:, 2].astype(int)
        
        # Get the pairwise distances for all hits in the cluster
        pairwise_distances = op_distance_array[cluster_opchannels[:, None], cluster_opchannels]
        pairwise_x_distances = op_x_distance_array[cluster_opchannels[:, None], cluster_opchannels]
        pairwise_y_distances = op_y_distance_array[cluster_opchannels[:, None], cluster_opchannels]
        pairwise_z_distances = op_z_distance_array[cluster_opchannels[:, None], cluster_opchannels]

        # Set the diagonal to a large number so that it's ignored by the minimum function
        np.fill_diagonal(pairwise_distances, 1e10)
        np.fill_diagonal(pairwise_x_distances, 1e10)
        np.fill_diagonal(pairwise_y_distances, 1e10)
        np.fill_diagonal(pairwise_z_distances, 1e10)

        # For a hit to be included in the cluster, the n=min_neighbours lowest distances must be less
        # than the maximum allowed distance
        # TODO: include a new parameter: minimum number of neighbours per hit

        # Find the lowest n values for each row
        min_distances = np.sort(pairwise_distances, axis=1)[:, :min_neighbours]
        min_x_distances = np.sort(pairwise_x_distances, axis=1)[:, :min_neighbours]
        min_y_distances = np.sort(pairwise_y_distances, axis=1)[:, :min_neighbours]
        min_z_distances = np.sort(pairwise_z_distances, axis=1)[:, :min_neighbours]

        for i in range(cluster.shape[0]):
            valid_hit = True
            if np.any(min_distances[i, :] > max_hit_distance):
                valid_hit = False
            if np.any(min_x_distances[i, :] > max_x_hit_distance):
                valid_hit = False
            if np.any(min_y_distances[i, :] > max_y_hit_distance):
                valid_hit = False
            if np.any(min_z_distances[i, :] > max_z_hit_distance):
                valid_hit = False
            
            if valid_hit:
                candidate_cluster.append(cluster[i])

        # Make the clusters a numpy array
        candidate_cluster = np.array(candidate_cluster)

        # Plot the clusters in 3D
        # plotter = plot_hits.HitPlotter(self.config)
        # plotter.plot_3d_clusters(cluster, candidate_cluster, plot_complement_cluster=True)

        return candidate_cluster


def compute_cluster_features(cluster, detector_type="VD", features_to_return="all"):
    # Number of features each hit in the cluster has
    # This is hardcoded and unlikely to change...
    number_of_hit_features = 11

    if cluster is None or len(cluster) == 0:
        logging.warning(f"Cluster is not valid: {cluster}, returning None")
        return None
    
    # Cluster of hit features
    features = {}

    features['hit_multiplicity'] = len(cluster[:, 0])

    # Time-related features
    times = cluster[:, 3]
    if len(times) > 1:
        time_diffs = np.diff(times)
        features['average_time_diff'] = np.mean(time_diffs)
        features['max_time_diff'] = np.max(time_diffs)
        features['max_time_extension'] = times[-1] - times[0]
    else:
        features['average_time_diff'] = 0
        features['max_time_diff'] = 0
        features['max_time_extension'] = 0

    # Spatial-related features
    x, y, z = cluster[:, 4], cluster[:, 5], cluster[:, 6]
    if len(x) > 1:
        diff_x = np.diff(x)
        diff_y = np.diff(y)
        diff_z = np.diff(z)
        features['average_distance_diff'] = np.mean(diff_x**2 + diff_y**2 + diff_z**2)

        features['max_x_extension'] = np.max(x) - np.min(x)
        features['max_y_extension'] = np.max(y) - np.min(y)
        features['max_z_extension'] = np.max(z) - np.min(z)

        ordered_x = np.sort(x)
        ordered_y = np.sort(y)
        ordered_z = np.sort(z)

        diff_ordered_x = np.diff(ordered_x)
        diff_ordered_y = np.diff(ordered_y)
        diff_ordered_z = np.diff(ordered_z)

        features['average_x_diff'] = np.mean(diff_ordered_x)
        features['average_y_diff'] = np.mean(diff_ordered_y)
        features['average_z_diff'] = np.mean(diff_ordered_z)

        features['max_x_diff'] = np.max(diff_ordered_x)
        features['max_y_diff'] = np.max(diff_ordered_y)
        features['max_z_diff'] = np.max(diff_ordered_z)
    else:
        features.update({
            'average_distance_diff': 0,
            'max_x_extension': 0,
            'max_y_extension': 0,
            'max_z_extension': 0,
            'average_x_diff': 0,
            'average_y_diff': 0,
            'average_z_diff': 0,
            'max_x_diff': 0,
            'max_y_diff': 0,
            'max_z_diff': 0,
        })

    # Charge-related features
    features['average_pe'] = np.mean(cluster[:, 10])
    features['average_width'] = np.mean(cluster[:, 7])
    features['average_amplitude'] = np.mean(cluster[:, 9])

    features['max_pe'] = np.max(cluster[:, 10])
    features['max_width'] = np.max(cluster[:, 7])
    features['max_amplitude'] = np.max(cluster[:, 9])

    features['total_pe'] = np.sum(cluster[:, 10])
    features['total_width'] = np.sum(cluster[:, 7])

    # Detector-related features
    # Number of different opchannels in the cluster
    unique_opchannels, unique_counts = np.unique(cluster[:, 2], return_counts=True)
    features['num_active_opchannels'] = len(unique_opchannels)
    # Number of hits in the most populated opchannel
    features['num_hits_most_populated_opchannel'] = np.max(unique_counts)
    # Number of hits in the second most populated opchannel
    if len(unique_counts) > 1:
        features['num_hits_second_most_populated_opchannel'] = np.sort(unique_counts)[-2]
    else:
        features['num_hits_second_most_populated_opchannel'] = 0

    if detector_type == "VD":
        wall_hits = np.sum(cluster[:, 4] > -325)
        features['wall_hit_fraction'] = wall_hits / len(cluster)
    else:
        features['wall_hit_fraction'] = 0

    # TODO: It is a waste of time to compute features that are not returned.
    # However, feature computing is fast enough that it is not a big deal.
    if features_to_return != "all":
        features = {feature: features[feature] for feature in features_to_return}

    return features


def group_compute_cluster_features(clusters, detector_type="VD", data_type_str="Chipmonc", features_to_return="all"):
    features_list = []

    status_fstring = f"[bold green]Computing features for {data_type_str} clusters"
    with gui.live_progress(console=console, status_fstring=status_fstring) as (progress, live, group):  
        task = progress.add_task(f'[cyan]Computing features for {len(clusters)} {data_type_str} clusters', total=len(clusters))
        for cluster in clusters:
            features_list.append(compute_cluster_features(cluster, detector_type, features_to_return))
            progress.update(task, advance=1)
    
        # Create a new group with just the progress bar
        group = Group(progress)
        live.update(group)

    # Convert list of dicts to a numpy array
    feature_names = list(features_list[0].keys())
    features_array = np.array([[features[feature] for feature in feature_names] for features in features_list])

    return features_array, feature_names