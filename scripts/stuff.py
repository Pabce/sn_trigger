'''
stuff.py

Contains some secret functions that are none of your concern.
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import factorial
from scipy import stats
import pickle
import uproot


# Generate a 2d array with the distance between each pair of optical channels.
# Also for each individual coordinate
def generate_distance_grids(version="v5", detector="VD"):

    if version == "v4":
        uproot_coords = uproot.open("/Users/pbarham/OneDrive/workspace/cern/ruth/prod_snnue_pds/prodmarley_nue_dune10kt_vd_1x8x14_larger_lowADC_1665753396.179956833_g4_detsim_xe_reco_hist.root")["opflashana/OpDetCoords"]
    
        coords_dict = uproot_coords.arrays(['X_OpDet', 'Y_OpDet', 'Z_OpDet'], library="np")
        coords = [coords_dict[key][0:168] for key in coords_dict]
        coords = np.array(coords).T
    
    elif version == "v5":
        # Load text file into numpy array, separating by columns (delimiter any number of spaces) and skipping the first row
        if detector == "VD":
            file_rel = "../pdpos_vd1x8x14v5.dat"
        elif detector == "HD":
            file_rel = "../dunehd1x2x6PDPos.txt"

        coords = np.genfromtxt(file_rel, skip_header=1, skip_footer=2)
    
        # Split into x, y, z arrays
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]
    
        coords = np.array([x, y, z]).T

    print(coords)
    print(coords.shape)
    print(np.min(np.abs(coords[:,0])))

    print(len(coords[coords[:, 0] > -325, 0]))
    print(len(coords[coords[:, 0] < -325, 0]))
    print(len(coords[:, 0]))

    # for i in range(1, 480):
    #     if (coords[0, :] == coords[i, :]).all():
    #         print(i, "--------->>ZXz>XZ")
    #         #break

    # print(coords[0,0], coords[48, 0], "C")
    # print(coords[0,1], coords[48, 1], "C")
    # print(coords[0,2], coords[48, 2], "C")

    op_channel_number = len(coords)
    #print(op_channel_number)

    distance_array = np.zeros((op_channel_number, op_channel_number))
    x_distance_array = np.zeros((op_channel_number, op_channel_number))
    y_distance_array = np.zeros((op_channel_number, op_channel_number))
    z_distance_array = np.zeros((op_channel_number, op_channel_number))

    for i in range(op_channel_number):
        #print(i)
        for j in range(op_channel_number):
            if i == j:
                continue

            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[j] 

            distance_array[i, j] = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            x_distance_array[i, j] = np.abs(x1 - x2)
            y_distance_array[i, j] = np.abs(y1 - y2)
            z_distance_array[i, j] = np.abs(z1 - z2)
    
    #print(distance_array)

    print("./aux_pickles/op_distance_array_{}_{}".format(detector, version))
    pickle.dump(distance_array, open("./aux_pickles/op_distance_array_{}_{}".format(detector, version), "wb"))
    pickle.dump(x_distance_array, open("./aux_pickles/op_x_distance_array_{}_{}".format(detector, version), "wb"))
    pickle.dump(y_distance_array, open("./aux_pickles/op_y_distance_array_{}_{}".format(detector, version), "wb"))
    pickle.dump(z_distance_array, open("./aux_pickles/op_z_distance_array_{}_{}".format(detector, version), "wb"))

    #print(y_distance_array)

# TODO
def generate_coordinate_arrays():
    pass


def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def negative_log_likelihood(params, data):
    return -np.sum(np.log(poisson(data, params[0])))

def time_order_hits(hits, remove_outliers=False):
    time_sort = np.argsort(hits[:,3])
    ordered_hits = hits[time_sort, :]
    
    if remove_outliers:
        a, b = remove_outliers
        ordered_hits = ordered_hits[ordered_hits[:,3] < b, :]
        ordered_hits = ordered_hits[ordered_hits[:,3] > a, :]

    return ordered_hits


def time_stack_hits(hit_list_per_event, list_interval=1000):
    time_stacked_hits = []
    for i, hits in enumerate(hit_list_per_event):
        hits[:, 3] += list_interval * i
        time_stacked_hits.append(hits)
    
    time_stacked_hits = np.concatenate(time_stacked_hits, axis=0)
    return time_stacked_hits


def select_hits_for_channel(hits, optical_channel):
    hits = hits[hits[:, 2] == optical_channel, :]
    return hits


# Some kind of time correlation between two hit series
def time_correlation(hits_1, hits_2, lamb_1, lamb_2, poisson_interval=1000, time_window_length=1):
    # We get the hits of the desired optical channel

    time_series_1 = hits_1[:, 3]
    time_series_2 = hits_2[:, 3]

    time_min = np.min(time_series_1)
    time_max = np.max(time_series_2)
    time_windows = np.arange(-time_min, time_max + time_window_length, time_window_length) #

    hist_1, _ = np.histogram(time_series_1, bins=time_windows)
    hist_2, _ = np.histogram(time_series_2, bins=time_windows)
    # hist_1, _, _ = plt.hist(time_series_1, bins=time_windows)
    # hist_2, _, _ = plt.hist(time_series_2, bins=time_windows)

    # plt.figure(2)
    # plt.plot(hist_2 * hist_1)
    # plt.show()

    score = np.sum(hist_1 * hist_2)

    expected_score = lamb_1 * lamb_2 / poisson_interval**2 * time_window_length**2 * len(hist_1)

    print("Expected score: {}, Score: {}".format(expected_score, score))
    pearson = stats.pearsonr(hist_1, hist_2)
    print("Pearson:", stats.pearsonr(hist_1, hist_2))
    #print("Total hits 1:", len(time_series_1), "Total hits 2:", len(time_series_2))
    #print("Distance:", hp.OP_DISTANCE_ARRAY[opch_1, opch_2])


        
    return pearson[0]

    


# Fit a Poisson distribution to BG hits. This is probably insanely stupid 
# Maybe it is a better idea to do this on an individual bg source basis (maybe not!)
def hits_poisson_bg(hits, interval=10, optical_channel="yo k se", to_plot=True):
    
    hit_interval_count = []

    # Per time interval?
    time_min, time_max = int(np.min(hits[:, 3])), int(np.max(hits[:, 3]))
    for i in range(time_min, time_max, interval):
        #print(i)
        hits_in_int = hits[hits[:,3] < i + interval, :]
        hits_in_int = hits_in_int[hits_in_int[:,3] > i, :]

        hit_interval_count.append(len(hits_in_int[:, 3]))

    # Do fit
    fit_result = minimize(negative_log_likelihood, x0=np.ones(1), args=(hit_interval_count,), method='Powell')
    #print(fit_result.x)
    #print(fit_result)
    parameters = fit_result.x

    if to_plot:
        plt.figure(2)
        bin_edges = np.arange(0, np.max(hit_interval_count) + 1) - 0.5
        n, bins, _ = plt.hist(hit_interval_count, bins=bin_edges, density=True, label='Observed distribution')
        
        
        plt.figure(2)
        x_plot = np.round(bins + 0.1)
        plt.plot(x_plot, stats.poisson.pmf(x_plot, *parameters), marker='o', linestyle='', label='Fit result (Poisson)')
        plt.title("Optical channel {}, mu={}".format(optical_channel, fit_result.x))

        #plt.yscale('log')
        plt.legend()

        plt.show()
    
    return fit_result.x


def find_background_clusters():
    
    return



if __name__ == '__main__':
    print("SKAJHDLKSJHADLKA")

    generate_distance_grids(version="v5", detector="HD")
    exit()

    import save_n_load as sl
    sn_limit = 18
    
    sn_total_hits, sn_hit_list_per_event, sn_info_per_event, _ = sl.load_all_sn_events_chunky(limit=sn_limit, event_num=1000, detector="VD")
    
    total_len = 0
    total_wall_hits = 0
    wall_hit_fractions = []
    for hit_list in sn_hit_list_per_event:
        hit_list = np.array(hit_list)
        if hit_list.shape == (0,):
            continue
        

        #hit_list = hit_list[np.abs(hit_list[:, 5]) > 675, :]

        if len(hit_list) == 0:
            continue

        total_wall_hits += len(hit_list[hit_list[:,4] > -325, 0])
        total_len += len(hit_list)
        
        wall_hit_fractions.append(len(hit_list[hit_list[:,4] > -325, 0]) / len(hit_list))
    
    average_wall_hit_fraction = total_wall_hits / total_len

    print("Wall hit fraction (average)", average_wall_hit_fraction)
    print("Min:", np.min(wall_hit_fractions), "Max:", np.max(wall_hit_fractions))
