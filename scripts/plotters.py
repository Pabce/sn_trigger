from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import hitting as hp
import pickle
from hitting import event_number_to_distance, distance_to_event_number, snb_trigger_efficiency

def plot_sn_event_efficiencies(detected_clusters_count, cts, hts, hds, opt_params, bg_rate, burst_time_window,
                                 sn_event_number=1000, plane="max_hit_distances", plane_val=450):
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.9)

    ct_opt, ht_opt, hd_opt, _, _, _, _ = opt_params
    to_show = None
    if plane == "max_hit_distances":
        index = hds.index(plane_val)
        to_show = detected_clusters_count[:, :, index]
        extent = [cts[0], cts[-1], hts[0], hts[-1]]
        x_label = "Max cluster duration (us)"
        y_label = "Max hit time difference (us)"
        title_complement = "Max hit dist. : {} cm".format(plane_val)
        center = ct_opt, ht_opt
    elif plane == "max_hit_time_diffs":
        index = hts.index(plane_val)
        to_show = detected_clusters_count[:, index, :]
        extent = [cts[0], cts[-1], hds[0], hds[-1]]
        x_label = "Max cluster duration (us)"
        y_label = "Max hit distance separation (cm)"
        title_complement = "Max hit time diff. : {} us".format(plane_val)
        center = ct_opt, hd_opt
    elif plane == "max_cluster_times":
        index = cts.index(plane_val)
        to_show = detected_clusters_count[index, :, :]
        extent = [hts[0], hts[-1], hds[0], hds[-1]]
        x_label = "Max hit time difference (us)"
        y_label = "Max hit distance separation (cm)"
        title_complement = "Max cluster duration. : {} us".format(plane_val)
        center = ht_opt, hd_opt

    mat = ax.imshow(to_show.T/sn_event_number, aspect="auto")

    # TODO: Change so it works for all "planes"
    ax.set_xticks(np.arange(0, len(cts)))
    ax.set_xticklabels(cts)
    ax.set_yticks(np.arange(0, len(hts)))
    ax.set_yticklabels(hts)#[::-1])
    fig.colorbar(mat)
    
    circle1 = plt.Circle(center, 0.01, color='r')
    #ax.add_patch(circle1)

    ax.text(0.5, 0.5, 'PRELIMINARY', transform=ax.transAxes,
        fontsize=30, color='gray', alpha=0.6,
        ha='center', va='center', rotation='30')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.title("BG Rate: {} Hz".format(bg_rate), size=10)
    plt.suptitle("SN neutrino event DE --- {}".format(title_complement))


def plot_trigger_efficiency(total_sn_numbers, detected_sn_numbers, trigger_efficiency):
    
    plt.figure(1)
    plt.plot(total_sn_numbers, trigger_efficiency)
    plt.xscale('log')

    plt.suptitle('SN trigger efficiency vs number of SN neutrino events')
    plt.title("BG Rate: {} Hz, BTW: {} s".format(bg_rate, burst_time_window/1e6), size=10)
    plt.xlabel('Event number')
    plt.ylabel('Trigger efficiency')

    plt.text(0.5, 0.5, 'PRELIMINARY',
        fontsize=20, color='gray', alpha=0.5)

    # ---------------------------

    plt.figure(2)
    tpc_size = 10
    distances_liv = event_number_to_distance(total_sn_numbers, model="LIVERMORE", tpc_size=tpc_size)
    distances_gar = event_number_to_distance(total_sn_numbers, model="GARCHING", tpc_size=tpc_size)
    distances_gkvm = event_number_to_distance(total_sn_numbers, model="GKVM", tpc_size=tpc_size)

    plt.plot(distances_liv, trigger_efficiency, label='Livermore')
    plt.plot(distances_gar, trigger_efficiency, label='Garching', linestyle='dashed')
    plt.plot(distances_gkvm, trigger_efficiency, label='GKVM', linestyle='dashed')

    plt.suptitle('SN trigger efficiency vs distance to SN')
    plt.title("BG Rate: {} Hz, BTW: {} s".format(bg_rate, burst_time_window/1e6), size=10)
    plt.xlabel('Distance (kpc)')
    plt.ylabel('Trigger efficiency')
    plt.xlim(7, 200)

    #plt.vlines([10, 20, 50], 0, 1, color='gray')
    plt.xscale('log')

    plt.text(45, 0.5, 'PRELIMINARY',
        fontsize=20, color='gray', alpha=0.5),
        #ha='left', va='top', rotation='0')

    plt.legend()




if __name__ == "__main__":
    bg_rate = 0.1
    detected_clusters_count = pickle.load(open("../saved_pickles/detected_clusters_{}".format(bg_rate), "rb"))["detected_clusters_count"]
    sim_parameters = pickle.load(open("../saved_pickles/sim_parameters_{}".format(bg_rate), "rb"))
    sim_results = pickle.load(open("../saved_pickles/sim_results_{}".format(bg_rate), "rb"))

    cts, hts, hds = sim_parameters["max_cluster_times"], sim_parameters["max_hit_time_diffs"], sim_parameters["max_hit_distances"]
    burst_time_window = sim_results["burst_time_window"]
    opt_parameters = sim_results["opt_params"]
    total_sn_numbers = sim_results["total_sn_numbers"]
    detected_sn_numbers = sim_results["detected_sn_numbers"]
    trigger_efficiency = sim_results["sn_trigger_eff"]
    distances = sim_results["distances"]


    print("----------------------")
    print("OPT", opt_parameters)
    print("SN DEff:", sim_results["sn_event_eff"])
    #print("Trigger eff:", trigger_efficiency)
    print("Min clust mult:", sim_results["minimum_cluster_mult"])
    print("BTW:", burst_time_window)
    #print(sim_results)

    for i in range(len(distances)):
        if np.abs(distances[i] - 50) < 1 or np.abs(distances[i] - 20) < 1 or np.abs(distances[i] - 10) < 1:
            print(distances[i], trigger_efficiency[i])


    print("-----------------")

    plot_sn_event_efficiencies(detected_clusters_count, cts, hts, hds, opt_parameters,
                                 bg_rate, burst_time_window, sn_event_number=10000, plane_val=230)
    plt.show()

    plot_trigger_efficiency(total_sn_numbers, detected_sn_numbers, trigger_efficiency)
    plt.show()
    exit()
    # ------------
    bg_rates = 0.1, 0.2, 0.5, 1

    plt.figure(10)
    for bg_rate in bg_rates:
        detected_clusters_count = pickle.load(open("../saved_pickles/detected_clusters_{}".format(bg_rate), "rb"))["detected_clusters_count"]
        sim_parameters = pickle.load(open("../saved_pickles/sim_parameters_{}".format(bg_rate), "rb"))
        sim_results = pickle.load(open("../saved_pickles/sim_results_{}".format(bg_rate), "rb"))

        cts, hts, hds = sim_parameters["max_cluster_times"], sim_parameters["max_hit_time_diffs"], sim_parameters["max_hit_distances"]
        burst_time_window = sim_results["burst_time_window"]
        opt_parameters = sim_results["opt_params"]
        total_sn_numbers = sim_results["total_sn_numbers"]
        detected_sn_numbers = sim_results["detected_sn_numbers"]
        trigger_efficiency = sim_results["sn_trigger_eff"]

        plt.plot(total_sn_numbers, trigger_efficiency, label="{} Hz".format(bg_rate))
        plt.xscale('log')

        print("-------------------------------")
        print("BG_RATE:", bg_rate)
        #print(sim_results)
        print("-------------------------------")


    #plt.vlines(x=[10, 20, 50])
    plt.suptitle('SN trigger efficiency vs number of SN neutrino events')
    #plt.title("BG Rate: {} Hz, BTW: {} s".format(bg_rate, burst_time_window/1e6), size=10)
    plt.xlabel('Event number')
    plt.ylabel('Trigger efficiency')

    plt.text(0.5, 0.5, 'PRELIMINARY',
    fontsize=20, color='gray', alpha=0.5)
    plt.legend()
    plt.show()
    