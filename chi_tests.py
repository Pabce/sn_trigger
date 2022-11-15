import numpy as np 
import matplotlib.pyplot as plt
import random

from stuff import *
import hitting

# Signals are a list of time-ordered hits
def chi_squared_test(signal, signal_limits, window, poisson_fit_param, poisson_fit_interval):
    # We compare this to the fitted background. Note that this can get more complex than a Poisson, 
    # we are just using that for now

    new_poisson_param = poisson_fit_param * window/poisson_fit_interval
    
    hit_interval_count = []

    # Per time interval?
    time_min, time_max = signal_limits
    for i in np.arange(time_min, time_max, window):
        #print(i)
        hits_in_int = signal[signal < i + window]
        hits_in_int = hits_in_int[hits_in_int > i]

        hit_interval_count.append(len(hits_in_int))
    
    bin_edges = np.arange(np.min(hit_interval_count), np.max(hit_interval_count) + 2) - 0.5
    #num, bins, _ = plt.hist(hit_interval_count, density=False, bins=bin_edges)
    num, bins = np.histogram(hit_interval_count, density=False, bins=bin_edges)
    
    x_plot = np.round(bins + 0.1)
    expected_num = stats.poisson.pmf(x_plot, new_poisson_param) * np.sum(num)

    chi_squared = stats.chisquare(num, expected_num[:-1])
    print(chi_squared)
    
    # plt.scatter(x_plot, expected_num, color='orange', zorder=10)
    # plt.yscale('log')
    # plt.show()
    return chi_squared
    



def build_supernova_event_times(sn_hit_list_per_event, time_profile_x, time_profile_y, total_event_number, burst_time_window):

    # First, find the detected event number
    detected_event_number = int(hitting.event_number_per_time(time_profile_x, time_profile_y, total_event_number, burst_time_window))

    # Then, draw this number of random times from the time profile 
    # (we account for times in between entries)
    i = 0
    sampled_times = []
    spacing = time_profile_x[1] - time_profile_x[0]
    while i < detected_event_number:
        sampled_time = random.choices(population=time_profile_x, weights=time_profile_y)

        sampled_time += np.random.rand() * spacing

        if sampled_time <= burst_time_window / 1000:  # CONVERT BTW TO MS!
            sampled_times.append(sampled_time * 1000) # But log sampled times in mus (yes, this sucks)
            i += 1

    # Finally, associate a random event to one of these times
    sampled_hits = []
    sampled_hit_list_per_event = []
    for j in range(detected_event_number):
        hit_list = np.random.choice(sn_hit_list_per_event)
        sampled_hit_list_per_event.append(hit_list)

        for k in range(hit_list.shape[0]):
            hit_list[k, 3] = sampled_times[j]
            sampled_hits.append(hit_list[k, :])

    # plt.hist(np.array(sampled_hits)[:,3], bins=100)
    # plt.yscale('log')
    #plt.show()
    return sampled_hit_list_per_event, np.array(sampled_hits), np.array(sampled_times)


if __name__ == "__main__":
    burst_time_window = int(1e6)
    poisson_window = 100
    op_channel = 40

    sn_sim_number = 10000
    sn_total_hits, sn_hit_list_per_event, hit_num_per_channel = hitting.load_hit_data(
                                                                file_name="psn-ll_g4_digi_reco_hist.root", event_num=sn_sim_number)

    time_profile_x, time_profile_y = hitting.load_time_profile()
    sp_hit_list_per_event, sampled_hits, sampled_times = build_supernova_event_times(sn_hit_list_per_event,
                                                                         time_profile_x, time_profile_y, 100, burst_time_window)

    bg_limit = 2
    bg_total_hits, bg_hit_list_per_event, hit_num_per_channel = hitting.load_all_backgrounds(limit=bg_limit)
    fit_parameters = pickle.load(open("saved_pickles/bg_poisson_fit_params", "rb"))
    
    time_stacked_hits = time_order_hits(time_stack_hits(bg_hit_list_per_event))

    bg_hits_1 = select_hits_for_channel(time_stacked_hits, op_channel)[:, 3]
    sn_hits_1 = select_hits_for_channel(time_order_hits(sampled_hits), op_channel)[:, 3]

    signal_plus_bg = np.concatenate([bg_hits_1, sn_hits_1], axis=0)

    print(sn_hits_1.shape)
    #fit_param = hits_poisson_bg(sn_hits_1, interval=1000, optical_channel=op_channel, to_plot=True)

    # chi_squared_test(signal_plus_bg, signal_limits=(0, burst_time_window), window=poisson_window, 
    #                 poisson_fit_param=fit_parameters[op_channel], poisson_fit_interval=1000)
    
    chi_sqs1 = []
    chi_sqs2 = []
    ps1 = []
    ps2 = []
    for i in range(168):
        print("OPCH", i)
        bg_hits_1 = select_hits_for_channel(time_stacked_hits, i)[:, 3]
        sn_hits_1 = select_hits_for_channel(time_order_hits(sampled_hits), i)[:, 3]
        signal_plus_bg = np.concatenate([bg_hits_1, sn_hits_1], axis=0)

        chi_sq1, p1 = chi_squared_test(bg_hits_1, signal_limits=(0, burst_time_window), window=poisson_window, 
                    poisson_fit_param=fit_parameters[i], poisson_fit_interval=1000)
        chi_sqs1.append(chi_sq1)
        ps1.append(p1)
        
        chi_sq2, p2 = chi_squared_test(signal_plus_bg, signal_limits=(0, burst_time_window), window=poisson_window, 
                    poisson_fit_param=fit_parameters[i], poisson_fit_interval=1000)
        chi_sqs2.append(chi_sq2)
        ps2.append(p2)

    chi_sqs1 = np.array(chi_sqs1)
    chi_sqs2 = np.array(chi_sqs2)
    ps1 = np.array(ps1)
    ps2 = np.array(ps2)
    #plt.plot(chi_sqs1)

    plt.figure(7)
    plt.plot(chi_sqs2 - chi_sqs1)

    mean_chi = np.mean(chi_sqs2 - chi_sqs1)
    plt.axhline(y=np.mean(chi_sqs1), color='blue')
    plt.axhline(y=np.mean(chi_sqs2), color='orange')

    print(np.mean(chi_sqs1), np.mean(chi_sqs2))

    plt.figure(2)
    plt.plot(np.array(ps1))
    plt.axhline(y=np.mean(np.array(ps1)))



    plt.show()
    