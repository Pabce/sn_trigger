import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.special import factorial, kolmogorov
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from numba import jit, vectorize, float64, int64
from copy import deepcopy
from parameters import *
import random


def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def negative_log_likelihood(params, data):
    return -np.sum(np.log(poisson(data, params[0])))


def chi_squared(observed, expected, dof=1):
    if dof == 0:
        dof = 1

    for i, ex in enumerate(expected):
        if ex == 0:
            expected[i] = 0.1
            if observed[i] == 0:
                observed[i] = 0.1

    chi_sq_statistic = np.sum((observed - expected)**2/expected)
    significance = stats.chisqprob(chi_sq_statistic, dof)

    return chi_sq_statistic, dof, significance



def kolmogorov_smirnov(observed, expected, ftr=3.8e-7):
    # Compute the cumulative probabilities
    if len(observed) == 0 or len(expected) == 0:
        return -1, 1

    cum_observed = np.cumsum(observed)
    cum_expected = np.cumsum(expected)
    
    n, m = cum_observed[-1], cum_expected[-1]
    cumprob_observed = cum_observed/n
    cumprob_expected = cum_expected/m

    # plt.plot(cumprob_observed)
    # plt.plot(cumprob_expected)
    # plt.show()

    # Test statistic
    ks = np.sqrt(n) * np.max(np.abs(cumprob_observed - cumprob_expected))

    # Significance
    alpha = ftr
    significance = 1 - kolmogorov(ks)
    
    # print(ks, "KS")
    # print(significance, "REJECTED??", alpha)

    return ks, significance



#@jit(nopython=True)
def poisson_likelihood(observed, expected):

    #log_likelihood = np.log(1 - stats.poisson.cdf(observed - 1, expected))
    log_likelihood = np.log(stats.poisson.pmf(observed, expected))

    #log_likelihood = numba_log_poisson_pdf(observed, expected)
    #log_likelihood = log_likelihood[np.isfinite(log_likelihood)]

    #print("LOGL", log_likelihood)
    sum_log_likelihood = np.sum(log_likelihood)

    return -sum_log_likelihood


def bunch_histogram(hist_to_bunch, limit=4, last_index=-1):

    hist = deepcopy(hist_to_bunch)

    if last_index == -1:
        last_index = 0

        for i, el in enumerate(hist):
            if el < limit:
                # Two possibilities: the sum of the last bins is >= limit, or it is not. 
                # In the latter case, we need to sum the previous bin too

                sum_one = np.sum(hist[i:])
                if sum_one > limit or i < 2:
                    hist[i] = sum_one
                    last_index = i
                else:
                    hist[i-1] = np.sum(hist[i-1:])
                    last_index = i - 1
                
                break
    else:
        hist[last_index] = np.sum(hist[last_index:])

    # if hist.size == 0:
    #     return hist_to_bunch, -1

    bunched = hist[0: last_index + 1]

    return bunched, last_index


def get_acceptable_likelihood(expected_bg_hist, fake_trigger_rate):
    # --- LIKELIHOOD ---
    # To compensate for low background statistics, we assume a conservative value of 2 where bg is found to be 0
    # (is this maybe not needed?)
    # expected_bg_hist[expected_bg_hist < 1] = 1

    likelihoods = []
    number_of_tests = int(1e7)
    
    rng = np.random.default_rng()
    fake_bg_hist = rng.poisson(expected_bg_hist, size=(number_of_tests, len(expected_bg_hist)))
    
    likelihoods = compute_likelihoods(fake_bg_hist, expected_bg_hist)

    nbins = 10000
    plt.hist(likelihoods, bins=nbins, density=True)
    distribution, bins = np.histogram(likelihoods, bins=nbins, density=True)
    diff = bins[1] - bins[0]
    
    for i in range(len(distribution), 0, -1):
        sum = np.sum(distribution[i:]) * diff # Do the integral (you are retarded)
        if sum > fake_trigger_rate:
            print(i, sum, np.sum(distribution[i + 1:]) * diff, bins[i])
            print(fake_trigger_rate)
            break

    plt.show()


def compute_likelihoods(fake_bg_hist, expected_bg_hist):
    length = fake_bg_hist.shape[0]
    likelihoods = np.zeros(length)

    for i in range(length):
        likelihoods[i] = poisson_likelihood(fake_bg_hist[i, :], expected_bg_hist)
    
    return likelihoods

@jit(nopython=True)
def numba_factorial(k):
    result = 1
    for i in range(1, k + 1):
        result *= i
    
    return result

@vectorize([float64(int64, float64)], nopython=True)
def numba_log_poisson_pdf(k, mu):

    return -mu + k * np.log(mu) - np.log(numba_factorial(k))


def pinched_spectrum(energies, average_e, alpha):
    spectrum = (energies/average_e)**alpha * np.exp(- (alpha + 1) * energies/average_e)
    norm = np.sum(spectrum * np.diff(energies)[0])
    return spectrum/norm


# Get indices for events that follow a particular energy distribution
def get_energy_indices(energies, energy_histo, sn_energies, size):
    # Energy histogram should be normalized to 1
    
    cdf = np.cumsum(energy_histo)
    cdf /= cdf[-1]

    rng = np.random.rand(size)
    cdf_indices = np.searchsorted(cdf, rng)


    event_indices = []
    for i, cdf_index in enumerate(cdf_indices):
        left_energy = energies[cdf_index - 1]
        right_energy = energies[cdf_index]

        # Get event indices with energy between left and right energy
        #indices = [i for i, e in enumerate(sn_energies) if e >= left_energy and e < right_energy]
        indices = np.where((sn_energies >= left_energy) & (sn_energies < right_energy))[0]


        # FIXME
        if len(indices) == 0:
            continue

        index = np.random.choice(indices)
        event_indices.append(index)

        #print(left_energy, right_energy, indices)

    return np.array(event_indices)


def event_number_per_time(time_profile_x, time_profile_y, total_event_number, burst_time_window):
    if burst_time_window == 0:
        return 0
    # We integrate the time profile in (0, time_window)
    # TIME PROFILE IS IN MS!
    spacing = time_profile_x[1] - time_profile_x[0]
    
    #total_integral = np.sum(time_profile_y) * spacing/1000 * 10 # From ms to s, times 9.9 seconds duration
    # The total integral is normalized to 1 so no need for ratios

    # We compute the ratio with the desired burst window and multiply by the total number of events
    # WE NEED TO CONVERT THE BURST TIME WINDOW TO MS!!!
    ms_burst_time_window = burst_time_window/1000
    b_index = 0
    interp = 0
    for i, t in enumerate(time_profile_x):
        if t > ms_burst_time_window:
            b_index = i
            interp = (ms_burst_time_window - time_profile_x[i - 1]) / spacing
            break

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


if __name__ == "__main__":
    a = np.array([100,30,20,21,18,10,3,2,1,0])
    b = np.array([101,31,22,21,17,11,4,2,2,1])
    ftr = 1/(60 * 60 * 24 * 30)

    kolmogorov_smirnov(a, b, ftr)
    print(chi_squared(a, b, dof=len(b)))
    #get_acceptable_likelihood(a, ftr)



