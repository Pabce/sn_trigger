'''
aux.py

Bunch of auxiliary functions used in the rest of modules
'''

import logging
from gui import console

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize, curve_fit
from scipy.special import factorial, kolmogorov
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#from numba import jit, vectorize, float64, int64
from copy import deepcopy
import random


def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def negative_log_likelihood(params, data):
    return -np.sum(np.log(poisson(data, params[0])))


def chi_squared(observed, expected, dof=1):
    # If we only have one entry (dof=0), we default to Poisson likelihood
    if dof == 0:
        return None, dof, 1 - stats.poisson.cdf(observed - 1, expected)[0]

    # for i, ex in enumerate(expected):
    #     if ex == 0:
    #         expected[i] = 0.1
    #         if observed[i] == 0:
    #             observed[i] = 0.1

    chi_sq_statistic = np.sum((observed - expected)**2/expected)
    significance = stats.chisqprob(chi_sq_statistic, dof)

    return chi_sq_statistic, dof, significance



def kolmogorov_smirnov(observed_rvs, expected, filter, hbins, fit=False):
    # The observed array is discrete (it is not a histogram, but a list of SN and fake BG hit multiplicities)
    # The expected array is also discrete, but we will fit it with a continuous function.
    # We do this in order to cover the low-statistics region of the spectrum.
    # This region will be well populated once DUNE is running, but for now we must rely on an approximation.

    # We do the fit on the unfiltered part of hist, but we will use the filtered hist to calculate the KS statistic.

    observed_rvs = observed_rvs[observed_rvs > 0.6]
    expected = expected[12:]
    hbins = hbins[12:]

    if fit:
        fcut = 0.0
        unfilt_expected = expected[filter >= fcut]
        popt, pcov = curve_fit(background_fit_function, hbins[:-1][filter >= fcut], unfilt_expected)

        true_expected = np.concatenate([expected[filter < fcut] * filter[filter < fcut], background_fit_function(hbins[:-1][filter >= fcut] * filter[filter >= fcut], *popt)])
        
        # # Plot the fit
        # obs_hist, _, _ = plt.hist(observed_rvs, bins=hbins, label="Observed", color="black", histtype="step", density=False)
        # plt.scatter(hbins[:-1], expected, label="Expected", color="blue", s=10)
        # plt.plot(hbins[:-1], background_fit_function(hbins[:-1], *popt), color="red")
        
        # plt.ylim(top=1.1*np.max(expected), bottom=0.01)
        # plt.xlim(left=0, right=1.1*np.max(hbins))
        # #plt.yscale("log")
        # print(popt)
        # print(pcov)

        #plt.plot(hbins[:-1], true_expected, color="green")
        #plt.show()
    else:
        true_expected = expected

        # Plot
        obs_hist, _= np.histogram(observed_rvs, bins=hbins, density=False)
        # plt.scatter(hbins[1:], expected, label="Expected", color="blue", s=10)
        # plt.yscale("log")
        
        #plt.show()

    # Compute the CDF of the expected distribution
    true_expected_cdf = np.concatenate((np.zeros(1), np.cumsum(true_expected))) # The first entry of the CDF must be 0! It corresponds to hbins[0]
    true_expected_cdf /= true_expected_cdf[-1] # Normalize the CDF
    
    obs_hist = np.concatenate((np.zeros(1), obs_hist))
    obs_cdf = np.cumsum(obs_hist) / np.sum(obs_hist)

    # plt.figure()
    # plt.plot(hbins, true_expected_cdf)
    # plt.plot(hbins, obs_cdf)
    # plt.show()

    # Compute the KS statistic
    ks = np.max(np.abs(true_expected_cdf - obs_cdf))

    # We need a callable for the CDF
    # bg_cdf = lambda x: background_cdf(x, hbins, true_expected_cdf)
    # ks2, pval2 = stats.kstest(observed_rvs, bg_cdf)

    pval = kolmogorov(ks * np.sqrt(len(observed_rvs)))
    # print(ks, pval)

    #print(ks, pval)
    #print(ks2, pval2)

    return ks, pval


def background_fit_function(x, a, sigma, mu):

    #poiss = stats.poisson.pmf(x, gamma)
    # Gaussian
    gauss = stats.norm.pdf(x, mu, sigma)
    exp = np.exp(-(x-mu)/sigma)

    return a * exp

def background_cdf(x, hbins, discrete_cdf):
    # Find the linear interpolation between the two closest bins
    interp = np.interp(x, hbins, discrete_cdf)

    interp[interp > 1] = 1
    interp[interp < 0] = 0

    return interp
    

    


#@jit(nopython=True)
def poisson_likelihood(observed, expected):

    #log_likelihood = np.log(1 - stats.poisson.cdf(observed - 1, expected))
    log_likelihood = np.log(stats.poisson.pmf(observed, expected))

    #log_likelihood = numba_log_poisson_pdf(observed, expected)
    #log_likelihood = log_likelihood[np.isfinite(log_likelihood)]

    #print("LOGL", log_likelihood)
    sum_log_likelihood = np.sum(log_likelihood)

    return -sum_log_likelihood


# Take a histogram and "bunch it" at the far end so all entries are above a certain limit
# This is a simplified version of the original algorithm, but works always 
def bunch_histogram(hist_to_bunch, hbins, limit=4, bunching_index=None):

    hist = deepcopy(hist_to_bunch)

    # New empty histogram
    new_hist = []
    new_bins = []

    if bunching_index is None:
        # Find first index where the histogram entry is smaller than the limit
        # This is the "bunching index"
        try:
            bunching_index = np.where(hist < limit)[0][0]
            #console.log(f"Bunching index: {bunching_index}")
            #console.log(hist)
            # console.log(len(hist)-1)
        # If no index is found, we don't bunch
        except IndexError:
            return hist, hbins, None

    if bunching_index > len(hist) - 1:
        raise ValueError(f"Bunching index {bunching_index} is out of bounds for the provided histogram of length {len(hist)}")
    
    # Is the bunching index the last entry?
    # In such case, we need to sum the last two entries
    if bunching_index == len(hist) - 1:
        new_entry = hist[-1] + hist[-2]
    else:
        # Sum all entries from this index to the end
        new_entry = np.sum(hist[bunching_index:])
    #console.log(f"New entry: {new_entry}")
    new_hist = np.concatenate((hist[:bunching_index], [new_entry]))
    new_bins = np.concatenate((hbins[:bunching_index], hbins[bunching_index : bunching_index + 2]))

    # Has this produced a histogram where all entries are above the limit?
    # If not, we bunch again!
    if new_hist[-1] < limit:
        return bunch_histogram(new_hist, new_bins, limit, bunching_index=None)

    return new_hist, new_bins, bunching_index


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

# @jit(nopython=True)
# def numba_factorial(k):
#     result = 1
#     for i in range(1, k + 1):
#         result *= i
    
#     return result

# @vectorize([float64(int64, float64)], nopython=True)
# def numba_log_poisson_pdf(k, mu):

#     return -mu + k * np.log(mu) - np.log(numba_factorial(k))


def pinched_spectrum(energies, average_e, alpha):
    spectrum = (energies/average_e)**alpha * np.exp(- (alpha + 1) * energies/average_e)
    norm = np.sum(spectrum * np.diff(energies)[0])
    return spectrum/norm

def sample_indices_by_energy_weighted(energy_histo, energy_bins, sn_energies, size, weights=None, energy_lower_limit=None):
    if size == 0:
        return np.array([])
    
    # if weights is not None:
    #     print(np.nanmax(weights), np.nanmin(weights), "WEIGHTS")
    
    # If we cut at a lower limit, remove all bins below that limit
    if energy_lower_limit is not None:
        energy_histo_to_sample = energy_histo.copy()[energy_bins[:-1] >= energy_lower_limit]
        energy_bins_to_sample = energy_bins.copy()[energy_bins >= energy_lower_limit]
    else:
        energy_histo_to_sample = energy_histo.copy()
        energy_bins_to_sample = energy_bins.copy()

    # 1. Give each event a weight proportional to the bin height
    spectrum_weights = np.zeros_like(sn_energies)
    for i, bin_height in enumerate(energy_histo_to_sample):
        e_mask = (sn_energies >= energy_bins_to_sample[i]) & (sn_energies < energy_bins_to_sample[i + 1])
        spectrum_weights[e_mask] = bin_height

    # 2. If we have some extra weights, apply them
    if weights is not None:
        full_weights = spectrum_weights * weights
    else:
        full_weights = spectrum_weights
    
    # Remove the nans (make them 0)
    #print(len(np.where(np.isnan(full_weights))[0]), "lasdhfjakshjfsa")
    full_weights[np.isnan(full_weights)] = 0
    
    # 3. Normalize the weights
    full_weights /= np.sum(full_weights)

    # print(weights)
    # print(spectrum_weights)
    # print(full_weights)

    # 4. Sample the indices according to the weights
    event_indices = np.random.choice(np.arange(len(sn_energies)), size=size, p=full_weights)

    return event_indices
    

# Get indices for events that follow a particular energy distribution
def sample_indices_by_energy(energy_histo, energy_bins, sn_energies, size, energy_lower_limit=None):
    if size == 0:
        return np.array([])
    # Energy histogram should be normalized to 1

    # If we cut at a lower limit, remove all bins below that limit
    if energy_lower_limit is not None:
        energy_histo_to_sample = energy_histo.copy()[energy_bins[:-1] >= energy_lower_limit]
        energy_bins_to_sample = energy_bins.copy()[energy_bins >= energy_lower_limit]
    else:
        energy_histo_to_sample = energy_histo.copy()
        energy_bins_to_sample = energy_bins.copy()

    cdf = np.cumsum(energy_histo_to_sample)
    cdf /= cdf[-1]
    # Append a 0 to the cdf at the left
    cdf = np.insert(cdf, 0, 0.)

    # print(len(energy_bins), len(energy_histo), "ENERGY BINS AND HISTO")
    # print(min(energy_bins), max(energy_bins), "MIN AND MAX ENERGY BINS")

    rng = np.random.rand(size)
    cdf_indices = np.searchsorted(cdf, rng)

    event_indices = []
    energies = []
    for i, cdf_index in enumerate(cdf_indices):
        left_energy = energy_bins_to_sample[cdf_index - 1]
        right_energy = energy_bins_to_sample[cdf_index]

        # Get event indices with energy between left and right energy
        #indices = [i for i, e in enumerate(sn_energies) if e >= left_energy and e < right_energy]
        indices = np.where((sn_energies >= left_energy) & (sn_energies < right_energy))[0]

        if len(indices) == 0:
            console.log(cdf[0])
            console.log(cdf_indices[i], rng[i])
            raise ValueError(f"No indices found for the given energy range. "
                             + f"Left: {left_energy}, Right: {right_energy}. "
                             + f"Try loading more SN events or making the energy histogram bins wider.")
            #indices = [0]

        index = np.random.choice(indices)
        event_indices.append(index)
        energies.append(sn_energies[index])
        #print(left_energy, right_energy, indices)

    event_indices = np.array(event_indices)
    energies = np.array(energies)
    #print(energies)

    return event_indices


def expected_event_number_in_time_window(time_profile_x, time_profile_y, total_event_number, burst_time_window):
    if burst_time_window == 0:
        return 0
    # We integrate the time profile in (0, time_window)
    # TIME PROFILE IS IN MS!
    spacing = np.diff(time_profile_x)
    
    #total_integral = np.sum(time_profile_y) * spacing/1000 * 10 # From ms to s, times 9.9 seconds duration
    # The total integral is normalized to 1 so no need for ratios

    # We compute the ratio with the desired burst window and multiply by the total number of events
    # WE NEED TO CONVERT THE BURST TIME WINDOW TO MS!!!
    ms_burst_time_window = burst_time_window/1000
    b_index = 0
    interp = 0
    for i, t in enumerate(time_profile_x):
        if t >= ms_burst_time_window:
            b_index = i
            interp = (ms_burst_time_window - time_profile_x[i - 1]) / spacing[i - 1]
            break

    # Do some interpolation
    int_1 = np.sum(time_profile_y[0: b_index - 1])
    int_2 = time_profile_y[b_index - 1]
    window_integral = int_1 + interp * int_2

    event_number = window_integral * total_event_number
    #print(event_number)
    #print(window_integral, total_integral, b_index, interp)

    return event_number


def event_number_to_distance(event_number, base_event_number_40kt, tpc_size=10):

    distance = np.sqrt(base_event_number_40kt * 10**2 * (tpc_size/40) * 1/event_number)

    return distance


def distance_to_event_number(distance, base_event_number_40kt, tpc_size=10):

    event_num = base_event_number_40kt * 10**2 * (tpc_size/40) * 1/distance**2

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



