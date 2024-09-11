'''
aux.py

Bunch of auxiliary functions used in the rest of modules
'''



import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize, curve_fit
from scipy.special import factorial, kolmogorov
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from numba import jit, vectorize, float64, int64
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


def bunch_histogram(hist_to_bunch, hbins, limit=4, bunched_indices=None, back_to_front=True):
    # We can bunch "back to front". I believe in this way we better harness the high hit multiplicity of the SN events

    hist = deepcopy(hist_to_bunch)

    if not back_to_front:
        pass
    
    else:
        # New empty histogram
        new_hist = []
        new_bins = []

        if bunched_indices is None:
            bunched_indices = []
            place = len(hist) - 1

            while place >= 0:
                cumul = 0
                prov_bunched_indices = []

                for i in range(place, -1, -1):
                    cumul += hist[i]
                    prov_bunched_indices.append(i)

                    if cumul >= limit:
                        new_hist.append(cumul)
                        bunched_indices.append(prov_bunched_indices)
                        place = i - 1
                        cumul = 0
                        break
                    elif i == 0:
                        place = -1
                        new_hist.append(cumul)
                        bunched_indices.append(prov_bunched_indices)
        else:
            for ind_list in bunched_indices:
                new_hist.append(np.sum(hist[ind_list]))

    # Convert to np array and reverse
    bunched = np.array(new_hist)[::-1]

    # Same for bins, and append the last bin
    for ind_list in bunched_indices[::-1]:
        #print(ind_list)
        new_bins.append(hbins[ind_list[-1]])
    new_bins.append(hbins[-1])
    bunched_bins = np.array(new_bins)

    # print(hbins)
    # print(bunched_bins)

    return bunched, bunched_indices, bunched_bins


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
            indices = [0]

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



