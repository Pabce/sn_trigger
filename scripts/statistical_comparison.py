import logging

import numpy as np
from scipy import stats, optimize
from rich.progress import track
from numba import jit, njit, prange, float64, int32
import multiprocessing as mp
from functools import partial
from sys import exit

from gui import console


class StatisticalComparison:

    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

    # Chi squared p-value
    @staticmethod
    def p_value_from_chi_squared_statistic(chi_squared_statistic, dof):
        # Use survival function (sf) instead of 1 - cdf for better numerical precision
        p_value = stats.chi2.sf(chi_squared_statistic, dof)
        
        return p_value
    
    # Inverse of the above function (chi squared statistic from p-value)
    @staticmethod
    def chi_squared_statistic_from_p_value(p_value, dof):
        # Ensure p_value is between 0 and 1
        p_value = max(0, min(1, p_value))
        
        # Use ppf (percent point function) to get the chi squared statistic
        chi_squared_statistic = stats.chi2.isf(p_value, dof)
        
        return chi_squared_statistic
    
    @staticmethod
    def pearson_chi_squared_statistic(observed, expected, **kwargs):
        # Compute chi-squared statistic
        chi_sq = np.sum((observed - expected)**2 / expected, axis=0)
        return chi_sq, None
    
    # Poissonian C statistic
    @staticmethod
    def cash_statistic(observed, expected, **kwargs):
        with np.errstate(divide='ignore', invalid='ignore'):
            c_array = 2 * np.where(observed > 0, expected - observed + observed * np.log(observed / expected), expected)
        
        c = np.sum(c_array, axis=0)
        return c, None

    # Log likelihood ratio statistic, with a strength parameter for the signal
    @staticmethod
    def log_likelihood_ratio_statistic_with_strength(observed, expected, **kwargs):
        signal_shape = kwargs.get("signal_shape", None)
        if signal_shape is None:
            raise ValueError("Signal shape must be provided for log likelihhod ratio statistic.")

        # 1. Evaluate the log likelihood for the null hypothesis (signal strength = 0)
        log_likelihood_h0 = poisson_log_likelihood(0.0, 1.0, observed, expected, signal_shape)
        #print(log_likelihood_h0)

        # 2. Find the maximum log likelihood for the alternative hypothesis (signal strength > 0)
        # We will maximise the log likelihood over the signal strength parameter
        # (300 is a reasonable upper limit for the signal strength)
        signal_strengths = np.arange(0, 300, 1)

        log_likelihoods_h1 = poisson_log_likelihood(signal_strengths, 1.0, observed, expected, signal_shape)
        optimal_log_likelihood_h1 = np.max(log_likelihoods_h1, axis=-1)
        optimal_signal_strength = signal_strengths[np.argmax(log_likelihoods_h1, axis=-1)]
        #print(f"Optimal log likelihood H1: {optimal_log_likelihood_h1.shape}")

        # 3. Compute the log likelihood ratio statistic
        llr = -2 * (log_likelihood_h0 - optimal_log_likelihood_h1)

        #print(np.argmax(log_likelihoods_h1))

        return llr, optimal_signal_strength
    

    def generate_statistic_distribution(self, expected, types="chi_squared", n_toys=1000, random_seed=None, chunk_size=1000, **kwargs):
        """
        Generate a distribution of the Cash statistic for a given expected distribution.
        """
        signal_shape = kwargs.get("signal_shape", None)

        # If type is a string, convert it to a list
        if isinstance(types, str):
            types = [types]

        expected = np.asarray(expected, dtype=np.float64).reshape(-1, 1)
        n_bins = len(expected)

        # We will generate the simulated data in chunks to avoid memory issues
        cash_statistics_chunks = []
        chisq_statistics_chunks = []
        log_likelihood_ratio_chunks = []
        llr_strengths_chunks = []

        if chunk_size > n_toys:
            chunk_sizes = [n_toys]
        else:
            chunk_sizes = [chunk_size] * (n_toys // chunk_size) + ([n_toys % chunk_size] if n_toys % chunk_size != 0 else [])

        for i, _ in track(enumerate(chunk_sizes), total=len(chunk_sizes), description="Generating simulated data"):

            # if i == len(chunk_sizes) - 1:
            #     tile = extra_tile

            #simulated_data = rng.poisson(expected, size=(n_bins, chunk_sizes[i]))
            simulated_data = jit_generate_poisson_data(expected, chunk_sizes[i], random_seed)
            #console.log(simulated_data.shape)
        
            if "chi_squared" in types:
                #statistics_chunks.append(StatisticalComparison.pearson_chi_squared_statistic(simulated_data, tile))
                chisq_statistics_chunks.append(jit_pearson_chi_squared_statistic(simulated_data, expected))
            if "cash" in types:
                #statistics_chunks.append(StatisticalComparison.cash_statistic(simulated_data, expected))
                cash_statistics_chunks.append(jit_cash_statistic(simulated_data, expected))
            if "log_likelihood_ratio" in types:
                llr, llr_strength = self.log_likelihood_ratio_statistic_with_strength(simulated_data, expected, signal_shape=signal_shape)
                log_likelihood_ratio_chunks.append(llr)
                llr_strengths_chunks.append(llr_strength)

        cash_statistics = None
        chisq_statistics = None
        log_likelihood_ratio_statistics = None
        llr_strengths = None
        if cash_statistics_chunks:
            cash_statistics = np.concatenate(cash_statistics_chunks, axis=0)
        if chisq_statistics_chunks:
            chisq_statistics = np.concatenate(chisq_statistics_chunks, axis=0)
        if log_likelihood_ratio_chunks:
            log_likelihood_ratio_statistics = np.concatenate(log_likelihood_ratio_chunks, axis=0)
            llr_strengths = np.concatenate(llr_strengths_chunks, axis=0)

        return cash_statistics, chisq_statistics, log_likelihood_ratio_statistics, llr_strengths


def fisher_method_statistic(observed, expected):
    return 0

def obj_function_sloped(x, gamma, observed, expected, sn_shape):
    mu, slope = x
    return -poisson_log_likelihood_sloped(mu, slope, gamma, observed, expected, sn_shape)

@njit(fastmath=True, cache=True)
def poisson_log_likelihood_sloped(mu, slope, gamma, observed, expected, sn_shape):
    # mu is the sn signal strength parameter
    # gamma is the background strength parameter

    #print(gamma, mu, "CHIRP")
    # Since we're maximising this over mu, gamma, we can ignore pure functions of the data
    mult = np.arange(0, observed.shape[0]) / (observed.shape[0] - 1) - 0.5
    sn_shape += slope * mult
    model = mu**1 * sn_shape + gamma * expected

    # log_term = np.where(observed > 0, observed * np.log(model), 0)
    # log_likelihood = np.sum(-model + log_term, axis=0)

    log_term = np.where(observed > 0, observed * np.log(observed / model), 0)
    log_likelihood = - np.sum(model - observed + log_term, axis=0)

    return log_likelihood
 
@njit(fastmath=True, cache=True)
def minus_poisson_log_likelihood(mu, gamma, observed, expected, sn_shape):
    # mu is the sn signal strength parameter
    # gamma is the background strength parameter

    #print(gamma, mu, "CHIRP")
    # Since we're maximising this over mu, gamma, we can ignore pure functions of the data
    model = mu**1 * sn_shape + gamma * expected

    # log_term = np.where(observed > 0, observed * np.log(model), 0)
    # log_likelihood = np.sum(-model + log_term, axis=0)

    log_term = np.where(observed > 0, observed * np.log(observed / model), 0)
    log_likelihood = - np.sum(model - observed + log_term, axis=0)

    return -log_likelihood

def poisson_log_likelihood(mu, gamma, observed, expected, sn_shape):
    # Useful reshaping if mu is a vector
    mu = np.atleast_1d(mu)
    observed, expected, sn_shape = np.atleast_2d(observed, expected, sn_shape)
    #print(observed.shape, expected.shape, sn_shape.shape, mu.shape)
    # mu is the sn signal strength parameter
    # gamma is the background strength parameter

    #print(gamma, mu, "CHIRP")
    # Since we're maximising this over mu, gamma, we can ignore pure functions of the data
    model = mu[None, None, :] * sn_shape[:, :, None] + gamma * expected[:, :, None]
    observed_3d = observed[:, :, None]

    # log_term = np.where(observed > 0, observed * np.log(model), 0)
    # log_likelihood = np.sum(-model + log_term, axis=0)

    log_term = np.where(observed_3d > 0, observed_3d * np.log(observed_3d / model), 0)
    log_likelihood = - np.sum(model - observed_3d + log_term, axis=1)

    #print(f"log likelihood: {log_likelihood.shape}, observed: {observed_3d.shape}, model: {model.shape}")

    return np.squeeze(log_likelihood)


def log_likelihood_ratio_statistic(observed, expected, sn_shape):
    expected_b = np.broadcast_to(expected, observed.shape).copy()
    sn_shape_b = np.broadcast_to(sn_shape, observed.shape).copy()

    #print(expected_b.shape, observed.shape, expected.shape)

    #print(observed.shape, observed.dtype, expected_b.shape, expected_b.dtype, sn_shape_b.shape, sn_shape_b.dtype)
    optimal_log_likelihood_h0 = -minus_poisson_log_likelihood(0.0, 1.0, observed, expected_b, sn_shape_b)
    #print(poisson_log_likelihood.inspect_types())

    llr_array = np.zeros(observed.shape[1])
    gamma_array = np.zeros(observed.shape[1])
    mu_array = np.zeros(observed.shape[1])
    slope_array = np.zeros(observed.shape[1])

    # console.log("Starting loop")
    # num_processes = mp.cpu_count()
    # with mp.Pool(num_processes) as pool:
    #     results = pool.map(partial(optimize.minimize_scalar, bounds=(0.0, 100), method='bounded'), 
    #                        [partial(minus_poisson_log_likelihood, gamma=1.0, observed=observed[:,i].reshape(-1, 1), expected=expected, sn_shape=sn_shape) for i in range(observed.shape[1])],
    #                        chunksize=10_000)
    #     mu_array = [float(res.x) for res in results]
    #     optimal_log_likelihood_h1 = -np.array([float(res.fun) for res in results])

    #     # results = pool.map(partial(optimize.minimize, x0=[0.0, 0.0], bounds=[(-100, 100), (0.0, 100)]), 
    #     #             [partial(obj_function_sloped, gamma=1.0, observed=observed[:,i].reshape(-1, 1), expected=expected, sn_shape=sn_shape) for i in range(observed.shape[1])],
    #     #             chunksize=10_000)
    #     # slope_array = np.array([res.x[0] for res in results])
    #     # mu_array = np.array([res.x[1] for res in results])
    #     # optimal_log_likelihood_h1 = -np.array([float(res.fun) for res in results])

    #     # print(mu_array)
    #     # print(slope_array, np.max(slope_array))
    #     #print(optimal_log_likelihood_h1)

    #     llr_array = -2 * (optimal_log_likelihood_h0 - optimal_log_likelihood_h1)
    
    # console.log("Loop finished")

    strengths = np.arange(0, 100, 1)
    for i in range(observed.shape[1]):
        if i % 50000 == 0:
            console.log(i)
        
        observed_i = observed[:,i].reshape(-1, 1)
        log_likelihoods_h1 = poisson_log_likelihood(strengths, 1.0, observed_i, expected, sn_shape)
        optimal_log_likelihood_h1 = np.max(log_likelihoods_h1)

        llr_array[i] = -2 * (optimal_log_likelihood_h0[i] - optimal_log_likelihood_h1)


    # for i in range(observed.shape[1]):
    #     if i % 5000 == 0:
    #         console.log(i)

    #     observed_i = observed[:,i].reshape(-1, 1)

    #     # Null hypothesis: mu = 0
    #     # f_0 = lambda gamma: -poisson_log_likelihood([gamma, 0], observed_i, expected, sn_shape)
    #     # res_0 = optimize.minimize(f_0, x0=[1.0], bounds=[(0.0, None)], method='L-BFGS-B')
    #     # # gamma_0 = res_0.x
    #     # optimal_log_likelihood_h0 = -res_0.fun
    #     #optimal_log_likelihood_h0 = poisson_log_likelihood([1, 0], observed_i, expected, sn_shape)
    #     optimal_log_likelihood_h0_i = optimal_log_likelihood_h0[i]


    #     # Alternative hypothesis: mu > 0
    #     # f_1 = lambda x: -poisson_log_likelihood(x, observed_i, expected, sn_shape)
    #     # res_1 = optimize.minimize(f_1, x0=[1.0, 0.0], bounds=[(0, None), (None, None)], method='L-BFGS-B')
    #     # gamma_1, mu_1 = res_1.x
        
    #     # f_1 = lambda mu: -poisson_log_likelihood([1, mu], observed_i, expected, sn_shape)
    #     # #print(f_1(np.linspace(0, 10, 100)))
    #     # res_1 = optimize.minimize(f_1, x0=[0.0], bounds=[(0, None)], method='L-BFGS-B')
    #     # mu_1 = res_1.x
    #     # gamma_1 = 1

    #     # res_1 = optimize.minimize(minus_poisson_log_likelihood_mu_only, x0=[-0.01], args=(observed_i, expected, sn_shape), bounds=[(None, None)])
    #     # mu_1 = res_1.x
    #     # gamma_1 = 1

    #     f_1 = lambda mu: -poisson_log_likelihood(1.0, mu, observed_i, expected, sn_shape)
    #     res_1 = optimize.minimize_scalar(f_1, bounds=(0.1, 100), method='bounded')
    #     mu_1 = res_1.x
    #     gamma_1 = 1

    #     optimal_log_likelihood_h1 = -res_1.fun
    #     gamma_array[i] = gamma_1
    #     mu_array[i] = mu_1

    #     # Likelihood ratio test statistic
    #     llr = -2 * (optimal_log_likelihood_h0_i - optimal_log_likelihood_h1)

    #     llr_array[i] = llr #if llr > 0 else 0

    #     # print(f"H0 gamma:{gamma_0}")
    #     # print(f"Optimal log likelihood H0: {optimal_log_likelihood_h0}")
    #     # print(f"H1 mu:{mu_1}, gamma:{gamma_1}")
    #     # print(f"Optimal log likelihood H1: {optimal_log_likelihood_h1}")
    #     # print("LLR:", llr)

    return llr_array, gamma_array, mu_array, slope_array

    


    


@njit(float64[:](float64[:,:], float64[:,:]), parallel=True, fastmath=True, cache=True)
def jit_log_likelihood_ratio_statistic(observed, expected):
    expected_b = np.broadcast_to(expected, observed.shape)

    # estimated_si = np.zeros_like(expected_b)
    # np.round_(observed - expected_b, 0, estimated_si)
    estimated_si = observed - expected_b
    
    estimated_si = np.where(estimated_si < 0, 0, estimated_si)
    #estimated_si = np.zeros_like(estimated_si) + 10
    
    llr = - 2 * np.sum( estimated_si - observed * np.log(1 + estimated_si / expected_b), axis=0)

    return llr


@njit(float64[:](float64[:,:], float64[:,:]), parallel=True, fastmath=True, cache=True)
def jit_pearson_chi_squared_statistic(observed, expected):
    expected_b = np.broadcast_to(expected, observed.shape)
    chi_sq = np.sum((observed - expected_b)**2 / expected_b, axis=0)
    return chi_sq

@njit(float64[:](float64[:,:], float64[:,:]), parallel=True, fastmath=True, cache=True)
def jit_cash_statistic(observed, expected):
    expected_b = np.broadcast_to(expected, observed.shape)
    log_term = np.where(observed > 0, observed * np.log(observed / expected_b), 0)
    c = 2 * np.sum(expected_b - observed + log_term, axis=0)
    return c

@njit(float64[:,:](float64[:,:], int32, int32), parallel=True, fastmath=True, cache=True)
def jit_generate_poisson_data(expected, chunk_size, seed):
    n_bins = expected.shape[0]
    simulated_data = np.empty((n_bins, chunk_size), dtype=np.float64)
    np.random.seed(seed)
    for i in prange(n_bins):
        lam = expected[i, 0]
        for j in prange(chunk_size):
            simulated_data[i, j] = np.random.poisson(lam)
    
    #console.log(simulated_data.dtype)

    return simulated_data

# @jit(parallel=True)
# def jit_poisson_random(expected, size):
#     return np.random.poisson(expected, size)