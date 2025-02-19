from copy import deepcopy

import numpy as np
from typing import Optional, Tuple, Union
import logging
from scipy.stats import gamma, poisson
from supernova_spectrum import SupernovaSpectrum
import supernova_spectrum

from gui import console

class EnhancedHistogram:
    def __init__(self, 
                 raw_data: Optional[Union[np.ndarray, list]] = None, 
                 bins: Union[int, np.ndarray, list] = 10, 
                 range: Optional[Tuple[float, float]] = None, 
                 weights: Optional[Union[np.ndarray, list]] = None,
                 density: bool = False,
                 values: Optional[Union[np.ndarray, list]] = None,
                 bunched_values: Optional[Union[np.ndarray, list]] = None,
                 bunched_bins: Optional[Union[np.ndarray, list]] = None,
                 bunch_threshold: Optional[int] = None,
                 multiplier: Optional[float] = None,
                 error_info: Optional[dict] = None,
                 to_generate_variations: Optional[Union[bool, int]] = False,
                 logging_level=logging.INFO):
        
        # Setup the logging for this class
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

        # Call np.histogram to compute counts and bin edges
        if values is not None and raw_data is not None:
            raise ValueError("You may only provide either the entry values or the raw data to bin, not both")

        self.weights = weights

        if values is None:
            values, bins = np.histogram(raw_data, bins=bins, range=range, weights=weights, density=density)
        else:
            bins = bins
        self.raw_data = raw_data
        self.bins = bins
        self.multiplier = multiplier if multiplier is not None else 1.0
        self.values = values * self.multiplier

        # if errors is not None:
        #     if len(errors) != len(values):
        #         raise ValueError("Length of errors must match the number of histogram bins")
        #     self.errors = np.array(errors)
        
        self.bunched_values = bunched_values
        self.bunched_bins = bunched_bins
        self.bunch_threshold = bunch_threshold
        
        if bunch_threshold is not None or bunched_bins is not None:
            # console.log(bunch_limit)
            # console.log(bunching_index)

            if bunch_threshold and bunched_bins:
                raise ValueError("You may only provide either a bunch threshold or bunched bins, not both")
            
            if bunched_values:
                self.log.warning("Bunched histogram values provided along with threshold/bins. These will now be overwritten!")

            self.set_bunched_histogram(self.values, self.bins, threshold=self.bunch_threshold, 
                                       new_bins=self.bunched_bins, raw_data=self.raw_data)
            
        if self.bunched_values is None:
            self.bunched_values = self.values
            self.bunched_bins = self.bins
        
        self.upper_errors = None
        self.lower_errors = None
        self.upper_bunched_errors = None
        self.lower_bunched_errors = None

        self.error_info = error_info
        if self.error_info:
            self.setup_error_algorithm()
            self.set_errors(**error_info.get("params", {}))

            self.upper_limit = self.values + self.upper_errors
            self.lower_limit = self.values - self.lower_errors
            self.bunched_upper_limit = self.bunched_values + self.upper_bunched_errors
            self.bunched_lower_limit = self.bunched_values - self.lower_bunched_errors
            
    
    def setup_error_algorithm(self):
        # TOOD: Add info logs to this
        self.error_algorithms = {
            "frequentist": self.set_frequentist_errors,
            "bayesian": self.set_bayesian_errors
        }
        # TODO: Add this to the config
        error_type = self.error_info.get("type")
        self.set_errors = self.error_algorithms.get(error_type)
        
        if not self.set_errors:
            raise ValueError(f"Invalid error type: {error_type}")


    def __getitem__(self, key):
        if key == 0:
            return self.values
        elif key == 1:
            return self.bins
        elif key == 2:
            return self.errors
        else:
            raise IndexError("Index out of range for EnhancedHistogram")
    
    def __len__(self):
        return 3  # Now returns counts, bin_edges, and errors
    
    def __iter__(self):
        yield self.values
        yield self.bins
        yield self.errors

    def plot(self, ax=None, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        
        values, bins, errors = self.values, self.bins, self.errors
        centers = (bins[:-1] + bins[1:]) / 2
        
        ax.bar(centers, values, width=np.diff(bins), yerr=errors, **kwargs)
        
        return ax

    # Take a histogram and "bunch it" at the far end so all entries are above a certain limit
    # This is a simplified version of the original algorithm, but works always 
    def old_set_bunched_histogram(self, values, hbins, bunch_limit=None, bunching_index=None):

        # values = deepcopy(self.values)
        # hbins = deepcopy(self.bins)
        checksum = np.sum(values)

        # Has the histogram too few values to be bunched?
        if bunch_limit:
            if np.sum(values) < bunch_limit:
                self.bunched_values = None
                self.bunched_bins = None
                self.bunching_index = None
                # Print warning
                self.log.warning(f"Histogram has too few values to be bunched! (Sum of entries is {np.sum(values)} < {bunch_limit})")
                return None, None, None

        # New empty histogram
        new_values = []
        new_bins = []

        if bunching_index is None:
            # Find first index where the histogram entry is smaller than the limit
            # This is the "bunching index"
            try:
                bunching_index = np.where(values < bunch_limit)[0][0]
                #console.log(f"Bunching index: {bunching_index}")
                #console.log(hist)
                # console.log(len(hist)-1)
            # If no index is found, we don't bunch
            except IndexError:
                return values, hbins, None

        if bunching_index > len(values) - 1:
            self.log.warning(f"Bunching index {bunching_index} is out of bounds for the provided histogram of length {len(values)}. No bunching will be performed.")
            return values, hbins, None

        # Is the bunching index the last entry?
        # In such case, we need to sum the last two entries
        if bunching_index == len(values) - 1:
            new_entry = values[-1] + values[-2]
            bunching_index -= 1
        else:
            # Sum all entries from this index to the end
            new_entry = np.sum(values[bunching_index:])
        #console.log(f"New entry: {new_entry}")
        new_values = np.concatenate((values[:bunching_index], [new_entry]))
        new_bins = np.concatenate((hbins[:bunching_index], hbins[bunching_index : bunching_index + 2]))
        
        checksum_new = np.sum(new_values)
        assert np.isclose(checksum, checksum_new), f"Checksums do not match: {checksum} != {checksum_new}"

        # Has this produced a histogram where all non-zero entries are above the limit?
        # If not, we bunch again!
        # TODO: This is wrong
        if bunch_limit is not None:   
            if new_values[-1] < bunch_limit:
                new_values, new_bins, bunching_index = self.set_bunched_histogram(new_values, new_bins, bunch_limit=bunch_limit, bunching_index=None)

        self.bunched_values = new_values
        self.bunched_bins = new_bins
        self.bunching_index = bunching_index


        return new_values, new_bins, bunching_index


    def set_bunched_histogram(self, values, bins, threshold=None, new_bins=None, raw_data=None):
        """
        Rebin the histogram so that all entries are above a certain threshold, 
        or rebin it to a new set of bins.
        """
        if threshold and new_bins:
            raise ValueError("You may only provide either a threshold or new bins, not both")
        if threshold is None and new_bins is None:
            raise ValueError("You must provide either a threshold or new bins")
        if threshold is None and np.all(new_bins == bins):
            self.log.debug("New bins are the same as the old bins. No rebinning will be performed.")
            self.bunched_values = values
            self.bunched_bins = bins
            return values, bins

        checksum = np.sum(values)
        new_values = np.copy(values)

        if threshold is not None:
            new_bins = np.copy(bins)
            if np.sum(values) < threshold:
                self.log.warning(f"Histogram has too few values to be rebinned! (Number of entries is {np.sum(values)} < {threshold})")
                return values, bins
            
            all_above_threshold = np.all(new_values >= threshold)
            while not all_above_threshold:
                # Find the smallest value and its index
                limit_index = len(new_values) - 1
                min_index = np.argmin(new_values)

                select_forward = False
                if min_index == 0:
                    select_forward = True
                elif min_index == limit_index:
                    select_forward = False
                elif new_values[min_index + 1] < new_values[min_index - 1]:
                    select_forward = True

                # Merge with the adjacent bin with the smallest value
                if select_forward:
                    new_values[min_index] += new_values[min_index + 1]
                    new_values = np.delete(new_values, min_index + 1)
                    new_bins = np.delete(new_bins, min_index + 1)
                else:
                    new_values[min_index] += new_values[min_index - 1]
                    new_values = np.delete(new_values, min_index - 1)
                    new_bins = np.delete(new_bins, min_index)
                
                all_above_threshold = np.all(new_values >= threshold)

        elif new_bins is not None:
            if raw_data is None:
                # If no raw data is provided, we can still rebin the histogram given that 
                # the new bin edges are a subset of the old bin edges, which will be the case
                # unless this is being used incorrectly!
                # Check it!:
                if not np.all(np.isin(new_bins, bins)):
                    raise ValueError("New bins edges are not a subset of the old bin edges, and no raw data is provided. Cannot rebin the histogram.")
                
                # To rebin, we create "fake" raw data by weighting the bin centers by the bin values
                bin_centers = (bins[:-1] + bins[1:]) / 2
                new_values, new_bins = np.histogram(bin_centers, bins=new_bins, weights=values)
            else:
                new_values, new_bins = np.histogram(raw_data, bins=new_bins, weights=self.weights)
                new_values = new_values * self.multiplier

        self.bunched_values = new_values
        self.bunched_bins = new_bins

        checksum_new = np.sum(new_values)
        assert np.isclose(checksum, checksum_new), f"Checksums do not match: {checksum} != {checksum_new}"
        
        return new_values, new_bins


    def rebunch(self, bunch_threshold=None, new_bins=None):
        if bunch_threshold is not None and new_bins is not None:
            raise ValueError("You must provide either a bunch threshold or new bins to rebunch the histogram")
        
        return self.set_bunched_histogram(self.bunched_values, self.bunched_bins, threshold=bunch_threshold,
                                           new_bins=new_bins, raw_data=self.raw_data)


    def set_frequentist_errors(self):
        self.upper_errors = np.sqrt(self.values)
        self.lower_errors = np.sqrt(self.values)
        self.upper_bunched_errors = np.sqrt(self.bunched_values)
        self.lower_bunched_errors = np.sqrt(self.bunched_values)

        return self.upper_errors, self.lower_errors, self.upper_bunched_errors, self.lower_bunched_errors
    
    def set_bayesian_errors(self, prior="flat", confidence_level=0.68):
        # Whether we have a flat or Jeffreys prior, 
        # we can model the posterior as a Gamma distribution
        # (in the case of the flat prior it will just be another Poisson)

        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile

        for vals in [self.values, self.bunched_values]:
            if vals is None:
                continue

            if prior == "flat":
                alpha = vals + 1
            elif prior == "jeffreys":
                alpha = vals + 0.5
            else:
                raise ValueError("Invalid prior type")
            beta = 1

            lower_limit = gamma.ppf(lower_percentile, alpha, scale=1/beta)
            upper_limit = gamma.ppf(upper_percentile, alpha, scale=1/beta)

            if vals is self.values:
                self.lower_errors = vals - lower_limit
                self.upper_errors = upper_limit - vals
            elif vals is self.bunched_values:
                self.lower_bunched_errors = vals - lower_limit
                self.upper_bunched_errors = upper_limit - vals
        
        # Necessary for a stupid technical reason
        if np.all(self.values == self.bunched_values):
            self.upper_bunched_errors = self.upper_errors
            self.lower_bunched_errors = self.lower_errors

        return self.upper_errors, self.lower_errors, self.upper_bunched_errors, self.lower_bunched_errors
    

    def generate_variations(self, n_variations=1000, random_seed=None):
        """
        Generate variations of the bunched histogram by sampling the posterior distribution
        """
        if self.error_info is None:
            raise ValueError("No error info provided. Cannot generate variations without error info.")
        if self.error_info.get('type') != "bayesian":
            raise ValueError("Only Bayesian errors are supported for generating variations.")

        # To correctly generate the variations, we need to know the original (non-multiplied)
        # values of the histogram! (as the uncertainty on the true rate parameter decreases as we generate more BG data)

        #original_values = self.values / self.multiplier
        original_bunched_values = self.bunched_values / self.multiplier

        # Setup the random number generator
        rng = np.random.default_rng(random_seed)

        # Setup the posterior for each bin
        posterior_distributions = []

        prior = self.error_info.get("params").get("prior")
        if prior == "flat":
            alpha_delta = 1
        elif prior == "jeffreys":
            alpha_delta = 0.5

        for val in original_bunched_values:
            alpha = val + alpha_delta
            beta = 1
            
            posterior = lambda x: gamma.pdf(x, alpha, scale=1/beta)
            posterior_distributions.append(posterior)
        
        # Generate the variations
        #random_numbers = rng.random(size=(n_variations, len(self.bunched_values)))
        variations = np.zeros((n_variations, len(self.bunched_values)))

        for i, val in enumerate(original_bunched_values):
            variations[:, i] = gamma.rvs(val + alpha_delta, scale=1/beta, size=n_variations)
        
        # Normalize the variations
        variations *= self.multiplier
        
        return variations