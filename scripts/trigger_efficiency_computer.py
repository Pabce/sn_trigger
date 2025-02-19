'''
trigger_efficiency_computer.py

This module...
'''
import itertools
from itertools import repeat
import copy
import multiprocessing as mp

import numpy as np
import logging
import scipy.stats as stats
from rich.logging import RichHandler
from rich.console import Group
import matplotlib.pyplot as plt

import gui
from gui import console
import aux
import statistical_comparison as sc
import classifier as cl
from enhanced_histogram import EnhancedHistogram as ehistogram


class TriggerEfficiencyComputer:

    def __init__(self, config, statistical_info, error_info=None, classifier=None, logging_level=logging.INFO):
        self.config = config
        self.statistical_info = statistical_info
        self.error_info = error_info
        self.classifier = classifier

        # TODO: make logger per class instance!
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

        self.setup_algorithms()


    def setup_algorithms(self):
        # TODO: Add info logs to this
        self.histogram_algorithms = {
            "hit_multiplicity": self.compute_hit_multiplicity_filtered_histogram,
            "bdt_output": self.compute_bdt_output_histogram
        }
        histogram_variable = self.statistical_info.get("histogram_variable")
        # histogram_type = "hit_multiplicity"
        # histogram_type = "bdt_output"
        
        self.compute_histogram = self.histogram_algorithms.get(histogram_variable)
        if not self.compute_histogram:
            raise ValueError(f"Invalid histogram type: {histogram_variable}")
        
        # ----------------------------------------

        self.statistical_comparison_algorithms = {
            "pearson_chi_squared": sc.StatisticalComparison.pearson_chi_squared_statistic,
            "cash": sc.StatisticalComparison.cash_statistic,
            "log_likelihood_ratio": sc.StatisticalComparison.log_likelihood_ratio_statistic_with_strength
        }
        self.statistical_comparison_dofs = {
            "pearson_chi_squared": None,
            "cash": None,
            "log_likelihood_ratio": 1
        }
        statistical_method = self.statistical_info.get("statistical_method")
        # statistical_method = "cash"

        self.statistical_comparison = self.statistical_comparison_algorithms.get(statistical_method)
        self.statistical_comparison_dof = self.statistical_comparison_dofs.get(statistical_method)
        if not self.statistical_comparison:
            raise ValueError(f"Invalid statistical method: {statistical_method}")


    def evaluate_trigger_efficiency(self, supernova_spectrum, distance, tpc_size, 
                                   burst_time_window, fake_trigger_rate,
                                   sn_clusters_per_event, sn_cluster_features_per_event,
                                   sn_info_per_event, expected_bg_hist,
                                   expected_sn_shape_hist,
                                   expected_bg_hist_variations=None,
                                   number_of_tests=1000,
                                   in_parallel=False):

        # Get the energies of our SN events
        sn_energies = sn_info_per_event[:, 0]

        # Compute the expected number of SN events in the time window
        event_number_full_burst = supernova_spectrum.distance_to_event_number(
            distance, tpc_size=tpc_size)

        expected_event_number_in_time_window = aux.expected_event_number_in_time_window(
                            supernova_spectrum.time_profile_x, supernova_spectrum.time_profile_y,
                            event_number_full_burst, burst_time_window)
        
        console.log(f"Evaluating distance: {distance}")
        console.log(f"Expected event number in full burst: {event_number_full_burst}")
        console.log(f"Expected SN event number in time window: {expected_event_number_in_time_window}")
        # This is inaccurate because it is not weighted by the SN spectrum
        # average_clusters_per_event = np.mean([len(event) for event in sn_clusters_per_event])
        # expected_clusters_in_time_window = expected_event_number_in_time_window * average_clusters_per_event
        # console.log(f"Expected SN cluster number in time window (average): {expected_clusters_in_time_window}")
        # Average number of clusters per event (weighted by the SN spectrum)
        weighted_sn_cluster_average = 0
        weight_sum = 0
        for i, cluster_list in enumerate(sn_clusters_per_event):
            energy = sn_energies[i]
            # Find the energy bin corresponding to the energy
            energy_bin_index = np.searchsorted(supernova_spectrum.energy_bins, energy)
            weighted_sn_cluster_average += len(cluster_list) * supernova_spectrum.spectrum_values[energy_bin_index-1]
            weight_sum += supernova_spectrum.spectrum_values[energy_bin_index-1]
        weighted_sn_cluster_average *= 1/weight_sum
        console.log(f"Weighted average number of clusters per event: {weighted_sn_cluster_average}")
        expected_clusters_in_time_window = expected_event_number_in_time_window * weighted_sn_cluster_average
        console.log(f"Expected SN cluster number in time window (weighted): {expected_clusters_in_time_window}")
        # ---------------------------------------- Temporary bullshit ----------------------------------------

        success_counter = 0
        results = []
        chi_squared_statistics = []
        p_values = []
        number_of_sampled_sn_clusters_list = []
        number_of_sampled_bg_clusters_list = []
        log_likelihood_ratios = []
        with gui.live_progress(console=console, status_fstring=f"[bold green]Evaluating trigger efficiency...") as (progress, live, group):
            
            if not in_parallel:
                task = progress.add_task(f'[cyan]Evaluating trigger efficiency...', total=number_of_tests)
                for i in range(number_of_tests):
                    self.log.debug(f"------- Test {i+1}/{number_of_tests} -------")

                    # Evaluate the trigger on a single burst
                    (success, chi_squared_statistic, p_value, 
                    number_of_sampled_sn_clusters, 
                    number_of_sampled_bg_clusters) = self.evaluate_trigger_on_single_burst(
                                                            expected_event_number_in_time_window, supernova_spectrum,
                                                            burst_time_window, fake_trigger_rate,
                                                            sn_clusters_per_event, sn_cluster_features_per_event,
                                                            sn_energies, expected_bg_hist, expected_sn_shape_hist,
                                                            expected_bg_hist_variations=expected_bg_hist_variations,
                                                            )

                    results.append(success)
                    chi_squared_statistics.append(chi_squared_statistic)
                    p_values.append(p_value)
                    number_of_sampled_sn_clusters_list.append(number_of_sampled_sn_clusters)
                    number_of_sampled_bg_clusters_list.append(number_of_sampled_bg_clusters)

                    progress.update(task, advance=1)
            
            # TODO: parallel is broken, fix it
            else:
                raise NotImplementedError("Parallel evaluation is currently broken")
                task = progress.add_task(f'[cyan]Evaluating trigger efficiency... (in parallel)', total=number_of_tests)
                num_processes = mp.cpu_count()

                with mp.Pool(num_processes) as pool:
                    chunk_size = int(0.5 * number_of_tests)
                    chunk_size = max(chunk_size, num_processes)
                    chunks = [np.arange(i, i+chunk_size) for i in range(0, number_of_tests, chunk_size)]

                    for i, chunk in enumerate(chunks):
                        console.log(i)
                        args = zip(chunk, repeat(expected_event_number_in_time_window), repeat(fake_trigger_rate),
                                 repeat(sn_clusters_per_event), repeat(sn_cluster_features_per_event),
                                 repeat(sn_energies), repeat(expected_bg_hist), repeat(spectrum_energy_bins),
                                 repeat(spectrum_energies_histogram))

                        results_chunk = pool.map(self.wrapped_evaluate_trigger_on_single_burst, args)
                        results.extend(results_chunk)

                        progress.update(task, advance=len(chunk))
            # Create a new group with just the progress bar
            group = Group(progress)
            live.update(group)

            success_counter = np.sum(results)
            print(success_counter/number_of_tests/chi_squared_statistic.size, "SUCCESS RATE")

            # If chi_squared_statistics is a list of arrays (because we have run with BG variations),
            # we need to concatenate them into one big array
            if isinstance(chi_squared_statistics[0], np.ndarray) or isinstance(chi_squared_statistics[0], list):
                chi_squared_statistics = np.vstack(chi_squared_statistics)
                p_values = np.vstack(p_values)
                number_of_sampled_bg_clusters_list = np.vstack(number_of_sampled_bg_clusters_list)
                number_of_sampled_sn_clusters_list = np.vstack(number_of_sampled_sn_clusters_list)
            else:
                chi_squared_statistics = np.array(chi_squared_statistics)

            print(f"CHISQ SHAPE {chi_squared_statistics.shape}")

            # If chi_squared_statistics is a 2D array...
            if len(chi_squared_statistics.shape) == 2:
                # Get the p-values that are below the limit
                limit_p_value = fake_trigger_rate * (burst_time_window / 1.0e6)
                # Make a matrix of 0s and 1s, where 1s are the p-values above the limit
                p_values_below_limit = p_values < limit_p_value
                average_success_rate_per_burst = np.mean(p_values_below_limit, axis=1)
                average_success_rate_per_bg_variation = np.mean(p_values_below_limit, axis=0)
                print("adfadf", average_success_rate_per_burst.shape)
                average_success_rate = np.mean(average_success_rate_per_bg_variation)
                print(f"Average success rate: {average_success_rate}")
                print(f"Average success rate std: {np.std(average_success_rate_per_bg_variation)}")
                success_rate = average_success_rate

                plt.figure()
                plt.hist(average_success_rate_per_burst, bins=np.linspace(0, 1, 50))
                plt.savefig("temp_pics/average_success_rate_per_burst.png")

                plt.figure()
                plt.hist(average_success_rate_per_bg_variation, bins=np.linspace(0, 1, 50))
                plt.savefig("temp_pics/average_success_rate_per_bg_variation.png")

                # Next up, get the 10% and 90% percentiles of the success rate
                # TODO: read the percentiles from the config
                success_rate_10th_percentile = np.percentile(average_success_rate_per_bg_variation, 10)
                success_rate_90th_percentile = np.percentile(average_success_rate_per_bg_variation, 90)
                print(f"10th percentile: {success_rate_10th_percentile}")
                print(f"90th percentile: {success_rate_90th_percentile}")
            
            else:
                success_rate = success_counter / number_of_tests
                success_rate_10th_percentile = None
                success_rate_90th_percentile = None


        return (success_rate, results, chi_squared_statistics, p_values, success_rate_10th_percentile, success_rate_90th_percentile,
                np.array(number_of_sampled_sn_clusters_list), np.array(number_of_sampled_bg_clusters_list))
    
    def wrapped_evaluate_trigger_on_single_burst(self, arg_list):
        arg_list = arg_list[1:]
        return self.evaluate_trigger_on_single_burst(*arg_list)

    def evaluate_trigger_on_single_burst(self, expected_event_number_in_time_window, supernova_spectrum,
                                    burst_time_window, fake_trigger_rate,
                                    sn_clusters_per_event, sn_cluster_features_per_event,
                                    sn_energies, expected_bg_hist, expected_sn_shape_hist,
                                    expected_bg_hist_variations=None):
        
        # Make a copy of the expected histograms, so we don't modify the original
        expected_bg_hist = copy.deepcopy(expected_bg_hist)
        expected_sn_shape_hist = copy.deepcopy(expected_sn_shape_hist)
        
        #expected_bg_hist_variations = None
        # Draw a "fake" background histogram sampled from the expected background histogram
        if expected_bg_hist_variations is not None:
            # print(expected_bg_hist_variations.shape, "cHIRP")
            # print(expected_bg_hist.values.shape, "chfdafasf")
            sampled_bg_histogram_values = np.round(stats.poisson.rvs(expected_bg_hist_variations, size=expected_bg_hist_variations.shape))
        else:
            sampled_bg_histogram_values = np.round(stats.poisson.rvs(expected_bg_hist.values, size=len(expected_bg_hist.values)))

        #print(sampled_bg_histogram_values.shape, "cHARP")

        # Draw a true event number from the expected event number (Poisson)
        r = int(stats.poisson.rvs(expected_event_number_in_time_window, size=1))
        if r == 0:
            self.log.debug("[bold red]Failure[/bold red]", extra={"markup": True})
            if expected_bg_hist_variations is None:
                success = 0
                chi_squared_statistic = 0
                p_value = 1
            else:
                success = np.zeros(len(expected_bg_hist_variations))
                chi_squared_statistic = np.zeros(len(expected_bg_hist_variations))
                p_value = np.ones(len(expected_bg_hist_variations))

            return success, chi_squared_statistic, p_value, 0, 0
        
        # Draw randomly from the SN event pool, while following the desired energy distribution
        sampled_indices = aux.sample_indices_by_energy(supernova_spectrum.spectrum_values, 
                                                       supernova_spectrum.energy_bins, sn_energies, size=r)
        
        # Get the clusters and features corresponding to the sampled indices
        sampled_sn_clusters_per_event = [sn_clusters_per_event[i] for i in sampled_indices]
        sampled_sn_cluster_features_per_event = [sn_cluster_features_per_event[i] for i in sampled_indices]
        # Put all clusters/features into one big list
        sampled_sn_clusters = list(itertools.chain(*sampled_sn_clusters_per_event))
        sampled_sn_cluster_features = np.array(list(itertools.chain(*sampled_sn_cluster_features_per_event)))

        # If we have no clusters, we skip
        if len(sampled_sn_clusters) == 0:
            self.log.debug("[bold red]Failure[/bold red]", extra={"markup": True})
            if expected_bg_hist_variations is None:
                success = 0
                chi_squared_statistic = 0
                p_value = 1
            else:
                success = np.zeros(len(expected_bg_hist_variations))
                chi_squared_statistic = np.zeros(len(expected_bg_hist_variations))
                p_value = np.ones(len(expected_bg_hist_variations))

            return success, chi_squared_statistic, p_value, 0, 0

        # Compute the signal histogram
        # We don't bunch here, only when we sum the SN and BG histograms
        # TODO: TEMPORARILY we bunch here because the expected BG histogram variations are bunched
        # (bins=expected_bg_hist.bins is the correct implementation)
        sampled_sn_histogram = self.compute_histogram(
            sampled_sn_clusters, sampled_sn_cluster_features, bins=expected_bg_hist.bunched_bins, 
            bunch_threshold=None, multiplier=1.0)
        
        #print(sampled_sn_histogram.values.shape, sampled_bg_histogram_values.shape, "ñasdkfhjasñfk")
        
        # Before we can add the sampled SN and BG histograms, we need to make sure they have the same size.
        # Compute the observed histogram with the same binning as the expected BG histogram
        bunch_threshold = self.statistical_info.get("histogram_bunch_threshold")
        #console.log(len(expected_bg_hist.bins), len(expected_bg_hist.bunched_bins))
        observed_histogram = ehistogram(values=sampled_sn_histogram.values + sampled_bg_histogram_values.astype(int),
                                        bins=expected_bg_hist.bunched_bins, bunched_bins=expected_bg_hist.bunched_bins)

        # In most cases, the observed histogram will have all entries above the bunch threshold.
        # However, for the cases it doesn't, we need to rebunch it *and* the expected BG histogram, in order to compare them.

        # SKIP UNTIL WE SORT THE VARIATIONS 
        # TODO: restore this
        # if np.any(observed_histogram.bunched_values < bunch_threshold):
        #     # Bunch the observed histogram
        #     observed_histogram.rebunch(bunch_threshold=bunch_threshold)
        #     # Bunch the expected BG histogram
        #     expected_bg_hist.rebunch(new_bins=observed_histogram.bunched_bins)
        #     self.log.debug("[yellow]Rebunching observed and expected BG histograms...", extra={"markup": True})
        #     # Bunch the expected SN shape histogram
        #     expected_sn_shape_hist.rebunch(new_bins=observed_histogram.bunched_bins)
        #     self.log.debug("[yellow]Rebunching expected SN shape histogram...", extra={"markup": True})


        # Compute the chi squared statistic
        self.log.debug(f"Expected BG histogram: {expected_bg_hist.bunched_values}")
        self.log.debug(f"Observed histogram: {observed_histogram.bunched_values}")

        if expected_bg_hist_variations is None:
            chi_squared_statistic, _ = self.statistical_comparison(
                        observed_histogram.bunched_values, 
                        expected_bg_hist.bunched_values,
                        signal_shape=expected_sn_shape_hist.bunched_values)
        else:
            chi_squared_statistic, _ = self.statistical_comparison(
                        observed_histogram.bunched_values, 
                        expected_bg_hist_variations,
                        signal_shape=expected_sn_shape_hist.bunched_values)
    
        #print(f"chi squared stat: {chi_squared_statistic.shape}")

        # Compute the p-value from the chi squared statistic
        if self.statistical_comparison_dof is None:
            dof = len(expected_bg_hist.bunched_values)
        else:
            dof = self.statistical_comparison_dof
        p_value = sc.StatisticalComparison.p_value_from_chi_squared_statistic(chi_squared_statistic, dof=dof)
        self.log.debug(f"Chi squared statistic: {chi_squared_statistic}, p-value: {p_value}, dof: {len(expected_bg_hist.bunched_values)}")
        
        # Fake trigger rate is in Hz (1/s). 
        # This means that we need to correct the limit p-value if the burst time window is smaller or larger than 1 s.
        limit_p_value = fake_trigger_rate * (burst_time_window / 1.0e6)

        success = p_value < limit_p_value
        if np.any(success):
            self.log.debug("[bold green]Success[/bold green]", extra={"markup": True})
        else:
            self.log.debug("[bold red]Failure[/bold red]", extra={"markup": True})
        
        # Some extra info to return
        number_of_sampled_sn_clusters = len(sampled_sn_clusters)
        number_of_sampled_bg_clusters = np.sum(sampled_bg_histogram_values) 
        
        return success, chi_squared_statistic, p_value, number_of_sampled_sn_clusters, number_of_sampled_bg_clusters


    def compute_hit_multiplicity_filtered_histogram(self, clusters, features, bins=None, bunch_threshold=None, bunched_bins=None,
                                                    multiplier=1.0, weights=None):
        # If we have a classifier, we filter the clusters through it
        if self.classifier:
            threshold = self.statistical_info["classifier_threshold"]
            clusters, features = cl.cluster_filter(
                self.classifier, clusters, features, threshold=threshold)
        
        # Get the hit multiplicities of the clusters
        hit_multiplicities = [len(cluster) for cluster in clusters]
        # Do we have bins?
        if bins is None:
            min_hm, max_hm = np.min(hit_multiplicities), np.max(hit_multiplicities)
            bins = np.arange(min_hm, max_hm + 2)

        # Bunch and multiply
        hit_multiplicity_histogram = ehistogram(hit_multiplicities, bins=bins, bunch_threshold=bunch_threshold, bunched_bins=bunched_bins,
                                                multiplier=multiplier, error_info=self.error_info, weights=weights)

        return hit_multiplicity_histogram

    
    def compute_bdt_output_histogram(self, clusters, features, bins=None, bunched_bins=None,
                                    bunch_threshold=None, multiplier=1.0, weights=None):
        # We obtain the clasifier prediction for the clusters (no filtering here!)
        if not self.classifier:
            raise ValueError("Classifier info is required for \"bdt_output\" histogram variable type")
        
        # Get the BDT output of the clusters
        bdt_output = self.classifier.predict_proba(features)[:, 1]
        
        # Per empirical tests, a reasonable minimim number of entries per bin so that the Cash statistic
        #  is well behaved IN THE TAIL is ~10.
        # Also, for high enough number of total entries, a good number of bins for well behaved stats is ~30.
        # TODO: For the final tests, we should abandon chisq and compute the empirical distribution of the Cash statistic.
        
        # Do we have bins?
        if bins is None:
            max_bins = 30
            bins = np.linspace(0, 1, max_bins + 1)

        # Bunch and multiply
        bdt_output_histogram = ehistogram(bdt_output, bins=bins, bunch_threshold=bunch_threshold, bunched_bins=bunched_bins,
                                        multiplier=multiplier, error_info=self.error_info, weights=weights)

        return bdt_output_histogram







