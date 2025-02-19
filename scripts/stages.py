import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from rich.table import Table
import plotille
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

import data_loader as dl
from gui import console
import gui
import saver as sv
import clustering as cl
import classifier
import trigger_efficiency_computer as tec
import aux
from utils import get_total_size
import supernova_spectrum
from supernova_spectrum import SupernovaSpectrum, weigh_by_supernova_spectrum

class Stage(ABC):

    def __init__(self, config, input_data, input_data_file=None, logging_level=logging.INFO, stage_str=""):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.logging_level = logging_level
        self.log.setLevel(logging_level)

        self.input_data_file = input_data_file
        self.input_data = input_data
        self.output_data = None
        self.info_output_data = None

        self.stage_str = stage_str
        console.print("\n")
        console.log(f"[bold]------------------------------------------------------------")
        console.log(f":chipmunk:  [bold]Starting STAGE {self.stage_str} :chipmunk:")
        console.log(f"[bold]------------------------------------------------------------")

    # Constructor for when the data is loaded from an input file
    @classmethod
    def from_input_file(cls, config, input_data_file, extra_input_data=None, logging_level=logging.INFO):
        input_object = sv.DataWithConfig.load_data_from_file(input_data_file, data_type_str="input data")
        input_data = input_object.data
        input_data_config = input_object.config
        
        # Check that the configurations match. If not, raise a warning
        # TODO: This is useless right now, implement properly...
        if input_data_config != config:
            logging.getLogger(cls.__name__).warning(f"Configurations do not match between input file {input_data_file} and current configuration")
        
        return cls(config, input_data, logging_level)

    @abstractmethod
    def run(self):
        pass

    def verify_input(self):
        # Check that the input data dictionary has all the required keys
        pass

    # TODO: Add some cleanup functionality
    def exit(self):
        console.log(f"[bold]------------------------------------------------------------")
        console.log(f":saxophone::chipmunk:  [bold]Finished STAGE {self.stage_str} :saxophone::chipmunk:")
        console.log(f"[bold]------------------------------------------------------------")



class LoadEvents(Stage):
    IS_OPTIONAL = False
    REQUIRED_INPUTS = []

    def __init__(self, config, input_data, logging_level=logging.INFO):
        super().__init__(config, input_data, logging_level=logging_level, stage_str="LOAD_EVENTS")
        self.log = logging.getLogger(self.__class__.__name__)

        # LoadEvents stage accepts no input file/data.
        # Raise an error if an input file/data is provided
        if self.input_data_file or self.input_data:
            console.log(self.input_data_file, self.input_data)
            raise ValueError("LoadEvents stage does not accept an input file or input data")
        
    def run(self):
        # CONFIG READS ----- Load parameters from the configuration file
        sn_file_limit_parameter_search = self.config.get('Simulation', 'load_events', 'sn_file_limit')
        bg_file_limit_parameter_search = self.config.get('Simulation', 'load_events', 'bg_file_limit')
        bg_sample_length = self.config.get('DataFormat', 'bg_sample_length')
        split_for_classifier = self.config.get("Simulation", "load_events", "split_for_classifier")
        classifier_energy_limit = self.config.get("Simulation", "load_events", "classifier_energy_limit")
        train_split = self.config.get("Simulation", "load_events", "train_split")
        # -----------------------------------------------
        
        # Load SN and BG hits for the parameter search
        # Create a dataloader object
        loader = dl.DataLoader(self.config, logging_level=logging.INFO)
        
        sn_hit_list_per_event, bg_hit_list_per_event, sn_info_per_event,\
        sn_eff_hit_list_per_event, sn_bdt_hit_list_per_event, sn_eff_info_per_event,\
        sn_bdt_info_per_event, bg_eff_hit_list_per_event, bg_bdt_hit_list_per_event, total_bg_time_window =\
            loader.load_and_split(sn_file_limit_parameter_search, bg_file_limit_parameter_search, shuffle=True)
        
        # Set the output data 
        self.output_data = {
            'sn_hit_list_per_event': sn_hit_list_per_event,
            'bg_hit_list_per_event': bg_hit_list_per_event,
            'sn_info_per_event': sn_info_per_event,
            'sn_eff_hit_list_per_event': sn_eff_hit_list_per_event,
            'sn_bdt_hit_list_per_event': sn_bdt_hit_list_per_event,
            'sn_eff_info_per_event': sn_eff_info_per_event,
            'sn_bdt_info_per_event': sn_bdt_info_per_event,
            'bg_eff_hit_list_per_event': bg_eff_hit_list_per_event,
            'bg_bdt_hit_list_per_event': bg_bdt_hit_list_per_event
        }

        # Set the info output data
        self.info_output_data = {
            'memory_usage': {
                'sn_hits': np.sum([a.nbytes for a in sn_hit_list_per_event]) / 1024**2,
                'sn_info': get_total_size(sn_info_per_event) / 1024**2 / 1024**2,
                'bg_hits': np.sum([a.nbytes for a in bg_hit_list_per_event]) / 1024**2
            },
            'data_loading_statistics': {
                'sn_hit_num': np.sum([len(event) for event in sn_hit_list_per_event]),
                'sn_eff_hit_num': np.sum([len(event) for event in sn_eff_hit_list_per_event]),
                'sn_bdt_hit_num': np.sum([len(event) for event in sn_bdt_hit_list_per_event]),
                'bg_hit_num': np.sum([len(event) for event in bg_hit_list_per_event]),
                'bg_eff_hit_num': np.sum([len(event) for event in bg_eff_hit_list_per_event]),
                'bg_bdt_hit_num': np.sum([len(event) for event in bg_bdt_hit_list_per_event]),
                'sn_event_num': len(sn_info_per_event),
                'sn_eff_event_num': len(sn_eff_info_per_event),
                'sn_bdt_event_num': len(sn_bdt_info_per_event),
                'bg_event_num': len(bg_hit_list_per_event),
                'bg_eff_event_num': len(bg_eff_hit_list_per_event),
                'bg_bdt_event_num': len(bg_bdt_hit_list_per_event),
                'total_bg_time_window': total_bg_time_window,
                'bg_eff_time_window': len(bg_eff_hit_list_per_event) * bg_sample_length,
                'bg_bdt_time_window': len(bg_bdt_hit_list_per_event) * bg_sample_length
            },
            'timestamp': f'{datetime.now()}'
        }

        # Information tables ---------
        # Print the memory usage of the loaded data
        memory_table = gui.get_custom_table(self.config, "memory_usage", **self.info_output_data)
        # Print the data loading statistics
        data_loading_stats_table = gui.get_custom_table(self.config, "data_loading_statistics", **self.info_output_data)

        console.log(memory_table)
        console.log(data_loading_stats_table)

        # Log where the split was done and the energy threshold for the classifier
        if split_for_classifier:
            self.log.info(f"Data was split for the classifier at {train_split:.0%}. Events above {classifier_energy_limit} MeV were discarded.")
        # ----------------------------
        
        return self.output_data, self.info_output_data


class Clustering(Stage):
    IS_OPTIONAL = False
    REQUIRED_INPUTS = ['sn_eff_hit_list_per_event',
                        'bg_eff_hit_list_per_event', 
                        'sn_bdt_hit_list_per_event', 
                        'bg_bdt_hit_list_per_event']

    def __init__(self, config, input_data, logging_level=logging.INFO):
        super().__init__(config, input_data, logging_level=logging_level, stage_str="CLUSTERING")
        self.log = logging.getLogger(self.__class__.__name__)

    def run(self):
        # CONFIG READS ----- Load parameters from the configuration file
        clustering_parameters = self.config.get('Simulation', 'clustering', 'parameters')
        # -----------------------------------------------
        sn_eff_hit_list_per_event = self.input_data['sn_eff_hit_list_per_event']
        bg_eff_hit_list_per_event = self.input_data['bg_eff_hit_list_per_event']
        sn_bdt_hit_list_per_event = self.input_data['sn_bdt_hit_list_per_event']
        bg_bdt_hit_list_per_event = self.input_data['bg_bdt_hit_list_per_event']

        # Print the clustering parameters
        clustering_parameters_table = gui.get_custom_table(self.config, "clustering_parameters", clustering_parameters=clustering_parameters)
        console.log(clustering_parameters_table)

        # Initialize the clustering object
        clustering = cl.Clustering(self.config, logging_level=logging.INFO)

        # Compute the efficiency and training clusters for the SN and BG hits
        (final_sn_eff_clusters, final_sn_eff_clusters_per_event, 
        final_sn_eff_hit_multiplicities, final_sn_eff_hit_multiplicities_per_event) = clustering.group_clustering(
            sn_eff_hit_list_per_event, spatial_filter=True, data_type_str="SN efficiency", in_parallel=True, **clustering_parameters
        )
        (final_bg_eff_clusters, final_bg_eff_clusters_per_event, 
        final_bg_eff_hit_multiplicities, final_bg_eff_hit_multiplicities_per_event) = clustering.group_clustering(
            bg_eff_hit_list_per_event, spatial_filter=True, data_type_str="SN efficiency", in_parallel=True, **clustering_parameters
        )

        (final_sn_bdt_clusters, final_sn_bdt_clusters_per_event,
        final_sn_bdt_hit_multiplicities, final_sn_bdt_hit_multiplicities_per_event) = clustering.group_clustering(
            sn_bdt_hit_list_per_event, spatial_filter=True, data_type_str="SN BDT", in_parallel=True, **clustering_parameters
        )
        (final_bg_bdt_clusters, final_bg_bdt_clusters_per_event,
        final_bg_bdt_hit_multiplicities, final_bg_bdt_hit_multiplicities_per_event) = clustering.group_clustering(
            bg_bdt_hit_list_per_event, spatial_filter=True, data_type_str="BG BDT", in_parallel=True, **clustering_parameters
        )

        # Set the output data
        self.output_data = {
            'final_sn_eff_clusters': final_sn_eff_clusters,
            'final_bg_eff_clusters': final_bg_eff_clusters,
            'final_sn_bdt_clusters': final_sn_bdt_clusters,
            'final_bg_bdt_clusters': final_bg_bdt_clusters,
            'final_sn_eff_clusters_per_event': final_sn_eff_clusters_per_event,
            'final_bg_eff_clusters_per_event': final_bg_eff_clusters_per_event,
            'final_sn_bdt_clusters_per_event': final_sn_bdt_clusters_per_event,
            'final_bg_bdt_clusters_per_event': final_bg_bdt_clusters_per_event
        }

        # Set the info output data
        self.info_output_data = {
            'sn_eff_clusters_num': len(final_sn_eff_clusters),
            'sn_bdt_clusters_num': len(final_sn_bdt_clusters),
            'bg_eff_clusters_num': len(final_bg_eff_clusters),
            'bg_bdt_clusters_num': len(final_bg_bdt_clusters),
            'average_sn_eff_clusters_per_event': np.mean([len(event) for event in final_sn_eff_clusters_per_event]),
            'average_sn_bdt_clusters_per_event': np.mean([len(event) for event in final_sn_bdt_clusters_per_event]),
            'average_bg_eff_clusters_per_event': np.mean([len(event) for event in final_bg_eff_clusters_per_event]),
            'average_bg_bdt_clusters_per_event': np.mean([len(event) for event in final_bg_bdt_clusters_per_event]),
            'average_sn_eff_hit_multiplicity': np.mean(final_sn_eff_hit_multiplicities),
            'average_sn_bdt_hit_multiplicity': np.mean(final_sn_bdt_hit_multiplicities),
            'average_bg_eff_hit_multiplicity': np.mean(final_bg_eff_hit_multiplicities),
            'average_bg_bdt_hit_multiplicity': np.mean(final_bg_bdt_hit_multiplicities),
            'timestamp': f'{datetime.now()}'
        }

        # Print the clustering statistics
        clustering_stats_table = gui.get_custom_table(self.config, "clustering_statistics", **self.info_output_data)
        
        console.log(clustering_stats_table)

        return self.output_data, self.info_output_data

    
    # TODO: Add support for different clustering algorithms
    def get_clustering_parameters(self):
        pass

class ClusterFeatureExtraction(Stage):
    IS_OPTIONAL = False
    REQUIRED_INPUTS = ['final_sn_eff_clusters_per_event',
                        'final_bg_eff_clusters_per_event',
                        'final_sn_bdt_clusters_per_event',
                        'final_bg_bdt_clusters_per_event']

    def __init__(self, config, input_data, logging_level=logging.INFO):
        super().__init__(config, input_data, logging_level=logging_level, stage_str="CLUSTER_FEATURE_EXTRACTION")
        self.log = logging.getLogger(self.__class__.__name__)
    

    def run(self):
        # CONFIG READS ----- Load parameters from the configuration file
        cluster_features = self.config.get('Simulation', 'cluster_feature_extraction', 'cluster_features')
        detector_type = self.config.get('Detector', 'type')
        # -----------------------------------------------
        final_sn_eff_clusters_per_event = self.input_data['final_sn_eff_clusters_per_event']
        final_bg_eff_clusters_per_event = self.input_data['final_bg_eff_clusters_per_event']
        final_sn_bdt_clusters_per_event = self.input_data['final_sn_bdt_clusters_per_event']
        final_bg_bdt_clusters_per_event = self.input_data['final_bg_bdt_clusters_per_event']

        sn_eff_features_array, sn_eff_features_per_event, _ = cl.group_compute_cluster_features(
            final_sn_eff_clusters_per_event, detector_type=detector_type, data_type_str="SN efficiency", 
            features_to_return=cluster_features, per_event=True)
        bg_eff_features_array, bg_eff_features_per_event, _ = cl.group_compute_cluster_features(
            final_bg_eff_clusters_per_event, detector_type=detector_type, data_type_str="BG efficiency",
            features_to_return=cluster_features, per_event=True)
        sn_eff_targets = np.ones(len(sn_eff_features_array))
        bg_eff_targets = np.zeros(len(bg_eff_features_array))

        sn_bdt_features_array, sn_bdt_features_per_event, cluster_feature_names = cl.group_compute_cluster_features(
            final_sn_bdt_clusters_per_event, detector_type=detector_type, data_type_str="SN train", 
            features_to_return=cluster_features, per_event=True)
        bg_bdt_features_array, bg_bdt_features_per_event, _ = cl.group_compute_cluster_features(
            final_bg_bdt_clusters_per_event, detector_type=detector_type, data_type_str="BG train",
            features_to_return=cluster_features, per_event=True)
        sn_bdt_targets = np.ones(len(sn_bdt_features_array))
        bg_bdt_targets = np.zeros(len(bg_bdt_features_array))

        # Print the cluster feature names
        console.log(f"[bold yellow]Cluster Feature Names:")
        console.log(cluster_feature_names)

        # Set the output data
        self.output_data = {
            'sn_eff_features_array': sn_eff_features_array,
            'bg_eff_features_array': bg_eff_features_array,
            'sn_bdt_features_array': sn_bdt_features_array,
            'bg_bdt_features_array': bg_bdt_features_array,
            'sn_eff_targets': sn_eff_targets,
            'bg_eff_targets': bg_eff_targets,
            'sn_bdt_targets': sn_bdt_targets,
            'bg_bdt_targets': bg_bdt_targets,
            'sn_eff_features_per_event': sn_eff_features_per_event,
            'bg_eff_features_per_event': bg_eff_features_per_event,
        }

        # Set the info output data
        self.info_output_data = {
            'cluster_feature_names': cluster_feature_names,
            'cluster_feature_num': len(cluster_feature_names),
            'sn_eff_features_array_num': sn_eff_features_array.shape[0],
            'sn_bdt_features_array_num': sn_bdt_features_array.shape[0],
            'bg_eff_features_array_num': bg_eff_features_array.shape[0],
            'bg_bdt_features_array_num': bg_bdt_features_array.shape[0],
            'timestamp': f'{datetime.now()}'
        }

        self.log.debug(f"sn_eff_features_array shape: {sn_eff_features_array.shape}")
        self.log.debug(f"bg_eff_features_array shape: {bg_eff_features_array.shape}")
        self.log.debug(f"sn_bdt_features_array shape: {sn_bdt_features_array.shape}")
        self.log.debug(f"bg_bdt_features_array shape: {bg_bdt_features_array.shape}")
        self.log.debug(f"sn_eff_features_per_event[0] shape: {sn_eff_features_per_event[0].shape}")

        return self.output_data, self.info_output_data


class BDTTraining(Stage):
    IS_OPTIONAL = True
    REQUIRED_INPUTS = ['sn_bdt_features_array',
                        'bg_bdt_features_array',
                        'sn_bdt_targets',
                        'bg_bdt_targets']

    def __init__(self, config, input_data, logging_level=logging.INFO):
        super().__init__(config, input_data, logging_level=logging_level, stage_str="BDT_TRAINING")
        self.log = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        # TODO: DO the energy filtering here, and not in the LoadData stage! (?)

        # CONFIG READS ----- Load parameters from the configuration file
        bdt_optimize_hyperparameters = self.config.get('Simulation', 'bdt_training', 'optimize_hyperparameters')
        bdt_optimize_hyperparameters_random_state = self.config.get('Simulation', 'bdt_training', 'optimize_hyperparameters_random_state')
        bdt_hyperparameters = self.config.get('Simulation', 'bdt_training', 'bdt_hyperparameters')
        # -----------------------------------------------
        sn_bdt_features_array = self.input_data['sn_bdt_features_array']
        bg_bdt_features_array = self.input_data['bg_bdt_features_array']
        # You don't really need to read this, it's just a vector of ones/zeros
        # of the same length as sn_bdt_features_array/bg_bdt_features_array
        sn_bdt_targets = self.input_data['sn_bdt_targets'] 
        bg_bdt_targets = self.input_data['bg_bdt_targets']

        # If either the SN or BG features are empty or very small, raise an error. We cannot proceed any further
        if len(sn_bdt_features_array) <= 50 or len(bg_bdt_features_array) <= 50:
            raise ValueError("Either the SN or BG features are empty or contain very few events. Cannot proceed with training.")

        # Combine the features and targets
        features_bdt = np.vstack([sn_bdt_features_array, bg_bdt_features_array])
        targets_bdt = np.concatenate([sn_bdt_targets, bg_bdt_targets])

        # Train a BDT
        if bdt_hyperparameters:
            self.log.info(f"Training BDT with fixed hyperparameters: {bdt_hyperparameters}")

        with console.status(f'[bold green]Training BDT... Optimize hyperparmeters: {bdt_optimize_hyperparameters}'):

            hist_boosted_tree, test_features, test_targets, test_score, train_features, train_targets, train_score =\
                classifier.hist_gradient_boosted_tree(features_bdt, targets_bdt, n_estimators=200,
                optimize_hyperparameters=bdt_optimize_hyperparameters, random_state=bdt_optimize_hyperparameters_random_state,
                hyperparameters=bdt_hyperparameters)
            
        # Set the output data
        self.output_data = {
            'hist_boosted_tree': hist_boosted_tree,
            'test_features': test_features,
            'test_targets': test_targets,
            'test_score': test_score,
            'train_features': train_features,
            'train_targets': train_targets,
            'train_score': train_score
        }

        # Set the info output data
        self.info_output_data = {
            'test_score': test_score,
            'train_score': train_score,
            'bdt_hyperparameters': hist_boosted_tree.get_params(),
            'optimize_hyperparameters': bdt_optimize_hyperparameters,
            'optimize_hyperparameters_random_state': bdt_optimize_hyperparameters_random_state,
            'timestamp': f'{datetime.now()}'
        }

        self.log.debug(f"Training features: {train_features.shape}")
        self.log.debug(f"Training targets: {train_targets.shape}")
        self.log.debug(f"Test features: {test_features.shape}")
        self.log.debug(f"Test targets: {test_targets.shape}")
        
        console.log(f"Train score: {train_score}")
        console.log(f"Test score: {test_score}")

        return self.output_data, self.info_output_data
    

class TriggerEfficiency(Stage):
    IS_OPTIONAL = False
    REQUIRED_INPUTS = ['hist_boosted_tree',
                       'sn_eff_info_per_event',
                       'final_sn_eff_clusters',
                       'final_bg_eff_clusters',
                       'final_sn_eff_clusters_per_event',
                       'final_bg_eff_clusters_per_event',
                       'sn_eff_features_array',
                       'bg_eff_features_array',
                       'sn_eff_features_per_event',
                       'bg_eff_features_per_event']

    def __init__(self, config, input_data, logging_level=logging.INFO):
        super().__init__(config, input_data, logging_level=logging_level, stage_str="TRIGGER_EFFICIENCY")
        self.log = logging.getLogger(self.__class__.__name__)

        self.setup_supernova_spectra()


    def setup_supernova_spectra(self):
        # Load the time profile
        # TODO: Do this in a less ridiculous way
        data_loader = dl.DataLoader(self.config, logging_level=logging.INFO)
        time_profile = data_loader.load_time_profile()

        sn_spectra_labels = self.config.get('Simulation', 'trigger_efficiency', 'physics', 'supernova_spectra')
        # This will be an individual spectrum or a list of spectra
        # If an individual spectrum, convert into a list
        if not isinstance(sn_spectra_labels, list):
            sn_spectra_labels = [sn_spectra_labels]
        
        supernova_spectra = []
        for spectrum in sn_spectra_labels:

            spectrum_type = spectrum.get("spec_type")
            
            if spectrum_type == "pinched":
                pinching_parameters = spectrum.get("pinching_parameters")
                supernova_spectra.append( SupernovaSpectrum.from_pinched_spectrum(pinching_parameters, time_profile=time_profile) )
            elif spectrum_type == "model_name":
                model_name = spectrum.get("model_name")
                supernova_spectra.append( SupernovaSpectrum.from_model_name(model_name, time_profile=time_profile) )
            else:
                raise ValueError(f"Invalid supernova spectrum type: {spectrum_type}")
        
        self.supernova_spectra = supernova_spectra
        
    
    def run(self):
        # CONFIG READS ----- Load parameters from the configuration file
        true_tpc_size = self.config.get('Detector', 'true_tpc_size') * self.config.get('Detector', 'tpc_size_correction_factor')
        used_tpc_size = self.config.get('Detector', 'used_tpc_size') * self.config.get('Detector', 'tpc_size_correction_factor')
        tpc_size_correction_factor = self.config.get('Detector', 'tpc_size_correction_factor')
        bg_sample_length = self.config.get("DataFormat", "bg_sample_length")
        bg_sample_number_per_file = self.config.get("DataFormat", "bg_sample_number_per_file")
        sn_event_multiplier = self.config.get('Detector', 'sn_event_multiplier')

        # TODO: Fix this!!!
        distance_to_evaluate = self.config.get('Simulation', 'trigger_efficiency', 'distance_to_evaluate')
        fake_trigger_rate = self.config.get('Simulation', 'trigger_efficiency', 'fake_trigger_rate')
        burst_time_window = self.config.get('Simulation', 'trigger_efficiency', 'burst_time_window') # Remember this is in mu_s # TODO: Change to ms

        use_classifier = self.config.get('Simulation', 'trigger_efficiency', 'use_classifier')
        error_info = self.config.get('Simulation', 'trigger_efficiency', 'error_info')
        statistical_info = self.config.get('Simulation', 'trigger_efficiency', 'statistical_info')
        number_of_tests = self.config.get('Simulation', 'trigger_efficiency', 'number_of_tests')

        # INPUT READS -----------------------------------------------
        hist_boosted_tree = self.input_data['hist_boosted_tree']
        sn_eff_info_per_event = self.input_data['sn_eff_info_per_event']
        final_sn_eff_clusters = self.input_data['final_sn_eff_clusters']
        final_bg_eff_clusters = self.input_data['final_bg_eff_clusters']
        final_sn_eff_clusters_per_event = self.input_data['final_sn_eff_clusters_per_event']
        final_bg_eff_clusters_per_event = self.input_data['final_bg_eff_clusters_per_event']
        sn_eff_features_array = self.input_data['sn_eff_features_array']
        bg_eff_features_array = self.input_data['bg_eff_features_array']
        sn_eff_features_per_event = self.input_data['sn_eff_features_per_event']
        bg_eff_features_per_event = self.input_data['bg_eff_features_per_event']
        # -----------------------------------------------

        # If distance_to_evaluate is a number, convert it to a list
        if isinstance(distance_to_evaluate, (int, float)):
            distance_to_evaluate = [distance_to_evaluate]
        distance_to_evaluate = np.array(distance_to_evaluate)

        # If we have too few clusters, the statistics will be unreliable. Raise an error
        if len(sn_eff_features_array) <= 50 or len(bg_eff_features_array) <= 50:
            raise ValueError("Either the SN or BG features are empty or contain very few events. Cannot proceed with trigger efficiency evaluation.")
        
        # The total time window spanned by the loaded background events
        total_bg_time_window = bg_sample_length * len(bg_eff_features_per_event)
        bg_histogram_multiplier = 1/total_bg_time_window * (burst_time_window/1000) * true_tpc_size/used_tpc_size * tpc_size_correction_factor
        
        console.log(f"BG histogram multiplier: {bg_histogram_multiplier}")
        console.log(f"Total BG time window: {total_bg_time_window} ms")
        console.log(f"Number of BG clusters: {len(final_bg_eff_clusters)}")
        console.log(f"Number of BG events: {len(bg_eff_features_per_event)}, {len(final_bg_eff_clusters_per_event)}")
        
        # Get the corrected TPC size, including the correction factor and the SN event multiplier
        corrected_tpc_size = true_tpc_size * tpc_size_correction_factor * sn_event_multiplier
        
        # Create the trigger efficiency computer object
        trigger_eff_computer = tec.TriggerEfficiencyComputer(self.config, logging_level=self.logging_level, 
                                                            statistical_info=statistical_info, error_info=error_info,
                                                            classifier=hist_boosted_tree)
    
        # Create the BG expected histogram
        # TODO: Bunch limit should be set by the individual algorithms
        expected_bg_histogram = trigger_eff_computer.compute_histogram(final_bg_eff_clusters, bg_eff_features_array, bins=None,
                                                                        bunch_threshold=statistical_info["histogram_bunch_threshold"], 
                                                                        multiplier=bg_histogram_multiplier)
        
        # TEMPORARY -----------------------------------------------
        # Generate variations of the expected BG histogram if required
        expected_bg_histogram_variations = None
        use_bg_variations = statistical_info["use_bg_variations"]
        if use_bg_variations is not False:
            expected_bg_histogram_variations = expected_bg_histogram.generate_variations(n_variations=use_bg_variations)

            # plt.figure()
            # sns.boxplot(data=expected_bg_histogram_variations)
            # plt.yscale('log')
            # plt.ylim(1e0, 1e3)
            # plt.savefig("temp_pics/expected_bg_histogram_variations.png")


            # plt.figure()
            # sns.boxplot(data=sampled_bg_histogram_values_from_variations)
            # plt.yscale('log')
            # plt.ylim(1e0, 1e3)
            # plt.savefig("temp_pics/expected_bg_histogram_variations_sampled.png")


        # --------------------------------------------------------
        
        # If the total number of events in the expected BG histogram is too small, statistics will be unreliable. Raise an error
        expected_bg_cluster_number_in_time_window = np.sum(expected_bg_histogram.values)
        # This limit is set by hand from empirical observations. It should be above ~30 in any case
        if expected_bg_cluster_number_in_time_window < 50:
            raise ValueError("The total number of clusters in the expected BG histogram is too small. Cannot proceed with trigger efficiency evaluation.")
        
        # Print the expected BG histogram to the console
        console.log(f"[bold yellow]Expected BG histogram:")
        print(plotille.hist_aggregated(expected_bg_histogram.bunched_values, expected_bg_histogram.bunched_bins, log_scale=True, lc='blue'))
        console.log(f"Total number of clusters in the expected BG histogram: {expected_bg_cluster_number_in_time_window}")
        console.log(f"Expected BG cluster rate: { expected_bg_cluster_number_in_time_window / (burst_time_window/1e6) } Hz")

        console.log(f"[bold yellow]Evaluating efficiency for distances:")
        console.log(distance_to_evaluate)
        console.log(f"[bold yellow]Evaluating efficiency for spectra:")
        console.log(f"{[str(s) for s in self.supernova_spectra]}")
        console.log("\n")

        trigger_efficiencies_dict = {}
        chi_squared_statistics_dict = {}
        p_values_dict = {}
        success_rate_10th_percentile_dict = {}
        success_rate_90th_percentile_dict = {}

        for spectrum in self.supernova_spectra:
            # Get the string representation of the spectrum
            spectrum_str = str(spectrum)
            trigger_efficiencies_dict[spectrum_str] = []
            chi_squared_statistics_dict[spectrum_str] = []
            p_values_dict[spectrum_str] = []
            success_rate_10th_percentile_dict[spectrum_str] = []
            success_rate_90th_percentile_dict[spectrum_str] = []

            # Create the expected SN-weighted shape histogram
            # For now this will only be used in the log likelihood ratio calculation
            sn_eff_energies_per_cluster = []
            for i, clusters in enumerate(final_sn_eff_clusters_per_event):
                sn_eff_energies_per_cluster.extend([sn_eff_info_per_event[i, 0]] * len(clusters))
                
            sn_spectrum_weights = weigh_by_supernova_spectrum(sn_eff_energies_per_cluster, spectrum)
            expected_sn_shape_histogram = trigger_eff_computer.compute_histogram(
                        final_sn_eff_clusters, sn_eff_features_array, bins=expected_bg_histogram.bins,
                        bunched_bins=expected_bg_histogram.bunched_bins, multiplier=1.0, 
                        weights=sn_spectrum_weights)
            
            console.log(f"[bold yellow]Expected SN shape histogram:")
            print(plotille.hist_aggregated(expected_sn_shape_histogram.values, 
                                           expected_sn_shape_histogram.bins, 
                                           log_scale=False, lc='yellow'))

            for distance in distance_to_evaluate:
                
                (trigger_efficiency, results, chi_squared_statistics, p_values,
                success_rate_10th_percentile, success_rate_90th_percentile,
                number_of_sampled_sn_clusters_list,
                number_of_sampled_bg_clusters_list) = trigger_eff_computer.evaluate_trigger_efficiency(
                                                        spectrum, distance, corrected_tpc_size, 
                                                        burst_time_window, fake_trigger_rate,
                                                        final_sn_eff_clusters_per_event,
                                                        sn_eff_features_per_event,
                                                        sn_eff_info_per_event, 
                                                        expected_bg_histogram,
                                                        expected_sn_shape_histogram,
                                                        expected_bg_hist_variations=expected_bg_histogram_variations,
                                                        number_of_tests=number_of_tests,
                                                        in_parallel=False)
                
                # plt.figure()
                # plt.hist(chi_squared_statistics, bins=50)
                # # Save the plot
                # plt.savefig(f"temp_pics/chi_squared_{spectrum_str}_{distance}.png")

                # plt.figure()
                # log_min = np.log10(np.min(p_values))
                # log_max = np.log10(np.max(p_values))
                # plt.hist(p_values, bins=np.logspace(log_min, log_max, 100))
                # plt.xscale('log')
                # #plt.yscale('log')
                # plt.axvline(x=fake_trigger_rate, color='r', linestyle='--')
                # plt.axvline(x=fake_trigger_rate*2, color='r', linestyle='--')
                # plt.savefig(f"temp_pics/p_values_{spectrum_str}_{distance}.png")

                # num_sampled_clusters_accepted_p_value = []
                # num_sampled_clusters_rejected_p_value = []
                # for i, p in enumerate(p_values):
                #     n_clusters = number_of_sampled_sn_clusters_list[i] + number_of_sampled_bg_clusters_list[i]
                #     if p < fake_trigger_rate:
                #         num_sampled_clusters_accepted_p_value.append(n_clusters)
                #     else:
                #         num_sampled_clusters_rejected_p_value.append(n_clusters)

                # plt.figure()
                # #plt.hist(number_of_sampled_bg_clusters_list, bins=50, alpha=0.5)
                # #plt.hist(number_of_sampled_sn_clusters_list, bins=50, alpha=0.5)
                # plt.hist(number_of_sampled_bg_clusters_list + number_of_sampled_sn_clusters_list, bins=50, alpha=0.5, label='total')
                # plt.hist(num_sampled_clusters_accepted_p_value, bins=50, alpha=0.5, label='accepted')
                # plt.hist(num_sampled_clusters_rejected_p_value, bins=50, alpha=0.5, label='rejected')
                # plt.axvline(x=expected_bg_cluster_number_in_time_window, color='r', linestyle='--')
                # plt.legend()
                # plt.savefig(f"temp_pics/cluster_numbers_{spectrum_str}_{distance}.png")

                # plt.figure()
                # plt.hist([num_sampled_clusters_accepted_p_value, num_sampled_clusters_rejected_p_value], stacked=True, bins=50)
                # plt.axvline(x=expected_bg_cluster_number_in_time_window, color='r', linestyle='--')
                # plt.savefig(f"temp_pics/cluster_numbers_stacked_{spectrum_str}_{distance}.png")

                # plt.figure()
                # plt.hist(log_likelihood_ratios, bins=50)
                # plt.savefig(f"temp_pics/log_likelihood_ratios_{spectrum_str}_{distance}.png")
                
                console.log(f"Trigger efficiency: {trigger_efficiency} for distance: {distance}")
                console.log(f"and spectrum: {spectrum}")
                console.log("\n")

                trigger_efficiencies_dict[spectrum_str].append(trigger_efficiency)
                chi_squared_statistics_dict[spectrum_str].append(chi_squared_statistics)
                p_values_dict[spectrum_str].append(p_values)
                success_rate_10th_percentile_dict[spectrum_str].append(success_rate_10th_percentile)
                success_rate_90th_percentile_dict[spectrum_str].append(success_rate_90th_percentile)

        # Set the output data
        self.output_data = {
            'trigger_efficiencies': trigger_efficiencies_dict,
            'chi_squared_statistics': chi_squared_statistics_dict,
            'p_values': p_values_dict,
            'expected_bg_histogram': expected_bg_histogram,
            'expected_bg_histogram_variations': expected_bg_histogram_variations,
        }

        # Set the info output data
        self.info_output_data = {
            'trigger_efficiencies': trigger_efficiencies_dict,
            'success_rate_10th_percentile': success_rate_10th_percentile_dict,
            'success_rate_90th_percentile': success_rate_90th_percentile_dict,
            'distance_to_evaluate': distance_to_evaluate,
            'bg_histogram_multiplier': bg_histogram_multiplier,
            'expected_bg_cluster_number_in_time_window': expected_bg_cluster_number_in_time_window,
            'expected_bg_histogram_bunched_bins': expected_bg_histogram.bunched_bins,
            'expected_bg_histogram_bunched_values': expected_bg_histogram.bunched_values,
            'number_of_tests': number_of_tests,
            'timestamp': f'{datetime.now()}'
        }

        return self.output_data, self.info_output_data


