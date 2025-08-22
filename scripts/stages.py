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
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")

        # Parameter to kill all wall-arapuca hits in the VD
        wall_veto = self.config.get("Simulation", "load_events", "wall_veto")
        # -----------------------------------------------
        
        # Load SN and BG hits for the parameter search
        # Create a dataloader object
        loader = dl.DataLoader(self.config, logging_level=logging.INFO)
        
        sn_hit_list_per_event, bg_hit_list_per_event, sn_info_per_event, bg_info_per_event,\
        sn_eff_hit_list_per_event, sn_bdt_hit_list_per_event, sn_eff_info_per_event, bg_eff_info_per_event,\
        sn_bdt_info_per_event, bg_eff_hit_list_per_event, bg_bdt_hit_list_per_event, bg_bdt_info_per_event, total_bg_time_window =\
            loader.load_and_split(sn_file_limit_parameter_search, bg_file_limit_parameter_search, shuffle=True, wall_veto=wall_veto)
        
        # Set the output data 
        self.output_data = {
            'sn_hit_list_per_event': sn_hit_list_per_event,
            'bg_hit_list_per_event': bg_hit_list_per_event,
            'sn_info_per_event': sn_info_per_event,
            'bg_info_per_event': bg_info_per_event,
            'sn_eff_hit_list_per_event': sn_eff_hit_list_per_event,
            'sn_bdt_hit_list_per_event': sn_bdt_hit_list_per_event,
            'sn_eff_info_per_event': sn_eff_info_per_event,
            'bg_eff_info_per_event': bg_eff_info_per_event,
            'sn_bdt_info_per_event': sn_bdt_info_per_event,
            'bg_bdt_info_per_event': bg_bdt_info_per_event,
            'bg_eff_hit_list_per_event': bg_eff_hit_list_per_event,
            'bg_bdt_hit_list_per_event': bg_bdt_hit_list_per_event
        }

        console.log("CHIPMONK")

        # Set the info output data
        self.info_output_data = {
            'memory_usage': {
                'bg_hits': np.sum([a.nbytes for a in bg_hit_list_per_event]) / 1024**2
            },
            'data_loading_statistics': {

                'bg_hit_num': np.sum([len(event) for event in bg_hit_list_per_event]),
                'bg_eff_hit_num': np.sum([len(event) for event in bg_eff_hit_list_per_event]),
                'bg_bdt_hit_num': np.sum([len(event) for event in bg_bdt_hit_list_per_event]),
                'bg_event_num': len(bg_hit_list_per_event),
                'bg_eff_event_num': len(bg_eff_hit_list_per_event),
                'bg_bdt_event_num': len(bg_bdt_hit_list_per_event),
                'total_bg_time_window': total_bg_time_window,
                'bg_eff_time_window': len(bg_eff_hit_list_per_event) * bg_sample_length,
                'bg_bdt_time_window': len(bg_bdt_hit_list_per_event) * bg_sample_length
            },
            'timestamp': f'{datetime.now()}'
        }
        for sn_channel in sn_channels:
            self.info_output_data['memory_usage'][f'sn_hits_{sn_channel}'] = np.sum([a.nbytes for a in sn_hit_list_per_event[sn_channel]]) / 1024**2
            self.info_output_data['memory_usage'][f'sn_info_{sn_channel}'] = get_total_size(sn_info_per_event[sn_channel]) / 1024**2 / 1024**2
            self.info_output_data['data_loading_statistics'][f'sn_hit_num_{sn_channel}'] = np.sum([len(event) for event in sn_hit_list_per_event[sn_channel]])
            self.info_output_data['data_loading_statistics'][f'sn_eff_hit_num_{sn_channel}'] = np.sum([len(event) for event in sn_eff_hit_list_per_event[sn_channel]])
            self.info_output_data['data_loading_statistics'][f'sn_bdt_hit_num_{sn_channel}'] = np.sum([len(event) for event in sn_bdt_hit_list_per_event[sn_channel]])
            self.info_output_data['data_loading_statistics'][f'sn_event_num_{sn_channel}'] = len(sn_info_per_event[sn_channel])
            self.info_output_data['data_loading_statistics'][f'sn_eff_event_num_{sn_channel}'] = len(sn_eff_info_per_event[sn_channel])
            self.info_output_data['data_loading_statistics'][f'sn_bdt_event_num_{sn_channel}'] = len(sn_bdt_info_per_event[sn_channel])
        
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
        # TODO: This is a bit of a hack. We should not need to read this from the config file (or put it in a separate section)
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
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

        # Compute the efficiency and training clusters for the SN (for each channel) and BG hits
        final_sn_eff_clusters = {}
        final_sn_eff_clusters_per_event = {}
        final_sn_eff_hit_multiplicities = {}
        final_sn_eff_hit_multiplicities_per_event = {}
        for sn_channel in sn_channels:
            (final_sn_eff_clusters[sn_channel], final_sn_eff_clusters_per_event[sn_channel], 
            final_sn_eff_hit_multiplicities[sn_channel], final_sn_eff_hit_multiplicities_per_event[sn_channel]) = clustering.group_clustering(
            sn_eff_hit_list_per_event[sn_channel], spatial_filter=True, data_type_str=f"SN efficiency ({sn_channel})", in_parallel=True, **clustering_parameters
        )
        (final_bg_eff_clusters, final_bg_eff_clusters_per_event, 
        final_bg_eff_hit_multiplicities, final_bg_eff_hit_multiplicities_per_event) = clustering.group_clustering(
            bg_eff_hit_list_per_event, spatial_filter=True, data_type_str="BG efficiency", in_parallel=True, **clustering_parameters
        )

        final_sn_bdt_clusters = {}
        final_sn_bdt_clusters_per_event = {}
        final_sn_bdt_hit_multiplicities = {}
        final_sn_bdt_hit_multiplicities_per_event = {}
        for sn_channel in sn_channels:
            (final_sn_bdt_clusters[sn_channel], final_sn_bdt_clusters_per_event[sn_channel],
            final_sn_bdt_hit_multiplicities[sn_channel], final_sn_bdt_hit_multiplicities_per_event[sn_channel]) = clustering.group_clustering(
            sn_bdt_hit_list_per_event[sn_channel], spatial_filter=True, data_type_str=f"SN BDT ({sn_channel})", in_parallel=True, **clustering_parameters
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
            'bg_eff_clusters_num': len(final_bg_eff_clusters),
            'bg_bdt_clusters_num': len(final_bg_bdt_clusters),
            'average_bg_eff_clusters_per_event': np.mean([len(event) for event in final_bg_eff_clusters_per_event]),
            'average_bg_bdt_clusters_per_event': np.mean([len(event) for event in final_bg_bdt_clusters_per_event]),
            'average_bg_eff_hit_multiplicity': np.mean(final_bg_eff_hit_multiplicities),
            'average_bg_bdt_hit_multiplicity': np.mean(final_bg_bdt_hit_multiplicities),
            'timestamp': f'{datetime.now()}'
        }
        for sn_channel in sn_channels:
            self.info_output_data[f'sn_eff_clusters_num_{sn_channel}'] = len(final_sn_eff_clusters[sn_channel])
            self.info_output_data[f'sn_bdt_clusters_num_{sn_channel}'] = len(final_sn_bdt_clusters[sn_channel])
            self.info_output_data[f'average_sn_eff_clusters_per_event_{sn_channel}'] = np.mean([len(event) for event in final_sn_eff_clusters_per_event[sn_channel]])
            self.info_output_data[f'average_sn_bdt_clusters_per_event_{sn_channel}'] = np.mean([len(event) for event in final_sn_bdt_clusters_per_event[sn_channel]])
            self.info_output_data[f'average_sn_eff_hit_multiplicity_{sn_channel}'] = np.mean(final_sn_eff_hit_multiplicities[sn_channel])
            self.info_output_data[f'average_sn_bdt_hit_multiplicity_{sn_channel}'] = np.mean(final_sn_bdt_hit_multiplicities[sn_channel])

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
        # TODO: hack to get the SN channels
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
        # -----------------------------------------------
        final_sn_eff_clusters_per_event = self.input_data['final_sn_eff_clusters_per_event']
        final_bg_eff_clusters_per_event = self.input_data['final_bg_eff_clusters_per_event']
        final_sn_bdt_clusters_per_event = self.input_data['final_sn_bdt_clusters_per_event']
        final_bg_bdt_clusters_per_event = self.input_data['final_bg_bdt_clusters_per_event']

        # Compute the features for the SN (per channel) and BG efficiency clusters
        sn_eff_features_array = {}
        sn_eff_features_per_event = {}
        sn_bdt_features_array = {}
        sn_bdt_features_per_event = {}        

        for sn_channel in sn_channels:
            sn_eff_features_array[sn_channel], sn_eff_features_per_event[sn_channel], _ = cl.group_compute_cluster_features(
                final_sn_eff_clusters_per_event[sn_channel], detector_type=detector_type, data_type_str=f"SN efficiency ({sn_channel})", 
                features_to_return=cluster_features, per_event=True)
            
            sn_bdt_features_array[sn_channel], sn_bdt_features_per_event[sn_channel], cluster_feature_names = cl.group_compute_cluster_features(
                final_sn_bdt_clusters_per_event[sn_channel], detector_type=detector_type, data_type_str=f"SN train ({sn_channel})", 
                features_to_return=cluster_features, per_event=True)
            
        bg_eff_features_array, bg_eff_features_per_event, _ = cl.group_compute_cluster_features(
            final_bg_eff_clusters_per_event, detector_type=detector_type, data_type_str="BG efficiency",
            features_to_return=cluster_features, per_event=True)
        
        bg_bdt_features_array, bg_bdt_features_per_event, _ = cl.group_compute_cluster_features(
            final_bg_bdt_clusters_per_event, detector_type=detector_type, data_type_str="BG train",
            features_to_return=cluster_features, per_event=True)
        
        # Set the targets
        sn_eff_targets = {}
        sn_bdt_targets = {}
        for sn_channel in sn_channels:
            sn_eff_targets[sn_channel] = np.ones(len(sn_eff_features_array[sn_channel]))
            sn_bdt_targets[sn_channel] = np.ones(len(sn_bdt_features_array[sn_channel]))
        bg_eff_targets = np.zeros(len(bg_eff_features_array))    
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
            'bg_eff_features_array_num': bg_eff_features_array.shape[0],
            'bg_bdt_features_array_num': bg_bdt_features_array.shape[0],
            'timestamp': f'{datetime.now()}'
        }
        for sn_channel in sn_channels:
            self.info_output_data[f'sn_eff_features_array_num_{sn_channel}'] = sn_eff_features_array[sn_channel].shape[0]
            self.info_output_data[f'sn_bdt_features_array_num_{sn_channel}'] = sn_bdt_features_array[sn_channel].shape[0]
        
        self.log.debug(f"sn_eff_features_array shape: {sn_eff_features_array[sn_channels[0]].shape}")
        self.log.debug(f"bg_eff_features_array shape: {bg_eff_features_array.shape}")
        self.log.debug(f"sn_bdt_features_array shape: {sn_bdt_features_array[sn_channels[0]].shape}")
        self.log.debug(f"bg_bdt_features_array shape: {bg_bdt_features_array.shape}")
        self.log.debug(f"sn_eff_features_per_event[0] (cc) shape: {sn_eff_features_per_event[sn_channels[0]][0].shape}")

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
        # TODO: hack to get the SN channels
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
        # -----------------------------------------------
        sn_bdt_features_array = self.input_data['sn_bdt_features_array']
        bg_bdt_features_array = self.input_data['bg_bdt_features_array']
        # You don't really need to read this, it's just a vector of ones/zeros
        # of the same length as sn_bdt_features_array/bg_bdt_features_array
        sn_bdt_targets = self.input_data['sn_bdt_targets'] 
        bg_bdt_targets = self.input_data['bg_bdt_targets']

        # If either the SN or BG features are empty or very small, raise an error. We cannot proceed any further
        if len(bg_bdt_features_array) <= 50:
            raise ValueError("The BG features are empty or contain very few entries. Cannot proceed with training.")
        for sn_channel in sn_channels:
            if len(sn_bdt_features_array[sn_channel]) <= 50:
                raise ValueError(f"The SN ({sn_channel}) features are empty or contain very few entries. Cannot proceed with training.")

        # Combine the features and targets
        features_bdt = np.vstack([sn_bdt_features_array[sn_channel] for sn_channel in sn_channels] + [bg_bdt_features_array])
        targets_bdt = np.concatenate([sn_bdt_targets[sn_channel] for sn_channel in sn_channels] + [bg_bdt_targets])

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
        self.setup_cross_sections()

        # Do we do reweighting?
        self.do_reweighting = self.config.get('Simulation', 'trigger_efficiency', 'do_reweighting')
        if self.do_reweighting:
            self.setup_reweighting()
        else:
            self.rw_weights = None
            self.rw_weights_discrete = None
            self.rw_weights_continuum = None
    
    def setup_reweighting(self):
        console.log(f"[bold green]Setting up reweighting...")

        # Import our bullshit python wrapper
        MARLEY_BASE_PATH = "/eos/user/p/pbarhama/marley-crpa/marley-crpa"
        import os, ctypes
        os.environ["MARLEY"] = MARLEY_BASE_PATH
        os.environ["MARLEY_DATA_PATH"] = f"{MARLEY_BASE_PATH}/data"
        # ctypes.CDLL(
        #     f"{MARLEY_BASE_PATH}/build/libMARLEY.so" # .dylib for mac, .so for linux
        # )
        # preload libMARLEY as a GLOBAL symbol provider
        libpath = os.path.join(MARLEY_BASE_PATH, "build", "libMARLEY.so")
        # use the OS‐provided constants
        flags = os.RTLD_GLOBAL | os.RTLD_NOW
        ctypes.CDLL(libpath, mode=flags)
        import py_marley
        # TODO: This only works for the CC channel right now 
        # (ES channel is never reweighted anyways, but RN it just breaks it)

        # Input reads ----------------------------------------------- 
        # 4 -> lepton momentum, 5 -> lepton dcosx, 6 -> lepton dcosy, 7 -> lepton dcosz, 0 -> neutrino energy
        sn_eff_info_per_event = self.input_data['sn_eff_info_per_event']

        base_path = 'marley_reaction_files'
        reaction_file_list_superallowed = [f"{base_path}/ve40ArCC_Bhattacharya2009-2.react"]
        reaction_file_list_discrete_only = [f"{base_path}/ve40ArCC_Bhattacharya2009-Discrete.react"]
        reaction_file_list_continuum_only = [f"{base_path}/ve40ArCC_HF-CRPA.react"]

        
        # Hard-coded for now
        unbound_threshold = 6.43847 # MeV for 40K
        omega_min = 1.504 # threshold for 40Ar CC reaction
        electron_mass = 0.511 # MeV
        mass_difference = omega_min - electron_mass # Difference in mass between 40K and 40Ar ground states
        omega_unbound_threshold = unbound_threshold + mass_difference # MeV

        #print(sn_eff_info_per_event['cc'].shape, "CHIPMUNCK")

        e_nu = sn_eff_info_per_event['cc'][:, 0]
        cos_theta = sn_eff_info_per_event['cc'][:, 5]
        p_lepton = sn_eff_info_per_event['cc'][:, 4] * 1000 # GeV -> MeV
        e_lepton = np.sqrt(p_lepton**2 + electron_mass**2)
        omega = e_nu - e_lepton
        # print(p_lepton, "PLEPTON")
        # print(e_lepton, "ELEPTON")
        # for i in range(len(e_nu)):
        #     print(e_nu[i], omega[i], e_lepton[i])

        # Get the indices of the events that have a neutrino energy greater than the unbound threshold
        b_mask = omega <= omega_unbound_threshold
        u_mask = omega > omega_unbound_threshold
        # Get a list of the indices
        b_indices = np.where(b_mask)[0]
        u_indices = np.where(u_mask)[0]

        # Get the discrete samples
        e_nu_discrete = e_nu[b_mask]
        cos_theta_discrete = cos_theta[b_mask]
        p_lepton_discrete = p_lepton[b_mask]
        e_lepton_discrete = e_lepton[b_mask]
        omega_discrete = omega[b_mask]
        # Get the continuum samples
        e_nu_continuum = e_nu[u_mask]
        cos_theta_continuum = cos_theta[u_mask]
        p_lepton_continuum = p_lepton[u_mask]
        e_lepton_continuum = e_lepton[u_mask]
        omega_continuum = omega[u_mask]
        
        # Get the cross-sections with the python library!
        # Import the python wrapper# Import our bullshit python wrapper
        MARLEY_BASE_PATH = "/eos/user/p/pbarhama/marley-crpa/marley-crpa"
        import os, ctypes
        os.environ["MARLEY"] = MARLEY_BASE_PATH
        os.environ["MARLEY_DATA_PATH"] = f"{MARLEY_BASE_PATH}/data"
        ctypes.CDLL(
            f"{MARLEY_BASE_PATH}/build/libMARLEY.so" # .dylib for mac, .so for linux
        )
        import py_marley
        # The above miraculously works, don't touch it!
        
        # ----------------------------------------------------------------
        # Discrete reweighting
        # ----------------------------------------------------------------
        single_diff_xsec_discrete_old = np.zeros_like(e_nu_discrete)
        single_diff_xsec_discrete_new = np.zeros_like(e_nu_discrete)

        print(len(e_nu_discrete), "len e_nu_discrete")
        print(len(e_nu_continuum), "len e_nu_continuum")
        print(len(e_nu), "len e_nu")

        # Let us define an array to save the possible excitation energies (i.e. the discrete levels)
        excitation_energies_list = None

        # "Old" cross-sections (for the model we have)
        for i, e in enumerate(e_nu_discrete):
            differential_xsec_list = py_marley.diff_xsec_files_discrete(
                reaction_file_list_superallowed,
                e,
                cos_theta_discrete[i],
                coulomb_mode='Fermi-MEMA',
                ff_scaling_mode='flat',
            superallowed=True,
            to_1e42_cm2=True,
            apply_lab_jacobian=True
            )
            # This is an array of tuples (excitation energy, xsec)
            # We need to find the one that matches with the omega_discrete[i]
            # The excitation energy is the difference between omega and the mass difference, barring 
            # the small recoil energy
            differential_xsecs = [dx[1] for dx in differential_xsec_list]
            excitation_energies = [dx[0] for dx in differential_xsec_list]

            if excitation_energies_list is None:
                excitation_energies_list = np.array(excitation_energies)

            # Find the index of the excitation energy that is closest to the approximate excitation energy
            # (we only have access to omega, but recoil is tiny)
            excitation_energy_approx = omega_discrete[i] - mass_difference
            idx = np.argmin(np.abs(excitation_energies - excitation_energy_approx))
            single_diff_xsec_discrete_old[i] = differential_xsecs[idx]

        #We have to compute the new cross sections in a separate loop,
        # because of the caching of the py_marley object
        for i, e in enumerate(e_nu_discrete):
            if i % 40000 == 0:
                print(f"{i} / {len(e_nu_discrete)}")

            differential_xsec_list = py_marley.diff_xsec_files_discrete(
                reaction_file_list_discrete_only,
                e,
                cos_theta_discrete[i],
                coulomb_mode='Fermi-MEMA',
                ff_scaling_mode='dipole',
                superallowed=False,
                to_1e42_cm2=True,
                apply_lab_jacobian=True
            )
            # This is an array of tuples (excitation energy, xsec)
            # We need to find the one that matches with the omega_discrete[i]
            # The excitation energy is the difference between omega and the mass difference, barring 
            # the small recoil energy
            differential_xsecs = [dx[1] for dx in differential_xsec_list]
            excitation_energies = [dx[0] for dx in differential_xsec_list]

            # Find the index of the excitation energy that is closest to the approximate excitation energy
            excitation_energy_approx = omega_discrete[i] - mass_difference
            idx = np.argmin(np.abs(excitation_energies - excitation_energy_approx))
            single_diff_xsec_discrete_new[i] = differential_xsecs[idx]

        weights_discrete = single_diff_xsec_discrete_new / single_diff_xsec_discrete_old
        # Plot the ratio of the cross sections (weights)
        # print(weights_discrete, "WD")
        # fig, ax = plt.subplots()
        # ax.scatter(e_nu_discrete, weights_discrete, label="New / Old", s=1)
        # ax.set_xlabel("Energy (MeV)")
        # ax.set_ylabel("Ratio")
        # plt.savefig("temp_pics/reweighting_discrete.png")
        # #exit("CHIPMUNCK")

        # ----------------------------------------------------------------
        # Continuum reweighting
        # ----------------------------------------------------------------
        single_diff_xsec_continuum_old = np.zeros_like(e_nu_continuum)
        double_diff_xsec_continuum_old = np.zeros_like(e_nu_continuum)
        double_diff_xsec_continuum_new = np.zeros_like(e_nu_continuum)

        # We wanna make some wrapping bins for the continuum part 
        excitation_energies_list_continuum = excitation_energies_list[excitation_energies_list > unbound_threshold]
        excitation_energies_bin_edges = (excitation_energies_list_continuum[:-1] + excitation_energies_list_continuum[1:]) / 2
        excitation_energies_bin_edges = np.insert(excitation_energies_bin_edges, 0, unbound_threshold)
        excitation_energies_bin_edges = np.append(excitation_energies_bin_edges, 70)

        # Compute old
        for i, e in enumerate(e_nu_continuum):
            if i % 40000 == 0:
                print(f"{i} / {len(e_nu_continuum)}")
            differential_xsec_list = py_marley.diff_xsec_files_discrete(
                reaction_file_list_superallowed,
                e,
                cos_theta_continuum[i],
                coulomb_mode='Fermi-MEMA',
                ff_scaling_mode='flat',
                superallowed=True,
                to_1e42_cm2=True,
                apply_lab_jacobian=True
            )
            # This is an array of tuples (excitation energy, xsec)
            # We need to find the one that matches with the omega_discrete[i]
            # The excitation energy is the difference between omega and the mass difference, barring 
            # the small recoil energy
            differential_xsecs = [dx[1] for dx in differential_xsec_list]
            excitation_energies = [dx[0] for dx in differential_xsec_list]

            # Find the index of the excitation energy that is closest to the approximate excitation energy
            excitation_energy_approx = omega_continuum[i] - mass_difference
            idx = np.argmin(np.abs(excitation_energies - excitation_energy_approx))

            # So, now we can divide by the bin width to get a "true" double-differential cross section
            idx_cont = np.where(excitation_energies_bin_edges > excitation_energy_approx)[0][0]
            bin_width = excitation_energies_bin_edges[idx_cont] - excitation_energies_bin_edges[idx_cont-1]
            single_diff_xsec_continuum_old[i] = differential_xsecs[idx]
            double_diff_xsec_continuum_old[i] = differential_xsecs[idx] / bin_width

            #print(excitation_energy_continuum[i], omega_continuum[i] - mass_difference, excitation_energies_bin_edges[idx_cont], excitation_energies_bin_edges[idx_cont-1])
            #print(differential_xsec_list[idx], excitation_energy_approx)

        # We have to compute the new cross sections in a separate loop,
        # because of the caching of the py_marley object
        # (and in this case it's just a different function lol)
        for i, e in enumerate(e_nu_continuum):
            if i % 40000 == 0:
                print(f"{i} / {len(e_nu_continuum)}")

            differential_xsec = py_marley.diff_xsec_files_d2(
                reaction_file_list_continuum_only,
                e,
                omega_continuum[i],
                cos_theta_continuum[i],
                coulomb_mode='Fermi-MEMA',
                ff_scaling_mode='dipole',
                crpa_discrete_mode='mirror',
                superallowed=False,
                to_1e42_cm2=True,
                apply_lab_jacobian=True
            )
            # This is not an array of tuples, it's just a single value
            double_diff_xsec_continuum_new[i] = differential_xsec
        
        weights_continuum = double_diff_xsec_continuum_new / double_diff_xsec_continuum_old
        # Plot the ratio of the cross sections (weights)
        # print(weights_continuum, "WC")
        # fig, ax = plt.subplots()
        # ax.scatter(e_nu_continuum, weights_continuum, label="New / Old", s=1)
        # ax.set_xlabel("Energy (MeV)")
        # ax.set_ylabel("Ratio")
        # ax.set_ylim(0, 25)
        # plt.savefig("temp_pics/reweighting_continuum.png")
        
        # exit("CHIPMUNCK")

        # Ok, now we have the continuum and discrete weights
        # We want to weave them into a single weight array, respecting the original ordering
        weights = np.zeros_like(e_nu)
        weights[b_mask] = weights_discrete
        weights[u_mask] = weights_continuum

        # Plot the weights
        fig, ax = plt.subplots()
        ax.scatter(e_nu, weights, label="New / Old", s=1)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0, 25)
        plt.savefig("temp_pics/reweighting_combined.png")

        # Do a cut?
        weights_max = 25
        weights_discrete[weights_discrete > weights_max] = 0
        weights_continuum[weights_continuum > weights_max] = 0
        weights[weights > weights_max] = 0

        self.rw_weights_discrete = weights_discrete
        self.rw_weights_continuum = weights_continuum
        self.rw_weights = weights

        console.log(f"[bold green]Reweighting setup complete.")
        #exit("CHIPMUNCK")

    def setup_supernova_spectra(self):
        # TODO: hack to get the SN channels
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
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
            
            if spectrum_type == "pinching":
                pinching_parameters = spectrum.get("pinching_parameters")
                interaction_number_10kpc = spectrum.get("interaction_number_10kpc")
                label = spectrum.get("label")
                supernova_spectra.append( SupernovaSpectrum.from_pinched_spectrum(**pinching_parameters,
                                                interaction_number_10kpc=interaction_number_10kpc, time_profile=time_profile,
                                                parameters={"model_name": label}) )
            elif spectrum_type == "model_name":
                model_name = spectrum.get("model_name")
                supernova_spectra.append( SupernovaSpectrum.from_model_name(model_name, time_profile=time_profile) )
            else:
                raise ValueError(f"Invalid supernova spectrum type: {spectrum_type}")
        
        self.supernova_spectra = supernova_spectra
    
    def setup_cross_sections(self):
        # Get the CC cross-section file
        cross_section_cc_file = self.config.get('Simulation', 'trigger_efficiency', 'physics', 'cross_section_cc_file')
        # For now, we used a fixed ES cross-section
        cross_section_es_file = "../cross-sections/xsec_sg_es_total.txt"
        # TODO: hack to get the SN channels
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
        
        # Load the cross-sections
        cross_sections = {}
        for sn_channel in sn_channels:
            if sn_channel == "cc":
                cross_sections[sn_channel] = np.loadtxt(cross_section_cc_file)
            elif sn_channel == "es":
                cross_sections[sn_channel] = np.loadtxt(cross_section_es_file)
            else:
                raise ValueError(f"Invalid SN channel: {sn_channel}")

        # Interpolate the cross-sections to the energy bins of the supernova spectra
        # TODO: We won't do this here, only when it's time to mulitply...
        # for spectrum in self.supernova_spectra:
        #     cross_sections["cc"] = np.interp(spectrum.energy_bins, cross_sections["cc"][:, 0], cross_sections["cc"][:, 1])
        #     cross_sections["es"] = np.interp(spectrum.energy_bins, cross_sections["es"][:, 0], cross_sections["es"][:, 1])
        
        self.cross_sections = cross_sections
    
    def run(self):
        # CONFIG READS ----- Load parameters from the configuration file
        true_tpc_size = self.config.get('Detector', 'true_tpc_size') * self.config.get('Detector', 'tpc_size_correction_factor')
        used_tpc_size = self.config.get('Detector', 'used_tpc_size') * self.config.get('Detector', 'tpc_size_correction_factor')
        tpc_size_correction_factor = self.config.get('Detector', 'tpc_size_correction_factor')
        bg_sample_length = self.config.get("DataFormat", "bg_sample_length")
        bg_sample_number_per_file = self.config.get("DataFormat", "bg_sample_number_per_file")
        sn_event_multiplier = self.config.get('Detector', 'sn_event_multiplier')

        do_reweighting = self.config.get('Simulation', 'trigger_efficiency', 'do_reweighting')

        # TODO: hack to get the SN channels
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")
        # ****
        # TODO: Fix this!!!
        distance_to_evaluate = self.config.get('Simulation', 'trigger_efficiency', 'distance_to_evaluate')
        number_of_interactions_to_evaluate = self.config.get('Simulation', 'trigger_efficiency', 'number_of_interactions_to_evaluate')
        energy_lower_limit = self.config.get('Simulation', 'trigger_efficiency', 'energy_lower_limit')

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
        if isinstance(number_of_interactions_to_evaluate, (int, float)):
            number_of_interactions_to_evaluate = [number_of_interactions_to_evaluate]
            number_of_interactions_to_evaluate = np.array(number_of_interactions_to_evaluate)

        # If we have too few clusters, the statistics will be unreliable. Raise an error
        for sn_channel in sn_channels:
            if len(sn_eff_features_array[sn_channel]) <= 50:
                raise ValueError(f"The SN ({sn_channel}) features are empty or contain very few events. Cannot proceed with trigger efficiency evaluation.")
        if len(bg_eff_features_array) <= 50:
            raise ValueError("The BG features are empty or contain very few events. Cannot proceed with trigger efficiency evaluation.")

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
                                                            classifier=hist_boosted_tree, rw_weights=self.rw_weights)
    
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

        if number_of_interactions_to_evaluate is not None:
            console.log(f"[bold yellow]Evaluating efficiency for number of interactions:")
            console.log(number_of_interactions_to_evaluate)
        else:
            console.log(f"[bold yellow]Evaluating efficiency for distances:")
            console.log(distance_to_evaluate)
        console.log(f"[bold yellow]Evaluating efficiency for spectra:")
        # Display supernova spectra information in a more readable format
        spectra_info = []
        for s in self.supernova_spectra:
            if isinstance(s, dict) and "cc" in s:
                spectra_info.append(str(s["cc"]))
            else:
                spectra_info.append(str(s))
        console.log(f"{spectra_info}")
        console.log("\n")

        trigger_efficiencies_dict = {}
        chi_squared_statistics_dict = {}
        p_values_dict = {}
        success_rate_10th_percentile_dict = {}
        success_rate_90th_percentile_dict = {}

        for spectrum in self.supernova_spectra:
            # Get the string representation of the spectrum
            # print(self.supernova_spectra, "CHIPMuNK")
            # print(spectrum, "CHIPMoNK")
            spectrum_str = spectrum["cc"].parameters.get("model_name")

            trigger_efficiencies_dict[spectrum_str] = []
            chi_squared_statistics_dict[spectrum_str] = []
            p_values_dict[spectrum_str] = []
            success_rate_10th_percentile_dict[spectrum_str] = []
            success_rate_90th_percentile_dict[spectrum_str] = []

            # Create the expected SN-weighted shape histogram
            # For now this will only be used in the log likelihood ratio calculation
            # We will only use the CC channel as it is dominant
            sn_eff_energies_per_cluster = []
            for i, clusters in enumerate(final_sn_eff_clusters_per_event["cc"]):
                sn_eff_energies_per_cluster.extend([sn_eff_info_per_event["cc"][i, 0]] * len(clusters))
                
            sn_spectrum_weights = weigh_by_supernova_spectrum(sn_eff_energies_per_cluster, spectrum["cc"], 
                                                              cross_section=self.cross_sections["cc"])
            expected_sn_shape_histogram = trigger_eff_computer.compute_histogram(
                        final_sn_eff_clusters["cc"], sn_eff_features_array["cc"], bins=expected_bg_histogram.bins,
                        bunched_bins=expected_bg_histogram.bunched_bins, multiplier=1.0, 
                        weights=sn_spectrum_weights)
            
            console.log(f"[bold yellow]Expected SN shape histogram:")
            print(plotille.hist_aggregated(expected_sn_shape_histogram.values, 
                                           expected_sn_shape_histogram.bins, 
                                           log_scale=False, lc='yellow'))
            
            iteration_list = number_of_interactions_to_evaluate if number_of_interactions_to_evaluate is not None else distance_to_evaluate

            reached_max_efficiency = False
            max_efficiency_tuple = None
            for val in iteration_list:

                distance = val if distance_to_evaluate is not None else None
                interaction_number = val if number_of_interactions_to_evaluate is not None else None

                if reached_max_efficiency:
                    (trigger_efficiency, results, chi_squared_statistics, p_values,
                        success_rate_10th_percentile, success_rate_90th_percentile,
                        number_of_sampled_sn_clusters_list,
                        number_of_sampled_bg_clusters_list,
                        expected_clusters_in_time_window,
                        weighted_sn_cluster_average) = max_efficiency_tuple
                else:
                    (trigger_efficiency, results, chi_squared_statistics, p_values,
                    success_rate_10th_percentile, success_rate_90th_percentile,
                    number_of_sampled_sn_clusters_list,
                    number_of_sampled_bg_clusters_list,
                    expected_clusters_in_time_window,
                    weighted_sn_cluster_average) = trigger_eff_computer.evaluate_trigger_efficiency(
                                                            spectrum, sn_channels, self.cross_sections,
                                                            distance, interaction_number,
                                                            energy_lower_limit,
                                                            corrected_tpc_size, sn_event_multiplier,
                                                            burst_time_window, fake_trigger_rate,
                                                            final_sn_eff_clusters_per_event,
                                                            sn_eff_features_per_event,
                                                            sn_eff_info_per_event, 
                                                            expected_bg_histogram,
                                                            expected_sn_shape_histogram,
                                                            expected_bg_hist_variations=expected_bg_histogram_variations,
                                                            number_of_tests=number_of_tests,
                                                            in_parallel=False)
                
                if not reached_max_efficiency:
                    if trigger_efficiency > 0.99999 and success_rate_10th_percentile > 0.99999 and success_rate_90th_percentile > 0.99999:
                        reached_max_efficiency = True
                        max_efficiency_tuple = (trigger_efficiency, results, chi_squared_statistics, p_values,
                                                success_rate_10th_percentile, success_rate_90th_percentile,
                                                number_of_sampled_sn_clusters_list,
                                                number_of_sampled_bg_clusters_list,
                                                expected_clusters_in_time_window,
                                                weighted_sn_cluster_average)
                        console.log(f"[bold green]Reached max efficiency: {trigger_efficiency} for value: {val} (model: {spectrum['cc'].parameters.get('model_name')})")
                    

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
                

                if distance is not None:
                    console.log(f"Trigger efficiency: {trigger_efficiency} for distance: {distance} (model: {spectrum['cc'].parameters.get('model_name')})")
                else:
                    console.log(f"Trigger efficiency: {trigger_efficiency} for number of interactions: {interaction_number} (model: {spectrum['cc'].parameters.get('model_name')})")
                console.log(f"and spectrum parameters: {spectrum['cc'].parameters}")
                console.log(f"using channels: {sn_channels}")
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
            'number_of_interactions_to_evaluate': number_of_interactions_to_evaluate,
            'bg_histogram_multiplier': bg_histogram_multiplier,
            'expected_bg_cluster_number_in_time_window': expected_bg_cluster_number_in_time_window,
            'expected_clusters_in_time_window': expected_clusters_in_time_window,
            'number_of_sampled_sn_clusters_list': number_of_sampled_sn_clusters_list,
            'number_of_sampled_bg_clusters_list': number_of_sampled_bg_clusters_list,
            'weighted_sn_cluster_average': weighted_sn_cluster_average,
            'expected_bg_histogram_bunched_bins': expected_bg_histogram.bunched_bins,
            'expected_bg_histogram_bunched_values': expected_bg_histogram.bunched_values,
            'number_of_tests': number_of_tests,
            'timestamp': f'{datetime.now()}'
        }

        return self.output_data, self.info_output_data


