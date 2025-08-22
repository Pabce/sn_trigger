'''
data_loader.py

This module...
'''

import sys
import numpy as np
import os
import multiprocessing as mp
import uproot
from itertools import repeat
import random
import string
import pickle
import logging
from sys import exit

from rich.logging import RichHandler
from rich.console import Console, Group
from rich.table import Table

import gui
from gui import console
from utils import get_total_size

class DataLoader:

    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

        # self.log.debug("This is a debug message")
        # self.log.info("Hello, World!")
        # self.log.warning("This is a warning")
        # self.log.error("This is an error")
        # self.log.exception("This is an exception")


    def load_and_split(self, sn_limit, bg_limit, shuffle=False, wall_veto=False):
        '''
        Load the signal and backgrounds and split them into train and efficiency computation sets
        '''

        # Config reads ----
        # ENERGY LIMIT TO TRAIN THE BDT (only usable if the BDT is used)
        split_for_classifier = self.config.get("Simulation", "load_events", "split_for_classifier")
        classifier_energy_limit = self.config.get("Simulation", "load_events", "classifier_energy_limit") if split_for_classifier else 1e5
        train_split = self.config.get("Simulation", "load_events", "train_split")
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")

        bg_sample_length = self.config.get("DataFormat", "bg_sample_length")
        bg_sample_number_per_file = self.config.get("DataFormat", "bg_sample_number_per_file")
        load_data_in_parallel = self.config.get("Simulation", "load_events", "load_data_in_parallel")
        # -----------------


        self.log.info(f"Loading SN data...")
        # These are now dictionaries, one entry per channel
        sn_total_hits, sn_hit_list_per_event, sn_info_per_event, _, _ = self.load_all_sn_events_chunky(limit=sn_limit, 
                                                                                                        load_data_in_parallel=load_data_in_parallel)
        self.log.info(f"SN data loaded")
        
        self.log.info("Loading BG data...")
        bg_total_hits, bg_hit_list_per_event, bg_info_per_event, _, _= self.load_all_backgrounds_chunky_type_separated(limit=bg_limit, 
                                                                                                    load_data_in_parallel=load_data_in_parallel)
        total_bg_time_window = len(bg_hit_list_per_event) * bg_sample_length # in miliseconds
        self.log.info("BG data loaded")

        del _
        #del sn_total_hits, bg_total_hits

        # Do this for convenient shuffling/energy filtering later
        for sn_channel in sn_channels:
            sn_info_per_event[sn_channel] = np.array(sn_info_per_event[sn_channel])
            sn_hit_list_per_event[sn_channel] = np.array(sn_hit_list_per_event[sn_channel], dtype=object)
        # Convert BG lists to numpy arrays for shuffling
        bg_hit_list_per_event = np.array(bg_hit_list_per_event, dtype=object)
        bg_info_per_event = np.array(bg_info_per_event, dtype=object)

        if shuffle:
            # Shuffle the SN and BG events. This seems especially important as the parallel loading of the BG files messes things up
            # TODO: better investigate this
            # TODO: move the shuffling to the loading functions
            # Shuffle the SN events
            for sn_channel in sn_channels:
                sn_shuffled_indices = np.arange(len(sn_info_per_event[sn_channel]))
                np.random.shuffle(sn_shuffled_indices)
                sn_info_per_event[sn_channel] = sn_info_per_event[sn_channel][sn_shuffled_indices]
                sn_hit_list_per_event[sn_channel] = sn_hit_list_per_event[sn_channel][sn_shuffled_indices]

            # Shuffle the BG events
            bg_shuffled_indices = np.arange(len(bg_info_per_event))
            np.random.shuffle(bg_shuffled_indices)
            bg_hit_list_per_event = bg_hit_list_per_event[bg_shuffled_indices]
            bg_info_per_event = bg_info_per_event[bg_shuffled_indices]
        
        if wall_veto:
            console.log("Applying wall veto")
            # Kill all wall-arapuca hits in the VD
            for sn_channel in sn_channels:
                for i, hit_list in enumerate(sn_hit_list_per_event[sn_channel]):
                    if hit_list.shape[0] == 0:
                        continue
                    x_hit_coordinates = hit_list[:, 4]
                    x_mask = x_hit_coordinates < 0
                    sn_hit_list_per_event[sn_channel][i] = hit_list[x_mask, :]
                    # You don't need to modify sn_info_per_event, as it just contains the underlying SN event truth information
                    # print(f"SN {sn_channel}", x_mask.sum(), len(hit_list), len(sn_hit_list_per_event[sn_channel][i]))
                    # print(sn_hit_list_per_event[sn_channel][i][:, 4:7])
            # Do the same for the BG events
            for i, hit_list in enumerate(bg_hit_list_per_event):
                if hit_list.shape[0] == 0:
                    continue
                x_hit_coordinates = hit_list[:, 4]
                x_mask = x_hit_coordinates < 0
                bg_hit_list_per_event[i] = hit_list[x_mask, :]
                # In this case, you DO need to modify bg_info_per_event, as it contains the background type information PER HIT
                #print(bg_info_per_event[i])
                bg_info_per_event[i] = np.array(bg_info_per_event[i])[x_mask]
                #print("bg", x_mask.sum(), len(hit_list), len(bg_hit_list_per_event[i]))

        # We need to spit the SN and BG events for training the BDT and efficiency evaluation 
        # (this is not a train/test split, that is done in the BDT training later)
        split_point_sn = {}
        if split_for_classifier:
            for sn_channel in sn_channels:
                split_point_sn[sn_channel] = int(len(sn_info_per_event[sn_channel]) * train_split)
            split_point_bg = int(len(bg_hit_list_per_event) * train_split)
        else:
            for sn_channel in sn_channels:
                split_point_sn[sn_channel] = 0
            split_point_bg = 0

        sn_bdt_info_per_event = {}
        sn_bdt_hit_list_per_event = {}
        sn_eff_info_per_event = {}
        sn_eff_hit_list_per_event = {}
        for sn_channel in sn_channels:
            sn_bdt_info_per_event[sn_channel] = sn_info_per_event[sn_channel][:split_point_sn[sn_channel]]
            sn_bdt_hit_list_per_event[sn_channel] = sn_hit_list_per_event[sn_channel][:split_point_sn[sn_channel]] 

            # Filter the SN events that are above the BDT energy limit
            sn_bdt_hit_list_per_event[sn_channel] = sn_bdt_hit_list_per_event[sn_channel][sn_bdt_info_per_event[sn_channel][:, 0] < classifier_energy_limit]
            sn_bdt_info_per_event[sn_channel] = sn_bdt_info_per_event[sn_channel][sn_bdt_info_per_event[sn_channel][:, 0] < classifier_energy_limit]

            # Do the second split
            sn_eff_info_per_event[sn_channel] = sn_info_per_event[sn_channel][split_point_sn[sn_channel]:]
            sn_eff_hit_list_per_event[sn_channel] = sn_hit_list_per_event[sn_channel][split_point_sn[sn_channel]:]
        
        bg_bdt_hit_list_per_event = bg_hit_list_per_event[:split_point_bg]
        bg_bdt_info_per_event = bg_info_per_event[:split_point_bg]
        bg_eff_hit_list_per_event = bg_hit_list_per_event[split_point_bg:]
        bg_eff_info_per_event = bg_info_per_event[split_point_bg:]
        #console.log(split_point_bg, len(bg_hit_list_per_event))

        return (sn_hit_list_per_event, bg_hit_list_per_event, sn_info_per_event, bg_info_per_event,
                sn_eff_hit_list_per_event, sn_bdt_hit_list_per_event, sn_eff_info_per_event, bg_eff_info_per_event,
                sn_bdt_info_per_event, bg_eff_hit_list_per_event, bg_bdt_hit_list_per_event, bg_bdt_info_per_event,
                total_bg_time_window)


    def load_all_backgrounds_chunky_type_separated(self, limit=1, offset=0, load_data_in_parallel=True):
        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        bg_data_dir = self.config.get("Simulation", "load_events", "bg_data_dir")
        sim_mode = self.config.get('Simulation', 'load_events', 'sim_mode')
        bg_types = self.config.get("Backgrounds")
        bg_sample_number_per_file = self.config.get("DataFormat", "bg_sample_number_per_file")
        startswith = self.config.get("Simulation", "load_events", "bg_hit_file_start_pattern")
        endswith = self.config.get("Simulation", "load_events", "bg_hit_file_end_pattern").format(sim_mode)
        # -----------------
        # We might have some backgrounds to multiply by a factor
        modified_bgs = self.config.get("Simulation", "load_events", "modified_bgs")
        # -------------------------------------------------------------------------

        bg_total_hits = []
        bg_hit_list_per_event = []
        bg_info_per_event = []

        bg_total_hits_per_type = {bg_type: [] for bg_type in bg_types}
        bg_hit_list_per_event_per_type = {bg_type: [] for bg_type in bg_types}
        
        # Get the directories for each background type
        directories = []
        for bg_type in bg_types:
            #directory = os.fsencode(bg_data_dir + bg_type + '/')
            directory = bg_data_dir + bg_type + '/'
            directories.append(directory)

        # Collect valid file names for each background type
        reco_file_names_per_type = []
        for i, directory in enumerate(directories):
            bg_type = bg_types[i]
            
            reco_file_names, _, _ = self.collect_valid_file_names(
                directory,
                startswith,
                endswith,
                limit=limit,
                offset=offset,
                data_type_str=f'{bg_type} BG data',
            )

            reco_file_names_per_type.append(reco_file_names)
        
        for i, directory in enumerate(directories):
            bg_type = bg_types[i]
            
            # Process files in parallel
            results = self.process_files(
                reco_file_names_per_type[i],
                event_num=bg_sample_number_per_file,
                sn_event=False,
                data_type_str=f'{bg_type} BG data',
                in_parallel=load_data_in_parallel
            ) 

            # Accumulate results
            (
                bg_total_hits_i,
                bg_hit_list_per_event_i,
                _, _, _,
            ) = self.accumulate_results(results, load_photon_info=False, sn_event=False)

            # If we have a modified background, just keep a random fraction of the hits
            if modified_bgs is not None:
                if bg_type in modified_bgs.keys():
                    keep_fraction = modified_bgs[bg_type]
                    console.log("BG type: ", bg_type)
                    console.log("Keep fraction: ", keep_fraction)
                    bg_total_hits_i = []
                    
                    for j in range(len(bg_hit_list_per_event_i)):
                        num_original_hits = len(bg_hit_list_per_event_i[j])
                        num_hits_to_keep = int(num_original_hits * keep_fraction)
                        bg_hit_list_per_event_i[j] = bg_hit_list_per_event_i[j][:num_hits_to_keep]

                        # console.log("BG type: ", bg_type)
                        # console.log(f"Keeping {num_hits_to_keep} hits from {num_original_hits} hits")

                        bg_total_hits_i.extend(bg_hit_list_per_event_i[j])

            bg_total_hits_per_type[bg_type].append(bg_total_hits_i)
            bg_hit_list_per_event_per_type[bg_type].extend(bg_hit_list_per_event_i)

            # Concatenate the total hit lists per type
            try:
                bg_total_hits_per_type[bg_type] = np.concatenate(bg_total_hits_per_type[bg_type], axis=0)
                self.log.debug("Loaded {} background hits".format(bg_type))
            except ValueError as e:
                self.log.warning("No background of type {} found".format(bg_type))


        # Create the total hit list per event, combining all bakcgrounds
        for i, _ in enumerate(bg_hit_list_per_event_per_type[bg_types[0]]):
            hit_list = []
            info_list = []
            for j, bg_type in enumerate(bg_types):
                hit_list.extend(bg_hit_list_per_event_per_type[bg_type][i])
                info_list.extend([bg_types[j]] * len(bg_hit_list_per_event_per_type[bg_type][i]))

            # Convert to numpy arrays for consistent indexing/sorting
            hit_list = np.array(hit_list)
            info_list = np.array(info_list)

            # Sort both the hits and their corresponding background-type labels by time so that
            # the mapping "hit - bg_type" is preserved after the time ordering.
            # NOTE: the original implementation only reordered `hit_list`, leaving `info_list`
            # unsorted, which broke the alignment and resulted in every bg_type apparently having
            # the same spatial distribution.  We fix that by applying the same permutation
            # (`time_sort`) to `info_list`.
            if hit_list.shape != (0,):
                time_sort = np.argsort(hit_list[:, 3])  # sort by time column
                hit_list = hit_list[time_sort, :]
                info_list = info_list[time_sort]

            bg_info_per_event.append(info_list.tolist())
            bg_hit_list_per_event.append(hit_list)
            bg_total_hits.extend(hit_list)
            
        # bg_total_hits: List[np.ndarray] - List containing all hits from all background events, potentially mixed types. Each element is one hit array.
        # bg_hit_list_per_event: List[np.ndarray] - List where each element is a numpy array of time-ordered hits for a single combined background event.
        # bg_total_hits_per_type: Dict[str, np.ndarray] - Dictionary mapping background type string to a single numpy array containing all hits for that type.
        # bg_hit_list_per_event_per_type: Dict[str, List[np.ndarray]] - Dictionary mapping background type string to a list of numpy arrays, where each array contains hits for one event of that specific type.
        return (
            bg_total_hits,
            bg_hit_list_per_event,
            bg_info_per_event,
            bg_total_hits_per_type,
            bg_hit_list_per_event_per_type
        )


    # Load SN events hits and neutrino (MC truth) information
    def load_all_sn_events_chunky(self, load_photon_info=False, limit=1, offset=0, load_data_in_parallel=True):
        """
        Load Supernova event hits and neutrino (MC truth) information.

        Args:
            load_photon_info (bool): Whether to load photon information.
            limit (int): Maximum number of events to load.
            event_num (int): Number of events per file to consider.
            offset (int): Number of files to skip before starting.
            verbose (int): Verbosity level.

        Returns:
            tuple: (sn_total_hits, sn_hit_list_per_event, sn_total_info,
                    sn_total_info_per_event, total_photons_per_channel)
        """

        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        event_num = self.config.get("DataFormat", "sn_event_number_per_file")
        sn_channels = self.config.get("Simulation", "load_events", "sn_channels")

        sn_data_dir = self.config.get("Simulation", "load_events", "sn_data_dir")
        sim_mode = self.config.get('Simulation', 'load_events', 'sim_mode')
        startswith = self.config.get("Simulation", "load_events", "sn_hit_file_start_pattern")
        endswith = self.config.get("Simulation", "load_events", "sn_hit_file_end_pattern")
        endswith = [el.format(sim_mode) for el in endswith]
        info_endswith = self.config.get("Simulation", "load_events", "sn_info_file_end_pattern")
        photon_endswith = self.config.get("Simulation", "load_events", "photon_file_end_pattern")
        # -----------------

        sn_total_hits = {}
        sn_hit_list_per_event = {}
        sn_total_info = {}
        sn_total_info_per_event = {}
        total_photons_per_channel = {}

        for i, sn_channel in enumerate(sn_channels):
            # Collect valid file names
            #console.log(sn_channel, sn_data_dir)
            reco_file_names, info_file_names, photon_file_names = self.collect_valid_file_names(
                sn_data_dir[i],
                startswith[i],
                endswith[i],
                info_endswith[i],
                event_num,
                photon_endswith[i],
                load_photon_info,
                limit[i],
                offset,
                data_type_str=f"{sn_channel} SN data"
            )

            # Process files in parallel
            results = self.process_files(
                reco_file_names,
                info_file_names,
                event_num,
                load_photon_info,
                photon_file_names,
                sn_event=True,
                data_type_str=f"{sn_channel} SN data",
                in_parallel=load_data_in_parallel
            )

            # Accumulate results
            (
                sn_total_hits[sn_channel],
                sn_hit_list_per_event[sn_channel],
                sn_total_info[sn_channel],
                sn_total_info_per_event[sn_channel],
                total_photons_per_channel[sn_channel],
            ) = self.accumulate_results(results, load_photon_info, sn_event=True)

        return (
            sn_total_hits,
            sn_hit_list_per_event,
            sn_total_info,
            sn_total_info_per_event,
            total_photons_per_channel,
        )

    def collect_valid_file_names(self, data_dir, startswith, endswith, 
                                info_endswith=None, event_num=1000, 
                                photon_endswith=None, load_photon_info=False, 
                                limit=1, offset=0, data_type_str="Chipmonc"):
        """
        Collect valid file names from the SN or BG data directory.

        Args:
            data_dir (str): Path to the SN/BG data directory.
            startswith (str): Start pattern for hit files.
            endswith (str): End pattern for hit files.
            info_endswith (str): End pattern for info files (*ana files).
            photon_endswith (str): End pattern for photon files.
            load_photon_info (bool): Whether to load photon info.
            limit (int): Maximum number of files to collect.
            offset (int): Number of files to skip before starting.
            event_num (int): Number of events per file to consider.
            verbose (int): Verbosity level.

        Returns:
            tuple: Lists of hit filenames, info filenames, and photon filenames.
        """
        reco_file_names = []
        info_file_names = []
        photon_file_names = []

        processed_files = 0

        status_fstring = f'[bold green] Verifying {data_type_str}.'
        with gui.live_progress(console=console, status_fstring=status_fstring) as (progress, live, group):
            task = progress.add_task(f'[cyan]Finding {data_type_str} files and verifying integrity...', total=limit)
            for filename in os.listdir(data_dir):
                filename = os.fsdecode(filename)

                if not (filename.startswith(startswith) and filename.endswith(endswith)):
                    continue

                if processed_files < offset:
                    processed_files += 1
                    continue

                self.log.debug(f"Validating file: {filename}")

                # Check if hit file contains required tree
                reco_filename = data_dir + filename
                try:
                    with uproot.open(reco_filename) as uproot_file:
                        uproot_ophit = uproot_file["opflashana/PerOpHitTree"]
                except uproot.exceptions.KeyInFileError as e:
                    logging.warning(f"Corrupted *reco file {filename}: {e}. Skipping.")
                    continue

                # Check if corresponding *ana file exists (if required)
                if info_endswith:
                    info_filename = data_dir + filename.replace(endswith, info_endswith)
                    if not os.path.isfile(info_filename):
                        self.log.warning(f"No corresponding *ana file for {filename}. Skipping.")
                        continue

                    # Check if ana file contains required tree
                    try:
                        with uproot.open(info_filename) as uproot_file:
                            uproot_neutrino = uproot_file["analysistree/anatree"]
                            # Skip if the tree has less entries than the event number we need
                            if len(uproot_neutrino['enu_truth'].array()) < event_num:
                                raise uproot.exceptions.KeyInFileError("Not enough events in the tree")
                    except uproot.exceptions.KeyInFileError as e:
                        logging.warning(f"Corrupted *ana file {info_filename}: {e}. Skipping.")
                        continue
                
                # Check if corresponding photon file exists (if required)
                photon_filename = None
                if load_photon_info:
                    photon_filename = os.path.join(
                        data_dir, filename.replace(endswith, photon_endswith)
                    )
                    if not os.path.isfile(photon_filename):
                        self.log.warning(f"No corresponding *photon file for {filename}. Skipping.")
                        continue
                    # Check if photon file contains required tree
                    try:
                        with uproot.open(photon_filename) as uproot_file:
                            uproot_photon = uproot_file["simphotonsana/PhotonsTree"]
                    except Exception as e:
                        self.log.warning(f"Corrupted *photon file {photon_filename}: {e}. Skipping.")
                        continue
                
                # --------------------------------

                processed_files += 1
                if processed_files > limit + offset:
                    break
                
                # Add files to lists
                reco_file_names.append(reco_filename)
                if info_endswith:
                    info_file_names.append(info_filename)
                if load_photon_info:
                    photon_file_names.append(photon_filename)

                progress.update(task, advance=1)

            # Create a new group with just the progress bar
            group = Group(progress)
            # Update the live display with the new group
            live.update(group)

        return reco_file_names, info_file_names, photon_file_names

    def process_files(self, reco_file_names, info_file_names=None, event_num=1000,
                        load_photon_info=False, photon_file_names=None, 
                        sn_event=True, data_type_str="Chipmonc", in_parallel=False):
        """
        Process the collected files in parallel.

        Args:
            file_names (list): List of hit filenames.
            info_file_names (list): List of info filenames.
            photon_file_names (list): List of photon filenames.
            event_num (int): Number of events per file to consider.
            load_photon_info (bool): Whether to load photon info.
            verbose (int): Verbosity level.

        Returns:
            list: Results from processing each file.
        """
        
        status_fstring = f'[bold green] Loading {data_type_str}.'
        results = []

        # For consistency in the loading order, and for testing purposes, we may load the files in a non-parallel way
        with gui.live_progress(console=console, status_fstring=status_fstring) as (progress, live, group):
            if not in_parallel:
                task = progress.add_task(f'[cyan]Loading {data_type_str} (NOT in parallel)...', total=len(reco_file_names))
                
                for i, reco_file_name in enumerate(reco_file_names):
                    info_file_name = info_file_names[i] if info_file_names else None
                    photon_file_name = photon_file_names[i] if photon_file_names else None

                    result = self.load_hit_and_neutrino_info(reco_file_name, info_file_name, event_num, load_photon_info, photon_file_name, sn_event)
                    progress.update(task, advance=1)
                    results.append(result)
                
            else:
                task = progress.add_task(f'[cyan]Loading {data_type_str} (in parallel)...', total=len(reco_file_names))
                num_processes = mp.cpu_count()
                with mp.Pool(num_processes) as pool:
                    results = []
                    
                    # Prepare arguments for parallel processing
                    args = zip(
                        reco_file_names,
                        info_file_names if info_file_names else repeat(None),
                        repeat(event_num),
                        repeat(load_photon_info),
                        photon_file_names if photon_file_names else repeat(None),
                        repeat(sn_event)
                    )

                    imap_results = pool.imap_unordered(self.wrapped_load_hit_and_neutrino_info, args)

                    for result in imap_results:
                        results.append(result)
                        progress.update(task, advance=1)
            
            # Create a new group with just the progress bar
            # Update the live display with the new group
            group = Group(progress)
            live.update(group)

        return results
        

    def accumulate_results(self, results, load_photon_info, sn_event=True):
        """
        Accumulate results from processing files.

        Args:
            results (list): List of results from each file.
            load_photon_info (bool): Whether photon info was loaded.

        Returns:
            tuple: Accumulated data arrays and lists.
        """
        total_hits = []
        hit_list_per_event = []
        sn_total_info = []
        sn_total_info_per_event = []
        total_photons_per_channel = []

        for result in results:
            (
                total_hits_i,
                hit_list_per_event_i,
                _,
                sn_info_i,
                sn_info_per_event_i,
                photons_per_channel,
            ) = result

            total_hits.append(total_hits_i)
            hit_list_per_event.extend(hit_list_per_event_i)

            if sn_event:
                sn_total_info.append(sn_info_i)
                sn_total_info_per_event.extend(sn_info_per_event_i)

            if load_photon_info and photons_per_channel is not None:
                total_photons_per_channel.append(photons_per_channel)

        total_hits = np.concatenate(total_hits, axis=0)
        if sn_event:
            sn_total_info = np.concatenate(sn_total_info, axis=0)
        else:
            sn_total_info = None

        if load_photon_info and total_photons_per_channel:
            total_photons_per_channel = np.concatenate(total_photons_per_channel, axis=0)
        else:
            total_photons_per_channel = None

        return (
            total_hits,
            hit_list_per_event,
            sn_total_info,
            sn_total_info_per_event,
            total_photons_per_channel,
        )



    def load_hit_and_neutrino_info(self, reco_file_name, info_file_name=None, event_num=-1, load_photon_info=False, photon_file_name=None, sn_event=True):
        total_hits, hit_list_per_event, _ = self.load_hit_data(reco_file_name, event_num)
        sn_info, sn_info_per_event = None, None

        if sn_event:
            sn_info, sn_info_per_event = self.load_neutrino_info(info_file_name)

        if load_photon_info:
            photons_per_channel = self.load_g4_photon_data(photon_file_name, event_num)

            return total_hits, hit_list_per_event, None, sn_info, sn_info_per_event, photons_per_channel

        return total_hits, hit_list_per_event, None, sn_info, sn_info_per_event, None

    def wrapped_load_hit_and_neutrino_info(self, arg_list):
            return self.load_hit_and_neutrino_info(*arg_list)
    
    def load_hit_data(self, reco_file_name, event_num=-1):
        # The hit info is imported from a .root file.
        # From the hit info, we find the clusters

        # Keep it simple for now (shape: X*7)
        # X and Y coordinates are switched!!!

        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        x_coords = self.config.loaded["x_coords"]
        y_coords = self.config.loaded["y_coords"]
        z_coords = self.config.loaded["z_coords"]
        # ----------------- 

        # Disabling loging from multiprocessing functions right now due to isssues with rich
        # TODO: Fix this
        #self.log.debug(f"Loading hit data from file: {file_name}")

        # TODO: fix this bullshit for Laura's files
        try:
            with uproot.open(reco_file_name) as uproot_file:
                uproot_ophit = uproot_file["opflashana/PerOpHitTree"]
                total_hits_dict = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime', 'X_OpDet_hit', 'Y_OpDet_hit', 'Z_OpDet_hit',
                                                        'Width', 'Area', 'Amplitude', 'PE'], library="np")
                total_hits = [total_hits_dict[key] for key in total_hits_dict]
                total_hits = np.array(total_hits).T

        except uproot.exceptions.KeyInFileError:
            with uproot.open(reco_file_name) as uproot_file:
                uproot_ophit = uproot_file["opflashana/PerOpHitTree"]
                total_hits_dict_1 = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime'], library="np")

                # print(np.sort(np.unique(total_hits_dict_1['OpChannel'])), len(np.sort(np.unique(total_hits_dict_1['OpChannel']))))
                # exit()

                # Edit the OpChannel array for the new configuration of two channels per X-ARAPUCA (for VD only).
                # (What used to be OpChannel 43 is now 430 and 431, etc.)
                if detector_type == "VD":
                    total_hits_dict_1['OpChannel'] = (total_hits_dict_1['OpChannel'] // 10).astype(int)
                    
                total_hits_dict_1['X_OpDet_hit'] = x_coords[total_hits_dict_1['OpChannel']]
                total_hits_dict_1['Y_OpDet_hit'] = y_coords[total_hits_dict_1['OpChannel']]
                total_hits_dict_1['Z_OpDet_hit'] = z_coords[total_hits_dict_1['OpChannel']]

                total_hits_dict_2 = uproot_ophit.arrays(['Width', 'Area', 'Amplitude', 'PE'], library="np")

                # Merge the two dictionaries preserving the order of the keys
                total_hits_dict = {**total_hits_dict_1, **total_hits_dict_2}

                total_hits = [total_hits_dict[key] for key in total_hits_dict]
                total_hits = np.array(total_hits).T

        hit_list_per_event, hit_num_per_channel = self.process_hit_data(total_hits, event_num)

        return total_hits, hit_list_per_event, hit_num_per_channel


    # Load neutrino (MC truth) info for SN events
    # TODO: Make sn_info_per_event a dictionary
    @staticmethod
    def load_neutrino_info(file_name):
        info_per_event = []

        # uproot_neutrino = uproot.open(file_name)["vdflashmatch/FlashMatchTree"]
        # info_dict = uproot_neutrino.arrays(['TrueE', 'TrueX', 'TrueY', 'TrueZ'], library="np")
        uproot_neutrino = uproot.open(file_name)["analysistree"]
        info_dict = uproot_neutrino["anatree"].arrays(['enu_truth', 'nuvtxx_truth', 
                                                       'nuvtxy_truth', 'nuvtxz_truth',
                                                       'lep_mom_truth', 'lep_dcosx_truth',
                                                       'lep_dcosy_truth', 'lep_dcosz_truth'], library="np")

        info = [np.concatenate(info_dict[key]) for key in info_dict]
        info_per_event_arr = np.array(info).T

        info_per_event_arr[:, 0] *= 1000 # Convert to MeV

        for i in range(info_per_event_arr.shape[0]):
            info_per_event.append(info_per_event_arr[i, :])

        return info_per_event_arr, info_per_event


    def process_hit_data(self, total_hits, event_num=-1):
        if event_num == -1:
            try:
                event_num = int(np.max(total_hits[:, 0]))
            except ValueError:
                self.log.debug("No events found!")
                return [[]], []

        hit_number_per_event = np.zeros(event_num)
        hit_list_per_event = [[] for _ in range(event_num)]
        # detection_per_event = np.zeros(event_num)

        for i in range(len(total_hits)):
            event_id = total_hits[i, 0]
            ev = int(event_id) - 1

            if ev >= event_num:
                continue

            hit_number_per_event[ev] += 1

            hit_list_per_event[ev].append(total_hits[i, :])

        for i in range(event_num):
            hit_list_per_event[i] = np.array(hit_list_per_event[i])

            # We order the hits in time
            if hit_list_per_event[i].shape != (0,):
                time_sort = np.argsort(hit_list_per_event[i][:,3])
                hit_list_per_event[i] = hit_list_per_event[i][time_sort, :]


        unique, counts = np.unique(total_hits[:, 2], return_counts=True)
        hit_num_per_channel = dict(zip(unique, counts))

        return hit_list_per_event, hit_num_per_channel


    # TIME PROFILE IS IN MS!!!!
    def load_time_profile(self):
        data_dir = self.config.get("DataFormat", "aux_data_dir")
        uproot_time_profile = uproot.open(data_dir + "TimeProfile.root")["h_MarlTime"]
        uproot_time_profile_zoomed = uproot.open(data_dir + "TimeProfile.root")["h_MarlTimeZoom"]
        time_profile = uproot_time_profile.to_numpy()
        time_profile_zoomed = uproot_time_profile_zoomed.to_numpy()

        # Time to do some stiching!
        # Step 1: Find the total number of events in the zoomed histogram
        total_zoomed = np.sum(time_profile_zoomed[0])
        # Step 2: Find the total number of events in the first bin of the non-zoomed histogram
        first_bin_non_zoomed = time_profile[0][0]
        # Step 3: Remove the total number of events in the zoomed histogram from the first bin of the non-zoomed histogram
        time_profile[0][0] = first_bin_non_zoomed - total_zoomed
        # Step 4: stitch the two histograms together
        new_time_profile_vals = np.concatenate((time_profile_zoomed[0], time_profile[0]))
        new_time_profile_bins = np.concatenate((time_profile_zoomed[1], time_profile[1][1:]))

        time_profile = (new_time_profile_vals, new_time_profile_bins)

        # print(time_profile[0].shape)
        # print(time_profile[1].shape)
        # plt.scatter(time_profile[1][:-1], time_profile[0][:])
        # plt.yscale('log')
        # plt.show()

        self.log.info(f"SN time profile loaded from {data_dir}TimeProfile.root")

        return time_profile[1][:], time_profile[0][:]

    # Photon information from SimPhotonsLite (before Detsim and Reco)
    @staticmethod
    def load_g4_photon_data(file_name, event_num):
        uproot_simphotonsana = uproot.open(file_name)["simphotonsana/PhotonsTree"]
        photons_dict = uproot_simphotonsana.arrays(['EventID', 'OpChannel', 'DetectedPhotonsCount'], library="np")
        photons = [photons_dict[key] for key in photons_dict]

        opchannel_num = np.max(photons_dict['OpChannel']) + 1

        # Create (event_num)x(opchannel_num) matrix
        photons_per_channel = np.zeros((event_num, opchannel_num))

        for i in range(len(photons[0])):
            event_id = photons[0][i]
            ev = int(event_id) - 1

            opchannel_id = photons[1][i]
            photons_per_channel[ev][opchannel_id] = photons[2][i]

        return photons_per_channel
