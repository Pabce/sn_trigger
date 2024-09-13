'''
save_n_load.py

This module contains the functions to load the hit data in parallel from th ROOT files.
Also to save and load the efficiency data and curves.
'''


import numpy as np
import os, re
import multiprocessing as mp
import uproot
from itertools import repeat
import random
import string
import pickle
from sys import exit
import logging

import rich
from rich.progress import track, Progress, TextColumn, TimeElapsedColumn,\
    BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.logging import RichHandler
import gui

#from parameters import *

class DataLoader:

    def __init__(self, config):
        self.config = config

        logging.basicConfig(
            level=logging.INFO, 
            format="%(message)s", 
            datefmt="[%X]", 
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        self.log = logging.getLogger("rich")

        # self.log.debug("This is a debug message")
        # self.log.info("Hello, World!")
        # self.log.warning("This is a warning")
        # self.log.error("This is an error")
        # self.log.exception("This is an exception")


    def load_and_split(self, sn_limit, bg_limit):
        '''
        Load the signal and backgrounds and split them into train adn efficiency computation sets
        '''

        # Config reads ----
        # ENERGY LIMIT TO TRAIN THE BDT
        bdt_energy_limit = self.config.get("Simulation", "bdt", "energy_limit")
        bg_sample_length = self.config.get("IO", "bg_sample_length")
        bg_sample_number_per_file = self.config.get("IO", "bg_sample_number_per_file")
        # -----------------

        self.log.info("Loading SN data...")
        sn_total_hits, sn_hit_list_per_event, sn_info_per_event , _, _ = self.load_all_sn_events_chunky(limit=sn_limit, event_num=1000)

        self.log.info("Loading BG data...")
        bg_length = bg_limit * bg_sample_length * bg_sample_number_per_file # in miliseconds
        bg_total_hits, bg_hit_list_per_event, _, _, _ = self.load_all_backgrounds_chunky_type_separated(limit=bg_limit)


        # We need to spit the SN and BG events for training the BDT and efficiency evaluation 
        # (this is not a train/test split, that is done in the BDT training later)
        # (we don't really need to shuffle this as the loading is paralelised anyways)
        sn_train_info_per_event = sn_info_per_event[:int(len(sn_info_per_event)*0.5)]
        sn_train_info_per_event = np.array(sn_train_info_per_event)
        sn_train_hit_list_per_event = sn_hit_list_per_event[:int(len(sn_hit_list_per_event)*0.5)] 
        sn_train_hit_list_per_event = np.array(sn_train_hit_list_per_event, dtype=object)
        sn_train_hit_list_per_event = sn_train_hit_list_per_event[sn_train_info_per_event[:, 0] < bdt_energy_limit]
        sn_train_info_per_event = sn_train_info_per_event[sn_train_info_per_event[:, 0] < bdt_energy_limit]

        sn_info_per_event = sn_info_per_event[int(len(sn_info_per_event)*0.5):]
        sn_hit_list_per_event = sn_hit_list_per_event[int(len(sn_hit_list_per_event)*0.5):]

        bg_train_hit_list_per_event = bg_hit_list_per_event[:int(len(bg_hit_list_per_event)*0.5)]
        bg_hit_list_per_event = bg_hit_list_per_event[int(len(bg_hit_list_per_event)*0.5):]


        return sn_hit_list_per_event, sn_train_hit_list_per_event, sn_info_per_event,\
                 sn_train_info_per_event, bg_hit_list_per_event, bg_train_hit_list_per_event, bg_length


    def load_all_backgrounds_chunky_type_separated(self, limit=1, offset=0, verbose=0):
        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        bg_data_dir = self.config.get("IO", "bg_data_dir")
        adc_mode = self.config.get("Simulation", "adc_mode")
        sim_mode = self.config.get("Simulation", "sim_mode")
        bg_types = self.config.get("Backgrounds")
        bg_sample_number_per_file = self.config.get("IO", "bg_sample_number_per_file")
        startswith = self.config.get("IO", "bg_hit_file_start_pattern")
        endswith = self.config.get("IO", "bg_hit_file_end_pattern").format(sim_mode)
        # -----------------
        
        directories = []
        for bg_type in bg_types:
            #directory = os.fsencode(bg_data_dir + bg_type + '/')
            directory = bg_data_dir + bg_type + '/'
            directories.append(directory)

        # Collect valid file names
        for j, directory in enumerate(directories):
            print(directory, type(directory))
            reco_file_names, _, _ = self.collect_valid_file_names(
                directory,
                startswith,
                endswith,
                limit=limit,
                offset=offset,
            )

        # Collect valid file names

    def load_all_backgrounds_chunky_type_separated_old(self, limit=1, offset=0, verbose=0):
        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        bg_data_dir = self.config.get("IO", "bg_data_dir")
        adc_mode = self.config.get("Simulation", "adc_mode")
        sim_mode = self.config.get("Simulation", "sim_mode")
        bg_types = self.config.get("Backgrounds")
        bg_sample_number_per_file = self.config.get("IO", "bg_sample_number_per_file")
        startswith = self.config.get("IO", "bg_hit_file_start_pattern")
        endswith = self.config.get("IO", "bg_hit_file_end_pattern").format(sim_mode)
        # -----------------

        bg_total_hits = []
        bg_hit_list_per_event = []

        directories = []
        for bg_type in bg_types:
            directory = os.fsencode(bg_data_dir + bg_type + '/')
            directories.append(directory)

        
        bg_total_hits_per_type = [[] for _ in range(len(bg_types))]
        bg_hit_list_per_event_per_type = [[] for _ in range(len(bg_types))]
        for j, directory in enumerate(directories):
            file_names = []
            k = 0
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(endswith) and filename.startswith(startswith):
                    
                    if k < offset:
                        k += 1
                        continue

                    k += 1
                    if k > limit + offset:
                        break

                    if verbose > 0:
                        print(filename)
                    
                    filename = bg_data_dir + bg_types[j] + '/' + filename
                    file_names.append(filename)

            with mp.Pool(mp.cpu_count()) as pool:
                # TODO: THIS IS THE NUMBER OF EVENTS IN BG FILES, REMEMBER TO CHANGE IT
                results = pool.starmap(self.load_hit_data, zip(file_names, repeat(bg_sample_number_per_file), repeat(False)) )
        
            for i, result in enumerate(results):
                bg_total_hits_i, bg_hit_list_per_event_i, _,  = result
                bg_total_hits_per_type[j].append(bg_total_hits_i)
                bg_hit_list_per_event_per_type[j].extend(bg_hit_list_per_event_i)

            try:
                bg_total_hits_per_type[j] = np.concatenate(bg_total_hits_per_type[j], axis=0)
                if verbose > 0:
                    print("Loaded {} background hits".format(bg_types[j]))
            except ValueError as e:
                print("No background of type {} found".format(bg_types[j]))
                print(e)
                #raise e

        #bg_total_hits = np.concatenate(bg_total_hits, axis=0)

        # Merge the hit lists, combining all the backgrounds
        # for i in range(len(BG_TYPES)):
        #     print(len(bg_hit_list_per_event_per_type[i]))
        # exit()

        for i, _ in enumerate(bg_hit_list_per_event_per_type[0]):
            hit_list = []
            for j in range(len(bg_types)):
                hit_list.extend(bg_hit_list_per_event_per_type[j][i])

            # Now, we need to order the hits in time once again
            hit_list = np.array(hit_list)
            if hit_list.shape != (0,):
                time_sort = np.argsort(hit_list[:,3])
                hit_list = hit_list[time_sort, :]

            bg_hit_list_per_event.append(np.array(hit_list))


        return bg_total_hits, bg_hit_list_per_event, None, bg_total_hits_per_type, bg_hit_list_per_event_per_type


    # Load SN events hits and neutrino (MC truth) information
    def load_all_sn_events_chunky(self, load_photon_info=False, limit=1, event_num=1000, offset=0):
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
        sn_data_dir = self.config.get("IO", "sn_data_dir")
        adc_mode = self.config.get("Simulation", "adc_mode")
        sim_mode = self.config.get("Simulation", "sim_mode")
        startswith = self.config.get("IO", "sn_hit_file_start_pattern")
        endswith = self.config.get("IO", "sn_hit_file_end_pattern").format(sim_mode)
        info_endswith = self.config.get("IO", "sn_info_file_end_pattern")
        photon_endswith = self.config.get("IO", "photon_file_end_pattern")
        #TODO: add to read: the number of events per file
        # -----------------

        # Collect valid file names
        reco_file_names, info_file_names, photon_file_names = self.collect_valid_file_names(
            sn_data_dir,
            startswith,
            endswith,
            info_endswith,
            photon_endswith,
            load_photon_info,
            limit,
            offset,
            event_num,
        )

        # Process files in parallel
        results = self.process_files_in_parallel(
            reco_file_names,
            info_file_names,
            photon_file_names,
            event_num,
            load_photon_info,
        )

        # Accumulate results
        (
            sn_total_hits,
            sn_hit_list_per_event,
            sn_total_info,
            sn_total_info_per_event,
            total_photons_per_channel,
        ) = self.accumulate_results(results, load_photon_info)

        return (
            sn_total_hits,
            sn_hit_list_per_event,
            sn_total_info,
            sn_total_info_per_event,
            total_photons_per_channel,
        )

    def collect_valid_file_names(self, data_dir, startswith, endswith, 
                                info_endswith=None, photon_endswith=None, 
                                load_photon_info=False, limit=1, offset=0, 
                                event_num=1000, data_type_str="Chipmonc"):
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
        with gui.get_custom_progress() as progress:
            task = progress.add_task(f'[cyan]Finding {data_type_str} files and verifying integrity...', total=limit)

            for filename in os.listdir(data_dir):
                filename = os.fsdecode(filename)

                if not (filename.startswith(startswith) and filename.endswith(endswith)):
                    continue

                if processed_files < offset:
                    processed_files += 1
                    continue

                self.log.info(f"Validating file: {filename}")

                # Check if hit file contains required tree
                reco_filename = data_dir + filename
                try:
                    uproot_ophit = uproot.open(reco_filename)["opflashana/PerOpHitTree"]
                except uproot.exceptions.KeyInFileError as e:
                    logging.warning(f"Corrupted *reco file {filename}: {e}. Skipping.")
                    continue

                reco_file_names.append(reco_filename)

                # Check if corresponding *ana file exists (if required)
                if info_endswith:
                    info_filename = data_dir + filename.replace(endswith, info_endswith)
                    if not os.path.isfile(info_filename):
                        self.log.warning(f"No corresponding *ana file for {filename}. Skipping.")
                        continue

                    # Check if ana file contains required tree
                    try:
                        uproot_neutrino = uproot.open(info_filename)["analysistree/anatree"]
                        # Skip if the tree has less entries than the event number we need
                        if len(uproot_neutrino['enu_truth'].array()) < event_num:
                            raise uproot.exceptions.KeyInFileError
                    except uproot.exceptions.KeyInFileError as e:
                        logging.warning(f"Corrupted *ana file {info_filename}: {e}. Skipping.")
                        continue
                    
                    info_file_names.append(info_filename)
                
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
                        uproot_file = uproot.open(photon_filename)
                        uproot_file["simphotonsana/PhotonsTree"]
                    except Exception as e:
                        self.log.warning(f"Corrupted *photon file {photon_filename}: {e}. Skipping.")
                        continue

                    photon_file_names.append(photon_filename)
                
                # --------------------------------

                processed_files += 1
                if processed_files > limit + offset:
                    break

                progress.update(task, advance=1)

        return reco_file_names, info_file_names, photon_file_names

    def process_files_in_parallel(self, file_names, info_file_names,
                                   photon_file_names, event_num, load_photon_info):
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
        num_processes = mp.cpu_count()

        # We need to pair the results corresponding to the same ids
        with gui.get_custom_progress() as progress:
            task = progress.add_task("[cyan]Loading SN data (in parallel)...", total=len(file_names))
            with mp.Pool(num_processes) as pool:
                results = []
                if load_photon_info:
                    args = zip(
                        file_names,
                        info_file_names,
                        repeat(event_num),
                        repeat(load_photon_info),
                        photon_file_names,
                    )
                else:
                    args = zip(
                        file_names,
                        info_file_names,
                        repeat(event_num),
                        repeat(load_photon_info),
                    )

                imap_results = pool.imap_unordered(self.wrapped_load_sn_hit_and_neutrino_info, args)

                for result in imap_results:
                    progress.update(task, advance=1)
                    results.append(result)

            return results

    def accumulate_results(self, results, load_photon_info):
        """
        Accumulate results from processing files.

        Args:
            results (list): List of results from each file.
            load_photon_info (bool): Whether photon info was loaded.

        Returns:
            tuple: Accumulated data arrays and lists.
        """
        sn_total_hits = []
        sn_hit_list_per_event = []
        sn_total_info = []
        sn_total_info_per_event = []
        total_photons_per_channel = []

        for result in results:
            (
                sn_total_hits_i,
                sn_hit_list_per_event_i,
                _,
                sn_info_i,
                sn_info_per_event_i,
                photons_per_channel,
            ) = result

            sn_total_hits.append(sn_total_hits_i)
            sn_hit_list_per_event.extend(sn_hit_list_per_event_i)

            sn_total_info.append(sn_info_i)
            sn_total_info_per_event.extend(sn_info_per_event_i)

            if load_photon_info and photons_per_channel is not None:
                total_photons_per_channel.append(photons_per_channel)

        sn_total_hits = np.concatenate(sn_total_hits, axis=0)
        sn_total_info = np.concatenate(sn_total_info, axis=0)

        if load_photon_info and total_photons_per_channel:
            total_photons_per_channel = np.concatenate(total_photons_per_channel, axis=0)
        else:
            total_photons_per_channel = None

        return (
            sn_total_hits,
            sn_hit_list_per_event,
            sn_total_info,
            sn_total_info_per_event,
            total_photons_per_channel,
        )



    def load_sn_hit_and_neutrino_info(self, file_name, info_file_name, event_num=-1, load_photon_info=False, photon_file_name=None):
        sn_total_hits, sn_hit_list_per_event, _ = self.load_hit_data(file_name, event_num, sn_event=True)
        sn_info, sn_info_per_event = self.load_neutrino_info(info_file_name)

        if load_photon_info:
            photons_per_channel = self.load_g4_photon_data(photon_file_name, event_num)

            return sn_total_hits, sn_hit_list_per_event, None, sn_info, sn_info_per_event, photons_per_channel

        return sn_total_hits, sn_hit_list_per_event, None, sn_info, sn_info_per_event, None

    def wrapped_load_sn_hit_and_neutrino_info(self, arg_list):
            return self.load_sn_hit_and_neutrino_info(*arg_list)
    
    def load_hit_data(self, file_name="pbg_g4_digi_reco_hist.root", event_num=-1, sn_event=False):
        # The hit info is imported from a .root file.
        # From the hit info, we find the clusters

        # Keep it simple for now (shape: X*7)
        # X and Y coordinates are switched!!!

        # Config reads ----
        detector = self.config.get("Detector", "type")
        x_coords = self.config.loaded["x_coords"]
        y_coords = self.config.loaded["y_coords"]
        z_coords = self.config.loaded["z_coords"]
        # ----------------- 

        self.log.debug(f"Loading hit data from file: {file_name}")

        # TODO: fix this bullshit for Laura's files
        try:
            uproot_ophit = uproot.open(file_name)["opflashana/PerOpHitTree"]
            total_hits_dict = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime', 'X_OpDet_hit', 'Y_OpDet_hit', 'Z_OpDet_hit',
                                                    'Width', 'Area', 'Amplitude', 'PE'], library="np")
            total_hits = [total_hits_dict[key] for key in total_hits_dict]
            total_hits = np.array(total_hits).T

        except uproot.exceptions.KeyInFileError:
            uproot_ophit = uproot.open(file_name)["opflashana/PerOpHitTree"]
            total_hits_dict_1 = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime'], library="np")

            # print(np.sort(np.unique(total_hits_dict_1['OpChannel'])), len(np.sort(np.unique(total_hits_dict_1['OpChannel']))))
            # exit()

            # Edit the OpChannel array for the new configuration of two channels per X-ARAPUCA (for VD only).
            # (What used to be OpChannel 43 is now 430 and 431, etc.)
            if detector == "VD":
                for i in range(len(total_hits_dict_1['OpChannel'])):
                    total_hits_dict_1['OpChannel'][i] = int(total_hits_dict_1['OpChannel'][i] / 10)
                
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
    @staticmethod
    def load_neutrino_info(file_name, verbose=0):
        if verbose > 0:
            print(file_name, mp.current_process())

        info_per_event = []

        # uproot_neutrino = uproot.open(file_name)["vdflashmatch/FlashMatchTree"]
        # info_dict = uproot_neutrino.arrays(['TrueE', 'TrueX', 'TrueY', 'TrueZ'], library="np")
        uproot_neutrino = uproot.open(file_name)["analysistree"]
        info_dict = uproot_neutrino["anatree"].arrays(['enu_truth', 'nuvtxx_truth', 'nuvtxy_truth', 'nuvtxz_truth'], library="np")

        info = [np.concatenate(info_dict[key]) for key in info_dict]
        info_per_event_arr = np.array(info).T

        info_per_event_arr[:, 0] *= 1000 # Convert to MeV

        for i in range(info_per_event_arr.shape[0]):
            info_per_event.append(info_per_event_arr[i, :])

        return info_per_event_arr, info_per_event


    @staticmethod
    def process_hit_data(total_hits, event_num=-1):
        if event_num == -1:
            try:
                event_num = int(np.max(total_hits[:, 0]))
            except ValueError:
                print("No events found!")
                return [[]], []

        hit_number_per_event = np.zeros(event_num)
        hit_list_per_event = [[] for _ in range(event_num)]
        detection_per_event = np.zeros(event_num)

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
        data_dir = self.config.get("IO", "aux_data_dir")
        uproot_time_profile = uproot.open(data_dir + "TimeProfile.root")["h_MarlTime"]
        time_profile = uproot_time_profile.to_numpy()

        # print(time_profile[0].shape)
        # print(time_profile[1].shape)
        # plt.scatter(time_profile[1][:-1], time_profile[0][:])
        # plt.yscale('log')
        # plt.show()

        return time_profile[1][:-1], time_profile[0][:]

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


# TODO: PUT THIS INTO A CLASS
#TODO: Make sim_parameters a dictionary!!!
def save_efficiency_data(eff_data, sim_parameters, file_name=None, data_type="data"):
    #ftr, btw, dist, sim_mode, adc_mode, detector, classify, avg_energy, alpha = sim_parameters
    sim_parameters = tuple(sim_parameters)

    # Generate a random string to identify the file
    if file_name is None:
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        file_name = "efficiency_{}_".format(data_type) + random_string + ".pcl"
    
    # Save this correspondence to a dictionary
    try:
        file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
        if not isinstance(file_names, dict):
                file_names = {}
    
        # Check if entry with these sim_parameters already exists
        if sim_parameters in file_names.keys():
            file_name = file_names[sim_parameters]

        file_names[sim_parameters] = file_name

    except FileNotFoundError:
        file_names = {}
        file_names[sim_parameters] = file_name

    pickle.dump(file_names, open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "wb"))

    # Save the data
    pickle.dump(eff_data, open("{}/{}".format(SAVE_PATH, file_name), "wb"))

    print("Saved efficiency {} to file:".format(data_type), file_name)

    return file_name


def load_efficiency_data(sim_parameters=[], file_name=None, data_type="data"):
    # Check if the dict file exists
    try:
        file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
    except FileNotFoundError:
        file_names = {}
        pickle.dump(file_names, open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "wb"))

    sim_parameters = tuple(sim_parameters)
    if file_name is None:
        file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
        # Find the file name corresponding to the sim_parameters
        file_name = file_names[sim_parameters]

    # Load the data
    eff_data = pickle.load(open("{}/{}".format(SAVE_PATH, file_name), "rb"))
    print("Loaded efficiency {} from file {}".format(data_type, file_name))

    return eff_data, file_name



