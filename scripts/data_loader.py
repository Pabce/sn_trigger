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

#from parameters import *

class DataLoader:

    def __init__(self, config):
        self.config = config

    def load_and_split(self, sn_limit, bg_limit, detector):
        '''
        Load the signal and backgrounds and split them into train adn efficiency computation sets
        '''

        # Config reads ----
        # ENERGY LIMIT TO TRAIN THE BDT
        bdt_energy_limit = self.config.get("Simulation", "bdt", "energy_limit")

        # -----------------

        print("Loading SN data...")
        sn_total_hits, sn_hit_list_per_event, sn_info_per_event , _, _ = self.load_all_sn_events_chunky(limit=sn_limit, event_num=1000)

        print("Loading BG data...")
        bg_length = bg_limit * BG_SAMPLE_LENGTHS[detector] # in miliseconds
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

        detector = self.config.get("Detector", "type")
        bg_data_dir = self.config.get("IO", "bg_data_dir")
        adc_mode = self.config.get("Simulation", "adc_mode")
        sim_mode = self.config.get("Simulation", "sim_mode")
        bg_types = self.config.get("Backgrounds", detector)

        # -----------------

        bg_total_hits = []
        bg_hit_list_per_event = []

        directories = []
        if detector == 'VD':
            dir = bg_data_dir
            for bg_type in bg_types:
                directory = os.fsencode(dir + bg_type + '/')
                directories.append(directory)

            endswith = '_detsim_{}_reco_hist.root'.format(sim_mode)
            if adc_mode == 'low':
                startswith = 'prodbg_radiological_dune10kt_vd_1x8x14_lowADC'
            elif adc_mode == 'normal':
                startswith = 'prodbg_radiological_decay0_vd_dune10kt_1x8x14'

        elif detector == 'HD':
            dir = bg_data_dir
            for bg_type in bg_types:
                directory = os.fsencode(dir + bg_type + '/')
                directories.append(directory)

            endswith = '_detsim_reco_hist.root'
            if adc_mode == 'low':
                startswith = 'fprodbg_radiological_dune10kt_v3_hd_1x2x6'
            elif adc_mode == 'normal':
                startswith = 'prodbg_radiological_decay0_dune10kt_1x2x6'

        
        bg_total_hits_per_type = [[] for _ in range(len(bg_types))]
        bg_hit_list_per_event_per_type = [[] for _ in range(len(bg_types))]
        for j, directory in enumerate(directories):
            file_names = []
            k = 0
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(endswith) and filename.startswith(startswith):
                    if verbose > 1:
                        print(filename)
                        
                    if adc_mode == 'normal' and 'lowADC' in filename:
                        continue
                    
                    if k < offset:
                        k += 1
                        continue

                    k += 1
                    if k > limit + offset:
                        break

                    if verbose > 0:
                        print(filename)
                    
                    filename = dir + bg_types[j] + '/' + filename
                    file_names.append(filename)

            with mp.Pool(mp.cpu_count()) as pool:
                # THIS IS THE NUMBER OF EVENTS IN BG FILES, REMEMBER TO CHANGE IT
                results = pool.starmap(self.load_hit_data, file_names, repeat(20), repeat(False))
        
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
    def load_all_sn_events_chunky(self, load_photon_info=False, limit=1, event_num=1000, offset=0, verbose=0):
        # Config reads ----

        detector = self.config.get("Detector", "type")
        sn_data_dir = self.config.get("IO", "sn_data_dir")
        adc_mode = self.config.get("Simulation", "adc_mode")
        sim_mode = self.config.get("Simulation", "sim_mode")

        # -----------------

        sn_total_hits = []
        sn_hit_list_per_event = []

        sn_total_info = []
        sn_total_info_per_event = []

        total_photons_per_channel = []

        if detector == 'VD':
            dir = sn_data_dir
            directory = os.fsencode(dir)
            endswith = '_g4_detsim_{}_reco_hist.root'.format(sim_mode)
            info_endswith = '_stana_hist.root'
            photon_endswith = '_g4_1_hist.root'

            # Laura
            #endswith = '_reco_hist.root'
            #info_endswith = '_laura_hist.root'

            if adc_mode == 'low':
                startswith = 'prodmarley_nue_dune10kt_vd_1x8x14_larger_lowADC'
            elif adc_mode == 'normal':
                startswith = 'prodmarley_nue_dune10kt_vd_1x8x14'

        elif detector == 'HD':
            dir = sn_data_dir

            directory = os.fsencode(dir)
            endswith = '_g4_detsim_reco_hist.root'
            info_endswith = '_stana_hist.root'

            if adc_mode == 'low':
                startswith = '....'
            elif adc_mode == 'normal':
                startswith = 'prodmarley_nue_dune10kt_1x2x6'
        
        file_names = []
        info_file_names = []
        photon_file_names = []
        i = 0
        # We need to make sure to read the *ana and the *reco files with the same ids!
        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.endswith(endswith) and filename.startswith(startswith):
                if adc_mode == 'normal' and 'lowADC' in filename:
                    continue
                
                if verbose > 0:
                    print(filename, i, len(os.listdir(directory)))

                if i < offset:
                    i += 1
                    continue

                print(i)
                # Check if we have the corresponding *ana file
                info_filename = dir + filename.replace(endswith, info_endswith)
                if not os.path.isfile(info_filename):
                    print("WARNING: no corresponding *ana file")
                    if verbose > 0:
                        print(info_filename)
                    continue # Skip if we don't have the corresponding *ana file
                
                if load_photon_info:
                    # Check if we have the corresponding *simphotonslite file
                    photon_filename = dir + filename.replace(endswith, photon_endswith)
                    if not os.path.isfile(photon_filename):
                        print("WARNING: no corresponding *photon file")
                        continue

                    # Skip if the file doesn't contain the tree we need
                    try:
                        uproot_neutrino = uproot.open(photon_filename)["simphotonsana/PhotonsTree"]
                        print("Chipmunk")
                    except uproot.exceptions.KeyInFileError:
                        print("WARNING: corrupted *photon file")
                        continue

                # Skip if the file doesn't contain the tree we need
                try:
                    uproot_neutrino = uproot.open(info_filename)["analysistree/anatree"]
                    # Skip if the tree has less entries than the event number we need
                    if len(uproot_neutrino['enu_truth'].array()) < event_num:
                        continue
                except uproot.exceptions.KeyInFileError:
                    print("WARNING: corrupted *ana file")
                    continue
                
                i += 1
                if i > limit + offset:
                    break
                #print(filename)
                
                filename = dir + filename 
                file_names.append(filename)

                info_file_names.append(info_filename)
                if load_photon_info:
                    photon_file_names.append(photon_filename)

        #print("CHIRP", len(file_names), len(info_file_names))
        
        # We need to pair the results corresponding to the same ids
        with mp.Pool(mp.cpu_count()) as pool:
            if load_photon_info:
                results = pool.starmap(self.load_snhit_and_neutrino_info, file_names, info_file_names, repeat(event_num), repeat(load_photon_info), photon_file_names)
            else:
                results = pool.starmap(self.load_sn_hit_and_neutrino_info, file_names, info_file_names, repeat(event_num))
        
        for i, result in enumerate(results):
            sn_total_hits_i, sn_hit_list_per_event_i, _, sn_info_i, sn_info_per_event_i, photons_per_channel = result

            sn_total_hits.append(sn_total_hits_i)
            sn_hit_list_per_event.extend(sn_hit_list_per_event_i)

            sn_total_info.append(sn_info_i)
            sn_total_info_per_event.extend(sn_info_per_event_i)

            if load_photon_info:
                total_photons_per_channel.append(photons_per_channel)

        sn_total_hits = np.concatenate(sn_total_hits, axis=0)
        sn_total_info = np.concatenate(sn_total_info, axis=0)
        if load_photon_info:
            total_photons_per_channel = np.concatenate(total_photons_per_channel, axis=0)

        return sn_total_hits, sn_hit_list_per_event, sn_total_info, sn_total_info_per_event, total_photons_per_channel


    def load_sn_hit_and_neutrino_info(self, file_name, info_file_name, event_num=-1, load_photon_info=False, photon_file_name=None, verbose=0):
        sn_total_hits, sn_hit_list_per_event, _ = self.load_hit_data(file_name, event_num, sn_event=True)
        sn_info, sn_info_per_event = load_neutrino_info(info_file_name)

        if load_photon_info:
            photons_per_channel = load_g4_photon_data(photon_file_name, event_num)

            return sn_total_hits, sn_hit_list_per_event, None, sn_info, sn_info_per_event, photons_per_channel

        return sn_total_hits, sn_hit_list_per_event, None, sn_info, sn_info_per_event, None

    # TODO: find a smart way to pass the X_COORDS, etc, as arguments (e.g. just pass the subset of the config where the loaded data is contained?)
    def load_hit_data(self, file_name="pbg_g4_digi_reco_hist.root", event_num=-1, sn_event=False, verbose=0):
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

        if verbose > 0:
            print(file_name, mp.current_process())

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

        hit_list_per_event, hit_num_per_channel = process_hit_data(total_hits, event_num)

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
        uproot_time_profile = uproot.open(aux_data_dir + "TimeProfile.root")["h_MarlTime"]
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



