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

from parameters import *

def load_all_backgrounds(limit=1, detector="VD"):
    bg_total_hits = []
    bg_hit_list_per_event = []
    directory = os.fsencode('../')

    if detector == 'VD':
        dir = '../'
        #dir = '../vd_backgrounds/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'pbg'
    elif detector == 'HD':
        dir = '../horizontaldrift/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'hd_pbg'

    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(endswith) and filename.startswith(startswith):
            i += 1
            if i > limit:
                break
            print(filename)
            
            filename = dir + filename 
            bg_total_hits_i, bg_hit_list_per_event_i, _ = load_hit_data(file_name=filename)
            bg_total_hits.append(bg_total_hits_i)
            bg_hit_list_per_event.extend(bg_hit_list_per_event_i)

    bg_total_hits = np.concatenate(bg_total_hits, axis=0)

    return bg_total_hits, bg_hit_list_per_event, None


def load_all_backgrounds_chunky_type_separated(limit=1, detector="VD", bg_data_dir=None, adc_mode=None, 
                                                sim_mode=None, offset=1, bg_types=BG_TYPES, verbose=0):
    if adc_mode is None:
        adc_mode = ADC_MODE
    if sim_mode is None:
        sim_mode = SIM_MODE

    bg_total_hits = []
    bg_hit_list_per_event = []

    directories = []
    if detector == 'VD':
        if bg_data_dir:
            dir = bg_data_dir
        else:
            dir = BG_DATA_DIR

        for bg_type in bg_types:
            directory = os.fsencode(dir + bg_type + '/')
            directories.append(directory)

        endswith = '_detsim_{}_reco_hist.root'.format(sim_mode)
        if adc_mode == 'low':
            startswith = 'prodbg_radiological_dune10kt_vd_1x8x14_lowADC'
        elif adc_mode == 'normal':
            startswith = 'prodbg_radiological_dune10kt_vd_1x8x14'

    elif detector == 'HD':
        if bg_data_dir:
            dir = bg_data_dir
        else:
            dir = BG_DATA_DIR # TODO: Add a specific HD BG data dir

        for bg_type in bg_types:
            directory = os.fsencode(dir + bg_type + '/')
            directories.append(directory)

        endswith = '_detsim_{}_reco_hist.root'.format(sim_mode)
        if adc_mode == 'low':
            startswith = 'prodbg_radiological_dune10kt_v3_hd_1x2x6'
        elif adc_mode == 'normal':
            startswith = 'prodbg_radiological_dune10kt_v3_hd_1x2x6'

    
    bg_total_hits_per_type = [[] for _ in range(len(bg_types))]
    bg_hit_list_per_event_per_type = [[] for _ in range(len(bg_types))]
    for j, directory in enumerate(directories):
        file_names = []
        k = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(endswith) and filename.startswith(startswith):
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
            results = pool.starmap(load_hit_data, zip(file_names, repeat(20), repeat(False)))
    
        for i, result in enumerate(results):
            bg_total_hits_i, bg_hit_list_per_event_i, _,  = result
            bg_total_hits_per_type[j].append(bg_total_hits_i)
            bg_hit_list_per_event_per_type[j].extend(bg_hit_list_per_event_i)

        try:
            bg_total_hits_per_type[j] = np.concatenate(bg_total_hits_per_type[j], axis=0)
        except ValueError as e:
            print("No background of type {} found".format(bg_types[j]))
            print(e)
            raise e

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



def load_all_backgrounds_chunky(limit=1, detector="VD", type_separated=False, verbose=0):
    bg_total_hits = []
    bg_hit_list_per_event = []

    directories = []
    if detector == 'VD':
        dir = BG_DATA_DIR

        if type_separated:
            for bg_type in BG_TYPES:
                directory = os.fsencode(dir + bg_type + '/')
                directories.append(directory)
        else:
            directories = [os.fsencode(dir)]

        endswith = '_digi_reco_hist.root'
        startswith = 'prodbg'
    elif detector == 'HD':
        dir = '../horizontaldrift/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'hd_pbg'

    file_names = []
    i = 0
    for j, directory in enumerate(directories):
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(endswith) and filename.startswith(startswith):
                i += 1
                if i > limit:
                    break

                if verbose > 0:
                    print(filename)
                
                filename = (dir + BG_TYPES[j] + '/' + filename) if type_separated else (dir + filename)
                file_names.append(filename)
    
    with mp.Pool(mp.cpu_count()) as pool:
        # THIS IS THE NUMBER OF EVENTS IN BG FILES, REMEMBER TO CHANGE IT
        results = pool.starmap(load_hit_data, zip(file_names, repeat(20)))
    
    for i, result in enumerate(results):
        bg_total_hits_i, bg_hit_list_per_event_i, _ = result
        bg_total_hits.append(bg_total_hits_i)
        bg_hit_list_per_event.extend(bg_hit_list_per_event_i)

    bg_total_hits = np.concatenate(bg_total_hits, axis=0)

    return bg_total_hits, bg_hit_list_per_event, None


# Load SN events hits and neutrino (MC truth) information
def load_all_sn_events_chunky(limit=1, event_num=1000, detector="VD", sn_data_dir=None, adc_mode=None, sim_mode=None, offset=1):
    
    if adc_mode is None:
        adc_mode = ADC_MODE
    if sim_mode is None:
        sim_mode = SIM_MODE
    
    sn_total_hits = []
    sn_hit_list_per_event = []

    sn_total_info = []
    sn_total_info_per_event = []

    if detector == 'VD':
        if sn_data_dir:
            dir = sn_data_dir
        else:
            dir = EVENT_DATA_DIR

        directory = os.fsencode(dir)
        endswith = '_g4_detsim_{}_reco_hist.root'.format(sim_mode)
        info_endswith = '_stana_hist.root'
        
        # Laura
        #endswith = '_reco_hist.root'
        #info_endswith = '_laura_hist.root'

        if adc_mode == 'low':
            startswith = 'prodmarley_nue_dune10kt_vd_1x8x14_larger_lowADC'
        elif adc_mode == 'normal':
            startswith = 'prodmarley_nue_dune10kt_vd_1x8x14_larger'

    elif detector == 'HD':
        if sn_data_dir:
            dir = sn_data_dir
        else:
            dir = EVENT_DATA_DIR # TODO: add HD data dir

        directory = os.fsencode(dir)
        endswith = '_g4_detsim_{}_reco_hist.root'.format(sim_mode)
        info_endswith = '_stana_hist.root'
        
        # Laura
        #endswith = '_reco_hist.root'
        #info_endswith = '_laura_hist.root'

        if adc_mode == 'low':
            startswith = '....'
        elif adc_mode == 'normal':
            startswith = 'prodmarley_nue_dune10kt_1x2x6'
    
    file_names = []
    info_file_names = [] 
    i = 0

    # We need to make sure to read the *ana and the *reco files with the same ids!
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(endswith) and filename.startswith(startswith):
            if adc_mode == 'normal' and 'lowADC' in filename:
                continue

            # Check if we have the corresponding *ana file
            info_filename = dir + filename.replace(endswith, info_endswith)
            
            if not os.path.isfile(info_filename):
                continue # Skip if we don't have the corresponding *ana file

            # Skip if the file doesn't contain the tree we need
            try:
                uproot_neutrino = uproot.open(info_filename)["analysistree/anatree"]
                # Skip if the tree has less entries than the event number we need
                if len(uproot_neutrino['enu_truth'].array()) < event_num:
                    continue
            except uproot.exceptions.KeyInFileError:
                print("WARNING: corrupted *ana file")
                continue
            
            if i < offset:
                i += 1
                continue

            i += 1
            if i > limit + offset:
                break
            #print(filename)
            
            filename = dir + filename 
            file_names.append(filename)

            info_file_names.append(info_filename)

    #print("CHIRP", len(file_names), len(info_file_names))
    
    # We need to pair the results corresponding to the same ids
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(load_sn_hit_and_neutrino_info, zip(file_names, info_file_names, repeat(event_num)))
    
    for i, result in enumerate(results):
        sn_total_hits_i, sn_hit_list_per_event_i, _, sn_info_i, sn_info_per_event_i = result

        sn_total_hits.append(sn_total_hits_i)
        sn_hit_list_per_event.extend(sn_hit_list_per_event_i)

        sn_total_info.append(sn_info_i)
        sn_total_info_per_event.extend(sn_info_per_event_i)

    sn_total_hits = np.concatenate(sn_total_hits, axis=0)
    sn_total_info = np.concatenate(sn_total_info, axis=0)

    return sn_total_hits, sn_hit_list_per_event, sn_total_info, sn_total_info_per_event


def load_sn_hit_and_neutrino_info(file_name, info_file_name, event_num=-1):
    sn_total_hits, sn_hit_list_per_event, _ = load_hit_data(file_name, event_num, sn_event=True)
    sn_info, sn_info_per_event = load_neutrino_info(info_file_name)

    return sn_total_hits, sn_hit_list_per_event, None, sn_info, sn_info_per_event


def load_hit_data(file_name="pbg_g4_digi_reco_hist.root", event_num=-1, sn_event=False, verbose=0):
    # The hit info is imported from a .root file.
    # From the hit info, we find the clusters

    # Keep it simple for now (shape: X*7)
    # X and Y coordinates are switched!!!

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

        #print(np.sort(np.unique(total_hits_dict_1['OpChannel'])))

        total_hits_dict_1['X_OpDet_hit'] = X_COORDS[total_hits_dict_1['OpChannel']]
        total_hits_dict_1['Y_OpDet_hit'] = Y_COORDS[total_hits_dict_1['OpChannel']]
        total_hits_dict_1['Z_OpDet_hit'] = Z_COORDS[total_hits_dict_1['OpChannel']]

        total_hits_dict_2 = uproot_ophit.arrays(['Width', 'Area', 'Amplitude', 'PE'], library="np")

        # Merge the two dictionaries preserving the order of the keys
        total_hits_dict = {**total_hits_dict_1, **total_hits_dict_2}

        total_hits = [total_hits_dict[key] for key in total_hits_dict]
        total_hits = np.array(total_hits).T

    hit_list_per_event, hit_num_per_channel = process_hit_data(total_hits, event_num)

    return total_hits, hit_list_per_event, hit_num_per_channel


# Load neutrino (MC truth) info for SN events
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
def load_time_profile():
    uproot_time_profile = uproot.open("../TimeProfile.root")["h_MarlTime"]
    time_profile = uproot_time_profile.to_numpy()

    # print(time_profile[0].shape)
    # print(time_profile[1].shape)
    # plt.scatter(time_profile[1][:-1], time_profile[0][:])
    # plt.yscale('log')
    # plt.show()

    return time_profile[1][:-1], time_profile[0][:]


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

