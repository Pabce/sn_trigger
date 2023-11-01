import numpy as np
import os
import multiprocessing as mp
import uproot
from itertools import repeat

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


def load_all_backgrounds_chunky(limit=1, detector="VD"):
    bg_total_hits = []
    bg_hit_list_per_event = []

    if detector == 'VD':
        dir = '../'
        dir = '../vd_backgrounds/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'pbg'
    elif detector == 'HD':
        dir = '../horizontaldrift/'
        directory = os.fsencode(dir)
        endswith = '_reco_hist.root'
        startswith = 'hd_pbg'
    
    file_names = []
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(endswith) and filename.startswith(startswith):
            i += 1
            if i > limit:
                break
            print(filename)
            
            filename = dir + filename 
            file_names.append(filename)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(load_hit_data, zip(file_names, repeat(-1)))
    
    for i, result in enumerate(results):
        bg_total_hits_i, bg_hit_list_per_event_i, _ = result
        bg_total_hits.append(bg_total_hits_i)
        bg_hit_list_per_event.extend(bg_hit_list_per_event_i)

    bg_total_hits = np.concatenate(bg_total_hits, axis=0)

    return bg_total_hits, bg_hit_list_per_event, None


def load_hit_data(file_name="pbg_g4_digi_reco_hist.root", event_num=-1):
    # The hit info is imported from a .root file.
    # From the hit info, we find the clusters

    # Keep it simple for now (shape: X*7)
    # X and Y coordinates are switched!!!

    print(file_name, mp.current_process())


    uproot_ophit = uproot.open(file_name)["opflashana/PerOpHitTree"]
    total_hits_dict = uproot_ophit.arrays(['EventID', 'HitID', 'OpChannel', 'PeakTime', 'X_OpDet_hit', 'Y_OpDet_hit', 'Z_OpDet_hit',
                                            'Width', 'Area', 'Amplitude', 'PE'], library="np")
    total_hits = [total_hits_dict[key] for key in total_hits_dict]
    total_hits = np.array(total_hits).T
    #print(total_hits.shape)    

    # Remove the signal data for the HD "backgrounds"
    # if is_bg:
    #     time_sort = np.argsort(total_hits[:,3])
    #     ordered_hits = total_hits[time_sort, :]
    
    #     zero_remove = np.where(np.abs(ordered_hits[:,3]) > 10)[0]
    #     ordered_hits = ordered_hits[zero_remove, :]

    #     total_hits = ordered_hits

    hit_list_per_event, hit_num_per_channel = process_hit_data(total_hits, event_num)

    return total_hits, hit_list_per_event, hit_num_per_channel


def process_hit_data(total_hits, event_num=-1):
    if event_num == -1:
        event_num = int(np.max(total_hits[:, 0]))

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