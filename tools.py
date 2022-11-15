import numpy as np
import matplotlib.pyplot as plt
import random


def display_hits(hits, three_d=False, time=False):
    # Weird notation to accomodate for list of individual hits
    x_coords, y_coords, z_coords = hits[:, 5], hits[:, 4], hits[:, 6]
    time_coords = hits[:, 3]

    fig = plt.figure()

    if three_d:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z_coords, x_coords, y_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        #ax.set_box_aspect((np.ptp(hits[:][6]), np.ptp(hits[:][5]), np.ptp(hits[:][4])))

    elif time:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z_coords, x_coords, time_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')
        ax.set_zlabel('time')
        #ax.set_box_aspect((np.ptp(hits[:,6]), np.ptp(hits[:,5]), np.ptp(hits[:,4])))
    
    else:
        ax = fig.add_subplot()
        ax.scatter(z_coords, x_coords)
        ax.set_xlabel('Z') # This is due to Z being in the direction of the beam
        ax.set_ylabel('X')

    plt.show()


def spice_sn_event(hit_list, bg_hit_list_per_event, bg_length_to_add, bg_length):
    # Select one random BG sample and time order it (it should already be time ordered, but to be safe and future-proof)
    bg_sample = random.choice(bg_hit_list_per_event)
    time_sort = np.argsort(bg_sample[:, 3])
    bg_sample = bg_sample[time_sort, :]

    # Select a random time interval within the bg sample
    #bg_hit_number = len(bg_sample[:, 3])
    starting_point = np.random.rand() * bg_length - bg_length / 2 - bg_length_to_add
    end_point = starting_point + bg_length_to_add

    bg_sample = bg_sample[bg_sample[:, 3] > starting_point, :]
    bg_sample = bg_sample[bg_sample[:, 3] < end_point, :]
    
    #print(bg_sample.shape, starting_point, end_point)

    # Shift the center of the interval to t = the centre of the SN event
    bg_sample[:, 3] -= (starting_point + end_point)/2 # to 0
    bg_sample[:, 3] += (hit_list[0, 3] + hit_list[-1, 3])/2

    #print(bg_sample[0, 3], bg_sample[-1, 3])

    # Now combine with the sn hit sample (and sort in time)
    spiced_hit_list = np.vstack((bg_sample, hit_list))

    time_sort = np.argsort(spiced_hit_list[:, 3])
    spiced_hit_list = spiced_hit_list[time_sort, :]

    return spiced_hit_list
