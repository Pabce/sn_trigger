from decimal import DivisionByZero
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pickle

# Generate a fake background for a given optical channel following a Poisson distribution
# This of course ignores spatial correlations
# lamb is Poisson parameter, interval (in mus) is the length of the interval the Poisson parameter refers to

def main(poisson_interval, generate_time):
    fit_parameters = pickle.load(open("saved_pickles/bg_poisson_fit_params", "rb"))
    print(len(fit_parameters))
    fake_bg_list = []
    for i, lamb in enumerate(fit_parameters):
        print("Generating fake background for OpDet", i)
        randp, time_list = generate_fake_background(lamb, poisson_interval, generate_time)
        fake_bg = format_fake_bg(time_list, optical_channel=i)

        fake_bg_list.append(fake_bg)

    fake_bgs = combine_fake_bgs(fake_bg_list)

    return fake_bgs, fake_bg_list




def generate_fake_background(lamb, poisson_interval, generate_time):
    # Draw randomly from Poisson distribution, as many times as the ratio between the intervals
    ratio = generate_time // poisson_interval # Integer division
    remainder = generate_time / poisson_interval - ratio
    if remainder > 10e-5:
        print("Please only use exact ratios")
        raise DivisionByZero

    rand_poisson = poisson.rvs(lamb, size=ratio)

    time_list = []
    for i in range(ratio):
        # Get hit number for interval
        hit_number = int(rand_poisson[i])

        # Generate that number of hits randomly distributed in interval
        time_list_i = np.random.uniform(low=i*poisson_interval, high=(i+1)*poisson_interval, size=hit_number)
        time_list.extend(time_list_i)

    return rand_poisson, time_list


# We also want a function to put the fake bg in "correct" format (i.e. readable by your useless script)
def format_fake_bg(time_list, optical_channel):
    hit_number = len(time_list)

    fake_bg = np.zeros((hit_number, 7))

    for i in range(hit_number):
        # 'EventID', 'HitID', 'OpChannel', 'PeakTime', 'X_OpDet_hit', 'Y_OpDet_hit', 'Z_OpDet_hit'
        # TODO: Add coordinate information
        fake_bg[i, :] = np.array([-1, i + 1, optical_channel, time_list[i], 0, 0, 0])
    
    return fake_bg


# A function to combine the fake bgs of all optical channels into one
def combine_fake_bgs(bg_list):
    return np.concatenate(bg_list, axis=0)


if __name__ == '__main__':
    # randp, time_list = generate_fake_background(40.1, 1000, 2*1000)

    # print(randp)

    # fake_bg = format_fake_bg(time_list, 0)
    # print(fake_bg.shape)

    fake_bgs, _ = main(interval=1000, generate_time=int(1e6))
    print(fake_bgs.shape)

    pickle.dump(fake_bgs, open("saved_pickles/fake_bg", "wb"))

