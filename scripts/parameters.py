'''
parameters.py

This file contains the parameters for the algortihm run in hit_stat.py.
Some of these are overridable by command line arguments (see hit_stat.py --help).
'''

import os
import pickle
import numpy as np
import uproot

# TODO: This is just a temporary fix for Laura's files, should be removed later OR made to read in the same directory as the other files
# (and moved to save_n_load.py)
def laura_load_coordinate_arrays(detector="VD"):
    try:
        file_name = "/Users/pbarham/Downloads/prodbg_radiological_dune10kt_vd_1x8x14_1663948146.336200596_Ar39GenInLAr_g4_detsim_highADC_digi_reco_hist.root"
        uproot_coords = uproot.open(file_name)["opflashana/OpDetCoords"]
        coords_dict = uproot_coords.arrays(['X_OpDet', 'Y_OpDet', 'Z_OpDet'], library="np")
    except FileNotFoundError:
        file_name = "/eos/project-e/ep-nu/pbarhama/sn_saves/coordinate_aux.root"
        uproot_coords = uproot.open(file_name)["opflashana/OpDetCoords"]
        coords_dict = uproot_coords.arrays(['X_OpDet', 'Y_OpDet', 'Z_OpDet'], library="np")

    x = coords_dict['X_OpDet'][0:168]
    y = coords_dict['Y_OpDet'][0:168]
    z = coords_dict['Z_OpDet'][0:168]

    #coords = np.array([x, y, z]).T

    return x, y, z

def v5_load_coordinate_arrays(detector="VD"):
    # Load text file into numpy array, separating by columns (delimiter any number of spaces) and skipping the first row
    
    if detector == "VD":
        file_rel = "../pdpos_vd1x8x14v5.dat"
    elif detector == "HD":
        file_rel = "../dunehd1x2x6PDPos.txt"

    file_abs = os.path.join(os.path.dirname(__file__), file_rel)

    coords = np.genfromtxt(file_abs, skip_header=1, skip_footer=2)

    # Split into x, y, z arrays
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]

    return x, y, z


# NO NEED TO TOUCH THESE IN PRINCIPLE -------------------------------------------------------------------------------------
# We need to read the geometry info for the FDVD detector
# Distance is in cm
# There are only 168 "real" optical channels that get hits
DETECTOR = "VD"
GEOMETRY_VERSION = "v5" # "v4" or "v5" 

file_dir = os.path.dirname(__file__)
X_COORDS, Y_COORDS, Z_COORDS = laura_load_coordinate_arrays() if GEOMETRY_VERSION == "v4" else v5_load_coordinate_arrays(detector=DETECTOR)
OP_DISTANCE_ARRAY_VD = pickle.load(open("{}/aux_pickles/op_distance_array_VD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_X_DISTANCE_ARRAY_VD = pickle.load(open("{}/aux_pickles/op_x_distance_array_VD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_Y_DISTANCE_ARRAY_VD = pickle.load(open("{}/aux_pickles/op_y_distance_array_VD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_Z_DISTANCE_ARRAY_VD = pickle.load(open("{}/aux_pickles/op_z_distance_array_VD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))

OP_DISTANCE_ARRAY_HD = pickle.load(open("{}/aux_pickles/op_distance_array_HD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_X_DISTANCE_ARRAY_HD = pickle.load(open("{}/aux_pickles/op_x_distance_array_HD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_Y_DISTANCE_ARRAY_HD = pickle.load(open("{}/aux_pickles/op_y_distance_array_HD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))
OP_Z_DISTANCE_ARRAY_HD = pickle.load(open("{}/aux_pickles/op_z_distance_array_HD_{}".format(file_dir, GEOMETRY_VERSION), "rb"))

OP_DISTANCE_ARRAY = {"VD": OP_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_X_DISTANCE_ARRAY = {"VD": OP_X_DISTANCE_ARRAY_VD, "HD": OP_X_DISTANCE_ARRAY_HD}
OP_Y_DISTANCE_ARRAY = {"VD": OP_Y_DISTANCE_ARRAY_VD, "HD": OP_Y_DISTANCE_ARRAY_HD}
OP_Z_DISTANCE_ARRAY = {"VD": OP_Z_DISTANCE_ARRAY_VD, "HD": OP_Z_DISTANCE_ARRAY_HD}

TRUE_TPC_SIZES = {"VD": 10 * 1.2, "HD": 10 * 1.2}
USED_TPC_SIZES = {"VD": 2.6 * 1.2, "HD": 1 * 1.2}
BG_SAMPLE_LENGTHS = {"VD": 8.5 * 20, "HD": 4.492 * 20} # In milliseconds

# Values of v_e-CC interactions at 10 kpc for a 40 kton LArTPC
INTERACTION_NUMBER_10KPC = {"LIVERMORE": 2684, "GKVM": 3295, "GARCHING": 882}
# Correction for extra volume not included in the original MARLEY events
SN_EVENT_MULTIPLIER = 1.17

# --------------------------------------------------------------------------------------------------------

# Event and bg data diretories
# EVENT_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/new_photon_lib_prod_snnue_pds/"
# BG_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/new_photon_lib_prod_background_pds/"

# EVENT_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/new_compgraph_prod_snnue_pds/"
# BG_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/new_compgraph_prod_background_pds/"

# EVENT_DATA_DIR = "/Users/pbarham/OneDrive/workspace/cern/ruth/new_compgraph_prod_snnue_pds/"
# BG_DATA_DIR = "/Users/pbarham/OneDrive/workspace/cern/ruth/new_compgraph_prod_background_pds/"

# EVENT_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/nov23_snnue_pds/"
# BG_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/nov23_background/test/"

EVENT_DATA_DIR_VD = "/eos/project-e/ep-nu/pbarhama/sn_saves/jun24_snnue_pds/"
BG_DATA_DIR_VD = "/eos/project-e/ep-nu/pbarhama/sn_saves/jun24_vd_background_juergen/"


# Saves path for computed efficiencies, etc
SAVE_PATH = "/eos/home-p/pbarhama/myway/scripts/saved_effs/"
OUTPUT_NAME_DATA = None
OUTPUT_NAME_CURVE = None
INPUT_NAME = None # Input for when calculating the efficiency curve

# Background types
BG_TYPES_VD = [
     'Ar39GenInLAr',
     'Kr85GenInLAr',
     'Ar42GenInLAr',
     'K42From42ArGenInLAr',
     'Rn222ChainRn222GenInLAr',
     'Rn222ChainPo218GenInLAr',
     'Rn222ChainPb214GenInLAr',
     'Rn222ChainBi214GenInLAr',
     'Rn222ChainPb210GenInLAr',
     'Rn220ChainPb212GenInLAr',
     'K40GenInCathode',
     'U238ChainGenInCathode',
     'Th232ChainGenInCathode',
     'K40GenInAnode',
     'U238ChainGenInAnode',
     'Th232ChainGenInAnode',
     'Rn222ChainGenInPDS',
     'K42From42ArGenInUpperMesh1x8x14',
     'Rn222ChainFromPo218GenInUpperMesh1x8x14',
     'Rn222ChainFromPb214GenInUpperMesh1x8x14',
     'Rn222ChainFromBi214GenInUpperMesh1x8x14',
     'Rn222ChainFromPb210GenInUpperMesh1x8x14',
     'Rn222ChainFromBi210GenInUpperMesh1x8x14',
     'Rn220ChainFromPb212GenInUpperMesh1x8x14',
     'CavernwallGammasAtLAr1x8x14',
     'foamGammasAtLAr1x8x14',
     'CavernwallNeutronsAtLAr1x8x14',
     'CryostatNGammasAtLAr1x8x14',
     'CavernNGammasAtLAr1x8x14']

BG_TYPES_HD = [
        "Ar39GenInLAr", "Kr85GenInLAr", "Ar42GenInLAr", "K42From42ArGenInLAr",
        "Rn222ChainRn222GenInLAr",
        "Rn222ChainPo218GenInLAr",
        "Rn222ChainPb214GenInLAr",
        "Rn222ChainBi214GenInLAr",
        "Rn222ChainPb210GenInLAr",
        "Rn220ChainPb212GenInLAr",
        "K40GenInCPA",
        "U238ChainGenInCPA",
        "K42From42ArGenInCPA",
        "Rn222ChainPo218GenInCPA",
        "Rn222ChainPb214GenInCPA",
        "Rn222ChainBi214GenInCPA",
        "Rn222ChainPb210GenInCPA",
        "Rn222ChainFromBi210GenInCPA",
        "Rn220ChainFromPb212GenInCPA",
        "Co60GenInAPA",
        "U238ChainGenInAPA",
        "Th232ChainGenInAPA",
        "K40GenInAPAboards",
        "U238ChainGenInAPAboards",
        "Th232ChainGenInAPAboards",
        "Rn222ChainGenInPDS",
        "CavernwallGammasAtLAr",
        "foamGammasAtLAr",
        "CavernwallNeutronsAtLAr",
        "CryostatNGammasAtLAr",
        "CavernNGammasAtLAr"
    ]

# "Old but newer (04/23) types"
# BG_TYPES = ["Ar39GenInLAr", "Kr85GenInLAr", "Ar42GenInLAr", "K42From42ArGenInLAr", "Rn222ChainGenInLAr",
#             "K42From42ArGenInCPA", "K40inGenInCPA", "U238ChainGenInCPA", "Co60inGenInAPA", "U238ChainGenInAPA",
#             "Rn222ChainGenInPDS", "NeutronGenInRock", "GammasGenInRock"]
# "Old" background types
#BG_TYPES = ["Ar39GenInLAr", "Kr85GenInLAr", "Ar42GenInLAr", "Rn222ChainGenInLAr"]

# Number of files to load (go pretty low on these if not sent to a job)
SN_FILE_LIMIT = 10
BG_FILE_LIMIT = 6
# Number of files to use for the parameter search
SN_FILE_LIMIT_SEARCH = 10
BG_FILE_LIMIT_SEARCH = 20

# Algorithm parameters
FAKE_TRIGGER_RATE = 1/(60 * 60 * 24 * 30) # 1 trigger per month
BURST_TIME_WINDOW = 1e6 # In microseconds
distance_to_evaluate = 20.123 # In kpc 
SIM_MODE = 'xe' # 'xe' or 'aronly'
ADC_MODE = 'normal' # 'low' or 'normal'
# DETECTOR = 'VD' # 'VD' or 'HD' (THIS IS ALREADY DEFINED ABOVE)
CLASSIFY = True # True or False (Do we use the BDT? If not, you should iterate over many more clustering parameters)
DISTANCES = np.arange(4, 40, 4) # In kpc (list of distances in which to compute the efficiencies)

# Spectrum parameters
# GKVM 23 5
# LIVERMORE 14.4 2.8
# GARCHING 12.2 4.5 (OR 11.44 and 3.2)
AVERAGE_ENERGY = 23.0 # MeV
ALPHA = 5.0 # Dimensionless 

# Clustering parameters over which to search (don't go over too many, the classification does most of the work after all...)
MAX_CLUSTER_TIMES = [0.2, 0.15] # In microseconds
MAX_HIT_TIME_DIFFS = [0.20, 0.15]#, 0.25] # In microseconds
MAX_HIT_DISTANCES = [230, 220, 200] # In cm
LOWER_MIN_HIT_MULTUPLICITY = 8 # Minimum number of hits in a cluster
UPPER_MIN_HIT_MULTUPLICITY = 13 # Minimum number of hits in a cluster
CLASSIFIER_THRESHOLDS = [0.5, 0.7, 0.9]# [0.95, 0.7]#, 0.8]#, 0.9] # Threshold for the BDT
CLASSIFIER_HIST_TYPE = "hit_multiplicity" # "hit_multiplicity" or "bdt"
STATISITCAL_METHOD = "chi2" # "ks" (only allowed in CLASSIFIER_HIST_TYPE = "hit_multiplicity") or "chi2"
