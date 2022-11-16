import pickle

# NO NEED TO TOUCH THESE IN PRINCIPLE -------------------------------------------------------------------------------------

# We need to read the geometry info for the FDVD detector
# Distance is in cm
# There are only 168 "real" optical channels that get hits
OP_DISTANCE_ARRAY_VD = pickle.load(open("./aux_pickles/op_distance_array_VD", "rb"))[0:168, 0:168]
OP_X_DISTANCE_ARRAY_VD = pickle.load(open("./aux_pickles/op_x_distance_array_VD", "rb"))[0:168, 0:168]
OP_Y_DISTANCE_ARRAY_VD = pickle.load(open("./aux_pickles/op_y_distance_array_VD", "rb"))[0:168, 0:168]
OP_Z_DISTANCE_ARRAY_VD = pickle.load(open("./aux_pickles/op_z_distance_array_VD", "rb"))[0:168, 0:168]

OP_DISTANCE_ARRAY_HD = None # pickle.load(open("./aux_pickles/op_distance_array_HD", "rb"))
OP_DISTANCE_ARRAY = {"VD": OP_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_X_DISTANCE_ARRAY = {"VD": OP_X_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_Y_DISTANCE_ARRAY = {"VD": OP_Y_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}
OP_Z_DISTANCE_ARRAY = {"VD": OP_Z_DISTANCE_ARRAY_VD, "HD": OP_DISTANCE_ARRAY_HD}

TRUE_TPC_SIZES = {"VD": 10, "HD": 10}
USED_TPC_SIZES = {"VD": 2.6, "HD": 1}
BG_SAMPLE_LENGTHS = {"VD": 8.5 * 20, "HD": 4.492 * 200} # In milliseconds

# Values of v_e-CC interactions at 10 kpc for a 40 kton LArTPC 
INTERACTION_NUMBER_10KPC = {"LIVERMORE": 2684, "GKVM": 3295, "GARCHING": 882} 
# Correction for extra volume not included in the original MARLEY events
SN_EVENT_MULTIPLIER = 1.17

# --------------------------------------------------------------------------------------------------------

# Event and bg data diretories
#EVENT_DATA_DIR = "/eos/user/c/ccuesta/Pablo/sn_saves/prod_snnue_pds/"
#BG_DATA_DIR = "/eos/user/c/ccuesta/Pablo/sn_saves/prod_background_pds/"
EVENT_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/prod_snnue_pds/"
BG_DATA_DIR = "/eos/project-e/ep-nu/pbarhama/sn_saves/prod_background_pds/"
EVENT_DATA_DIR = "/Users/pbarham/OneDrive/workspace/cern/ruth/prod_snnue_pds/"
BG_DATA_DIR = "/Users/pbarham/OneDrive/workspace/cern/ruth/prod_background_pds/prod_background_pds/lmem_prod_background_pds/"

# Saves path for computed efficiencies, etc
SAVE_PATH = "./saved_effs/"

# Background types
BG_TYPES = ["Ar39GenInLAr", "Kr85GenInLAr", "Ar42GenInLAr", "K42From42ArGenInLAr", "Rn222ChainGenInLAr",
            "K42From42ArGenInCPA", "K40inGenInCPA", "U238ChainGenInCPA", "Co60inGenInAPA", "U238ChainGenInAPA",
            "Rn222ChainGenInPDS", "NeutronGenInRock", "GammasGenInRock"]
# "Old" background types
#BG_TYPES = ["Ar39GenInLAr", "Kr85GenInLAr", "Ar42GenInLAr", "Rn222ChainGenInLAr"]

# Number of files to load (go pretty low on these if not sent to a job)
SN_FILE_LIMIT = 20
BG_FILE_LIMIT = 20

# Algorithm parameters
FAKE_TRIGGER_RATE = 1/(60 * 60 * 24 * 30) # 1 trigger per month
BURST_TIME_WINDOW = 1e6 # In microseconds
DISTANCE_TO_OPTIMIZE = 12 # In kpc
SIM_MODE = 'xe' # 'xe' or 'aronly'
ADC_MODE = 'normal' # 'low' or 'normal'
DETECTOR = 'VD' # 'VD' or 'HD'
CLASSIFY = True # True or False (Do we use the BDT? If not, you should iterate over many more clustering parameters)

# Spectrum parameters
# GKVM 23 5
# LIVERMORE 14.4 2.8
# GARCHING 12.2 4.5
AVERAGE_ENERGY = 63.0 # MeV
ALPHA = 5.0 # Dimensionless 

# Clustering parameters over which to search (don't go over too many, the classification does most of the work after all...)
MAX_CLUSTER_TIMES = [0.3] # In microseconds
MAX_HIT_TIME_DIFFS = [0.2] # In microseconds
MAX_HIT_DISTANCES = [250, 525] # In cm
LOWER_MIN_HIT_MULTUPLICITY = 10 # Minimum number of hits in a cluster
UPPER_MIN_HIT_MULTUPLICITY = 12 # Minimum number of hits in a cluster
CLASSIFIER_THRESHOLDS = [0.3, 0.4] # Threshold for the BDT

