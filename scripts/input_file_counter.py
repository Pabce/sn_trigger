import logging

import numpy as np

from stages import *
from gui import console
import gui
import config as cf
from argument_parser import parse_arguments
import data_loader as dl

# AVAILABLE_STAGES = {
#     "load_events": LoadEvents,
# }

# Read the command line for the configuration file, and possibly the input and output files
config_path, input_path, output_path, output_info_path = parse_arguments()

# Create the configuration object, including the "loaded" parameters
config = cf.Configurator.from_file(config_path, logging_level=logging.ERROR) # Instance of the config class

# Create a LoadEvents object
# load_events_stage = LoadEvents(config, input_data=None, logging_level=logging.WARNING)

# Create a dataloader object
loader = dl.DataLoader(config, logging_level=logging.ERROR)

# Config reads ----
detector_type = config.get("Detector", "type")
sn_data_dir = config.get("Simulation", "load_events", "sn_data_dir")
sim_mode = config.get('Simulation', 'load_events', 'sim_mode')
event_num = config.get("DataFormat", "sn_event_number_per_file")
startswith = config.get("Simulation", "load_events", "sn_hit_file_start_pattern")
endswith = config.get("Simulation", "load_events", "sn_hit_file_end_pattern").format(sim_mode)
info_endswith = config.get("Simulation", "load_events", "sn_info_file_end_pattern")
photon_endswith = config.get("Simulation", "load_events", "photon_file_end_pattern")
# -----------------
bg_data_dir = config.get("Simulation", "load_events", "bg_data_dir")
bg_types = config.get("Backgrounds")
bg_sample_number_per_file = config.get("DataFormat", "bg_sample_number_per_file")
bg_startswith = config.get("Simulation", "load_events", "bg_hit_file_start_pattern")
bg_endswith = config.get("Simulation", "load_events", "bg_hit_file_end_pattern").format(sim_mode)
# -----------------

# Count the number of SN input files available
# Collect valid file names
reco_file_names, info_file_names, photon_file_names = loader.collect_valid_file_names(
    sn_data_dir,
    startswith,
    endswith,
    info_endswith,
    event_num,
    photon_endswith,
    load_photon_info=False,
    limit=1e6,
    offset=0,
    data_type_str="SN data"
)

#print(reco_file_names)
print(f'Number of SN files available: {len(reco_file_names)}')

# Count the number of BG input files available   
directories = []
for bg_type in bg_types:
    #directory = os.fsencode(bg_data_dir + bg_type + '/')
    directory = bg_data_dir + bg_type + '/'
    directories.append(directory)

# Collect valid file names
reco_file_names_per_type = []
bg_files_available = []
for i, directory in enumerate(directories):
    bg_type = bg_types[i]
    
    reco_file_names, _, _ = loader.collect_valid_file_names(
        directory,
        bg_startswith,
        bg_endswith,
        limit=1e6,
        offset=0,
        data_type_str=f'{bg_type} BG data',
    )

    reco_file_names_per_type.append(reco_file_names)
    print(f'{bg_type}: {len(reco_file_names)}')
    bg_files_available.append(len(reco_file_names))

print(f'Number of BG files available (min of all types): {np.min(bg_files_available)}')
