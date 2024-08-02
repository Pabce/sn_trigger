'''
config.py

Contains the Config class, which is used to load configuration files and access their values.
Also contains auxiliary functions to load the global parameters used in the simulation.
'''

import yaml
import numpy as np
import pickle
import argparse

class Config:
    def __init__(self, config_path, default_config_path):
        # Load the active config file AND the default config file, 
        # which is used to fill in missing values
        with open(default_config_path, 'r') as file:
            self.default_config = yaml.safe_load(file)
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, *keys, default=None):
        # Get the value from the active config file if it exists, 
        # otherwise get it from the default config file
        value = self.config
        default_value = self.default_config
        try:
            for key in keys:
                value = value[key]
        except KeyError:
            # If we can't find the key in the active config file,
            # we look in the default config file
            value = default_value
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    raise KeyError(f"Configuration value {keys} not found in active or default config file")
        return value
    
    def load_coordinate_arrays(self, detector):
        if detector == "VD":
            file = self.get("IO", "aux_coordinate_dir") + "pdpos_vd1x8x14v5.dat"
        elif detector == "HD":
            file = self.get("IO", "aux_coordinate_dir") + "pdpos_hd1x2x6.dat"

        coords = np.genfromtxt(file, skip_header=1, skip_footer=2)

        # Split into x, y, z arrays
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]

        return x, y, z
    
    def load_op_distance_arrays(self, detector):
        file_dir = self.get("IO", "aux_pickle_dir")
        geometry_version = self.get("Detector", "geometry_version")

        op_distance_array = pickle.load(open("{}/op_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_x_distance_array = pickle.load(open("{}/op_x_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_y_distance_array = pickle.load(open("{}/op_y_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_z_distance_array = pickle.load(open("{}/op_z_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))

        return op_distance_array, op_x_distance_array, op_y_distance_array, op_z_distance_array
    
    def parse_arguments():
        parser = argparse.ArgumentParser()

        # For now, only one argument (the configuration file path) is needed
        # TODO: add the possibility to override configuration values from the command line
        parser.add_argument("--config", type=str, help="Path to the configuration file")

        args = parser.parse_args()

        # Raise an error if no configuration file is provided
        if not args.config:
            parser.error("No configuration file provided")
        
        return args.config
        


# Example usage
if __name__ == "__main__":
    config_instance = Config("../configs/test_config_vd.yaml", "../configs/default_config.yaml")

    print(type(config_instance.config))
    print(config_instance)

    print(config_instance.get("Detector", "geometry_version"))
    print(config_instance.get("Simulation", "fake_trigger_rate"))
    print(config_instance.get("Simulation", "chipmunk"))
    