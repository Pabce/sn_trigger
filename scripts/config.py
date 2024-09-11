'''
config.py

Contains the Config class, which is used to load configuration files and access their values.
Also contains auxiliary functions to load the global parameters used in the simulation.
'''

import yaml
import numpy as np
import pickle
import argparse

class Configurator:
    DEFAULT_CONFIG_PATH = "../configs/default_config.yaml"

    def __init__(self, config_path="../nope.yaml", default_config_path=DEFAULT_CONFIG_PATH):
        # Load the active config file AND the default config file, 
        # which is used to fill in missing values
        with open(default_config_path, 'r') as file:
            self.default_config = yaml.safe_load(file)
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # We will also load extra arrays that are used in the simulation
        # under the "loaded" member variables
        self.loaded = {}
        self.load_coodinates()

    def load_coodinates(self):
        # Load coordinate arrays
        detector = self.get("Detector", "type")
        x, y, z = self.load_coordinate_arrays(detector)

        self.loaded["x_coords"] = x
        self.loaded["y_coords"] = y
        self.loaded["z_coords"] = z

        # Load optical distance arrays
        op_distance_array, op_x_distance_array, op_y_distance_array, op_z_distance_array = self.load_op_distance_arrays(detector)

        self.loaded["op_distance_array"] = op_distance_array
        self.loaded["op_x_distance_array"] = op_x_distance_array
        self.loaded["op_y_distance_array"] = op_y_distance_array
        self.loaded["op_z_distance_array"] = op_z_distance_array
    
    # Constructor for when the config file is grabbed from the command line
    @classmethod
    def file_from_command_line(cls):
        config_path = cls.parse_arguments()
        return cls(config_path=config_path)
    
    # Constructor for using the default config file
    @classmethod
    def default_config(cls):
        return cls(config_path=cls.DEFAULT_CONFIG_PATH)
        

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
    
    def set_value(self, *keys, value):
        # Set a value in the active config file
        config = self.config
        default_config = self.default_config

        try:
            for key in keys[:-1]:
                config = config[key]
        except KeyError:
            # If we can't find the key in the active config file,
            # we look in the default config file
            config = default_config
            for key in keys[:-1]:
                if key in config:
                    config = config[key]
                else:
                    raise KeyError(f"Configuration value {keys} not found in active or default config file")

        config[keys[-1]] = value

        # Just in case we have modified a relevant path, we reload the coordinates
        self.load_coodinates()

    
    def load_coordinate_arrays(self, detector):
        if detector == "VD":
            file = self.get("IO", "aux_coordinate_dir") + "pdpos_vd1x8x14v5.dat"
        elif detector == "HD":
            file = self.get("IO", "aux_coordinate_dir") + "pdpos_hd1x2x6.dat"

        print("Loading coordinates from file: ", file)
        coords = np.genfromtxt(file, skip_header=1, skip_footer=2)

        # Split into x, y, z arrays
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]

        return x, y, z
    
    def load_op_distance_arrays(self, detector):
        file_dir = self.get("IO", "aux_data_dir")
        geometry_version = self.get("Detector", "geometry_version")

        op_distance_array = pickle.load(open("{}/op_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_x_distance_array = pickle.load(open("{}/op_x_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_y_distance_array = pickle.load(open("{}/op_y_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))
        op_z_distance_array = pickle.load(open("{}/op_z_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb"))

        return op_distance_array, op_x_distance_array, op_y_distance_array, op_z_distance_array
    
    @staticmethod 
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
    