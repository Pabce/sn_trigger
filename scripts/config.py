'''
config.py

Contains the Config class, which is used to load configuration files and access their values.
Also contains auxiliary functions to load the global parameters used in the simulation.
'''

import os
import yaml
import numpy as np
import pickle
import argparse
import logging
import mergedeep

class Configurator:
    DEFAULT_CONFIG_FILE_VD = "../configs/default_config_vd.yaml"
    DEFAULT_CONFIG_FILE_HD = "../configs/default_config_hd.yaml"

    def __init__(self, config_path="../nope.yaml", default_config_path=None, logging_level=logging.INFO):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)
        
        # Load the active config file AND the default config file, 
        # which is used to fill in missing values
        with open(config_path, 'r') as file:
            self.yaml_dict = yaml.safe_load(file)
            self.log.info(f"Configuration file loaded from: {config_path}")

        with open(default_config_path, 'r') as file:
            self.default_yaml_dict = yaml.safe_load(file)
            self.log.info(f"Default config file is: {default_config_path}")
        # We will also load extra arrays that are used in the simulation
        # under the "loaded" member variables
        self.loaded = {}
        self.load_coodinates()

    # Constructor for when the config file is grabbed from the command line
    @classmethod
    def from_file(cls, config_path, logging_level=logging.INFO):
        # Get the absolute path of the config file, if it is not already
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

        # Get the default config file from the active config file
        with open(config_path, 'r') as file:
            temp_yaml_dict = yaml.safe_load(file)

            # If the default config file is None, use the active config file as the default
            if temp_yaml_dict["DEFAULT_CONFIG_FILE"] is None:
                default_config_path = config_path
            else:
                default_config_path = temp_yaml_dict["DEFAULT_CONFIG_FILE"]
        
        # Get the absolute path of the default config file, if it is not already
        if not os.path.isabs(default_config_path):
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_config_path)

        return cls(config_path=config_path, default_config_path=default_config_path, logging_level=logging_level)
    
    # Constructor for using the default config file for either detector
    @classmethod
    def detector_default(cls, detector_type, logging_level=logging.INFO):
        if detector_type == "VD":
            return cls(config_path=cls.DEFAULT_CONFIG_FILE_VD, default_config_path=cls.DEFAULT_CONFIG_FILE_VD, logging_level=logging_level)
        elif detector_type == "HD":
            return cls(config_path=cls.DEFAULT_CONFIG_FILE_HD, default_config_path=cls.DEFAULT_CONFIG_FILE_HD, logging_level=logging_level)
        else:
            raise ValueError("Invalid detector type")

    # Return the full dictionary, with the active config file values overriding the default config file values
    # Note: does not include the "loaded" member variables
    def get_dict(self):
        merged_dict = mergedeep.merge({}, self.default_yaml_dict, self.yaml_dict)

        return merged_dict

    def get(self, *keys, default=None):
        # Get the value from the active config file if it exists, 
        # otherwise get it from the default config file
        value = self.yaml_dict
        default_value = self.default_yaml_dict
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
    
    def set_value(self, *keys, value=None):
        # Set a value in the active config file
        yaml_dict = self.yaml_dict
        default_yaml_dict = self.default_yaml_dict

        try:
            for key in keys[:-1]:
                yaml_dict = yaml_dict[key]
        except KeyError:
            # If we can't find the key in the active config file,
            # we look in the default config file
            yaml_dict = default_yaml_dict
            for key in keys[:-1]:
                if key in yaml_dict:
                    yaml_dict = yaml_dict[key]
                else:
                    raise KeyError(f"Configuration value {keys} not found in active or default config file")

        yaml_dict[keys[-1]] = value

        # Just in case we have modified a relevant path, we reload the coordinates
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

    # TODO: make the file names configurable
    def load_coordinate_arrays(self, detector):
        if detector == "VD":
            file = self.get("DataFormat", "aux_coordinate_dir") + "pdpos_vd1x8x14v5.dat"
        elif detector == "HD":
            file = self.get("DataFormat", "aux_coordinate_dir") + "pdpos_hd1x2x6.dat"

        # Make path absolute if it is not already
        if not os.path.isabs(file):
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)

        coords = np.genfromtxt(file, skip_header=1, skip_footer=2)
        self.log.info(f"Loaded detector coordinates from file: {file}")

        # Split into x, y, z arrays
        x = coords[:, 1]
        y = coords[:, 2]
        z = coords[:, 3]

        return x, y, z
    
    def load_op_distance_arrays(self, detector):
        file_dir = self.get("DataFormat", "aux_data_dir")
        geometry_version = self.get("Detector", "geometry_version")

        # Make path absolute if it is not already
        if not os.path.isabs(file_dir):
            file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_dir)

        # TODO: switch from pickle to text file, make function to create these accessible
        with open("{}/op_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb") as f:
            op_distance_array = pickle.load(f)
        with open("{}/op_x_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb") as f:
            op_x_distance_array = pickle.load(f)
        with open("{}/op_y_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb") as f:
            op_y_distance_array = pickle.load(f)
        with open("{}/op_z_distance_array_{}_{}".format(file_dir, detector, geometry_version), "rb") as f:
            op_z_distance_array = pickle.load(f)

        return op_distance_array, op_x_distance_array, op_y_distance_array, op_z_distance_array
        


# Example usage
if __name__ == "__main__":
    config_instance = Configurator("../configs/test_config_vd.yaml", "../configs/default_config.yaml")

    print(type(config_instance.config))
    print(config_instance)

    print(config_instance.get("Detector", "geometry_version"))
    print(config_instance.get("Simulation", "fake_trigger_rate"))
    print(config_instance.get("Simulation", "chipmunk"))
    