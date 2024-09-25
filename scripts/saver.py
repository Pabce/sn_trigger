import pickle
import logging

class DataWithConfig:
    def __init__(self, data, config, logging_level=logging.INFO):
        self.data = data
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    @classmethod
    def load_data_from_file(cls, file_name):
        with open(file_name, "rb") as file:
            data = pickle.load(file)
            data_with_config = cls(data.data, data.config)
            data_with_config.logger.info(f"Loaded data from file: {file_name}")
            return data_with_config

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)
        self.logger.info(f"Saved data to file: {file_name}")

# TODO: PUT THIS INTO A CLASS
# TODO: Make sim_parameters a dictionary!!!
# def save_efficiency_data(eff_data, sim_parameters, file_name=None, data_type="data"):
#     #ftr, btw, dist, sim_mode, adc_mode, detector, classify, avg_energy, alpha = sim_parameters
#     sim_parameters = tuple(sim_parameters)

#     # Generate a random string to identify the file
#     if file_name is None:
#         random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
#         file_name = "efficiency_{}_".format(data_type) + random_string + ".pcl"
    
#     # Save this correspondence to a dictionary
#     try:
#         file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
#         if not isinstance(file_names, dict):
#                 file_names = {}
    
#         # Check if entry with these sim_parameters already exists
#         if sim_parameters in file_names.keys():
#             file_name = file_names[sim_parameters]

#         file_names[sim_parameters] = file_name

#     except FileNotFoundError:
#         file_names = {}
#         file_names[sim_parameters] = file_name

#     pickle.dump(file_names, open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "wb"))

#     # Save the data
#     pickle.dump(eff_data, open("{}/{}".format(SAVE_PATH, file_name), "wb"))

#     print("Saved efficiency {} to file:".format(data_type), file_name)

#     return file_name


# def load_efficiency_data(sim_parameters=[], file_name=None, data_type="data"):
#     # Check if the dict file exists
#     try:
#         file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
#     except FileNotFoundError:
#         file_names = {}
#         pickle.dump(file_names, open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "wb"))

#     sim_parameters = tuple(sim_parameters)
#     if file_name is None:
#         file_names = pickle.load(open("{}/{}".format(SAVE_PATH, "file_names_dict_{}.pcl".format(data_type)), "rb"))
#         # Find the file name corresponding to the sim_parameters
#         file_name = file_names[sim_parameters]

#     # Load the data
#     eff_data = pickle.load(open("{}/{}".format(SAVE_PATH, file_name), "rb"))
#     print("Loaded efficiency {} from file {}".format(data_type, file_name))

#     return eff_data, file_name