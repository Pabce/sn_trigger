import pickle
import logging

class DataWithConfig:
    def __init__(self, data, config, logging_level=logging.INFO):
        self.data = data
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    @classmethod
    def load_data_from_file(cls, file_name, data_type_str="data"):
        with open(file_name, "rb") as file:
            data = pickle.load(file)
            data_with_config = cls(data.data, data.config)
            data_with_config.logger.info(f"Loaded {data_type_str} from file: {file_name}")
            return data_with_config

    def save(self, file_name, data_type_str="data"):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)
        self.logger.info(f"Saved {data_type_str} to file: {file_name}")

