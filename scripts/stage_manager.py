import logging

import numpy as np

from stages import *
from gui import console
import gui

AVAILABLE_STAGES = {
    "load_events": LoadEvents,
    "clustering": Clustering,
    "cluster_feature_extraction": ClusterFeatureExtraction,
    "bdt_training": BDTTraining,
    "trigger_efficiency": TriggerEfficiency,
    "wtf": None
}

class StageManager:

    def __init__(self, config, stage_name_list, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)
        self.logging_level = logging_level

        self.stage_name_list = stage_name_list
        self.initial_stage_name = stage_name_list[0]
        self.final_stage_name = stage_name_list[-1]

        # Assert that the stages are in the correct order, etc
        self.assert_order()

        self.stages_with_saved_output_data = self.config.get("Output", "stages_with_saved_output_data")
        self.output_data_file = self.config.get("Output", "output_data_file")
        self.error_behavior = self.config.get("Output", "error_behavior")
    
    def check_input(self):
        # Make sure that the input provided for the first stage is correct
        pass
    
    def assert_order(self):
        # Check that the stages in the stage_list are in the correct order
        available_stages_list = list(AVAILABLE_STAGES.keys())
        
        # First, are there any stages that are not in AVAILABLE_STAGES?
        for stage in self.stage_name_list:
            if stage not in available_stages_list:
                logging.error(f"Stage [b]{stage}[/b] is not available.", extra={"markup": True})
                logging.error(f"Available stages are: {available_stages_list}")
                raise ValueError(f"Stage {stage} is not available.")

        # Second, are the stages ordered correctly?
        ordered_stage_name_list = [stage for stage in available_stages_list if stage in self.stage_name_list]
        if self.stage_name_list != ordered_stage_name_list:
            logging.error(f"Stages are not in the correct order.")
            logging.error(f"Provided stages are: {self.stage_name_list}")
            logging.error(f"Available stages are (in this order): {available_stages_list}")
            raise ValueError(f"Stages are not in the correct order.")
        
        # Third, are there any mandatory stages missing between the first and last stage?
        first_stage_index = available_stages_list.index(self.stage_name_list[0])
        last_stage_index = available_stages_list.index(self.stage_name_list[-1])

        for i in range(first_stage_index, last_stage_index):
            if AVAILABLE_STAGES[available_stages_list[i]].IS_OPTIONAL:
                continue
            
            if available_stages_list[i] not in self.stage_name_list:
                logging.error(f"Stage [b]{available_stages_list[i]}[/b] is mandatory between stage [b]{self.stage_name_list[0]}[/b] and stage [b]{self.stage_name_list[-1]}[/b].",
                               extra={"markup": True})
                logging.error(f"Provided stages are: {self.stage_name_list}")
                raise ValueError(f"Stage {available_stages_list[i]} is mandatory.")
    
    def run_stages(self):
        # Run the stages in the stage_list
        # If the initial stage is "load_events", we load the class without input data.
        # Else, we need to assume that an input file has been provided.

        cumulative_stage_output_data = {}
        cumulative_stage_info_output_data = {}
        # Variable to store a possible exception string
        exception = None

        for i, stage_name in enumerate(self.stage_name_list):
            stage_class = AVAILABLE_STAGES[stage_name]
            
            # If this is the first stage, we'll need to load the data from the input file
            # (unless it's the "load_events" stage, which doesn't require input data)
            if i == 0:
                if self.initial_stage_name == "load_events":
                    stage = stage_class(self.config, input_data=None, logging_level=self.logging_level)
                else:
                    input_data_file = self.config.get("Input", "input_data_file")
                    if input_data_file is None:
                        logging.error("No input file provided. Required for stage {stage_name}.")
                        raise ValueError("No input file provided. Required for stage {stage_name}.")
                    stage = stage_class.from_input_file(self.config, input_data_file, logging_level=self.logging_level)
                    cumulative_stage_output_data.update(stage.input_data)

            else:
                input_data = {ri: cumulative_stage_output_data[ri] for ri in stage_class.REQUIRED_INPUTS}
                stage = stage_class(self.config, input_data=input_data, logging_level=self.logging_level)
            
            try:
                stage_output_data, stage_info_output_data = stage.run()
            except Exception as e:
                logging.error(f"Error in stage {stage_name}: {e}")
                exception = f'Error in stage {stage_name}: {e}'

                # TODO: Fix this
                if self.error_behavior == "kill":
                    raise e
                elif self.error_behavior == "graceful":
                    break
                else:
                    raise ValueError(f"Invalid error behavior: {self.error_behavior}")

            cumulative_stage_output_data.update(stage_output_data)
            cumulative_stage_info_output_data[stage_name] = stage_info_output_data

            # Save the output data if required
            if self.stages_with_saved_output_data is not None:
                if self.stages_with_saved_output_data == "all"\
                or stage_name in self.stages_with_saved_output_data\
                or (self.stages_with_saved_output_data == "last" and stage_name == self.final_stage_name):
                    
                    output_with_config = sv.DataWithConfig(cumulative_stage_output_data, self.config)

                    console.log(f"[bold yellow]Saving the following output data:")
                    console.log(list(cumulative_stage_output_data.keys())) 

                    try:
                        output_with_config.save(self.output_data_file[stage_name], data_type_str=f"{stage_name} output data")
                    except TypeError:
                        output_with_config.save(self.output_data_file, data_type_str=f"{stage_name} output data")

            stage.exit()
        
        if self.stages_with_saved_output_data is None:
            self.log.info("[bold]No output file provided, not saving output data", extra={"markup": True})
        
        return cumulative_stage_output_data, cumulative_stage_info_output_data, exception

