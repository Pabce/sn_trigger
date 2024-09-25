'''
trigger_efficiency_computer.py

This module...
'''

import numpy as np
import logging

from rich.logging import RichHandler

from gui import console


class TriggerEfficiencyComputer:

    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)
    

    def evaluate_trigger_efficiency(self, event_num_per_time, sn_event_num, sn_clusters, sn_hit_multiplicities, sn_energies, sn_features, sn_info_per_event,
                            expected_bg_hist, filter_bg_ratios, hbins, fake_trigger_rate, to_plot=False, number_of_tests=1000, classify=False, 
                            tree=None, threshold=0.5, classifier_hist_type="hit_multiplicity"):
        
        # Config reads ----
        pass

        # -----------------

    


    