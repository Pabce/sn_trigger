from contextlib import contextmanager
import time
import random
import logging
import warnings
import sys

import numpy as np
import rich
from rich.progress import track, Progress, TextColumn, TimeElapsedColumn,\
    BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install
from contextlib import redirect_stdout, redirect_stderr

from rich.panel import Panel
from rich.align import Align
from rich.text import Text

from utils import get_total_size

# Create a Rich console instance
console = Console()
# Install rich traceback handler
install()

# Set up global logger
# TODO: read logging level from config
logging.basicConfig(
    level=logging.INFO, 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)

def get_custom_progress() -> Progress:
    """
    Returns a Progress object with custom configuration.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

@contextmanager
def live_progress(console, status_fstring):
    progress = get_custom_progress()
    spinner = Spinner("dots", text=status_fstring, style="green")
    group = Group(progress, spinner)

    with Live(group, console=console, refresh_per_second=60) as live:
        yield progress, live, group


def get_custom_table(config, table_name, **kwargs) -> Table:
    table = None

    if table_name == "sim_parameters":
        table = Table(title="Parameters", title_style="bold yellow", title_justify="left")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Detector", f"{config.get('Detector', 'type')}")
        
        stages = config.get("Stages")

        if 'load_events' in stages:
            table.add_row("Simulation mode", f"{config.get('Simulation', 'load_events', 'sim_mode')}")

        if 'clustering' in stages:
            clustering_params = config.get("Simulation", "clustering", "parameters")
            for key, value in clustering_params.items():
                table.add_row(key, str(value))
        
        if 'trigger_efficiency' in stages:
            sn_spectra = config.get('Simulation', 'trigger_efficiency', 'physics', 'supernova_spectra')
            # This will be an individual spectrum or a list of spectra
            # If an individual spectrum, convert into a list
            if not isinstance(sn_spectra, list):
                sn_spectra = [sn_spectra]
            
            model_names = []
            average_energies = []
            alphas = []

            for spectrum in sn_spectra:
                spec_type = spectrum.get("spec_type")
                if spec_type == "pinching":
                    average_energy = spectrum.get("pinching_parameters").get("average_energy")
                    alpha = spectrum.get("pinching_parameters").get("alpha")
                    model_name = spectrum.get("label")
                elif spec_type == "model_name":
                    import supernova_spectrum
                    model_name = spectrum.get("model_name")
                    average_energy = supernova_spectrum.MODEL_PINCHED_PARAMETERS[model_name]["average_energy"]
                    alpha = supernova_spectrum.MODEL_PINCHED_PARAMETERS[model_name]["alpha"]
                else:
                    raise ValueError("Invalid supernova spectrum type")
            
                model_names.append(model_name)
                average_energies.append(average_energy)
                alphas.append(alpha)

            if model_names:
                table.add_row("Model names", f"{model_names}")
            table.add_row("Average energies", f"{average_energies} MeV")
            table.add_row("Alphas", f"{alphas}")

            sn_channels = config.get("Simulation", "load_events", "sn_channels")
            table.add_row("SN channels", f"{sn_channels}")
            
            distance_to_evaluate = config.get("Simulation", "trigger_efficiency", "distance_to_evaluate")
            number_of_interactions_to_evaluate = config.get("Simulation", "trigger_efficiency", "number_of_interactions_to_evaluate")
            if number_of_interactions_to_evaluate is not None:
                table.add_row("Number of interactions to evaluate", f"{number_of_interactions_to_evaluate}")
            elif distance_to_evaluate is not None:
                table.add_row("Distance to evaluate", f"{distance_to_evaluate} kpc")
            table.add_row("Burst time window", f"{config.get('Simulation', 'trigger_efficiency', 'burst_time_window')/1000} ms")
            table.add_row("Fake trigger rate", f"{config.get('Simulation', 'trigger_efficiency', 'fake_trigger_rate')} Hz")

    
    elif table_name == "memory_usage":
        sn_channels = config.get("Simulation", "load_events", "sn_channels")

        for sn_channel in sn_channels:
            sn_hits = kwargs.get("memory_usage").get(f"sn_hits_{sn_channel}")
            sn_info = kwargs.get("memory_usage").get(f"sn_info_{sn_channel}")
        bg_hits = kwargs.get("memory_usage").get(f"bg_hits")

        table = Table(title="Memory Usage of Loaded Data", title_style="bold yellow", title_justify="left")

        table.add_column("Data Type", justify="left", style="green", no_wrap=True)
        table.add_column("Memory Usage (MB)", justify="right", style="red")

        for sn_channel in sn_channels:
            table.add_row(f"SN hits ({sn_channel})", f"{sn_hits:.2f}")
            table.add_row(f"SN event info ({sn_channel})", f"{sn_info:.2f}")
        table.add_row("BG hits", f"{bg_hits:.2f}")

    elif table_name == "data_loading_statistics":
        sn_channels = config.get("Simulation", "load_events", "sn_channels")

        sn_hit_num = {}
        sn_eff_hit_num = {}
        sn_bdt_hit_num = {}
        sn_event_num = {}
        sn_eff_event_num = {}
        sn_bdt_event_num = {}
        for sn_channel in sn_channels:
            sn_hit_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_hit_num_{sn_channel}")
            sn_eff_hit_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_eff_hit_num_{sn_channel}")
            sn_bdt_hit_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_bdt_hit_num_{sn_channel}")
            sn_event_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_event_num_{sn_channel}")
            sn_eff_event_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_eff_event_num_{sn_channel}")
            sn_bdt_event_num[sn_channel] = kwargs.get("data_loading_statistics").get(f"sn_bdt_event_num_{sn_channel}")

        bg_hit_num = kwargs.get("data_loading_statistics").get("bg_hit_num")
        bg_eff_hit_num = kwargs.get("data_loading_statistics").get("bg_eff_hit_num")
        bg_bdt_hit_num = kwargs.get("data_loading_statistics").get("bg_bdt_hit_num")

        bg_event_num = kwargs.get("data_loading_statistics").get("bg_event_num")
        bg_eff_event_num = kwargs.get("data_loading_statistics").get("bg_eff_event_num")
        bg_bdt_event_num = kwargs.get("data_loading_statistics").get("bg_bdt_event_num")
        total_bg_time_window = kwargs.get("data_loading_statistics").get("total_bg_time_window")
        bg_eff_time_window = kwargs.get("data_loading_statistics").get("bg_eff_time_window")
        bg_bdt_time_window = kwargs.get("data_loading_statistics").get("bg_bdt_time_window")

        table = Table(title="Data Loading Statistics", title_style="bold yellow", title_justify="left")

        table.add_column("Category", justify="left", style="green", no_wrap=True)
        table.add_column("Total", justify="right", style="bold magenta")
        table.add_column("Efficiency", justify="right", style="bold red")
        table.add_column("Training", justify="right", style="bold blue")

        for sn_channel in sn_channels:
            table.add_row(f"Number of SN hits ({sn_channel})", 
                        f"{sn_hit_num[sn_channel]:,}",
                        f"{sn_eff_hit_num[sn_channel]:,}",
                        f"{sn_bdt_hit_num[sn_channel]:,}")
        table.add_row("Number of BG hits", 
                    f"{bg_hit_num:,}",
                    f"{bg_eff_hit_num:,}",
                    f"{bg_bdt_hit_num:,}")
        for sn_channel in sn_channels:
            table.add_row(f"Total SN events ({sn_channel})", 
                        f"{sn_event_num[sn_channel]:,}",
                        f"{sn_eff_event_num[sn_channel]:,}",
                        f"{sn_bdt_event_num[sn_channel]:,}")
        table.add_row("Total BG \"events\"", 
                    f"{bg_event_num:,}",
                    f"{bg_eff_event_num:,}",
                    f"{bg_bdt_event_num:,}")
        table.add_row("Total BG time window (ms)",
                    f"{total_bg_time_window:,}",
                    f"{bg_eff_time_window:,}",
                    f"{bg_bdt_time_window:,}")
    
    elif table_name == "clustering_parameters":
        clustering_parameters = kwargs.get("clustering_parameters")
        table = Table(title="Clustering Parameters", title_style="bold yellow", title_justify="left")
        table.add_column("Parameter", justify="left", style="green", no_wrap=True)
        table.add_column("Value", justify="right", style="red")
        for key, value in clustering_parameters.items():
            table.add_row(key, str(value))
    
    elif table_name == "clustering_statistics":
        sn_channels = config.get("Simulation", "load_events", "sn_channels")

        sn_eff_clusters_num = {}
        sn_bdt_clusters_num = {}
        average_sn_eff_clusters_per_event = {}
        average_sn_bdt_clusters_per_event = {}
        average_sn_eff_hit_multiplicity = {}
        average_sn_bdt_hit_multiplicity = {}
        for sn_channel in sn_channels:
            sn_eff_clusters_num[sn_channel] = kwargs.get(f"sn_eff_clusters_num_{sn_channel}")
            sn_bdt_clusters_num[sn_channel] = kwargs.get(f"sn_bdt_clusters_num_{sn_channel}")
            average_sn_eff_clusters_per_event[sn_channel] = kwargs.get(f"average_sn_eff_clusters_per_event_{sn_channel}")
            average_sn_bdt_clusters_per_event[sn_channel] = kwargs.get(f"average_sn_bdt_clusters_per_event_{sn_channel}")
            average_sn_eff_hit_multiplicity[sn_channel] = kwargs.get(f"average_sn_eff_hit_multiplicity_{sn_channel}")
            average_sn_bdt_hit_multiplicity[sn_channel] = kwargs.get(f"average_sn_bdt_hit_multiplicity_{sn_channel}")

        bg_eff_clusters_num = kwargs.get("bg_eff_clusters_num")
        bg_bdt_clusters_num = kwargs.get("bg_bdt_clusters_num")
        average_bg_eff_clusters_per_event = kwargs.get("average_bg_eff_clusters_per_event")
        average_bg_bdt_clusters_per_event = kwargs.get("average_bg_bdt_clusters_per_event")
        average_bg_eff_hit_multiplicity = kwargs.get("average_bg_eff_hit_multiplicity")
        average_bg_bdt_hit_multiplicity = kwargs.get("average_bg_bdt_hit_multiplicity")

        table = Table(title="Clustering Statistics", title_style="bold yellow", title_justify="left")
        table.add_column("Category", style="green", no_wrap=True)
        table.add_column("Efficiency", style="red")
        table.add_column("Training", style="blue")

        for sn_channel in sn_channels:
            table.add_row(f"Number of SN clusters ({sn_channel})",
                        f"{sn_eff_clusters_num[sn_channel]:,}",
                        f"{sn_bdt_clusters_num[sn_channel]:,}")
        table.add_row("Number of BG clusters",
                    f"{bg_eff_clusters_num:,}",
                    f"{bg_bdt_clusters_num:,}")
        
        for sn_channel in sn_channels:
            table.add_row(f"Average clusters per event (SN {sn_channel})",
                        f"{average_sn_eff_clusters_per_event[sn_channel]:.2f}",
                        f"{average_sn_bdt_clusters_per_event[sn_channel]:.2f}")
        table.add_row("Average clusters per \"event\" (BG)",
                    f"{average_bg_eff_clusters_per_event:.2f}",
                    f"{average_bg_bdt_clusters_per_event:.2f}")
        
        for sn_channel in sn_channels:
            table.add_row(f"Average hit multiplicity (SN {sn_channel})",
                        f"{average_sn_eff_hit_multiplicity[sn_channel]:.2f}",
                        f"{average_sn_bdt_hit_multiplicity[sn_channel]:.2f}")
        table.add_row("Average hit multiplicity (BG)",
                    f"{average_bg_eff_hit_multiplicity:.2f}",
                    f"{average_bg_bdt_hit_multiplicity:.2f}")

    return table



if __name__ == "__main__":
    pass

