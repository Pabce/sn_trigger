from contextlib import contextmanager
import time
import random
import logging
import warnings
import sys

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



if __name__ == "__main__":
    pass

