import os
import yaml
import csv
from typing import NamedTuple
from netket.utils.types import PyTree, Any
from timeit import default_timer as timer
from datetime import timedelta, time


def save_config(workdir, config):
    os.makedirs(workdir, exist_ok=True)
    filepath = os.path.join(workdir, "config.yaml")
    with open(filepath, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

def read_config(workdir):
    if os.path.isdir(workdir):
        filepath = os.path.join(workdir, "config.yaml")
    else:
        raise ValueError(f"The argument {workdir} is not a valid path to an existing directory.")
    if not os.path.isfile(filepath):
        raise ValueError(f"No config file found at {filepath}")
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = ['Step']+list(fieldnames)
        file_exists = os.path.isfile(filename)
        if not file_exists:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, mode='w') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def __call__(self, step, metrics):
        with open(self.filename, mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({'Step': step, **metrics})

class VMCState(NamedTuple):
    parameters: PyTree
    opt_state: Any
    step: int

class Timer:
    """
    A timer object to keep track of runtime, elapsed time and remaining time in an optimization loop
    """

    def __init__(self, total_steps : int) -> None:
        self._total_steps : int = total_steps
        self._prev_step : int = 0
        self._elapsed_time : time = None
        self._runtime : timedelta = None
        self._remaining_time : time = None
        self._start : float = timer()
        self._prev : float = self._start

    def update(self, step : int):
        """Updates runtime, elapsed time and remaining time"""
        now = timer()
        self._elapsed_time = timedelta(seconds=now-self._start)
        self._runtime = timedelta(seconds=now-self._prev)/(step-self._prev_step)
        self._remaining_time = self._runtime*(self._total_steps-step)
        self._prev = now
        self._prev_step = step

    @property
    def elapsed_time(self) -> str:
        """Time elapsed since beginning of loop"""
        return self._strftimedelta(self._elapsed_time)

    @property
    def runtime(self) -> float:
        """Current runtime of a single loop iteration"""
        return self._runtime.total_seconds()

    @property
    def remaining_time(self) -> str:
        """Estimated time left until completion"""
        return self._strftimedelta(self._remaining_time)

    def __repr__(self) -> str:
        return f'[{self.elapsed_time}<{self.remaining_time}, {self.runtime:.2f}s/it]'

    def _strftimedelta(self, delta: timedelta) -> str:
        """Returns a timedelta object formatted as a string"""
        days, seconds = delta.days, delta.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = (seconds % 60)
        s = ""
        if days > 0:
            s += f"{days}d "
        s += f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return s