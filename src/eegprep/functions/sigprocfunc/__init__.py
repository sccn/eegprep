"""EEGLAB-style signal processing function modules."""

from .celltomat import celltomat
from .eyelike import eyelike
from .fastif import fastif
from .matsel import matsel
from .mattocell import mattocell
from .nan_mean import nan_mean
from .openbdf import openbdf
from .quantile import quantile
from .readbdf import readbdf
from .shuffle import shuffle

__all__ = [
    "celltomat",
    "eyelike",
    "fastif",
    "matsel",
    "mattocell",
    "nan_mean",
    "openbdf",
    "quantile",
    "readbdf",
    "shuffle",
]
