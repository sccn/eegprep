"""EEGLAB-style signal processing function modules."""

from .cbar import cbar
from .celltomat import celltomat
from .copyaxis import copyaxis
from .eyelike import eyelike
from .fastif import fastif
from .forcelocs import forcelocs
from .headplot import headplot
from .matsel import matsel
from .mattocell import mattocell
from .nan_mean import nan_mean
from .openbdf import openbdf
from .plotcurve import plotcurve
from .quantile import quantile
from .readbdf import readbdf
from .sbplot import sbplot
from .shuffle import shuffle
from .slider import slider
from .writegdf import writegdf

__all__ = [
    "cbar",
    "celltomat",
    "copyaxis",
    "eyelike",
    "fastif",
    "forcelocs",
    "headplot",
    "matsel",
    "mattocell",
    "nan_mean",
    "openbdf",
    "plotcurve",
    "quantile",
    "readbdf",
    "sbplot",
    "shuffle",
    "slider",
    "writegdf",
]
