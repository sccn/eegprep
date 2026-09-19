"""EEGLAB-style miscellaneous numerical functions."""

from .abspeak import abspeak
from .averef import averef
from .covary import covary
from .datlim import datlim
from .eucl import eucl
from .gabor2d import gabor2d
from .gauss import gauss
from .gauss2d import gauss2d
from .gauss3d import gauss3d
from .hungarian import hungarian
from .laplac2d import laplac2d
from .mapcorr import mapcorr
from .matcorr import matcorr
from .matperm import matperm
from .means import means
from .nan_std import nan_std
from .pcexpand import pcexpand
from .pcsquash import pcsquash
from .perminv import perminv
from .scanfold import scanfold
from .uniquef import uniquef
from .vectdata import vectdata

__all__ = [
    "abspeak",
    "averef",
    "covary",
    "datlim",
    "eucl",
    "gabor2d",
    "gauss",
    "gauss2d",
    "gauss3d",
    "hungarian",
    "laplac2d",
    "mapcorr",
    "matcorr",
    "matperm",
    "means",
    "nan_std",
    "pcexpand",
    "pcsquash",
    "perminv",
    "scanfold",
    "uniquef",
    "vectdata",
]
