"""EEGLAB-style time-frequency helper functions."""

from eegprep.functions.timefreqfunc.bootstat import BootstrapResult, bootstat
from eegprep.functions.timefreqfunc.correct_mc import correct_mc
from eegprep.functions.timefreqfunc.correctfit import correctfit
from eegprep.functions.timefreqfunc.crossf import crossf
from eegprep.functions.timefreqfunc.dftfilt import dftfilt
from eegprep.functions.timefreqfunc.dftfilt2 import dftfilt2
from eegprep.functions.timefreqfunc.dftfilt3 import dftfilt3
from eegprep.functions.timefreqfunc.angtimewarp import angtimewarp
from eegprep.functions.timefreqfunc.newcrossf import CrossFrequencyResult, newcrossf
from eegprep.functions.timefreqfunc.newtimef import TimeFrequencyResult, newtimef
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.newtimefitc import newtimefitc
from eegprep.functions.timefreqfunc.newtimefpowerunit import newtimefpowerunit
from eegprep.functions.timefreqfunc.newtimeftrialbaseln import newtimeftrialbaseln
from eegprep.functions.timefreqfunc.pac import PacResult, pac
from eegprep.functions.timefreqfunc.pac_cont import PacContResult, pac_cont
from eegprep.functions.timefreqfunc.rsadjust import rsadjust
from eegprep.functions.timefreqfunc.rsfit import rsfit
from eegprep.functions.timefreqfunc.rsget import rsget
from eegprep.functions.timefreqfunc.rspdfsolv import rspdfsolv
from eegprep.functions.timefreqfunc.rspfunc import rspfunc
from eegprep.functions.timefreqfunc.tf_cycle_calc import TFCycleCalcResult, tf_cycle_calc
from eegprep.functions.timefreqfunc.timef import timef
from eegprep.functions.timefreqfunc.timefreq import TimeFrequencyDecomposition, timefreq
from eegprep.functions.timefreqfunc.timewarp import timewarp

__all__ = [
    "BootstrapResult",
    "CrossFrequencyResult",
    "PacContResult",
    "PacResult",
    "TFCycleCalcResult",
    "TimeFrequencyDecomposition",
    "TimeFrequencyResult",
    "angtimewarp",
    "bootstat",
    "correct_mc",
    "correctfit",
    "crossf",
    "dftfilt",
    "dftfilt2",
    "dftfilt3",
    "newcrossf",
    "newtimef",
    "newtimefbaseln",
    "newtimefitc",
    "newtimefpowerunit",
    "newtimeftrialbaseln",
    "pac",
    "pac_cont",
    "rsadjust",
    "rsfit",
    "rsget",
    "rspdfsolv",
    "rspfunc",
    "tf_cycle_calc",
    "timef",
    "timefreq",
    "timewarp",
]
