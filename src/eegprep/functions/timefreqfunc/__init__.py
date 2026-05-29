"""EEGLAB-style time-frequency helper functions."""

from eegprep.functions.timefreqfunc.newcrossf import CrossFrequencyResult, newcrossf
from eegprep.functions.timefreqfunc.newtimef import TimeFrequencyResult, newtimef

__all__ = ["CrossFrequencyResult", "TimeFrequencyResult", "newcrossf", "newtimef"]
