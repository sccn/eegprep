"""EEGLAB-style STUDY helper functions."""

from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.pop_studydesign import pop_studydesign
from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp
from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard
from eegprep.functions.studyfunc.std_checkset import std_checkdatasetinfo, std_checkset
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign

__all__ = [
    "pop_loadstudy",
    "pop_savestudy",
    "pop_study",
    "pop_studydesign",
    "pop_studyerp",
    "pop_studywizard",
    "std_checkdatasetinfo",
    "std_checkset",
    "std_editset",
    "std_makedesign",
    "std_selectdesign",
]
