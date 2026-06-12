"""Helper script for re-referencing EEG data."""

import sys

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_saveset import pop_saveset

if __name__ == "__main__":
    # check if a parameter is present and if it is assign eeglab_file_path to it
    if len(sys.argv) > 2:
        eeglab_file_path_in = sys.argv[1]
        eeglab_file_path_out = sys.argv[2]
    else:
        eeglab_file_path_in = './eeglab_data_with_ica_tmp.set'
        eeglab_file_path_out = './eeglab_data_with_ica_tmp_averef.set'

    EEG = pop_loadset(eeglab_file_path_in)
    EEG = pop_reref(EEG, [])
    pop_saveset(EEG, eeglab_file_path_out)
