"""
Extract Data Epochs
===================

Epoch the continuous tutorial dataset around the ``square`` stimulus events,
remove the pre-stimulus baseline, cut a shorter sub-epoch window, then select
epochs by index and by event field.

Everything runs headless with no GUI window: every ``pop_*`` call below passes
explicit arguments, so the dialogs never open.
"""

# %%
# Load the continuous tutorial dataset.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import eegprep
from eegprep import (
    eeg_checkset,
    pop_epoch,
    pop_loadset,
    pop_rmbase,
    pop_select,
    pop_selectevent,
)

REPO_ROOT = Path(eegprep.__file__).resolve().parents[2]  # sphinx-gallery defines no __file__
input_file = REPO_ROOT / "sample_data" / "eeglab_data.set"

EEG = pop_loadset(input_file)
print("continuous:", EEG["data"].shape, "trials:", EEG["trials"])
print("event types:", sorted({str(event["type"]) for event in EEG["event"]}))

# %%
# Extract epochs time locked to the ``square`` events
# (``Tools > Extract epochs``). ``pop_epoch`` returns the epoched dataset plus
# the history command the GUI and console record.

EEG, epoch_com = pop_epoch(EEG, ["square"], [-1, 2], newname="Square epochs", return_com=True)
print("epoched:", EEG["data"].shape, "trials:", EEG["trials"])
print("epoch range (s):", EEG["xmin"], EEG["xmax"])
print(epoch_com)

# %%
# Remove the pre-stimulus baseline (``Tools > Remove epoch baseline``). The
# range uses the units of ``EEG["times"]``, milliseconds for epoched data.

EEG, rmbase_com = pop_rmbase(EEG, [-200, 0], return_com=True)
print(rmbase_com)

# %%
# Cut a shorter sub-epoch window, -500 ms to 1000 ms, with ``pop_select``
# (``Edit > Select data``). The ``time`` option is in seconds.

EEG, select_time_com = pop_select(EEG, time=[-0.5, 1.0], return_com=True)
print("sub-epoch:", EEG["data"].shape, "range (ms):", EEG["times"][0], EEG["times"][-1])
print(select_time_com)

# %%
# Select epochs by index. Epoch selectors are EEGLAB-facing 1-based indices, so
# ``trial=[1, ..., 10]`` keeps the first ten epochs.

EEG_first10, select_trial_com = pop_select(EEG, trial=list(range(1, 11)), return_com=True)
print("first 10 epochs:", EEG_first10["data"].shape)
print(select_trial_com)

# %%
# Select epochs by event field (``Edit > Select epochs or events``). Keep the
# ``square`` events with ``position == 1`` and drop epochs that no selected
# event refers to.

EEG_pos1, selectevent_com = pop_selectevent(
    EEG,
    type=["square"],
    position=1,
    deleteepochs="on",
    deleteevents="off",
    return_com=True,
)
EEG_pos1 = eeg_checkset(EEG_pos1)
print("position 1 epochs:", EEG_pos1["data"].shape, "trials:", EEG_pos1["trials"])
print(selectevent_com)
