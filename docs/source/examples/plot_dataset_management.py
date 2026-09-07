"""
Dataset Management
==================

Load the tutorial dataset, modify it, store the result as a second dataset,
switch between datasets, save one to disk, and delete it from memory. This is
the scripted form of ``File > Load existing dataset``, ``Datasets > ...``,
``File > Save current dataset as``, and ``Edit > Delete dataset(s) from memory``.
"""

# %%
# Load the tutorial dataset into an empty dataset list.

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eegprep import (  # noqa: E402
    EEGPrepSession,
    eeg_retrieve,
    pop_copyset,
    pop_delset,
    pop_editset,
    pop_loadset,
    pop_newset,
    pop_resample,
    pop_saveset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_SET = REPO_ROOT / "sample_data" / "eeglab_data.set"

EEG = pop_loadset(str(TUTORIAL_SET))
ALLEEG, EEG, CURRENTSET, com = pop_newset([], EEG, 0, setname="tutorial continuous")
print("loaded:", EEG["setname"], EEG["data"].shape, "srate", EEG["srate"])
print("CURRENTSET:", CURRENTSET, "datasets in memory:", len(ALLEEG))
print("history:", com)

# %%
# Modify the dataset, then store the result as a *new* dataset instead of
# overwriting dataset 1. This is what the pop_newset dialog does when you leave
# "Overwrite it in memory" unchecked.

EEG_resampled, resample_com = pop_resample(EEG, 64, return_com=True)
ALLEEG, EEG, CURRENTSET, com = pop_newset(ALLEEG, EEG_resampled, CURRENTSET, setname="tutorial 64 Hz", overwrite="off")
print(resample_com)
print("CURRENTSET:", CURRENTSET, "datasets in memory:", len(ALLEEG))

# %%
# The Datasets menu is built from this list.

session = EEGPrepSession()
session.ALLEEG = ALLEEG
session.store_current(EEG, index=CURRENTSET)
for index, label, selected in session.dataset_summaries():
    print(("* " if selected else "  ") + label)

# %%
# Switch back to dataset 1 (``Datasets > Dataset 1:...``), then forward again.

EEG, ALLEEG, CURRENTSET = eeg_retrieve(ALLEEG, 1)
print("retrieved:", EEG["setname"], EEG["data"].shape, "srate", EEG["srate"])
ALLEEG, EEG, CURRENTSET, com = pop_newset(ALLEEG, EEG, CURRENTSET, retrieve=2)
print("retrieved:", EEG["setname"], EEG["data"].shape, "srate", EEG["srate"])
print("history:", com)

# %%
# Rename the current dataset (``Edit > Dataset info``). Editing ``EEG`` alone
# does not update ``ALLEEG``; store it back with ``overwrite="on"``, which is the
# "Overwrite it in memory" checkbox of the pop_newset dialog.

EEG, editset_com = pop_editset(EEG, setname="tutorial 64 Hz (renamed)", return_com=True)
print(editset_com)
print("EEG:", EEG["setname"], "| ALLEEG[2]:", ALLEEG[1]["setname"])

ALLEEG, EEG, CURRENTSET, com = pop_newset(ALLEEG, EEG, CURRENTSET, overwrite="on")
print("after overwrite -> ALLEEG[2]:", ALLEEG[1]["setname"], "| datasets:", len(ALLEEG))
print("history:", com)

# %%
# Copy a dataset to a new slot (``Edit > Copy current dataset``).

ALLEEG, EEG, CURRENTSET, copy_com = pop_copyset(ALLEEG, 2, 3, return_com=True)
print("datasets in memory:", len(ALLEEG), "CURRENTSET:", CURRENTSET)
print(copy_com)

# %%
# Save the current dataset (``File > Save current dataset as``). A plain save
# writes a single ``.set`` file; ``savemode="twofiles"`` writes ``.set`` plus a
# float32 ``.fdt`` sidecar.

with tempfile.TemporaryDirectory() as tmpdir:
    one_file = Path(tmpdir) / "dataset_mgmt_one.set"
    two_file = Path(tmpdir) / "dataset_mgmt_two.set"
    pop_saveset(EEG, str(one_file))
    pop_saveset(EEG, str(two_file), savemode="twofiles")
    print("one-file:", sorted(p.name for p in Path(tmpdir).glob("dataset_mgmt_one.*")))
    print("two-file:", sorted(p.name for p in Path(tmpdir).glob("dataset_mgmt_two.*")))
    reloaded = pop_loadset(str(two_file))
    print("reloaded:", reloaded["setname"], reloaded["data"].shape)

# %%
# Delete datasets from memory (``Edit > Delete dataset(s) from memory`` or
# ``File > Clear dataset(s)``).

ALLEEG, delete_com = pop_delset(ALLEEG, [3])
print("datasets in memory:", len(ALLEEG))
print(delete_com)
