"""
Group Analysis with STUDY
=========================

Build a STUDY from several loaded datasets, add a design, precompute channel
measures, read a grand-average ERP, cluster ICA components, and save/reload the
``.study`` file. Everything here runs headless; the same steps are reachable
from ``File > Create study`` and the ``Study`` menu in the GUI.
"""

# %%
# Headless plotting and imports.

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

from eegprep import (  # noqa: E402
    pop_clust,
    pop_listfactors,
    pop_loadset,
    pop_loadstudy,
    pop_preclust,
    pop_precomp,
    pop_savestudy,
    pop_study,
    pop_studydesign,
    std_erpplot,
    std_makedesign,
    std_maketrialinfo,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE = REPO_ROOT / "sample_data" / "eeglab_data_epochs_ica.set"

# %%
# Load the epoched + ICA tutorial dataset once per pseudo-subject and tag the
# STUDY metadata EEGLAB uses: ``subject``, ``condition``, ``group``.

ALLEEG = []
for index in range(1, 5):
    EEG = pop_loadset(str(SAMPLE))
    EEG["setname"] = f"S{index:02d} tutorial"
    EEG["subject"] = f"S{index:02d}"
    EEG["condition"] = "tutorial"
    EEG["group"] = "young" if index < 3 else "older"
    ALLEEG.append(EEG)

print(f"datasets={len(ALLEEG)} nbchan={ALLEEG[0]['nbchan']} trials={ALLEEG[0]['trials']}")

# %%
# Create the STUDY from the loaded datasets (``File > Create study > Using all
# loaded datasets``).

STUDY, ALLEEG, com = pop_study(None, ALLEEG, name="Tutorial group study", return_com=True)
print("history:", com)
print("subjects:", STUDY["subject"])

# %%
# Trial metadata supplies the independent variables a design can use.

STUDY, trialinfo = std_maketrialinfo(STUDY, ALLEEG)
factors = pop_listfactors(STUDY, constant="off")
print("trials in dataset 1:", len(trialinfo[0]))
print("factors:", [f"{f['label']}={f['value']}" for f in factors])

# %%
# Add a second design contrasting the two groups and select it
# (``Study > Select/Edit study design(s)``).

STUDY, com = std_makedesign(
    STUDY, ALLEEG, 2, name="group contrast", variable1="group", return_com=True
)
STUDY, ALLEEG, com = pop_studydesign(STUDY, ALLEEG, 2, return_com=True)
print("designs:", [d.get("name") for d in STUDY["design"]])
print("current design:", STUDY["currentdesign"])

# %%
# Precompute channel ERP and spectrum measures
# (``Study > Precompute channel measures``).

STUDY, ALLEEG, com = pop_precomp(
    STUDY, ALLEEG, "channels", erp="on", spec="on", return_com=True
)
print("cached channel groups:", len(STUDY["changrp"]))
print("cached fields:", sorted(k for k in STUDY["changrp"][0] if k.endswith("data")))

# %%
# Read the cached ERP for one channel without opening a figure. Cells are
# ``(subjects, times)``; drop ``noplot`` to get a Matplotlib figure back.

STUDY, erpdata, erptimes, _ = std_erpplot(STUDY, ALLEEG, channels=[1], noplot="on")
print("erp cells:", len(erpdata), "cell shape:", erpdata[0].shape)
print("times:", f"{erptimes[0]:.1f} .. {erptimes[-1]:.1f} ms")

# %%
# Component measures, preclustering array, and k-means clustering
# (``Study > Precompute component measures``, ``Study > PCA clustering``).

STUDY, ALLEEG, com = pop_precomp(
    STUDY, ALLEEG, "components", scalp="on", erp="on", return_com=True
)
STUDY, ALLEEG, com = pop_preclust(
    STUDY,
    ALLEEG,
    preproc=[{"measure": "scalp", "npca": 3, "norm": 1, "weight": 1}],
    return_com=True,
)
STUDY, com = pop_clust(STUDY, ALLEEG, clus_num=3, random_state=0, return_com=True)
print("clusters:", [c.get("name") for c in STUDY["cluster"]])
print("cluster sizes:", [len(c.get("comps") or []) for c in STUDY["cluster"]])

# %%
# Save and reload the ``.study`` file (``File > Save current study as`` and
# ``File > Load existing study``).

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "tutorial.study"
    STUDY, com = pop_savestudy(STUDY, ALLEEG, filename=str(path), return_com=True)
    RELOADED, RELOADED_ALLEEG, com = pop_loadstudy(str(path), return_com=True)
    print("reloaded:", RELOADED["name"], "datasets:", len(RELOADED_ALLEEG))
    print("reloaded clusters:", len(RELOADED["cluster"]))
