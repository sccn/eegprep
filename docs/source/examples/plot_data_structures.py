"""
EEGPrep Data Structures
=======================

A guided tour of the objects EEGPrep passes between functions: the ``EEG``
dataset dictionary, its ``chanlocs`` / ``event`` / ``urevent`` / ``epoch``
tables, the ICA fields, and the ``ALLEEG`` / ``CURRENTSET`` dataset list.

Nothing here modifies data. The script only reads two datasets from
``sample_data/`` and prints their anatomy, so it doubles as a reference for
what each field actually contains.
"""

# %%
# An empty dataset: the field inventory
# -------------------------------------
# ``eeg_emptyset`` returns the canonical dictionary every EEGPrep function
# expects. Every dataset you load, create, or process has these keys.

from pathlib import Path

import numpy as np

import eegprep
from eegprep import (
    eeg_emptyset,
    eeg_findboundaries,
    eeg_getica,
    eeg_lat2point,
    eeg_point2lat,
    eeg_retrieve,
    eeg_store,
    pop_loadset,
    pop_select,
)

EEG_EMPTY = eeg_emptyset()
print("fields:", ", ".join(sorted(EEG_EMPTY)))

# %%
# Continuous data: shape and time base
# ------------------------------------
# Continuous data is channel-major ``(nbchan, pnts)``. ``trials`` is 1.
# ``xmin`` / ``xmax`` are seconds; ``times`` is milliseconds.

SAMPLE_DIR = Path(eegprep.__file__).resolve().parents[2] / "sample_data"  # sphinx-gallery defines no __file__

EEG = pop_loadset(str(SAMPLE_DIR / "eeglab_data.set"))
print("setname:", EEG["setname"])
print("data:", EEG["data"].shape, EEG["data"].dtype)
print(f"nbchan={EEG['nbchan']} pnts={EEG['pnts']} trials={EEG['trials']}")
print(f"srate={EEG['srate']} xmin={EEG['xmin']} xmax={EEG['xmax']:.4f}")

# %%
# ``chanlocs``: one record per channel
# ------------------------------------
# A per-channel table with EEGLAB's field names, so topographic plotting,
# interpolation and DIPFIT all read the same metadata.

chan = EEG["chanlocs"][0]
print("chanlocs keys:", ", ".join(sorted(chan)))
print("labels:", [c["labels"] for c in EEG["chanlocs"][:6]])
print(f"channel 0: labels={chan['labels']} theta={chan['theta']} radius={chan['radius']}")
print(f"           X={chan['X']:.4f} Y={chan['Y']:.4f} Z={chan['Z']:.4f}")

# %%
# ``event`` and ``urevent``: what happened, and what originally happened
# ---------------------------------------------------------------------
# ``event`` describes the events of the *current* dataset. ``urevent`` keeps
# the original continuous-recording events; each ``event`` carries a
# ``urevent`` pointer so provenance survives rejection and epoching.

print("n events:", len(EEG["event"]), " n urevents:", len(EEG["urevent"]))
print("event keys:", ", ".join(sorted(EEG["event"][0])))
for ev in EEG["event"][:3]:
    print(f"  type={ev['type']:<7} latency={ev['latency']:.3f} urevent={ev['urevent']}")

types, counts = np.unique([ev["type"] for ev in EEG["event"]], return_counts=True)
print("event types:", dict(zip(types.tolist(), counts.tolist())))

# %%
# Latency units
# -------------
# ``latency`` is a sample position, not a time. Convert explicitly with
# ``eeg_point2lat`` (samples to time) and ``eeg_lat2point`` (time to samples)
# rather than dividing by ``srate`` by hand.

timewin = [EEG["xmin"], EEG["xmax"]]
first_latency = EEG["event"][0]["latency"]
seconds = eeg_point2lat([first_latency], None, EEG["srate"], timewin)
back, _outofrange = eeg_lat2point(seconds, 1, EEG["srate"], timewin)
print(f"latency {first_latency:.3f} samples -> {seconds[0]:.4f} s -> {back[0]:.3f} samples")

# %%
# Boundary events and provenance
# ------------------------------
# Removing a stretch of continuous data inserts a ``boundary`` event whose
# ``duration`` records how many samples disappeared. ``urevent`` is *not*
# rewritten, so the original event list survives the edit and every remaining
# event still points into it.

TRIMMED = pop_select(EEG, nopoint=[[1000, 2000]])
boundaries = eeg_findboundaries(EEG=TRIMMED)
print(f"pnts {EEG['pnts']} -> {TRIMMED['pnts']}")
print("boundary event indices (0-based into EEG['event']):", boundaries)
print("boundary record:", {k: TRIMMED["event"][boundaries[0]][k] for k in ("type", "latency", "duration")})
print(f"events {len(EEG['event'])} -> {len(TRIMMED['event'])}, urevents unchanged: {len(TRIMMED['urevent'])}")

# %%
# Epoched data and the ``epoch`` table
# ------------------------------------
# Epoching adds a third dimension, ``(nbchan, pnts, trials)``, and an
# ``epoch`` table holding the events falling inside each trial. Event
# latencies inside ``epoch`` are milliseconds relative to the time-locking
# event, which is why ``eventlatency[0]`` is 0.

EPOCHED = pop_loadset(str(SAMPLE_DIR / "eeglab_data_epochs_ica.set"))
print("data:", EPOCHED["data"].shape)
print(f"pnts={EPOCHED['pnts']} trials={EPOCHED['trials']}")
print(f"xmin={EPOCHED['xmin']} xmax={EPOCHED['xmax']:.4f}")

epoch0 = EPOCHED["epoch"][0]
print("epoch keys:", ", ".join(sorted(epoch0)))
print("epoch 0 eventtype:", epoch0["eventtype"])
print("epoch 0 eventlatency (ms):", [round(float(x), 2) for x in epoch0["eventlatency"]])
print("event 0 belongs to epoch:", EPOCHED["event"][0]["epoch"])

# %%
# ICA fields
# ----------
# ICA is stored as the unmixing transform plus its inverse, never as a
# separate dataset. ``icaact`` is a cache that can be recomputed from
# ``icaweights``, ``icasphere`` and the data.

for field in ("icaweights", "icasphere", "icawinv", "icaact"):
    print(f"{field}: {np.asarray(EPOCHED[field]).shape}")
print("icachansind (0-based):", np.asarray(EPOCHED["icachansind"])[:8], "...")

# icaact is just weights @ sphere @ data, so it can always be regenerated.
unmix = np.asarray(EPOCHED["icaweights"]) @ np.asarray(EPOCHED["icasphere"])
recomputed = unmix @ np.asarray(EPOCHED["data"]).reshape(EPOCHED["nbchan"], -1)
cached = np.asarray(EPOCHED["icaact"]).reshape(EPOCHED["nbchan"], -1)
scale = float(np.abs(cached).std())
print(f"|icaact - weights @ sphere @ data| / std(icaact): {np.abs(recomputed - cached).max() / scale:.2e}")

# eeg_getica returns activations; its component selector is 1-based, like the GUI.
print("eeg_getica(EEG):", eeg_getica(EPOCHED).shape)
print("eeg_getica(EEG, 1):", eeg_getica(EPOCHED, 1).shape)

# %%
# ``ALLEEG`` and ``CURRENTSET``
# -----------------------------
# Loaded datasets live in ``ALLEEG``, a plain Python list. ``CURRENTSET``
# holds 1-based dataset numbers, matching what the GUI Datasets menu and the
# console show. ``eeg_store`` fills the lowest empty slot or overwrites;
# ``eeg_retrieve`` selects.

ALLEEG, _EEG, CURRENTSET = eeg_store([], EEG, 0)
ALLEEG, _EEG, CURRENTSET = eeg_store(ALLEEG, EPOCHED, 0)
print("len(ALLEEG):", len(ALLEEG), " CURRENTSET:", CURRENTSET)

EEG, ALLEEG, CURRENTSET = eeg_retrieve(ALLEEG, 1)
print("retrieved dataset 1:", EEG["setname"], EEG["data"].shape, "CURRENTSET:", CURRENTSET)
print("dataset names:", [ds["setname"] for ds in ALLEEG])
