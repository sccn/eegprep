"""
Import Data
===========

Import continuous and epoched recordings, attach events, read channel
locations, and round-trip a BIDS folder. The example runs headless and only
uses files from ``sample_data`` plus a temporary directory.
"""

# %%
# Continuous data: load an existing dataset.

import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np

import eegprep


def find_sample_data() -> Path:
    """Return the repository ``sample_data`` directory."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "sample_data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("sample_data directory not found")


SAMPLE_DATA = find_sample_data()

continuous = eegprep.pop_loadset(str(SAMPLE_DATA / "eeglab_data.set"))
print(
    "continuous:",
    continuous["nbchan"],
    "channels x",
    continuous["pnts"],
    "points x",
    continuous["trials"],
    "trial @",
    continuous["srate"],
    "Hz",
)

# %%
# Continuous data: import a raw array or an ASCII/float file, giving the
# sampling rate explicitly. Rows are channels and columns are samples.

tmpdir = Path(tempfile.mkdtemp(prefix="eegprep_import_"))
array = np.asarray(continuous["data"])[:4, :512]
array_file = tmpdir / "raw.tsv"
np.savetxt(array_file, array, delimiter="\t")

imported, com = eegprep.pop_importdata(
    "data",
    str(array_file),
    "srate",
    continuous["srate"],
    return_com=True,
)
print("pop_importdata ->", imported["nbchan"], "channels x", imported["pnts"], "points")
print("history:", com.replace(str(tmpdir), "<tmpdir>"))

# A NumPy array works the same way; only the history string differs.
from_array = eegprep.pop_importdata("data", array, "srate", continuous["srate"])
print("from array ->", from_array["nbchan"], "channels x", from_array["pnts"], "points")

# %%
# Epoched data: an epoched dataset reports ``trials > 1`` and
# ``data`` shaped ``(nbchan, pnts, trials)``.

epoched = eegprep.pop_loadset(str(SAMPLE_DATA / "eeglab_data_epochs_ica.set"))
print("epoched:", np.shape(epoched["data"]), "trials =", epoched["trials"])

# %%
# Events: import a text table. Latencies are given in the file's own unit and
# ``timeunit`` converts them to samples (NaN means the values are samples).

events_file = tmpdir / "events.tsv"
events_file.write_text(
    "type\tlatency\tduration\nstim\t100\t0\nresp\t250\t0\n",
    encoding="utf-8",
)

with_events = eegprep.pop_importevent(
    continuous,
    "event",
    str(events_file),
    "timeunit",
    np.nan,
)
print("replaced events:", [(e["type"], e["latency"]) for e in with_events["event"]])

appended = eegprep.pop_importevent(
    continuous,
    "event",
    str(events_file),
    "timeunit",
    np.nan,
    "append",
    "yes",
)
print("original:", len(continuous["event"]), "-> appended:", len(appended["event"]))

# %%
# Events: extract event onsets from a stimulus channel instead of a file.

pulse = eegprep.eeg_emptyset()
pulse.update(
    {
        "data": np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
        "nbchan": 1,
        "pnts": 8,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.07,
        "chanlocs": [{"labels": "TRIG"}],
    }
)
from_channel = eegprep.pop_chanevent(pulse, 1, "edge", "leading", "delchan", "off")
print("channel events:", [(e["type"], e["latency"]) for e in from_channel["event"]])

# %%
# Epoch info: attach one metadata value per epoch.

epoch_file = tmpdir / "epochinfo.tsv"
epoch_file.write_text("condition\n" + "rare\n" * epoched["trials"], encoding="utf-8")
with_epochinfo = eegprep.pop_importepoch(epoched, str(epoch_file), return_com=False)
print("epoch fields:", sorted(with_epochinfo["epoch"][0].keys()))

# %%
# Channel locations: read a packaged montage, then load locations into a
# dataset through ``pop_chanedit``.

montage = files("eegprep").joinpath("resources", "montages", "standard-10-5-342ch.locs")
locs = eegprep.readlocs(montage)
print("montage:", len(locs), "channels, first =", locs[0]["labels"])
print(
    "coordinate keys:",
    sorted(k for k in locs[3] if k in {"theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius"}),
)

small_locs = [
    {"labels": "Fz", "theta": 0.0, "radius": 0.25},
    {"labels": "Cz", "theta": 0.0, "radius": 0.0},
]
loc_file = tmpdir / "demo.locs"
eegprep.writelocs(small_locs, loc_file)
two_chan = eegprep.pop_importdata("data", array[:2, :], "srate", continuous["srate"])
relocated, chan_com = eegprep.pop_chanedit(two_chan, "load", str(loc_file), return_com=True)
print("chanlocs labels:", [c["labels"] for c in relocated["chanlocs"]])
print("history:", chan_com.replace(str(tmpdir), "<tmpdir>"))

# %%
# BIDS: export a dataset to a BIDS folder and import it back.

bids_root = tmpdir / "bids"
eegprep.pop_exportbids(continuous, bids_root, subject="01", task="tutorial")
print("bids files:", sorted(p.relative_to(bids_root).as_posix() for p in bids_root.rglob("*") if p.is_file())[:6])

reimported = eegprep.pop_importbids(bids_root)
datasets = reimported if isinstance(reimported, list) else [reimported]
print("imported from BIDS:", len(datasets), "dataset(s);", datasets[0]["nbchan"], "channels")

# %%
# Clean up the temporary directory used by this example.

shutil.rmtree(tmpdir)
