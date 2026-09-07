"""
Preprocess Data: Filtering, Re-referencing, Resampling
======================================================

Mirrors the EEGLAB tutorial section "Preprocess data" on the checked-in
tutorial dataset. The script runs headless: no dialog, no figure window.
"""

# %%
# Load the tutorial dataset.

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eegprep import pop_eegfiltnew, pop_loadset, pop_reref, pop_resample, pop_spectopo

REPO_ROOT = Path(__file__).resolve().parents[3]
EEG = pop_loadset(REPO_ROOT / "sample_data" / "eeglab_data.set")

print(f"loaded    : {EEG['nbchan']} chan, {EEG['pnts']} pnts, {EEG['srate']:g} Hz")
print(f"reference : {EEG.get('ref')!r}")
print(f"events    : {len(EEG['event'])}, first latency {EEG['event'][0]['latency']:g}")

# %%
# Filtering. Apply the high-pass and the low-pass as separate passes so each
# transition band is designed for its own cutoff.

EEG, hp_com = pop_eegfiltnew(EEG, locutoff=1.0, plotfreqz=False, return_com=True)
EEG, lp_com = pop_eegfiltnew(EEG, hicutoff=40.0, plotfreqz=False, return_com=True)
print(f"highpass  : {hp_com}")
print(f"lowpass   : {lp_com}")

# %%
# Re-referencing. An empty reference means the average of all channels.

filtered = copy.deepcopy(EEG)
EEG, ref_com = pop_reref(EEG, [], return_com=True)
print(f"reref     : {ref_com}")
print(f"reference : {EEG.get('ref')!r}")
print(f"max abs channel sum per sample: {abs(EEG['data'].sum(axis=0)).max():.3e}")

# %%
# Re-referencing to one channel, keeping that channel in the data. The retained
# reference channel becomes flat, which is the expected single-channel result.
# Channel labels avoid any index-base ambiguity; numeric ``ref`` values are
# 0-based in EEGPrep.

single = pop_reref(filtered, "Cz", keepref="on")
cz = [chan["labels"] for chan in single["chanlocs"]].index("Cz")
print(f"kept nbchan: {single['nbchan']}, Cz std {single['data'][cz].std():.3e}")
dropped = pop_reref(filtered, "Cz")
print(f"dropped nbchan: {dropped['nbchan']}")

# %%
# Resampling. The anti-alias low-pass, the time vector, and the event latencies
# are updated together.

first_latency_before = EEG["event"][0]["latency"]
EEG, rs_com = pop_resample(EEG, 64, return_com=True)
print(f"resample  : {rs_com}")
print(f"after     : {EEG['pnts']} pnts, {EEG['srate']:g} Hz, xmax {EEG['xmax']:.3f} s")
print(f"first event latency {first_latency_before:g} -> {EEG['event'][0]['latency']:g} samples")

# %%
# Check the result in the frequency domain without opening a window.

spectra = pop_spectopo(EEG, 1, [], freqs=[10], plot="off")
print(f"spectra shape {spectra['spectra'].shape}, freqs up to {spectra['freqs'].max():g} Hz")
