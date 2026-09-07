"""
Plot data: ERPs, ERP-images, spectra, time-frequency, ICA components
====================================================================

Headless walkthrough of the EEGPrep ``Plot`` menu on the epoched tutorial
dataset ``sample_data/eeglab_data_epochs_ica.set``. Every call is the console
equivalent of one ``Plot`` menu item, and each passes ``plot='off'`` so the
figure is built and returned instead of opening a window.
"""

# %%
# Load the epoched dataset with an ICA decomposition
# --------------------------------------------------

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import eegprep
from eegprep import (
    eeg_pvaf,
    pop_envtopo,
    pop_erpimage,
    pop_headplot,
    pop_loadset,
    pop_newtimef,
    pop_plotdata,
    pop_plottopo,
    pop_prop,
    pop_spectopo,
    pop_timtopo,
    pop_topoplot,
)

REPO_ROOT = Path(eegprep.__file__).resolve().parents[2]  # sphinx-gallery defines no __file__
SAMPLE_DATA = REPO_ROOT / "sample_data"

EEG = pop_loadset(SAMPLE_DATA / "eeglab_data_epochs_ica.set")
EPOCH_MS = [EEG["xmin"] * 1000.0, EEG["xmax"] * 1000.0]
print(f"data {EEG['data'].shape} srate {EEG['srate']:g} trials {EEG['trials']}")
print(f"components {np.asarray(EEG['icaweights']).shape[0]} epoch {EPOCH_MS[0]:g}..{EPOCH_MS[1]:g} ms")

# %%
# a. ERPs -- Plot > Channel ERPs, Plot > ERP map series
# -----------------------------------------------------

fig_timtopo, com_timtopo = pop_timtopo(EEG, [100, 300], timerange=[-100, 600], plot="off", return_com=True)
fig_plottopo = pop_plottopo(EEG, chans=list(range(1, 33)), plot="off")
figs_erp2d = pop_topoplot(EEG, 1, [0, 100, 200, 300], plot="off")
figs_erp3d = pop_headplot(EEG, 1, [100, 300], load=SAMPLE_DATA / "eeglab_data_epochs_ica.spl", plot="off")
print("timtopo history:", com_timtopo)
print(f"ERP map series: {len(figs_erp2d)} 2-D figure(s), {len(figs_erp3d)} 3-D figure(s)")

# %%
# b. ERP-image -- Plot > Channel ERP image
# ----------------------------------------

raw_image = pop_erpimage(EEG, 1, 14, smooth=1, decimate=1, plot="off")
smooth_image = pop_erpimage(EEG, 1, 14, smooth=10, decimate=2, plot="off")
sorted_image = pop_erpimage(EEG, 1, 14, sortingeventfield="latency", sortingtype=["rt"], smooth=10, plot="off")
print(f"raw image {raw_image['image'].shape} smoothed {smooth_image['image'].shape}")
print(f"latency-sorted image {sorted_image['image'].shape}")

# %%
# c. Spectra -- Plot > Channel spectra and maps, Plot > Channel properties
# -----------------------------------------------------------------------

spectra = pop_spectopo(EEG, 1, EPOCH_MS, freqs=[6, 10, 22], winsize=256, overlap=128, plot="off")
mean_spectrum = np.nanmean(spectra["spectra"], axis=0)
band = spectra["freqs"] >= 2.0
peak = float(spectra["freqs"][band][int(np.argmax(mean_spectrum[band]))])
fig_prop = pop_prop(EEG, 1, 14, plot="off")
print(f"spectra {spectra['spectra'].shape} over {spectra['freqs'][0]:.1f}-{spectra['freqs'][-1]:.1f} Hz")
print(f"strongest mean spectral peak above 2 Hz: {peak:.1f} Hz")

# %%
# d. Time-frequency -- Plot > Time-frequency transforms > Channel time-frequency
# -----------------------------------------------------------------------------

tf = pop_newtimef(EEG, 1, 14, EPOCH_MS, [3, 0.8], plot="off")
tf_masked = pop_newtimef(EEG, 1, 14, EPOCH_MS, [3, 0.8], alpha=0.05, mcorrect="fdr", plot="off")
tf_curves = pop_newtimef(EEG, 1, 1, EPOCH_MS, [3, 0.8], plottype="curve", freqs=[5, 10, 20], plot="off")
print(f"ERSP {tf.ersp.shape} ITC {tf.itc.shape} freqs {tf.freqs[0]:.1f}-{tf.freqs[-1]:.1f} Hz")
print(
    f"FDR-masked bins: ERSP {int(np.count_nonzero(tf_masked.ersp_significant))}, "
    f"ITC {int(np.count_nonzero(tf_masked.itc_significant))}"
)
print(f"curve mode ERSP {tf_curves.ersp.shape} at freqs {np.round(tf_curves.freqs, 1).tolist()}")

# %%
# e. ICA components -- Plot > Component spectra/maps/ERPs/ERP image/time-frequency
# -------------------------------------------------------------------------------

comp_spectra = pop_spectopo(EEG, 0, EPOCH_MS, freqs=[10], icacomps=[1, 2, 3, 4, 5], plot="off")
fig_comp_erps = pop_plotdata(EEG, components=[1, 2, 3, 4, 5], plot="off")
fig_envtopo, com_envtopo = pop_envtopo(EEG, [-100, 600], compnums=list(range(1, 11)), plot="off", return_com=True)
comp_image = pop_erpimage(EEG, 0, 1, smooth=10, decimate=2, plot="off")
comp_tf = pop_newtimef(EEG, 0, 1, EPOCH_MS, [3, 0.8], plot="off")
figs_comp2d = pop_topoplot(EEG, 0, [1, 2, 3, 4, 5], plot="off")
figs_comp3d = pop_headplot(EEG, 0, [1, 2], load=SAMPLE_DATA / "eeglab_data_epochs_ica.spl", plot="off")
pvaf, _, _ = eeg_pvaf(EEG, [1, 2, 3, 4, 5])
print(f"component spectra {comp_spectra['spectra'].shape}")
print(f"pvaf of components 1-5: {float(np.atleast_1d(pvaf)[0]):.2f}%")
print(f"envtopo history: {com_envtopo}")
print(f"component ERP image {comp_image['image'].shape} component ERSP {comp_tf.ersp.shape}")
print(f"component maps: {len(figs_comp2d)} 2-D figure(s), {len(figs_comp3d)} 3-D figure(s)")
