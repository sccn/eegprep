# coregister

Coregisters EEG channel locations to a reference electrode montage and optional
head mesh for `headplot` spline setup.

The manual editor follows the EEGLAB workflow:

- user electrodes are shown in green;
- reference electrodes are shown in brown;
- the head mesh can be toggled on and off;
- labels and electrode subsets can be toggled;
- transform boxes edit the Talairach-model translation, rotation, and scale;
- `Align montages` fits a shared-scale transform to common labels;
- `Warp montage` fits the 9-parameter EEGLAB-style transform to common labels.

Press `Ok` to return the transform to the calling dialog, or `Cancel` to leave
the original transform unchanged.
