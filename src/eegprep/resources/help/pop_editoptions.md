# POP_EDITOPTIONS - Edit EEGPrep options

`pop_editoptions` updates EEGPrep's EEGLAB-style option registry.

File > Preferences controls disk storage of inactive datasets, menu
visibility, numerical boundary events, and native file dialogs.

These EEGLAB options have no meaning in Python and are not in the window:

- Two-file saving: EEGPrep saves a dataset in one file.
- Saving fields as individual variables (the 2021 format): always on.
- Parallel toolbox, EEG object, HTML in graphics: MATLAB features.
- Memory-mapped arrays and single precision: not preferences in Python.
- RMS scaling of ICA activities: always on.
- Show advanced options and plugins pending approval: nothing to show.

Their keys stay in `EEG_OPTIONS` with their defaults, so EEGLAB scripts and
saved option files still load.

Disabled preferences have no runtime connection yet: ICA caching, remembered
folders, startup update checks, STUDY cache size, and the MATLAB save version.
Their registry values remain available to scripts but do not change these
workflows. Use explicit function arguments where available. Use Help > Check
for EEGPrep updates to check manually. The MATLAB toolbox switch is also
disabled in standalone EEGPrep.

GUI changes are saved to `~/.eegprep/options.json` and restored on the next
launch. Cancel leaves current and saved preferences unchanged. Menu visibility
changes immediately. MATLAB's `eeg_options.m` is not modified.

Programmatic calls accept named options:

```python
com = pop_editoptions(option_allmenus=1)
com = pop_editoptions("option_storedisk", 1)
com = pop_editoptions(option_native_dialogs=0)
com = pop_editoptions(option_allmenus=1, persist=True)
```

Scripted changes are process-local unless `persist=True`. Native file-dialog
overrides passed to the launcher still take precedence over the global setting.

See also: EEG_OPTIONS
