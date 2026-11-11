# POP_EDITOPTIONS - Edit EEGPrep options

`pop_editoptions` updates EEGPrep's EEGLAB-style option registry.

File > Preferences controls disk storage of inactive datasets, two-file saves,
menu visibility, numerical boundary events, and native file dialogs. Show
advanced options reveals memory mapping when you reopen the dialog.

Disabled preferences have no runtime connection yet: precision, ICA caching
and RMS scaling, remembered folders, startup update checks, STUDY cache size,
save-version/struct choices, pending plugins, and HTML in graphics. Their
registry values remain available to scripts but do not change these workflows.
Use explicit function arguments where available. Use Help > Check for EEGPrep
updates to check manually. MATLAB-only toolbox and EEG-object switches are
also disabled in standalone EEGPrep.

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
