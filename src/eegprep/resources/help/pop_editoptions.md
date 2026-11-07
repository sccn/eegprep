# POP_EDITOPTIONS - Edit EEGPrep options

`pop_editoptions` updates EEGPrep's EEGLAB-style option registry.

File > Preferences opens memory, storage, ICA, folder, menu, boundary-event,
update, and cache settings. Show advanced options reveals the precision and
memory-mapping controls when you reopen the dialog. Options affect workflows
that support them; processing functions retain their explicit parameters.
MATLAB-only toolbox and EEG-object switches are disabled in standalone EEGPrep.

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
