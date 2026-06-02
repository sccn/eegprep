# POP_EDITOPTIONS - Edit EEGPrep options

`pop_editoptions` updates EEGPrep's EEGLAB-style option registry.

The main GUI currently exposes the advanced-menu preference. Enabling advanced
menus shows legacy/full EEGLAB menu items represented in EEGPrep's menu
inventory; disabling it returns to the default simplified menu.

Programmatic calls accept named options:

```python
com = pop_editoptions(option_allmenus=1)
```

Reopen the main window after changing menu-mode options so the menu tree is
rebuilt.

See also: EEG_OPTIONS
