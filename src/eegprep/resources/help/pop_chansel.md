# POP_CHANSEL - Select channels from a list

`pop_chansel` opens an EEGLAB-style channel selector and returns selected
one-based channel indices, a display string, and the selected channel labels.

Usage:

```python
chanlist, chanliststr, labels = pop_chansel(EEG["chanlocs"], withindex="on")
```

Use `field="type"` to select from channel types instead of labels. Use
`selectionmode="single"` when only one channel or type should be selected.

The helper is used by GUI dialogs such as clean_rawdata and ICA channel/type
selectors. Programmatic scripts can use `pop_chansel_display_values` and
`pop_chansel_selected_string` to format channel choices without opening a GUI.

See also: LISTDLG2, POP_SELECT, POP_CHANEDIT
