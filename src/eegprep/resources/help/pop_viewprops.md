POP_VIEWPROPS - View channel or component properties.

Usage:

    figures = pop_viewprops(EEG, typecomp, chanorcomp)
    figures, command = pop_viewprops(EEG, return_com=True)

Set `typecomp=1` for channels and `typecomp=0` for ICA components. Indices are
EEGLAB-facing 1-based values.

Property overview figures include browser-backed activity views for each
visible channel or component. The dialog's event checkbox controls whether
events are included in those scrolling activity views.
