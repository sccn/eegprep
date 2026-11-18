POP_VIEWPROPS - View channel or component properties.

Usage:

    figures = pop_viewprops(EEG, typecomp, chanorcomp)
    figures, command = pop_viewprops(EEG, return_com=True)

Set `typecomp=1` for channels and `typecomp=0` for ICA components. Indices are
EEGLAB-facing 1-based values.

Property overview figures include browser-backed activity views for each
visible channel or component. The dialog's event checkbox controls whether
events are included in those scrolling activity views.

Channel mode shows the extended property dashboard: topography, ERP image,
and spectrum. Component mode shows that dashboard together with the
topography grid. Clicking a map opens the dashboard for that component.
The window is displayed when the plotting backend can show figures.

Component mode requires `EEG.icaweights`. Run `pop_runica` first. ICLabel
rows that do not match the current decomposition are refused; run
`pop_iclabel` again. A dataset with no classifier still shows the topography
grid and the dashboard.
