POP_SELECTCOMPS - Mark ICA components for rejection.

Usage:

    EEG = pop_selectcomps(EEG, compnum, reject=[...])
    EEG, command = pop_selectcomps(EEG, return_com=True)

This EEGPrep implementation provides the component-selection and marking path.
Marked components are stored in `EEG.reject.gcompreject` and can be inspected
alongside component activity in the scrolling EEGPlot/EEGBrowser workflows.

Use `pop_subcomp` to remove marked components from the data.
