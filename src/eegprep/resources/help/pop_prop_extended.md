POP_PROP_EXTENDED - View extended channel or component properties.

Usage:

    figure = pop_prop_extended(EEG, typecomp, chanorcomp)
    figure, command = pop_prop_extended(EEG, typecomp, chanorcomp, return_com=True)

Set `typecomp=1` for channels and `typecomp=0` for ICA components. Indices are
EEGLAB-facing 1-based values.

For ICA components, the dashboard shows the component map, activity, ERP/image
summary, power spectrum, percent variance accounted for, and ICLabel-style
classification probabilities when classifier data are available in
`EEG.etc.ic_classification`.

`scroll_event=1` includes events in the attached browser-backed activity view.
`scroll_event=0` hides events in that browser view. Multiple selected
components are shown in one navigable dashboard.

Notes:

- EEGPrep consumes existing classifier results and does not run ICLabel from
  this viewer.
- If classifier data are absent, component property plotting falls back to the
  non-classifier property display used by `pop_viewprops`.
- This implementation is EEGPrep-owned and does not require an EEGLAB checkout
  at runtime.
