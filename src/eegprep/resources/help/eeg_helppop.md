# EEG_HELPPOP - Interactive pop_ functions

`pop_*` functions are the primary EEGLAB-style user-facing wrappers in EEGPrep.
They either open a GUI dialog or run directly from Python when enough
arguments are provided.

Conventions:

- Continuous EEG data is channel-major with shape `(nbchan, pnts)`.
- Epoched EEG data is channel-major with shape `(nbchan, pnts, trials)`.
- User-facing dataset, channel, component, epoch, and event indices follow
  EEGLAB's 1-based conventions unless a function explicitly documents a
  Python-only 0-based parameter.
- `return_com=True` returns the modified object and the history command where
  the workflow supports history.

The GUI dispatcher stores successful dataset changes through
`EEGPrepSession`, so GUI actions and `eegprep-console` commands update the
same workspace.

See also: EEG_HELPADMIN, EEG_HELPMENU
