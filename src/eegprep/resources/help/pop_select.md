POP_SELECT - Select a subset of EEG data.

Usage:
  EEG = pop_select(EEG, 'key', value, ...)

Common options:
  'time'       - time range to retain, in seconds.
  'rmtime'     - time range to remove, in seconds.
  'point'      - sample range to retain.
  'rmpoint'    - sample range to remove.
  'trial'      - epoch indices to retain. With sorttrial off, a repeated
                 index is kept at its first position. trial [5 4 3 2 1 5]
                 stays [5 4 3 2 1]. EEGLAB unique_bc keeps the last copy,
                 [4 3 2 1 5] (pop_select.m lines 282-286).
  'rmtrial'    - epoch indices to remove.
  'channel'    - channel labels or EEGPrep channel indices to retain.
  'rmchannel'  - channel labels or EEGPrep channel indices to remove.
  'chantype'   - channel types to retain.
  'rmchantype' - channel types to remove.

Calling pop_select(EEG) opens the EEGPrep GUI dialog.
