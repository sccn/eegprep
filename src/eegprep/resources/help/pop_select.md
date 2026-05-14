POP_SELECT - Select a subset of EEG data.

Usage:
  EEG = pop_select(EEG, 'key', value, ...)

Common options:
  'time'       - time range to retain, in seconds.
  'rmtime'     - time range to remove, in seconds.
  'point'      - sample range to retain.
  'rmpoint'    - sample range to remove.
  'trial'      - epoch indices to retain.
  'rmtrial'    - epoch indices to remove.
  'channel'    - channel labels or EEGPrep channel indices to retain.
  'rmchannel'  - channel labels or EEGPrep channel indices to remove.
  'chantype'   - channel types to retain.
  'rmchantype' - channel types to remove.

Calling pop_select(EEG) opens the EEGPrep GUI dialog.
