# pop_copyset

Copy an EEG dataset inside `ALLEEG`.

Dataset indices are EEGLAB-facing 1-based indices. For example,
`pop_copyset(ALLEEG, 1, 3)` copies dataset 1 into slot 3 and returns updated
`ALLEEG`, `EEG`, `CURRENTSET`, and the history command when `return_com=True`.

The main-window menu uses this helper for **Edit > Copy current dataset** and
keeps the GUI and `eegprep-console` workspace synchronized.
