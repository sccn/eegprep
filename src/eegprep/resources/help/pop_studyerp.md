# Create a simple ERP STUDY

`pop_studyerp` asks how many conditions and subjects to include, then shows one dataset box per subject and condition. Each box has a `...` button that browses for a `.set` file. Fill at least one box and press Ok to build the STUDY. Cancel on either dialog leaves the workspace unchanged.

Datasets on one line belong to the same subject. Subject codes are filled in as `S01`, `S02`, and so on. Dataset commands are ordered condition by condition, with the same index formula EEGLAB uses (`nConds * subject + condition - 1`).

## Usage

```matlab
[STUDY ALLEEG] = pop_studyerp; % pop up interface
```

## See also

`std_editset`, `std_checkset`
