# Create a STUDY from dataset files

`pop_studywizard` lists the files you selected. Each row is numbered and has the file, a browse button, subject, session, run, condition, group, task, and a `Clear` button that empties that row. The headers include `browse` and leave the Clear column unlabeled. Subject codes start at `S1`. Edit any cell, then press Ok. Cancel leaves the workspace unchanged.

The history command is `[STUDY ALLEEG] = std_editset(STUDY, [], 'commands', {...}, 'updatedat', 'on')` followed by `std_checkset`. Each command cell keeps the subject, session, run, condition, group, and task from that row.

## Usage

```matlab
[STUDY ALLEEG] = pop_studywizard; % pop up window
```

## See also

`pop_study`, `std_editset`, `std_checkset`
