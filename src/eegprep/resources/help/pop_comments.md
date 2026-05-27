POP_COMMENTS - Edit comments for an EEG dataset.

Usage:

    EEG = pop_comments(EEG)
    EEG = pop_comments(EEG, title, newcomments, concat)
    comments = pop_comments(comments, title, newcomments, concat)

Calling `pop_comments(EEG)` opens the interactive comments dialog. Passing
`newcomments` edits comments without opening a dialog.

Inputs:

- `EEG`: EEG dataset dictionary, or a string/list of existing comments.
- `title`: optional dialog title.
- `newcomments`: replacement comments.
- `concat`: when true, append `newcomments` to the existing comments.

Outputs:

- Updated EEG dataset when the first input is an EEG dictionary.
- Updated comment text when the first input is comment text.

See also: POP_EDITSET
