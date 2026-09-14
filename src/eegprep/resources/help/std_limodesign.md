# STD_LIMODESIGN - LIMO-compatible design matrices

`std_limodesign` builds categorical and continuous design matrices from STUDY
factor descriptors and trial metadata.

It supports categorical interactions, split continuous regressors, description
only mode, and optional export of `categorical_variables.txt` and
`continuous_variables.txt`. Use `pop_limo` to fit the active STUDY design in
standalone EEGPrep.

See also: POP_LISTFACTORS, STD_BUILDDESIGNMAT, POP_LIMO
