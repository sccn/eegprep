
% --- this pipeline is used to run the MATLAB side of the test_bids_preproc() unit test ---

% import BIDS
[STUDY, ALLEEG] = pop_importbids(...    
    '/home/christian/data/OpenNeuro/ds003061-download', ...
    'subjects', {'sub-001','sub-002'}, ...
    'runs', {'run-001'})


