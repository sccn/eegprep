

if __name__ == "__main__":
    import os
    import sys
    from eegprep import pop_load_frombids, bids_list_eeg_files, pop_loadset, bids_preproc


    if True:
        # no events
        fn = '/home/christian/data/OpenNeuro/ds005815-download/sub-01/ses-2/eeg/sub-01_ses-2_task-rest2_eeg.vhdr'

        # this dataset does not have channel locations
        fn = '/home/christian/data/OpenNeuro/ds006018-download/sub-002/eeg/sub-002_task-visualoddball_eeg.vhdr'

        # giant files that need to be resampled
        fn = '/home/christian/data/OpenNeuro/ds004517-download/sub-01/eeg/sub-01_task-eeg_eeg.bdf'

        # file with associated electrode locations (but very large file!)
        fn = '/home/christian/data/OpenNeuro/ds004324-download/sub-01/ses-01/eeg/sub-01_ses-01_task-RSVP_run-01_eeg.edf'
        EEG = pop_load_frombids(fn, apply_bids_events=True)

    if True:

        # good test case for coord import
        # rt = '/home/christian/data/OpenNeuro/ds004324-download'

        # good test case for coord inference
        rt = '/home/christian/data/OpenNeuro/ds005815-download'

        print(f'parsing BIDS files in {rt}...')
        filelist = bids_list_eeg_files(rt)
        for fn in filelist:
            print(' - ' + fn)

        reserve = '4CPU,4GB-5GB,16max' # ''
        bids_preproc(rt, ReservePerJob=reserve, WithICLabel=True, SkipIfPresent=False)
