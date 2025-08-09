

if __name__ == "__main__":
    import os
    import sys
    from eegprep import pop_load_frombids, pop_biosig

    fn = '/Users/arno/Python/eegprep/data/test_file.edf'
    EEG1 = pop_load_frombids(fn, apply_bids_events=True)
    
    # import using eeglabcompat
    EEG2 = pop_biosig(fn)
    
    # compare using eeg_compare
    from eegprep import eeg_compare
    eeg_compare(EEG1, EEG2, verbose_level=1)
    
    