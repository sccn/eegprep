"""A script that reports the accuracy vs MATLAB (in terms of absolute error)."""

import logging
import socket
import os

import numpy as np

from eegprep import bids_preproc, pop_loadset, eeg_checkset_strict_mode
from eegprep.eeglabcompat import get_eeglab

logger = logging.getLogger(__name__)

curhost = socket.gethostname()

# add your host to this list if you want to run things in parallel
if curhost in ['ck-carbon']:
    reservation = '8GB'
else:
    reservation = ''


# list of studies and subsets thereof to run the statistics on
studies = [
    {
        'studyname': 'ds003061',
        'subjects': ['001', '002'],
        'runs': [1],
    },
    {
        'studyname': 'ds002680',
        'subjects': ['002'],  # first subject, has 2 sessions
        'runs': [10],  # needs to be >= 10 otherwise MATLAB-side filtering by run fails
    }
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # root path of all OpenNeuro datasets on this host
    if curhost == 'ck-carbon':
        root_path = os.path.expanduser('~/data/OpenNeuro')
    else:
        root_path = None
        raise ValueError(f"Skipping report generation on unknown test host {curhost}; "
                         f"please add support for your hostname to the above list to "
                         f"enable this test.")

    # run up to and including the given numeric stage
    # (1=import, 2=select channels, 3=resample, ... up to and including 14=CAR)
    StageNames = {
        1:  'BIDS Import',
        2:  'ChannelSelection',
        3:  'Resampling',
        4:  'FlatlineRemoval',
        5:  'HighpassFilter',
        6:  'BadChannelRemoval',
        7:  'BurstRemoval (ASR)',
        8:  'BadWindowRemoval',
        9:  'PICARD',
        10: 'ICLabel',
        11: 'Reinterpolate',
        12: 'Epoching',
        13: 'BaselineRemoval',
        14: 'CommonAverageRef',
    }

    # Initialize results structure indexed by stage
    results = {}
    for stage in range(1, max(StageNames.keys()) + 1):
        results[stage] = {'all': []}
        for study in studies:
            results[stage][study['studyname']] = []

    # compare the first k trials of each recording
    max_trials = 30

    for study in studies:
        # subset of subjects/runs to compare
        studyname = study['studyname']
        subjects = study['subjects']
        runs = study['runs']

        candidates = [studyname, studyname + '-download']
        retain = [d for d in os.listdir(root_path) if d in candidates]
        if len(retain) == 0:
            raise ValueError(f"None of the candidate study folders {candidates} was "
                             f"found in {root_path}. Cannot generate report.")

        study_path = os.path.join(root_path, retain[0])

        for to_stage in range(1, max(StageNames.keys()) + 1):
            print(f"Running bids_preproc() on {study_path} to stage {to_stage} ({StageNames[to_stage]})...")

            ALLEEG_py = bids_preproc(
                study_path,
                ReservePerJob=reservation,
                # just the first few subjects of the main task
                Subjects=subjects, Runs=runs,
                # reuse results for for quicker re-runs
                SkipIfPresent=True, UseHashes=True, MinimizeDiskUsage=False,
                # parse events from BIDS, use value column
                ApplyEvents=True, EventColumn='value', # <- needed for study ds3061 to match pop_importbids() in MATLAB

                # determine arguments in accordance with to_stage
                OnlyChannelsWithPosition=True if to_stage >= 2 else False,
                OnlyModalities=() if to_stage >= 2 else (),
                SamplingRate=128 if to_stage >= 3 else None,
                FlatlineCriterion=5.0 if to_stage >= 4 else 'off',
                Highpass=(0.25, 0.75) if to_stage >= 5 else 'off',
                ChannelCriterion=0.8 if to_stage >= 6 else 'off',
                LineNoiseCriterion=4.0 if to_stage >= 6 else 'off',
                BurstCriterion=5.0 if to_stage >= 7 else 'off',
                WindowCriterion=0.25 if to_stage >= 8 else 'off',
                WithPicard=to_stage >= 9,
                WithICLabel=to_stage >= 10,
                WithInterp=to_stage >= 11,
                EpochEvents=[] if to_stage >= 12 else None,
                EpochBaseline=[-0.2, 0] if to_stage >= 13 else None,
                CommonAverageReference=to_stage >= 14,

                # misc params
                EpochLimits=[-0.2, 0.5],
                # return results so we can compare things
                ReturnData=True)

            print(f"Running bids_pipeline() on {study_path} to stage {to_stage} ({StageNames[to_stage]})...")
            eeglab = get_eeglab('MATLAB')
            result_paths = eeglab.bids_pipeline(
                study_path,
                [f'sub-{s}' for s in subjects],
                [f'{r}' for r in runs],
                to_stage)

            with eeg_checkset_strict_mode(False):
                ALLEEG_mat = [pop_loadset(p.item()) for p in result_paths.flatten()]
            for p in result_paths.flatten():
                p = p.item()
                if os.path.exists(p):
                    os.remove(p)

            # quantify the absolute difference for this stage
            stage_errs = []
            for k in range(min(len(ALLEEG_py), len(ALLEEG_mat))):
                EEG_py = ALLEEG_py[k]
                EEG_mat = ALLEEG_mat[k]

                if EEG_py['data'].ndim == 3:
                    A = EEG_py['data'][:, :, :max_trials]
                    B = EEG_mat['data'][:, :, :max_trials]
                else:
                    A = EEG_py['data']
                    B = EEG_mat['data']
                abs_err = np.abs(A - B)
                max_err = np.amax(abs_err)
                stage_errs.append(max_err)

            # Store results for this stage and study
            results[to_stage][studyname].extend(stage_errs)
            results[to_stage]['all'].extend(stage_errs)

    # Print summary table
    print("\n\n" + "="*80)
    print("Summary of maximum absolute errors (in µV) between Python and MATLAB results:")
    print("="*80)
    
    # Build header
    header = ["Stage", "StageName", "MaxErr(all)"]
    for study in studies:
        header.append(f"MaxErr({study['studyname']})")
    print("\t".join(header))
    
    # Build rows
    for stage in range(1, max(StageNames.keys()) + 1):
        row = [str(stage), StageNames[stage]]
        
        # Max error across all studies
        all_errs = results[stage]['all']
        if len(all_errs) > 0:
            row.append(f"{np.max(all_errs)}")
        else:
            row.append("N/A")
        
        # Max error per study
        for study in studies:
            studyname = study['studyname']
            study_errs = results[stage][studyname]
            if len(study_errs) > 0:
                row.append(f"{np.max(study_errs)}")
            else:
                row.append("N/A")
        
        print("\t".join(row))
    
    print("="*80)
    print("\nDone.")
