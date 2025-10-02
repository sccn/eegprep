"""End-to-end test vs MATLAB."""
from eegprep import bids_preproc
import os

if __name__ == '__main__':
    root_path = '../../../openneuro'
    root_path = '/System/Volumes/Data/data/data/STUDIES/'
    root_path = '/System/Volumes/Data/data/matlab/pca_averef/'
    outputdir = './bids_preproc_output'
    bids_collection = ['ds003061'] #'ds002680'] # 'ds003061']
    retain = [d for d in os.listdir(root_path) if d in bids_collection]
    if len(retain) != len(bids_collection):
        self.skipTest(f"Skipping test_end2end because neither {bids_collection} exist in {root_path}")
        
    for bids_study in bids_collection:
        study_path = os.path.join(root_path, bids_study)
        outputdir = os.path.join(outputdir, bids_study)
        os.makedirs(outputdir, exist_ok=True)
        
        print(f"Running bids_preproc() on {study_path}...")
        ALLEEG_py = bids_preproc(
            study_path,
            outputdir=outputdir,
            ReservePerJob='1CPU',
            # just the first 2 subjects of the main task
            subjects=[0,1], 
            runs=[1],
            SkipIfPresent=True, # <- for quicker re-runs
            bidsevent=True,
            SamplingRate=128,
            WithInterp=True, 
            EpochEvents=[], 
            EpochLimits=[-0.2, 0.5], 
            EpochBaseline=[None, 0],
            WithPicard=True, WithICLabel=True,
            MinimizeDiskUsage=False,
            ReturnData=True)