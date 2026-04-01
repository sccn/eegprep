## EEGDash Verification Scripts

This folder contains helper scripts to:
- discover EEGDash datasets that contain eegprep-compatible raw EEG files,
- run `eegprep.bids_preproc` on each selected dataset via Slurm array jobs,
- parse Slurm logs into a per-dataset status report.

The workflow is designed for the Expanse environment paths currently used in the scripts.

## Report

Link to the Google Sheets:
https://docs.google.com/spreadsheets/d/1HROV8je_9HjcZTYWtB-paMWHowcIrdWg6_lsrmz9jFY/edit?usp=sharing

<img src="Screenshot%202026-04-01%20at%207.41.08%E2%80%AFAM.png" alt="Slurm Job Screenshot" width="700" />



## Files

- `get_all_eegdash_datasets.py`
  - Queries EEGDash (`EEGDash().find_datasets()`), groups datasets by modality, and scans local dataset folders under `dataset_root`.
  - Counts eegprep-compatible extensions in one `os.walk` pass per dataset:
    - `.set`, `.edf`, `.bdf`, `.vhdr`
  - Writes:
    - `reports/eegdash_stats.json` (modality -> dataset -> extension counts)
    - `reports/eegdash_datasets.csv` (flat table with extension counts)
    - `reports/eegdash_datasets.txt` (dataset IDs with at least one compatible file)

- `test_dataset.py`
  - Runs `bids_preproc` on one dataset directory.
  - Key CLI args:
    - `--dataset-name`
    - `--dataset-root`
    - `--output-root`
    - `--reserve-per-job` (default `4GB,1CPU`)
  - Sets BLAS/OpenMP thread env vars to `1` by default to avoid oversubscription in multiprocess runs.

- `test_dataset.slurm`
  - Slurm array launcher for `test_dataset.py`.
  - Reads dataset IDs from `reports/eegdash_datasets.txt`, one per array task.
  - Uses `#SBATCH --array=...%...` for controlled concurrency.

- `parse_logs.py`
  - Parses Slurm output/error logs.
  - Detects NFS/Slurm-level failures.
  - Produces `reports/eegdash_datasets_with_status.csv` with status categories such as:
    - `ok`, `skipped`, `partial`, `ignored`, `error`, `nfs_slurm_error`, `no_files`

- `reports/`
  - Generated artifacts and summary outputs from the scripts above.

## Typical Run Order

1. Build candidate dataset list:
   - run `python scripts/verification/get_all_eegdash_datasets.py`
2. Submit array verification:
   - run `sbatch scripts/verification/test_dataset.slurm`
3. Parse logs and summarize outcomes:
   - run `python scripts/verification/parse_logs.py`

## Notes

- Paths in these scripts are environment-specific (currently `/expanse/projects/nemar/...`).
- If `reports/eegdash_datasets.txt` line count changes, update `#SBATCH --array=1-N%K` in `test_dataset.slurm` accordingly.
