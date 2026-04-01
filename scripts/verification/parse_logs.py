import os
import glob
from tqdm import tqdm

if __name__ == "__main__":
    log_dir = "/expanse/projects/nemar/aman/eegprep/eegprep_slurm"

    error_logs = os.path.join(log_dir, "error")
    output_logs = os.path.join(log_dir, "output")

    assert os.path.exists(error_logs), "Error logs directory does not exist"
    assert os.path.exists(output_logs), "Output logs directory does not exist"

    assert len(os.listdir(error_logs)) == len(os.listdir(output_logs)), "Number of error and output logs do not match"

    print('Renaming files before parsing...')
    for output_log in tqdm(glob.glob(os.path.join(output_logs, "*_*.out"))):
        basename = os.path.basename(output_log)
        with open(output_log, "r") as f:
            lines = f.readlines()
            dataset_name = lines[6].strip().split(" ")[-1]
            os.rename(output_log, os.path.join(output_logs, "{}.out".format(dataset_name)))
            os.rename(os.path.join(error_logs, basename.replace(".out", ".err")), os.path.join(error_logs, "{}.err".format(dataset_name)))


    print('Found {} logs'.format(len(glob.glob(os.path.join(output_logs, "*.out")))))

    print('Checking for NFS and slurm issues...')
    nfs_slurm_issues = []
    for error_log in tqdm(glob.glob(os.path.join(error_logs, "*.err"))):
        basename = os.path.basename(error_log).split(".")[0]
        with open(error_log, "r") as f:
            for line in f.readlines():
                line = line.strip().lower()
                if "stale file handle" in line or "slurm" in line:
                    nfs_slurm_issues.append(basename)
                    break


    if len(nfs_slurm_issues) > 0:
        print('Found {} NFS and slurm issues'.format(len(nfs_slurm_issues)))
        with open('nfs_slurm_issues.txt', 'w') as f:
            for issue in nfs_slurm_issues:
                f.write(issue + '\n')
    else:
        print('No NFS and slurm issues found')
