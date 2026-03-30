#!/bin/bash
#SBATCH --job-name=verify-eegprep
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --account=csd403
#SBATCH --array=1-433%25
#SBATCH --output=/expanse/projects/nemar/aman/eegprep/eegprep_slurm/output/%A_%a.out
#SBATCH --error=/expanse/projects/nemar/aman/eegprep/eegprep_slurm/error/%A_%a.err

# One array task per line in eegdash_dataset.txt; runs scripts/verification/test_dataset.py
# Node assumed: 32 CPUs, 128GB — ReservePerJob 4GB,1CPU caps parallel workers by RAM + CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p /expanse/projects/nemar/aman/eegprep/eegprep_slurm/output
mkdir -p /expanse/projects/nemar/aman/eegprep/eegprep_slurm/error

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

source /home/ajaiswal4/eegdash-dataset-verification/.venv/bin/activate

echo ""
echo "Python: $(command -v python3)"
echo ""

cd /home/ajaiswal4/github/eegprep

DATASET_LIST=/home/ajaiswal4/github/eegprep/scripts/verification/eegdash_dataset.txt
dataset=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${DATASET_LIST}")

if [[ -z "${dataset}" ]]; then
  echo "Error: No dataset found for array task ID ${SLURM_ARRAY_TASK_ID} in ${DATASET_LIST}"
  exit 1
fi

echo "Processing dataset: ${dataset}"

python3 scripts/verification/test_dataset.py \
  --dataset-name "${dataset}" \
  --dataset-root /expanse/projects/nemar/openneuro \
  --output-root /expanse/projects/nemar/aman/eegprep/eegprep_test \
  --reserve-per-job "${RESERVE_PER_JOB:-4GB,1CPU}"

echo "Finished ${dataset} at $(date)"
