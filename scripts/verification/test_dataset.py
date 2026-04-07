import os
import argparse

# Set before importing numpy/eegprep so each worker does not multiply threads across processes.
for _key, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_key, _val)

from eegprep import bids_preproc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bids_preproc on one BIDS dataset under dataset-root."
    )
    parser.add_argument("--dataset-name", "-d", type=str, required=True)
    parser.add_argument("--dataset-root", "-p", type=str, nargs="+", required=True)
    parser.add_argument("--output-root", "-o", type=str, required=True)
    parser.add_argument(
        "--reserve-per-job",
        type=str,
        default="4GB,1CPU",
        help=(
            "bids_preproc ReservePerJob: caps parallel workers by RAM and CPU. "
            "Default '4GB,1CPU' suits ~32 CPUs and ~128GB RAM (tune GB after measuring peak RSS per file). "
            "Use '8GB,1CPU' if workers OOM. Empty string uses a single worker."
        ),
    )
    return parser.parse_args()


def test_eegprep(dataset_name, dataset_root, output_root, reserve_per_job: str):
    print(f"Testing {dataset_name} from {dataset_root}")

    dataset_dir = None
    for root in dataset_root:
        if os.path.exists(os.path.join(root, dataset_name)):
            dataset_dir = os.path.join(root, dataset_name)
            break

    if dataset_dir is None:
        raise FileNotFoundError(
            f"Dataset directory {dataset_name} does not exist in any of the dataset roots"
        )

    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    kwargs = dict(
        outputdir=output_dir,
        ReservePerJob=reserve_per_job or "",
        SkipIfPresent=True,
        bidsevent=True,
        SamplingRate=128,
        WithInterp=True,
        EpochEvents=[],
        EpochLimits=[-0.2, 0.5],
        EpochBaseline=[None, 0],
        WithICA=True,
        WithICLabel=True,
        MinimizeDiskUsage=False,
        ReturnData=False,
    )

    bids_preproc(dataset_dir, **kwargs)
    print(f"Finished {dataset_name}; outputs under {output_dir}")


def main():
    args = parse_args()
    test_eegprep(
        args.dataset_name,
        args.dataset_root,
        args.output_root,
        args.reserve_per_job,
    )


if __name__ == "__main__":
    main()
