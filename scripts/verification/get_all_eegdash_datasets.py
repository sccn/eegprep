import os
import shutil
import logging
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from eegdash import EEGDash
from collections import defaultdict
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def download_dataset(dataset_id: str, download_dir: str):
    dataset_dir = os.path.join(download_dir, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    subprocess.run(
        [
            "s5cmd",
            "--no-sign-request",
            "cp",
            f"s3://openneuro.org/{dataset_id}/*",
            f"{dataset_dir}/",
        ],
        check=True,
    )
    return dataset_dir


def _scan_dataset_for_eegprep_extensions(
    data_paths: [list[str] | str], dataset_id: str, valid_extensions: frozenset[str]
) -> tuple[str, dict[str, int] | None]:
    """One os.walk pass; count files whose extension is in valid_extensions."""
    dataset_dir = None
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    for data_path in data_paths:
        if os.path.exists(os.path.join(data_path, dataset_id)):
            dataset_dir = os.path.join(data_path, dataset_id)
            break

    if dataset_dir is None:
        return dataset_id, None

    counts = defaultdict(int)
    for _, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_extensions:
                counts[ext] += 1
    return dataset_id, dict(counts)


def parse_args():
    parser = argparse.ArgumentParser(description="Get all EEGDash datasets")
    parser.add_argument(
        "--data-paths",
        "-d",
        type=str,
        nargs="+",
        default=[
            "/expanse/projects/nemar/aman/openneuro",
            "/expanse/projects/nemar/openneuro",
        ],
    )
    parser.add_argument("--save-dir", "-s", type=str, default="reports")
    parser.add_argument("--n-jobs", "-j", type=int, default=-1)
    parser.add_argument(
        "--valid-extensions",
        "-e",
        type=str,
        nargs="+",
        default=[".vhdr", ".edf", ".bdf", ".set"],
    )
    parser.add_argument("--download-missing-datasets", "-m", action="store_true")
    parser.add_argument(
        "--download-dir",
        "-dd",
        type=str,
        default="/expanse/projects/nemar/aman/openneuro",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.download_missing_datasets:
        os.makedirs(args.download_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, "EEGDash_datasets.log")
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

    logger.info("--------------------------------")
    # Get the datasets from EEGDash and build a dataframe with dataset_id, source, and recorded_modality

    eegdash = EEGDash()
    datasets = eegdash.find_datasets()

    df = pd.DataFrame(
        [
            {
                "dataset_id": d["dataset_id"],
                "source": d.get("source", "unknown"),
                "recorded_modality": (
                    "_".join(d["recording_modality"])
                    if isinstance(d.get("recording_modality", "unknown"), list)
                    else d.get("recording_modality", "unknown")
                ),
            }
            for d in datasets
        ],
        columns=["dataset_id", "source", "recorded_modality"],
    )
    df = df.sort_values(by="dataset_id")

    logger.info("Found %d datasets", len(df))
    for modality, ds_list in df.groupby("recorded_modality"):
        logger.info("%s: %d", modality, len(ds_list))

    logger.info("--------------------------------")
    # Scan the datasets in parallel for the valid extensions

    #download missing openneuro datasets one-by-one to not overwhelm the system
    if args.download_missing_datasets:
        for _, row in df[df["source"] == "openneuro"].iterrows():
            dataset_id = row["dataset_id"]
            if any(
                [
                    os.path.exists(os.path.join(data_path, dataset_id))
                    for data_path in args.data_paths
                ]
            ):
                continue
            else:
                try:
                    download_dataset(dataset_id, args.download_dir)
                except subprocess.CalledProcessError as e:
                    logger.warning("Failed to download dataset %s: %s", dataset_id, e)
                    shutil.rmtree(os.path.join(args.download_dir, dataset_id))
                    continue

    scan_results_dict = {}
    scan_results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(_scan_dataset_for_eegprep_extensions)(
            args.data_paths,
            row["dataset_id"],
            frozenset(args.valid_extensions),
        )
        for _, row in df.iterrows()
    )
    for dataset_id, file_types in tqdm(
        scan_results, total=len(df), desc="Scanning datasets", unit="dataset"
    ):
        scan_results_dict[dataset_id] = file_types

    for col in ["missing"] + list(args.valid_extensions):
        df[col] = 0 if col != "missing" else False

    for idx, row in df.iterrows():
        file_types = scan_results_dict[row["dataset_id"]]
        if file_types is None:
            df.at[idx, "missing"] = True
            logger.warning(
                "Dataset %s does not exist in any of the data paths",
                row["dataset_id"],
            )
        else:
            for ext in args.valid_extensions:
                df.at[idx, ext] = file_types.get(ext, 0)

    with open(os.path.join(args.save_dir, "EEGDash_datasets.csv"), "w") as f:
        df.to_csv(f, index=False)

    logger.info("--------------------------------")
    # Write the dataset IDs of the eegprep-compatible datasets to a text file

with open(os.path.join(args.save_dir, "eegprep_compatible_datasets.txt"), "w") as f:
    for idx, row in df.iterrows():
        if (
            not row["missing"]
            and "eeg" in row["recorded_modality"].split("_")
            and sum(row[col] for col in args.valid_extensions) > 0
        ):
            f.write(row["dataset_id"] + "\n")
