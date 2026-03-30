import json
import os
from collections import defaultdict

from joblib import Parallel, delayed
from tqdm import tqdm
from eegdash import EEGDash


def _scan_dataset_for_eegprep_extensions(
    dataset_root: str,
    dataset_id: str,
    valid_extensions: frozenset[str],
) -> tuple[str, dict[str, int] | None]:
    """One os.walk pass; count files whose extension is in valid_extensions."""
    dataset_dir = os.path.join(dataset_root, dataset_id)
    if not os.path.exists(dataset_dir):
        return dataset_id, None

    counts = defaultdict(int)
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_extensions:
                counts[ext] += 1
    return dataset_id, dict(counts)


if __name__ == "__main__":
    dataset_root = "/expanse/projects/nemar/openneuro"
    valid_eegprep_extensions = [".vhdr", ".edf", ".bdf", ".set"]
    valid_ext_frozen = frozenset(valid_eegprep_extensions)
    n_jobs = -1

    eegdash = EEGDash()
    datasets = eegdash.find_datasets()

    modalities = defaultdict(list)
    for dataset in tqdm(datasets):
        recording_modality = dataset.get("recording_modality", "unknown")

        if isinstance(recording_modality, list):
            recording_modality = "".join(recording_modality)

        modalities[recording_modality.lower()].append(dataset["dataset_id"])

    print(f"Found {len(datasets)} datasets")
    for modality, ds_list in modalities.items():
        print(f"{modality}: {len(ds_list)}")

    modality_stats = defaultdict(dict)
    for modality, ds_list in modalities.items():
        print(f"\nProcessing {modality} modality datasets")
        # return_as='generator_unordered' yields each result as a worker finishes (not when
        # tasks are submitted), so tqdm tracks completion. Order is arbitrary; we key by dataset_id.
        results_iter = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_scan_dataset_for_eegprep_extensions)(
                dataset_root, ds_id, valid_ext_frozen
            )
            for ds_id in ds_list
        )
        results = list(
            tqdm(results_iter, total=len(ds_list), desc=modality, unit="dataset")
        )

        dataset_file_types = {}
        for dataset_id, file_types in results:
            if file_types is None:
                print(
                    f"Dataset directory {os.path.join(dataset_root, dataset_id)} does not exist"
                )
                continue
            dataset_file_types[dataset_id] = file_types
        modality_stats[modality] = dataset_file_types

    with open("eegdash_stats.json", "w") as f:
        json.dump(modality_stats, f, indent=4)

    with open("eegdash_dataset.txt", "w") as f:
        for modality, stats in sorted(modality_stats.items()):
            if not len(stats):
                continue

            for dataset_id, file_types in stats.items():
                if len(file_types):
                    f.write(f"{dataset_id}\n")
