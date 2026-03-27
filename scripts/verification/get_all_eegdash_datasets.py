from tqdm import tqdm
from eegdash import EEGDash

if __name__ == "__main__":
    eegdash = EEGDash()
    datasets = eegdash.find_datasets()
    with open('eegdash_dataset.txt', 'w') as f:
        for dataset in tqdm(datasets):
            f.write(dataset['dataset_id'] + '\n')