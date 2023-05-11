import os
import h5py
import shutil
import numpy as np

from tqdm.auto import tqdm

DATA_PATH = os.environ.get('KATAKOMBA_DATA_DIR', os.path.expanduser('~/.katakomba/datasets'))
CACHE_PATH = os.environ.get('KATAKOMBA_CACHE_DIR', os.path.expanduser('~/.katakomba/cache'))


def _flush_to_memmap(filename: str, array: np.ndarray):
    mmap = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
    mmap[:] = array
    mmap.flush()
    return mmap


# TODO: add desc to the tqdm
def load_nld_aa_dataset(character, mode="in_memory"):
    # TODO: check if exists and download dataset first, for now just loading from the path
    # os.makedirs(DATA_PATH, exist_ok=True)

    dataset_path = os.path.join(DATA_PATH, f"data-{character}-any.hdf5")
    df = h5py.File(dataset_path, "r")
    if mode == "in_memory":
        trajectories = {}
        for episode in tqdm(df["/"].keys()):
            episode_data = {
                k: df[episode][k][()] for k in df[episode].keys()
            }
            trajectories[episode] = episode_data

    elif mode == "memmap":
        os.makedirs(CACHE_PATH, exist_ok=True)

        trajectories = {}
        for episode in tqdm(df["/"].keys()):
            episode_cache_path = os.path.join(CACHE_PATH, f"memmap-data-{character}-any", str(episode))

            os.makedirs(episode_cache_path, exist_ok=True)
            episode_data = {
                k: _flush_to_memmap(
                    filename=os.path.join(episode_cache_path, str(k)),
                    array=df[episode][k][()]
                )
                for k in df[episode].keys()
            }
            trajectories[episode] = episode_data

    elif mode == "compressed":
        trajectories = {}
        for episode in tqdm(df["/"].keys()):
            # we do not copy data here! it will decompress it during reading or slicing
            episode_data = {k: df[episode][k] for k in df[episode].keys()}
            trajectories[episode] = episode_data
    else:
        raise RuntimeError("Unknown mode for dataset loading!")

    return df, trajectories


class NLDDataset:
    def __init__(self, character, mode="compressed"):
        self.hdf5_file, self.data = load_nld_aa_dataset(character, mode=mode)
        self.gameids = list(self.data.keys())

        self.character = character
        self.mode = mode

    def __getitem__(self, idx):
        gameid = self.gameids[idx]
        return self.data[gameid]

    def __len__(self):
        return len(self.gameids)

    def metadata(self, idx):
        gameid = self.gameids[idx]
        return dict(self.hdf5_file[gameid].attrs)

    def close(self):
        self.hdf5_file.close()
        # remove memmap files from the disk upon closing
        if self.mode == "memmap":
            shutil.rmtree(os.path.join(CACHE_PATH, f"memmap-data-{self.character}-any"))


def dict_slice(data, start, end):
    return {k: v[start:end] for k, v in data.items()}


def dict_concat(datas):
    return {k: np.concatenate([d[k] for d in datas]) for k in datas[0].keys()}


def dict_stack(datas):
    return {k: np.stack([d[k] for d in datas]) for k in datas[0].keys()}

