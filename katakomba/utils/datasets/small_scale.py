import os
import h5py
import shutil
import numpy as np

import urllib
from typing import Optional
from katakomba.utils.roles import Role, Race, Alignment, ALLOWED_COMBOS
from tqdm.auto import tqdm

BASE_REPO_ID = os.environ.get('KATAKOMBA_REPO_ID', os.path.expanduser('Howuhh/katakomba'))
DATA_PATH = os.environ.get('KATAKOMBA_DATA_DIR', os.path.expanduser('~/.katakomba/datasets'))
CACHE_PATH = os.environ.get('KATAKOMBA_CACHE_DIR', os.path.expanduser('~/.katakomba/cache'))


# similar to huggingface_hub function hf_hub_url
def download_dataset(
        repo_id: str,
        filename: str,
        subfolder: Optional[str] = None
):
    dataset_path = os.path.join(DATA_PATH, filename)
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    print(f"Downloading dataset: {dataset_url} to {DATA_PATH}")
    urllib.request.urlretrieve(dataset_url, dataset_path)

    if not os.path.exists(os.path.join(DATA_PATH, filename.split("/")[-1])):
        raise IOError(f"Failed to download dataset from {dataset_url}")


def _flush_to_memmap(filename: str, array: np.ndarray):
    if os.path.exists(filename):
        mmap = np.load(filename, mmap_mode="r")
    else:
        mmap = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
        mmap[:] = array
        mmap.flush()

    return mmap


def load_nld_aa_small_dataset(
        role: Role,
        race: Race,
        align: Alignment,
        mode="in_memory"
):
    os.makedirs(DATA_PATH, exist_ok=True)
    if (role, race, align) not in ALLOWED_COMBOS:
        raise RuntimeError(
            "Invalid character combination! "
            "Please see all allowed combos in the katakomba/utils/roles.py"
        )
    dataset_name = f"data-{role.value}-{race.value}-{align.value}-any.hdf5"
    if not os.path.exists(os.path.join(DATA_PATH, dataset_name)):
        download_dataset(
            repo_id=BASE_REPO_ID,
            subfolder="data",
            filename=dataset_name,
        )

    dataset_path = os.path.join(DATA_PATH, dataset_name)
    df = h5py.File(dataset_path, "r")

    if mode == "in_memory":
        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False):
            episode_data = {
                k: df[episode][k][()] for k in df[episode].keys()
            }
            trajectories[episode] = episode_data

    elif mode == "memmap":
        os.makedirs(CACHE_PATH, exist_ok=True)

        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False):
            cache_name = f"memmap-{dataset_name.split('.')[0]}"
            episode_cache_path = os.path.join(CACHE_PATH, cache_name, str(episode))

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
        for episode in tqdm(df["/"].keys(), leave=False):
            # we do not copy data here! it will decompress it during reading or slicing
            episode_data = {k: df[episode][k] for k in df[episode].keys()}
            trajectories[episode] = episode_data
    else:
        raise RuntimeError("Unknown mode for dataset loading! Please use one of: 'compressed', 'in_memory', 'memmap'")

    # TODO: or return NLDSmallDataset here similar to nld loading?
    return df, trajectories


class NLDSmallDataset:
    def __init__(
            self,
            role: Role,
            race: Race,
            align: Alignment,
            mode="compressed"
    ):
        self.hdf5_file, self.data = load_nld_aa_small_dataset(role, race, align, mode=mode)
        self.gameids = list(self.data.keys())

        self.role = role
        self.race = race
        self.align = align
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
            cache_name = f"memmap-data-{self.role.value}-{self.race.value}-{self.align.value}-any"
            shutil.rmtree(os.path.join(CACHE_PATH, cache_name))
