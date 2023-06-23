import os
import h5py
import shutil
import numpy as np

import urllib.request
from typing import Optional
from katakomba.utils.roles import Role, Race, Alignment, ALLOWED_COMBOS
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Any

BASE_REPO_ID = os.environ.get(
    "KATAKOMBA_REPO_ID", os.path.expanduser("Howuhh/katakomba")
)
DATA_PATH = os.environ.get(
    "KATAKOMBA_DATA_DIR", os.path.expanduser("~/.katakomba/datasets")
)
CACHE_PATH = os.environ.get(
    "KATAKOMBA_CACHE_DIR", os.path.expanduser("~/.katakomba/cache")
)


# similar to huggingface_hub function hf_hub_url
def download_dataset(repo_id: str, filename: str, subfolder: Optional[str] = None):
    dataset_path = os.path.join(DATA_PATH, filename)
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    print(f"Downloading dataset: {dataset_url} to {DATA_PATH}")
    with tqdm(unit="B", unit_scale=True, miniters=1, desc="Progress") as t:

        def progress_hook(block_num=1, block_size=1, total_size=None):
            if total_size is not None:
                t.total = total_size
            t.update(block_num * block_size - t.n)

        urllib.request.urlretrieve(dataset_url, dataset_path, reporthook=progress_hook)

    if not os.path.exists(os.path.join(DATA_PATH, filename.split("/")[-1])):
        raise IOError(f"Failed to download dataset from {dataset_url}")


def _flush_to_memmap(filename: str, array: np.ndarray) -> np.ndarray:
    if os.path.exists(filename):
        mmap = np.load(filename, mmap_mode="r")
    else:
        mmap = np.memmap(filename, mode="w+", dtype=array.dtype, shape=array.shape)
        mmap[:] = array
        mmap.flush()

    return mmap


def load_nld_aa_small_dataset(
    role: Role, race: Race, align: Alignment, mode: str = "in_memory"
) -> Tuple[h5py.File, List[Dict[str, Any]]]:
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
        for episode in tqdm(
            df["/"].keys(), leave=False, desc="Preparing (RAM Decompression)"
        ):
            episode_data = {k: df[episode][k][()] for k in df[episode].keys()}
            trajectories[episode] = episode_data

    elif mode == "memmap":
        os.makedirs(CACHE_PATH, exist_ok=True)
        trajectories = {}
        for episode in tqdm(
            df["/"].keys(), leave=False, desc="Preparing (Drive Decompression)"
        ):
            cache_name = f"memmap-{dataset_name.split('.')[0]}"
            episode_cache_path = os.path.join(CACHE_PATH, cache_name, str(episode))

            os.makedirs(episode_cache_path, exist_ok=True)
            episode_data = {
                k: _flush_to_memmap(
                    filename=os.path.join(episode_cache_path, str(k)),
                    array=df[episode][k][()],
                )
                for k in df[episode].keys()
            }
            trajectories[episode] = episode_data

    elif mode == "compressed":
        trajectories = {}
        for episode in tqdm(df["/"].keys(), leave=False, desc="Preparing"):
            # we do not copy data here! it will decompress it during reading or slicing
            episode_data = {k: df[episode][k] for k in df[episode].keys()}
            trajectories[episode] = episode_data
    else:
        raise RuntimeError(
            "Unknown mode for dataset loading! Please use one of: 'compressed', 'in_memory', 'memmap'"
        )

    # TODO: or return NLDSmallDataset here similar to nld loading?
    return df, trajectories


class NLDSmallDataset:
    def __init__(
        self, role: Role, race: Race, align: Alignment, mode: str = "compressed"
    ):
        self.hdf5_file, self.data = load_nld_aa_small_dataset(
            role, race, align, mode=mode
        )
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

    def close(self, clear_cache=True):
        self.hdf5_file.close()
        if self.mode == "memmap" and clear_cache:
            print("Cleaning memmap cache...")
            # remove memmap cache files from the disk upon closing
            cache_name = f"memmap-data-{self.role.value}-{self.race.value}-{self.align.value}-any"
            shutil.rmtree(os.path.join(CACHE_PATH, cache_name))
