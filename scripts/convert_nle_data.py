import os
import json
import h5py
import zipfile
import numpy as np
import nle.dataset as nld

import pyrallis
from dataclasses import dataclass
from typing import Optional

from tqdm.auto import tqdm
from collections import defaultdict
from nle.nethack.actions import ACTIONS

ACTION_MAPPING = np.zeros(256)
for i, a in enumerate(ACTIONS):
    ACTION_MAPPING[a.value] = i


@dataclass
class Config:
    data_path: str = "data/nle_data"
    save_path: str = "data/nle_data_converted"
    race: Optional[str] = None
    role: Optional[str] = None
    alignment: Optional[str] = None
    gender: Optional[str] = None


def reward_as_score_diff(scores):
    rewards = np.zeros(len(scores))
    for i in range(len(scores) - 1):
        # score at step i: the in-game score at this timestep (the result of the action at the previous timestep)
        rewards[i] = scores[i + 1] - scores[i]
    # last reward will be repeated (it is not defined, as we don't have t + 1 score for last state)
    rewards[-1] = rewards[-2]
    # clip as for some reason last steps after death can have zero scores
    return rewards.clip(0)


def load_game(dataset, game_id):
    raw_data = defaultdict(list)
    for step in dataset.get_ttyrec(game_id, 1)[:-1]:
        # check that this step is not padding
        assert step["gameids"][0, 0] != 0
        raw_data["tty_chars"].append(step["tty_chars"].squeeze())
        raw_data["tty_colors"].append(step["tty_colors"].squeeze())
        raw_data["tty_cursor"].append(step["tty_cursor"].squeeze())

        raw_data["actions"].append(ACTION_MAPPING[step["keypresses"].item()])
        # raw_data["done"].append(step["done"].item())
        raw_data["scores"].append(step["scores"].item())

    data = {
        "tty_chars": np.stack(raw_data["tty_chars"]),
        "tty_colors": np.stack(raw_data["tty_colors"]),
        "tty_cursor": np.stack(raw_data["tty_cursor"]),
        "actions": np.array(raw_data["actions"]).astype(np.int16),
        "rewards": reward_as_score_diff(raw_data["scores"]).astype(np.int32),
        # dones are broken in NLD-AA, so we just rewrite them with always done at last step
        # see: https://github.com/facebookresearch/nle/issues/355
        "dones": np.zeros(len(raw_data["actions"]), dtype=bool)
    }
    data["dones"][-1] = True
    return data


def optional_eq(x, cond):
    if cond is not None:
        return x == cond
    return True


def name(role, race, align, gender):
    return f"{role or 'any'}-{race or 'any'}-{align or 'any'}-{gender or 'any'}"


@pyrallis.wrap()
def main(config: Config):
    os.makedirs(config.save_path, exist_ok=True)

    dbfilename = "tmp_ttyrecs.db"
    if not nld.db.exists(dbfilename):
        nld.db.create(dbfilename)
        nld.add_nledata_directory(config.data_path, "autoascend", dbfilename)

    dataset = nld.TtyrecDataset(
        "autoascend",
        batch_size=1,
        seq_length=1,
        dbfilename=dbfilename,
    )
    metadata = [dict(dataset.get_meta(game_id)) for game_id in dataset._gameids]
    metadata = list(filter(
        lambda k: (
                optional_eq(k["role"].lower(), config.role) and
                optional_eq(k["race"].lower(), config.race) and
                optional_eq(k["align"].lower(), config.alignment) and
                optional_eq(k["gender"].lower(), config.gender)
        ),
        metadata
    ))
    file_name = name(config.role, config.race, config.alignment, config.gender)

    # saving episode metadata as json
    with open(os.path.join(config.save_path, f"metadata-{file_name}.json"), "w") as f:
        json.dump(metadata, f)

    # saving episodes data as compressed hdf5
    with h5py.File(os.path.join(config.save_path, f"data-{file_name}.hdf5"), "w", track_order=True) as df:
        for ep in tqdm(metadata):
            data = load_game(dataset, game_id=ep["gameid"])

            g = df.create_group(str(ep["gameid"]))
            g.create_dataset("tty_chars", data=data["tty_chars"], compression="gzip")
            g.create_dataset("tty_colors", data=data["tty_colors"], compression="gzip")
            g.create_dataset("tty_cursor", data=data["tty_cursor"], compression="gzip")
            g.create_dataset("actions", data=data["actions"], compression="gzip")
            g.create_dataset("rewards", data=data["rewards"], compression="gzip")
            g.create_dataset("dones", data=data["dones"], compression="gzip")

    with zipfile.ZipFile(os.path.join(config.save_path, f"data-{file_name}.hdf5.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(config.save_path, f"data-{file_name}.hdf5"))

    os.remove(dbfilename)
    os.remove(os.path.join(config.save_path, f"data-{file_name}.hdf5"))


if __name__ == "__main__":
    main()
