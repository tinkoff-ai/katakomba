import os
import json
import h5py
import argparse
import numpy as np
import nle.dataset as nld

from tqdm.auto import tqdm, trange
from collections import defaultdict
from nle.nethack.actions import ACTIONS
from nle.nethack import tty_render

ACTION_MAPPING = np.zeros(256)
for i, a in enumerate(ACTIONS):
    ACTION_MAPPING[a.value] = i


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
    metadata = dict(dataset.get_meta(game_id))

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

    return data, metadata


def main(args):
    os.makedirs(os.path.join(args.save_path, "metadata"))
    os.makedirs(os.path.join(args.save_path, "data"))

    dbfilename = "tmp_ttyrecs.db"
    if not nld.db.exists(dbfilename):
        nld.db.create(dbfilename)
        nld.add_nledata_directory(args.dataset_path, "autoascend", dbfilename)

    dataset = nld.TtyrecDataset(
        "autoascend",
        batch_size=1,
        seq_length=1,
        dbfilename=dbfilename,
    )

    start = args.start_game_id
    end = args.end_game_id if args.end_game_id > 0 else max(dataset._gameids)

    for game_id in tqdm(range(start, end + 1)):
        data, metadata = load_game(dataset, game_id)

        # saving episode metadata as json
        with open(os.path.join(args.save_path, "metadata", f"{game_id}.json"), "w") as f:
            json.dump(metadata, f)

        # saving episode data as compressed hdf5
        with h5py.File(os.path.join(args.save_path, "data", f"{game_id}.hdf5"), "w") as df:
            df.create_dataset("tty_chars", data=data["tty_chars"], compression="gzip")
            df.create_dataset("tty_colors", data=data["tty_colors"], compression="gzip")
            df.create_dataset("tty_cursor", data=data["tty_cursor"], compression="gzip")
            df.create_dataset("actions", data=data["actions"], compression="gzip")
            df.create_dataset("rewards", data=data["rewards"], compression="gzip")
            df.create_dataset("dones", data=data["dones"], compression="gzip")

    os.remove(dbfilename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converter from NLD dataset ttyrec format to the hdf5 datasets.')
    parser.add_argument("--dataset_path", type=str, default="data/nle_data")
    parser.add_argument("--save_path", type=str, default="data/nle_data_hdf5")
    parser.add_argument("--start_game_id", type=int, default=1)
    parser.add_argument("--end_game_id", type=int, default=-1)

    args = parser.parse_args()
    main(args)
