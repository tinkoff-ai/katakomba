import os
import wandb
import numpy as np

import pickle
import pyrallis
from typing import Optional
from dataclasses import dataclass
from tqdm.auto import tqdm

from katakomba.utils.roles import ALLOWED_COMBOS
from katakomba.utils.datasets.small_scale import load_nld_aa_small_dataset
from katakomba.utils.scores import MEAN_SCORES_AUTOASCEND


@dataclass
class Config:
    bc_wandb_group: Optional[str] = "small_scale_bc_chaotic_lstm_multiseed-v0"
    cql_wandb_group: Optional[str] = "small_scale_cql_chaotic_lstm_multiseed-v0"
    awac_wandb_group: Optional[str] = "small_scale_awac_chaotic_lstm_multiseed-v0"
    iql_wandb_group: Optional[str] = "small_scale_iql_chaotic_lstm_multiseed-v0"
    rem_wandb_group: Optional[str] = "small_scale_rem_chaotic_lstm_multiseed-v0"
    checkpoint: int = 500000
    cache_path: str = "cached_algo_stats.pkl"


def get_character_scores(runs, character, filename):
    multiseed_scores = []

    runs = [run for run in runs if run.config["character"] == character]
    for run in runs:
        run.file(filename).download(replace=True)
        multiseed_scores.append(np.load(filename))

    os.remove(filename)
    return np.array(multiseed_scores)


def get_autoascend_scores():
    characters_metrics = {}
    for role, race, align in ALLOWED_COMBOS:
        df, traj = load_nld_aa_small_dataset(role=role, race=race, align=align, mode="compressed")
        returns = np.array([df[gameid].attrs["points"] for gameid in list(traj.keys())])
        norm_scores = returns / MEAN_SCORES_AUTOASCEND[(role, race, align)]
        depths = np.array([df[gameid].attrs["deathlev"] for gameid in list(traj.keys())])

        characters_metrics[f"{role.value}-{race.value}-{align.value}"] = {
            "normalized_scores": norm_scores,
            "returns": returns,
            "depths": depths,
        }
        df.close()

    return characters_metrics


@pyrallis.wrap()
def main(config: Config):
    algo_groups = {
        "BC": config.bc_wandb_group,
        "CQL": config.cql_wandb_group,
        "AWAC": config.awac_wandb_group,
        "IQL": config.iql_wandb_group,
        "REM": config.rem_wandb_group,
    }

    if not os.path.exists(config.cache_path):
        algorithms_scores = {algo_name: {} for algo_name in algo_groups.keys()}
        algorithms_scores["AUTOASCEND"] = get_autoascend_scores()

        api = wandb.Api()
        for algo, group in tqdm(algo_groups.items(), desc="Downloading algorithms scores"):
            algo_runs = [run for run in api.runs("tlab/Nethack") if run.group == group]

            for role, race, align in tqdm(ALLOWED_COMBOS, desc="Downloading character scores", leave=False):
                character = f"{role.value}-{race.value}-{align.value}"
                algorithms_scores[algo][character] = {
                    "normalized_scores": get_character_scores(algo_runs, character, f"{config.checkpoint}_normalized_scores.npy"),
                    "returns": get_character_scores(algo_runs, character, f"{config.checkpoint}_returns.npy"),
                    "depths": get_character_scores(algo_runs, character, f"{config.checkpoint}_depths.npy"),
                }

        with open(config.cache_path, "wb") as f:
            pickle.dump(algorithms_scores, f)


if __name__ == "__main__":
    main()
