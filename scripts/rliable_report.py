import os
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyrallis
from typing import Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
from itertools import product
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from katakomba.utils.roles import ALLOWED_COMBOS
from katakomba.utils.datasets.small_scale import load_nld_aa_small_dataset
from katakomba.utils.scores import MEAN_SCORES_AUTOASCEND


@dataclass
class Config:
    bc_wandb_group: Optional[str] = "small_scale_bc_chaotic_lstm_multiseed-v0"
    cql_wandb_group: Optional[str] = None
    iql_wandb_group: Optional[str] = None
    metric_name: str = "normalized_scores"  # normalized_scores | returns | depths
    checkpoint: int = 500000


def get_scores(runs, character, filename):
    multiseed_scores = []

    runs = [run for run in runs if run.config["character"] == character]
    for run in runs:
        run.file(filename).download(replace=True)
        multiseed_scores.append(np.load(filename))

    os.remove(filename)
    return np.array(multiseed_scores)


def get_autoascend_scores(metric_name):
    characters_metrics = []
    for role, race, align in ALLOWED_COMBOS:
        df, traj = load_nld_aa_small_dataset(role=role, race=race, align=align, mode="compressed")
        if metric_name == "returns":
            metric = np.array([df[gameid].attrs["points"] for gameid in list(traj.keys())])
        elif metric_name == "normalized_scores":
            metric = np.array([df[gameid].attrs["points"] for gameid in list(traj.keys())])
            metric = metric / MEAN_SCORES_AUTOASCEND[(role, race, align)]
        elif metric_name == "depths":
            metric = np.array([df[gameid].attrs["maxlvl"] for gameid in list(traj.keys())])
        else:
            raise RuntimeError("Unknown metric!")

        characters_metrics.append(metric.mean())
        df.close()

    return np.array(characters_metrics)[None, :]


@pyrallis.wrap()
def main(config: Config):
    algo_groups = {
        "BC": config.bc_wandb_group,
        "IQL": config.iql_wandb_group,
        "CQL": config.cql_wandb_group
    }
    filename = f"{config.checkpoint}_{config.metric_name}.npy"
    xlabels = {
        "normalized_scores": "Normalized Score",
        "returns": "Score",
        "depths": "Depth"
    }
    metrics_thresholds = {
        "normalized_scores": 1.0,
        "returns": 5000,
        "depths": 5
    }

    api = wandb.Api()
    algorithms_scores = {
        "AUTOASCEND": get_autoascend_scores(config.metric_name)
    }
    for algo, group in tqdm(algo_groups.items(), desc="Downloading algorithms scores"):
        if group is None:
            continue

        algo_runs = [run for run in api.runs("tlab/Nethack") if run.group == group]

        characters_scores = []
        for role, race, align in tqdm(ALLOWED_COMBOS, desc="Downloading character scores", leave=False):
            character = f"{role.value}-{race.value}-{align.value}"
            characters_scores.append(
                get_scores(algo_runs, character=character, filename=filename).mean(-1)
            )
        algorithms_scores[algo] = np.array(characters_scores).T

    # plotting aggregate metrics with 95% stratified bootstrap CIs
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x)
    ])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        algorithms_scores, aggregate_func, reps=50000
    )
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=list(algorithms_scores.keys()),
        xlabel=xlabels[config.metric_name],
        xlabel_y_coordinate=-1.0,
    )
    plt.savefig("reliable_metrics.png", bbox_inches="tight")

    # plotting performance profiles
    thresholds = np.linspace(0.0, metrics_thresholds[config.metric_name], 64)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        algorithms_scores, thresholds
    )
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
      score_distributions, thresholds,
      performance_profile_cis=score_distributions_cis,
      colors=dict(zip(
          list(algorithms_scores.keys()), sns.color_palette('colorblind')
      )),
      xlabel=fr"{xlabels[config.metric_name]} $(\tau)$",
      ax=ax
    )
    plt.legend()
    plt.savefig("reliable_performance_profile.png", bbox_inches="tight")

    # plotting probability of improvement
    paired_scores = {}
    for algo_x, algo_y in product(algorithms_scores.keys(), algorithms_scores.keys()):
        if algo_x != algo_y:
            paired_scores[f"{algo_x},{algo_y}"] = (
                algorithms_scores[algo_x],
                algorithms_scores[algo_y]
            )

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        paired_scores, metrics.probability_of_improvement, reps=2000
    )
    plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
    plt.savefig("reliable_probability_of_improvement.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
