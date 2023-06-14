import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import pyrallis
from dataclasses import dataclass
from itertools import product
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from katakomba.utils.roles import Role, Race, Alignment
from katakomba.utils.roles import (
    ALLOWED_COMBOS, BASE_COMBOS, EXTENDED_COMBOS, COMPLETE_COMBOS
)

@dataclass
class Config:
    scores_path: str = "cached_algo_stats.pkl"
    metric_name: str = "normalized_scores"  # normalized_scores | returns | depths
    setting: str = "full"


@pyrallis.wrap()
def main(config: Config):
    setting_combos = {
        "full": ALLOWED_COMBOS,
        "base": BASE_COMBOS,
        "extended": EXTENDED_COMBOS,
        "complete": COMPLETE_COMBOS,
    }

    with open(config.scores_path, "rb") as f:
        cached_stats = pickle.load(f)

        algorithms_scores = {}
        for algo in cached_stats:
            all_metrics = []
            for character in cached_stats[algo]:
                role, race, align = character.split("-")
                role, race, align = Role(role), Race(race), Alignment(align)

                if (role, race, align) not in setting_combos[config.setting]:
                    continue

                character_metrics = cached_stats[algo][character][config.metric_name].ravel()
                if config.metric_name == "depths":
                    # levels start from 1, not 0
                    character_metrics = character_metrics + 1
                elif config.metric_name == "normalized_scores":
                    character_metrics = character_metrics * 100.0

                if algo == "AUTOASCEND":
                    # sample min trajectories, to align shapes
                    np.random.seed(32)
                    character_metrics = np.random.choice(character_metrics, size=675, replace=False)

                all_metrics.append(character_metrics)

            algorithms_scores[algo] = np.stack(all_metrics, axis=0).T

        # exclude it for now
        algorithms_scores.pop("AUTOASCEND")

    xlabels = {
        "normalized_scores": "Normalized Score",
        "returns": "Score",
        "depths": "Death Level"
    }
    metrics_thresholds = {
        "normalized_scores": np.linspace(0.0, 100.0, 16),
        "returns": np.linspace(0.0, 5000.0, 16),
        "depths": np.linspace(1.0, 3.0, 6)
    }
    metrics_ticks = {
        "normalized_scores": {"xticks": None, "yticks": [0.0, 0.10, 0.25, 0.5, 0.75, 1.0]},
        "returns": {"xticks": None, "yticks": None},
        "depths": {"xticks": None, "yticks": np.linspace(0.0, 0.1, 5)},
    }

    # plotting aggregate metrics with 95% stratified bootstrap CIs
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x)
    ])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        algorithms_scores, aggregate_func, reps=10000
    )
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=list(algorithms_scores.keys()),
        xlabel=xlabels[config.metric_name],
        xlabel_y_coordinate=-0.25,
    )
    plt.savefig(f"reliable_metrics_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")

    # plotting performance profiles
    thresholds = metrics_thresholds[config.metric_name]
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
        ax=ax,
        yticks=metrics_ticks[config.metric_name]["yticks"],
        xticks=metrics_ticks[config.metric_name]["xticks"]
    )
    plt.legend()
    plt.savefig(f"reliable_performance_profile_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")

    # plotting probability of improvement
    paired_scores = {}
    for algo_y in algorithms_scores.keys():
        if algo_y != "BC":
            paired_scores[f"BC,{algo_y}"] = (
                algorithms_scores["BC"],
                algorithms_scores[algo_y]
            )
    # for algo_x, algo_y in product(algorithms_scores.keys(), algorithms_scores.keys()):
    #     if algo_x != algo_y:
    #         paired_scores[f"{algo_x},{algo_y}"] = (
    #             algorithms_scores[algo_x],
    #             algorithms_scores[algo_y]
    #         )

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        paired_scores, metrics.probability_of_improvement, reps=500
    )
    plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis, xticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.savefig(f"reliable_probability_of_improvement_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    main()
