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
        "normalized_scores": 1.0,
        "returns": 5000,
        "depths": 5
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
        xlabel_y_coordinate=-1.0,
    )
    plt.savefig(f"reliable_metrics_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")

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
    plt.savefig(f"reliable_performance_profile_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")

    # plotting probability of improvement
    paired_scores = {}
    for algo_x, algo_y in product(algorithms_scores.keys(), algorithms_scores.keys()):
        if algo_x != algo_y:
            paired_scores[f"{algo_x},{algo_y}"] = (
                algorithms_scores[algo_x],
                algorithms_scores[algo_y]
            )

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        paired_scores, metrics.probability_of_improvement, reps=500
    )
    plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
    plt.savefig(f"reliable_probability_of_improvement_{config.metric_name}.pdf", bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    main()
