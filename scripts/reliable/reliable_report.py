"""
This is a script for extracting the scores (used for populating values in utils/scores.py)
"""
import nle.dataset as nld
import numpy as np
import random
import matplotlib


from nle.dataset.db import db as nld_database
from katakomba.utils.roles import ALLOWED_COMBOS, Role, Race, Alignment, Sex

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


db_path = "ttyrecs.db"
db_connection = nld.db.connect(filename=db_path)

with nld_database(conn=db_connection) as connection:
    c = connection.execute(
        "SELECT games.role, games.race, games.align0, games.gender0, games.points "
        "FROM games "
        "JOIN datasets ON games.gameid=datasets.gameid "
        "WHERE datasets.dataset_name='autoascend' "
    )

    all_scores = {}
    for row in c:
        role, race, alignment, gender, score = row
        key = (role, race, alignment, gender)
        if key not in all_scores:
            all_scores[key] = [score]
        else:
            all_scores[key].append(score)

min_games = min([len(all_scores[key]) for key in all_scores])

# Sampling minimum games results in around the same average score (but it's lower than all aggregated by around 500)
# for _ in range(100):
#     aggregated_scores = []
#     for key in all_scores:
#         aggregated_scores.extend(random.choices(all_scores[key], k=min_games))
#     print(np.mean(aggregated_scores))

aligned_scores = []
for key in all_scores:
    aligned_scores.append(random.choices(all_scores[key], k=min_games))

all_scores = {
    "AutoAscend-Bot": np.array(aligned_scores).transpose() / 9600.0,
    "Zero": np.array(aligned_scores).transpose() * 0.0,
    "Half": np.array(aligned_scores).transpose() / 9600 * 2,
}
print(all_scores["AutoAscend-Bot"].shape)

aggregate_func = lambda x: np.array(
    [
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ]
)
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    all_scores, aggregate_func, reps=500
)
print(aggregate_scores)


# hotfix for the image, does not work otherwise
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_score_cis,
    metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
    algorithms=["AutoAscend-Bot", "Zero", "Half"],
    xlabel="",
    xlabel_y_coordinate=-0.16,
)
plt.savefig("reliable.png", bbox_inches="tight")

thresholds = np.linspace(0.0, 8.0, 81)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    all_scores, thresholds
)
print("mkay?!")
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
plot_utils.plot_performance_profiles(
    score_distributions,
    thresholds,
    performance_profile_cis=score_distributions_cis,
    colors=dict(
        zip(["AutoAscend-Bot", "Zero", "Half"], sns.color_palette("colorblind"))
    ),
    xlabel=r"AutoAscend Normalized Score $(\tau)$",
    ax=ax,
)
plt.savefig("reliable1.png", bbox_inches="tight")
