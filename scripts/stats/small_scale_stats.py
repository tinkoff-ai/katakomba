import numpy as np

from katakomba.utils.roles import ALLOWED_COMBOS
from katakomba.utils.datasets.small_scale import load_nld_aa_small_dataset


def main():
    target_role = "wiz"
    target_race = "hum"
    target_align = "cha"

    for role, race, align in ALLOWED_COMBOS:
        # if not (role.value == target_role and race.value == target_race and align.value == target_align):
        #     continue

        df, traj = load_nld_aa_small_dataset(role=role, race=race, align=align, mode="compressed")
        transitions = [t["actions"].shape[0] for t in traj.values()]
        median_score = np.median([df[gameid].attrs["points"] for gameid in list(traj.keys())])
        median_depth = np.median([df[gameid].attrs["deathlev"] for gameid in list(traj.keys())])
        median_length = np.median(transitions)

        total_transitions = np.sum(transitions)
        total_trajectories = len(traj.keys())

        print(f"{role.value}-{race.value}-{align.value}", median_depth)

        # total_bytes = 0
        # for episode in traj.values():
        #     total_bytes += sum([arr.nbytes for arr in episode.values()])
        #
        # total_gb = round(total_bytes / 1e+9, 1)
        #
        # print(f"{role.value}-{race.value}-{align.value}", total_gb)

        # print(f"{role.value}-{race.value}-{align.value}")
        # print(f"& {total_transitions} & {median_length} & {median_score} & {median_depth} & {total_gb} \\")

        df.close()


if __name__ == "__main__":
    main()