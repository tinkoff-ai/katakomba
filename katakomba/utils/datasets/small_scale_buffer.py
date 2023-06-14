import random
import numpy as np
from itertools import cycle
from katakomba.utils.datasets.small_scale import NLDSmallDataset

from typing import Dict, List


# simple utility functions, you can also use map_tree analogs from dm-tree or optree
def dict_slice(
        data: Dict[str, np.ndarray],
        start: int,
        end: int
) -> Dict[str, np.ndarray]:
    return {k: v[start:end] for k, v in data.items()}


def dict_concat(datas: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([d[k] for d in datas]) for k in datas[0].keys()}


def dict_stack(datas: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([d[k] for d in datas]) for k in datas[0].keys()}


class SequentialBuffer:
    def __init__(
            self,
            dataset: NLDSmallDataset,
            batch_size: int,
            seq_len: int,
            add_next_step: bool = False,
            seed: int = 0
    ):
        self.traj = dataset
        self.traj_idxs = list(range(len(self.traj)))
        # shuffle starting trajectories indices
        random.seed(seed)
        random.shuffle(self.traj_idxs)
        # iterator over next free trajectories to pick
        self.free_traj = cycle(self.traj_idxs)
        # index of the current trajectory for each row in batch
        self.curr_traj = np.array([next(self.free_traj) for _ in range(batch_size)], dtype=int)
        # index withing the current trajectory for each row in batch
        self.curr_idx = np.zeros(batch_size, dtype=int)

        self.batch_size = batch_size
        # it will return seq_len + 1, but also will start next traj from seq_len + 1, not seq_len + 2 as in nle
        # this is very useful for DQN-like algorithms training with RNNs
        self.add_next_step = add_next_step
        self.seq_len = seq_len + 1 if add_next_step else seq_len

    def sample(self):
        batch = []
        for i in range(self.batch_size):
            traj_idx = self.curr_traj[i]
            start_idx = self.curr_idx[i]
            data = dict_slice(self.traj[traj_idx], start_idx, start_idx + self.seq_len)

            if len(data["actions"]) < self.seq_len:
                # if next traj will have total_len < seq_len, then get next until data is seq_len
                while len(data["actions"]) < self.seq_len:
                    traj_idx = next(self.free_traj)
                    len_diff = self.seq_len - len(data["actions"])

                    data = dict_concat([
                        data,
                        dict_slice(self.traj[traj_idx], 0, len_diff),
                    ])
                    self.curr_traj[i] = traj_idx
                    self.curr_idx[i] = len_diff - 1 if self.add_next_step else len_diff
            else:
                self.curr_idx[i] += self.seq_len - 1 if self.add_next_step else self.seq_len

            batch.append(data)

        return dict_stack(batch)

    def close(self, clear_cache=True):
        return self.traj.close(clear_cache=clear_cache)
