import itertools
from copy import deepcopy

import numpy as np
from nle.dataset.dataset import TtyrecDataset

from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.actions import ascii_actions_to_gym_actions
from d5rl.utils.observations import tty_to_numpy

from nle.dataset.dataset import TtyrecDataset

from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.observations import tty_to_numpy


class _SARSAutoAscendTTYIterator:
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        self._ttyrecdata = iter(ttyrecdata)

    def __iter__(self):
        # A note: I provided how the sequences look like below (+3 is an example)
        # so it's easier to understand what's happening with the alignment
        cur_batch = deepcopy(self._convert_batch(next(self._ttyrecdata)))
        
        while True:
            # [s_n, s_n+1, s_n+2, s_n+3]
            # [a_n, a_n+1, a_n+2, a_n+3]
            # [r_n-1, r_n, r_n+1, r_n+2]
            # [d_n-1, d_n, d_n+1, d_n+2]
            states, actions, rewards, dones = cur_batch

            # Alignment for rewards, dones, and next_states
            # [r_n, r_n+1, r_n+2, r_n-1]
            # [d_n, d_n+1, d_n+2, d_n-1]
            # [s_n+1, s_n+2, s_n+3, s_n]
            rewards = np.roll(rewards, shift=-1, axis=1)
            dones = np.roll(dones, shift=-1, axis=1)
            next_states = np.roll(deepcopy(states), shift=-1, axis=1)
            
            # Replace the last element using the information from the next batch
            # [r_n, r_n+1, r_n+2, r_n+3]
            # [d_n, d_n+1, d_n+2, d_n+3]
            # [s_n+1, s_n+2, s_n+3, s_n+4]
            next_batch = deepcopy(self._convert_batch(next(self._ttyrecdata)))
            rewards[:, -1] = next_batch[2][:, -1]
            dones[:, -1] = next_batch[3][:, -1]
            next_states[:, -1] = next_batch[0][:, 0]

            # Move on
            cur_batch = next_batch

            # states: [batch_size, seq_len, 24, 80, 3]
            # actions: [batch_size, seq_len]
            # rewards: [batch_size, seq_len]
            # dones: [batch_size, seq_len]
            # next_states: [batch_size, seq_len, 24, 80, 3]
            yield states, actions, rewards, dones, next_states

    def _convert_batch(self, batch):
        # [batch_size, seq_len, 24, 80, 3]
        states = tty_to_numpy(
            tty_chars=batch["tty_chars"].squeeze(),
            tty_colors=batch["tty_colors"].squeeze(),
            tty_cursor=batch["tty_cursor"].squeeze(),
        )

        # [batch_size, seq_len, 1]
        actions = ascii_actions_to_gym_actions(
            batch["keypresses"]
        )

        # [batch_size, seq_len, 1]
        rewards = batch["scores"]

        # [batch_size, seq_len, 1]
        dones = batch["done"]

        return states, actions, rewards, dones


class SARSAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        super().__init__(
            _SARSAutoAscendTTYIterator, ttyrecdata, batch_size
        )


# class _SARSAutoAscendTTYIterator:
#     def __init__(
#         self,
#         ttyrecdata: TtyrecDataset,
#         batch_size: int,
#         n_prefetched_batches: int,
#         seq_len: int = 1,
#     ):
#         self._ttyrecdata = ttyrecdata
#         self._iterator = iter(ttyrecdata)
#         self._batch_size = batch_size
#         self._n_prefetched_batches = n_prefetched_batches
#         self._seq_len = seq_len
#         self._prev_batch = None

#     def __iter__(self):
#         # Initialize buffers
#         self._states = np.zeros(
#             (self._batch_size, self._n_prefetched_batches, 24, 80, 3)
#         )
#         self._actions = np.zeros((self._batch_size, self._n_prefetched_batches, 1))
#         self._rewards = np.zeros((self._batch_size, self._n_prefetched_batches, 1))
#         self._dones = np.zeros((self._batch_size, self._n_prefetched_batches, 1))

#         # Prefetch states for faster/random access
#         self._init_prefetch()

#         return self

#     def __next__(self):
#         """
#         Returns a usual (s, a, s', r, done), where
#             - s is a tty-screen [batch_size, 80, 24, ?] (uint8)
#             - a is an action [batch_size, 1] (uint8)
#             - s' is a tty-screen [batch_size, 80, 24, ?] (uint8)
#             - r is the change in the game score
#             - whether the episode ended (game-over usually) (bool)
#         """
#         # Prefetch samples if we iterated over
#         if self._cur_ind >= self._total_samples:
#             self._init_prefetch()

#         b_s = np.zeros((self._batch_size, self._seq_len, 24, 80, 3), dtype=np.uint8)
#         b_a = np.zeros((self._batch_size, self._seq_len, 1), dtype=np.uint8)
#         b_r = np.zeros((self._batch_size, self._seq_len, 1), dtype=np.float32)
#         b_d = np.zeros((self._batch_size, self._seq_len, 1), dtype=bool)
#         b_ns = np.zeros((self._batch_size, self._seq_len, 24, 80, 3), dtype=np.uint8)
#         for b_ind in range(self._batch_size):
#             cur_ind = self._2d_indices[
#                 self._indices[self._cur_ind % self._total_samples]
#             ]
#             batch_ind = cur_ind[0]
#             prefetch_ind = cur_ind[1]

#             # for seq_ind in range(self._seq_len):
#             b_s[b_ind] = self._states[
#                 batch_ind, prefetch_ind : prefetch_ind + self._seq_len
#             ]
#             b_a[b_ind] = self._actions[
#                 batch_ind, prefetch_ind : prefetch_ind + self._seq_len
#             ]
#             b_r[b_ind] = self._rewards[
#                 batch_ind, prefetch_ind : prefetch_ind + self._seq_len
#             ]
#             b_d[b_ind] = self._dones[
#                 batch_ind, prefetch_ind : prefetch_ind + self._seq_len
#             ]
#             b_ns[b_ind] = self._states[
#                 batch_ind, prefetch_ind + 1 : prefetch_ind + 1 + self._seq_len
#             ]

#             self._cur_ind += 1

#         return b_s, b_a, b_r, b_d, b_ns

#     def _init_prefetch(self):
#         # Indices to iterate over
#         self._2d_indices = list(
#             itertools.product(
#                 range(self._batch_size),
#                 range(self._n_prefetched_batches - self._seq_len),
#             )
#         )
#         self._total_samples = len(self._2d_indices)
#         self._indices = np.random.choice(
#             a=range(self._total_samples), size=self._total_samples, replace=False
#         )
#         self._cur_ind = 0

#         # Just started
#         start_index = 0
#         if self._prev_batch is None:
#             self._prev_batch = next(self._iterator)
#         else:
#             start_index

#         cur_batch = next(self._iterator)
#         for ind in range(self._n_prefetched_batches):
#             # [batch_size, 24, 80, 3]
#             state = tty_to_numpy(
#                 tty_chars=self._prev_batch["tty_chars"].squeeze(),
#                 tty_colors=self._prev_batch["tty_colors"].squeeze(),
#                 tty_cursor=self._prev_batch["tty_cursor"].squeeze(),
#             )
#             # [batch_size, 1]
#             action = ascii_actions_to_gym_actions(
#                 self._prev_batch["keypresses"].reshape(-1, 1)
#             )
#             # [batch_size, 1]
#             reward = (cur_batch["scores"] - self._prev_batch["scores"]).reshape(
#                 -1, 1
#             )  # potentials are better
#             # [batch_size, 1]
#             done = cur_batch["done"].reshape(-1, 1)

#             # Reward fix for an episode end, there are two cases
#             # 1 - new episode started (reward should be zero)
#             # 2 - we died (there are some states we score is set to zero)
#             reward[reward < 0] = 0.0

#             self._states[:, ind] = state
#             self._actions[:, ind] = action
#             self._rewards[:, ind] = reward
#             self._dones[:, ind] = done

#             self._prev_batch = deepcopy(cur_batch)
#             cur_batch = next(self._iterator)


# class SARSAutoAscendTTYDataset(BaseAutoAscend):
#     def __init__(
#         self,
#         ttyrecdata: TtyrecDataset,
#         batch_size: int,
#         seq_len: int,
#         n_prefetched_batches: int,
#     ):
#         super().__init__(
#             _SARSAutoAscendTTYIterator,
#             ttyrecdata,
#             batch_size,
#             seq_len=seq_len,
#             n_prefetched_batches=n_prefetched_batches,
#         )
