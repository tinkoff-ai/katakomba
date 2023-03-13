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
        cur_batch = self._convert_batch(next(self._ttyrecdata))
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
            # TODO: gigantic overhead over original loader, somehow we need to optimimze this!
            rewards = np.roll(rewards, shift=-1, axis=1)
            dones = np.roll(dones, shift=-1, axis=1)
            next_states = np.roll(deepcopy(states), shift=-1, axis=1)

            # Replace the last element using the information from the next batch
            # [r_n, r_n+1, r_n+2, r_n+3]
            # [d_n, d_n+1, d_n+2, d_n+3]
            # [s_n+1, s_n+2, s_n+3, s_n+4]
            next_batch = self._convert_batch(next(self._ttyrecdata))
            rewards[:, -1] = next_batch[2][:, 0]
            dones[:, -1] = next_batch[3][:, 0]
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
            tty_chars=batch["tty_chars"],
            tty_colors=batch["tty_colors"],
            tty_cursor=batch["tty_cursor"],
        )
        # [batch_size, seq_len]
        actions = ascii_actions_to_gym_actions(batch["keypresses"])

        # [batch_size, seq_len]
        rewards = deepcopy(batch["scores"])

        # [batch_size, seq_len]
        dones = deepcopy(batch["done"])

        return states, actions, rewards, dones


class SARSAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        super().__init__(_SARSAutoAscendTTYIterator, ttyrecdata, batch_size)
