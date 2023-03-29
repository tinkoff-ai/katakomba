import numpy as np
from copy import deepcopy

from katakomba.datasets.base import BaseAutoAscend
from nle.dataset.dataset import TtyrecDataset
from nle.nethack.actions import ACTIONS
from concurrent.futures import ThreadPoolExecutor
from katakomba.utils.render import render_screen_image


class _SARSChaoticAutoAscendTTYIterator:
    def __init__(
        self, ttyrecdata: TtyrecDataset, batch_size: int, threadpool: ThreadPoolExecutor
    ):
        self._ttyrecdata = iter(ttyrecdata)
        self._threadpool = threadpool
        self._batch_size = batch_size

        # Mapping from ASCII keypresses to the gym env actions
        self.action_mapping = np.zeros((256, 1))
        for i, a in enumerate(ACTIONS):
            self.action_mapping[a.value][0] = i

    def __iter__(self):
        # A note: I provided how the sequences look like below (+3 is an example)
        # so it's easier to understand what's happening with the alignment
        cur_batch = self._convert_batch(next(self._ttyrecdata))
        while True:
            # [s_n, s_n+1, s_n+2, s_n+3]
            # [a_n, a_n+1, a_n+2, a_n+3]
            # [r_n-1, r_n, r_n+1, r_n+2]
            # [d_n-1, d_n, d_n+1, d_n+2]
            screen_image, tty_chars, actions, rewards, dones = cur_batch

            # Alignment for rewards, dones, and next_states
            # [r_n, r_n+1, r_n+2, r_n-1]
            # [d_n, d_n+1, d_n+2, d_n-1]
            # [s_n+1, s_n+2, s_n+3, s_n]
            # TODO: gigantic 4x overhead over original loader, somehow we need to optimimze this!
            rewards = np.roll(rewards, shift=-1, axis=1)
            dones = np.roll(dones, shift=-1, axis=1)
            next_screen_image = np.roll(screen_image, shift=-1, axis=1)
            next_tty_chars = np.roll(tty_chars, shift=-1, axis=1)

            # Replace the last element using the information from the next batch
            # [r_n, r_n+1, r_n+2, r_n+3]
            # [d_n, d_n+1, d_n+2, d_n+3]
            # [s_n+1, s_n+2, s_n+3, s_n+4]
            next_batch = self._convert_batch(next(self._ttyrecdata))

            rewards[:, -1] = next_batch[3][:, 0]
            dones[:, -1] = next_batch[4][:, 0]
            next_screen_image[:, -1] = next_batch[0][:, 0]
            next_tty_chars[:, -1] = next_batch[1][:, 0]

            # Move on
            cur_batch = next_batch

            # Method of potentials
            # [r_n+1, r_n+2, r_n+3] - [r_n, r_n+1, r_n+2]
            rewards[:, :-1] = rewards[:, 1:] - rewards[:, :-1]
            # [r_n+4] - [r_n+3]
            rewards[:, -1] = next_batch[3][:, 1] - rewards[:, -1]
            # As in DD, to the best of my knowledge there should not be transitions leading to lower score
            # however, due to the nature of the data, there is an abrupt move from cur_score to zero in the end of the game
            rewards = np.clip(rewards, a_min=0, a_max=None)

            yield screen_image, tty_chars, actions, rewards, next_screen_image, next_tty_chars, dones

    def _convert_batch(self, batch):
        screen_image = render_screen_image(
            tty_chars=batch["tty_chars"],
            tty_colors=batch["tty_colors"],
            tty_cursor=batch["tty_cursor"],
            threadpool=self._threadpool,
        )
        tty_chars = deepcopy(batch["tty_chars"])

        actions = np.take_along_axis(self.action_mapping, batch["keypresses"], axis=0)
        # TODO: score difference as reward
        rewards = deepcopy(batch["scores"]).astype(np.float32)
        dones = deepcopy(batch["done"])
        return screen_image, tty_chars, actions, rewards, dones


class SARSChaoticAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(
        self, ttyrecdata: TtyrecDataset, batch_size: int, threadpool: ThreadPoolExecutor
    ):
        super().__init__(
            _SARSChaoticAutoAscendTTYIterator,
            ttyrecdata,
            batch_size,
            threadpool=threadpool,
        )
