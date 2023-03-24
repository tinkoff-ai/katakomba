import numpy as np
from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.actions import ascii_actions_to_gym_actions
from nle.dataset.dataset import TtyrecDataset
from nle.nethack.actions import ACTIONS


class _SAAutoAscendTTYIterator:
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int, prev_action: bool = False):
        self.batch_size = batch_size
        self.prev_action = prev_action

        self._ttyrecdata = iter(ttyrecdata)
        # Mapping from ASCII keypresses to the gym env actions
        self.action_mapping = np.zeros((256, 1))
        for i, a in enumerate(ACTIONS):
            self.action_mapping[a.value][0] = i

    def __iter__(self):
        if self.prev_action:
            prev_action = np.zeros(self.batch_size)

        while True:
            batch = next(self._ttyrecdata)
            # actions = ascii_actions_to_gym_actions(batch["keypresses"])
            # actions = batch["keypresses"]
            actions = np.take_along_axis(self.action_mapping, batch["keypresses"], axis=0)

            curr_batch = (
                batch["tty_chars"],
                batch["tty_colors"],
                batch["tty_cursor"],
                actions,
            )
            if self.prev_action:
                curr_batch = curr_batch + (prev_action,)
                prev_action = actions[:, -1]

            yield curr_batch


class SAAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int, prev_action: bool = False):
        super().__init__(_SAAutoAscendTTYIterator, ttyrecdata, batch_size, prev_action=prev_action)
