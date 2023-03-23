import numpy as np
from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.actions import ascii_actions_to_gym_actions
from nle.dataset.dataset import TtyrecDataset
from nle.nethack.actions import ACTIONS


class _SAAutoAscendTTYIterator:
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        self._ttyrecdata = iter(ttyrecdata)

        # Mapping from ASCII keypresses to the gym env actions
        self.action_mapping = np.zeros((256, 1))
        for i, a in enumerate(ACTIONS):
            self.action_mapping[a.value][0] = i

    def __iter__(self):
        while True:
            batch = next(self._ttyrecdata)
            actions = ascii_actions_to_gym_actions(batch["keypresses"])
            # actions = batch["keypresses"]
            # actions = np.take_along_axis(self.action_mapping, batch["keypresses"], axis=0)

            yield (
                batch["tty_chars"],
                batch["tty_colors"],
                batch["tty_cursor"],
                actions
            )


class SAAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        super().__init__(_SAAutoAscendTTYIterator, ttyrecdata, batch_size)
