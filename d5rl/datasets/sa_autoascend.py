import numpy as np
from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.actions import ascii_actions_to_gym_actions
from nle.dataset.dataset import TtyrecDataset
from nle.nethack.actions import ACTIONS
import numba


@numba.njit(parallel=True)
def convert_actions(keypresses, mapping):
    actions = np.zeros_like(keypresses)

    for i in numba.prange(keypresses.shape[0]):
        for j in numba.prange(keypresses.shape[1]):
            actions[i, j] = mapping[keypresses[i, j]]
    return actions


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
            # actions = ascii_actions_to_gym_actions(batch["keypresses"])
            actions = batch["keypresses"]
            # actions = np.take_along_axis(self.action_mapping, batch["keypresses"], axis=0)
            # actions = convert_actions(batch["keypresses"], self.action_mapping.squeeze())

            yield (
                batch["tty_chars"],
                batch["tty_colors"],
                batch["tty_cursor"],
                actions
            )


class SAAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int):
        super().__init__(_SAAutoAscendTTYIterator, ttyrecdata, batch_size)
