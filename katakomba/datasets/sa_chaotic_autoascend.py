import numpy as np
from katakomba.datasets.base import BaseAutoAscend
from nle.dataset.dataset import TtyrecDataset
from nle.nethack.actions import ACTIONS
from concurrent.futures import ThreadPoolExecutor
from katakomba.utils.render import render_screen_image


class _SAChaoticAutoAscendTTYIterator:
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
        while True:
            batch = next(self._ttyrecdata)

            actions = np.take_along_axis(
                self.action_mapping, batch["keypresses"], axis=0
            )
            screen_image = render_screen_image(
                tty_chars=batch["tty_chars"],
                tty_colors=batch["tty_colors"],
                tty_cursor=batch["tty_cursor"],
                threadpool=self._threadpool,
            )

            yield (
                batch["tty_chars"],
                batch["tty_colors"],
                batch["tty_cursor"],
                screen_image,
                actions,
            )


class SAChaoticAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(
        self, ttyrecdata: TtyrecDataset, batch_size: int, threadpool: ThreadPoolExecutor
    ):
        super().__init__(
            _SAChaoticAutoAscendTTYIterator,
            ttyrecdata,
            batch_size,
            threadpool=threadpool,
        )
