from nle.dataset.dataset import TtyrecDataset

from d5rl.datasets import BaseAutoAscend
from d5rl.utils.observations import tty_to_numpy


class _StateAutoAscendIterator:
    def __init__(
        self,
        ttyrecdata: TtyrecDataset,
        batch_size: int,
    ):
        self._ttyrecdata = ttyrecdata
        self._batch_size = batch_size

    def __iter__(self):
        for batch in self._ttyrecdata:
            yield tty_to_numpy(
                tty_chars=batch["tty_chars"].squeeze(),
                tty_colors=batch["tty_colors"].squeeze(),
                tty_cursor=batch["tty_cursor"].squeeze(),
            )


class StateAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(
        self,
        ttyrecdata: TtyrecDataset,
        batch_size: int,
    ):
        super().__init__(
            _StateAutoAscendIterator,
            ttyrecdata,
            batch_size,
        )
