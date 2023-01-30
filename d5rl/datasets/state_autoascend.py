from nle.dataset.dataset import TtyrecDataset

from d5rl.datasets.base import BaseAutoAscend
from d5rl.utils.observations import tty_to_numpy


class _StateAutoAscendIterator:
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int, yield_freq: float):
        self._ttyrecdata = ttyrecdata
        self._batch_size = batch_size
        self._yield_freq = yield_freq

    def __iter__(self):
        for i, batch in enumerate(self._ttyrecdata):
            if i % self._yield_freq == 0:
                yield tty_to_numpy(
                    tty_chars=batch["tty_chars"][:, -1],
                    tty_colors=batch["tty_colors"][:, -1],
                    tty_cursor=batch["tty_cursor"][:, -1],
                )


class StateAutoAscendTTYDataset(BaseAutoAscend):
    def __init__(self, ttyrecdata: TtyrecDataset, batch_size: int, yield_freq: float):
        super().__init__(
            _StateAutoAscendIterator, ttyrecdata, batch_size, yield_freq=yield_freq
        )
