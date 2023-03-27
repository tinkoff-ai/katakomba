from nle.dataset.dataset import TtyrecDataset
from torch.utils.data import IterableDataset


class BaseAutoAscend(IterableDataset):
    def __init__(
        self,
        autoascend_iterator_cls,
        ttyrecdata: TtyrecDataset,
        batch_size: int,
        **kwargs
    ):
        self._autoascend_iterator_cls = autoascend_iterator_cls
        self._ttyrecdata = ttyrecdata
        self._batch_size = batch_size
        self._kwargs = kwargs

    def __iter__(self):
        return iter(
            self._autoascend_iterator_cls(
                ttyrecdata=self._ttyrecdata, batch_size=self._batch_size, **self._kwargs
            )
        )
