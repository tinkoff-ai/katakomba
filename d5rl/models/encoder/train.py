from torch.utils.data import DataLoader
from tqdm import tqdm

from d5rl.datasets import AutoAscendDatasetBuilder
from d5rl.datasets.state_autoascend import StateAutoAscendTTYDataset

path_to_nld_aa = "/app/data/nld-aa-encoder/nld-aa/nle_data"


if __name__ == "__main__":
    builder = AutoAscendDatasetBuilder(path_to_nld_aa)
    dataset = builder.build(256, StateAutoAscendTTYDataset)
    loader = DataLoader(dataset, pin_memory=True, prefetch_factor=10, num_workers=2)

    for i, v in tqdm(enumerate(loader)):
        pass
        if i == 1000:
            break
