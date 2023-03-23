import time

from torch.utils.data import DataLoader

from d5rl.tasks import make_task_builder
from d5rl.datasets.sa_autoascend import SAAutoAscendTTYDataset


NUM_BATCHES = 200
BATCH_SIZE = 256
SEQ_LEN = 32
N_WORKERS = 8
DEVICE = "cpu"

env_builder, dataset_builder = make_task_builder("NetHackScore-v0-tty-bot-v0", data_path="../nethack/nle_data")

dataset = dataset_builder.build(
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    n_workers=N_WORKERS,
    auto_ascend_cls=SAAutoAscendTTYDataset
)

loader = DataLoader(
    dataset=dataset,
    # Disable automatic batching
    batch_sampler=None,
    batch_size=None,
)

start = time.time()
for ind, batch in enumerate(loader):
    device_batch = [t.to(DEVICE) for t in batch]

    if (ind + 1) == NUM_BATCHES:
        break
end = time.time()
elapsed = end - start
print(
    f"Fetching {NUM_BATCHES} batches of [batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}] took: {elapsed} seconds."
)
print(f"1 batch takes around {elapsed / NUM_BATCHES} seconds.")
print(f"Total frames fetched: {NUM_BATCHES * BATCH_SIZE * SEQ_LEN}")
print(f"Frames / s: {NUM_BATCHES * BATCH_SIZE * SEQ_LEN / elapsed}")
