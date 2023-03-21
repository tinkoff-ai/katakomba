import time

from torch.utils.data import DataLoader

from d5rl.tasks import make_task_builder

<<<<<<< HEAD
NUM_BATCHES = 50
BATCH_SIZE = 128
SEQ_LEN = 512
N_WORKERS = 8
DEVICE = "cpu"
=======
NUM_BATCHES = 10
BATCH_SIZE = 256
SEQ_LEN = 1000
N_WORKERS = 32
DEVICE = "cuda"
>>>>>>> main

env_builder, dataset_builder = make_task_builder("NetHackScore-v0-tty-bot-v0", data_path="../nethack/nle_data")

dataset = dataset_builder.build(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, n_workers=N_WORKERS)

loader = DataLoader(
    dataset=dataset,
    # Disable automatic batching
    batch_sampler=None,
    batch_size=None,
)

start = time.time()
for ind, batch in enumerate(loader):
    states, actions, rewards, dones, next_states = batch
    states.to(DEVICE)
    actions.to(DEVICE)
    rewards.to(DEVICE)
    dones.to(DEVICE)
    next_states.to(DEVICE)

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
