import time

from torch.utils.data import DataLoader

from d5rl.tasks import make_task_builder

NUM_BATCHES = 10
BATCH_SIZE = 256
SEQ_LEN = 1000
DEVICE = "cpu"

env_builder, dataset_builder = make_task_builder("NetHackScore-v0-tty-bot-v0")

dataset = dataset_builder.build(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

loader = DataLoader(
    dataset=dataset,
    # Disable automatic batching
    batch_sampler=None,
    batch_size=None,
)

start = time.time()
for ind, batch in enumerate(loader):
    states, actions, rewards, dones, next_states, prev_actions = batch
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
