
![Katakomba: Tools and Benchmarks for Data-Driven NetHack](katakomba.png)
<p align="center"><b>Katakomba</b> is an open-source benchmark for data-driven NetHack. At the moment, it provides a set of standardized datasets with familiar interfaces and offline RL baselines augmented with recurrence. Full training logs synced to the Weights&Biases are included.</p>

<div align="center">
  
  [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/vladkurenkov/status/1669826074488782849)
  [![arXiv](https://img.shields.io/badge/arXiv-2306.08772-b31b1b.svg)](https://arxiv.org/abs/2306.08772)
</div>

## Installation

For now, Katakomba is not pip installable. However, the installation is easy. 
We also provide an alternative with the Dockerfile already set up to work (this is the preferred way!).
```bash
git clone https://github.com/tinkoff-ai/katakomba.git && cd katakomba
pip install -r requirements.txt

# or alternatively, you could use docker
docker build -t katakomba .
docker run --gpus all -it --rm --name katakomba katakomba
```

One last step is the installation of additional utils used for faster rendering of `tty` observations as images:
```bash
# use pip3 inside the docker container
pip install -e katakomba/utils/render_utils/
```

## Getting Started

```python
from katakomba.env import NetHackChallenge, OfflineNetHackChallengeWrapper
from katakomba.utils.datasets import SequentialBuffer

# The task is specified using the character field
env = NetHackChallenge (
  character = "mon-hum-neu",
  observation_keys = ["tty_chars", "tty_colors", "tty_cursor"]
)

# A convenient wrapper that provides interfaces for dataset loading, score normalization, and deathlevel extraction
env = OfflineNetHackChallengeWrapper(env)

# Several options for dataset reading (check the paper for details): 
# - from RAM, decompressed ("in_memory"): fast but requires a lot of RAM, takes 5-10 minutes for decompression first
# - from Disk, decompressed ("memmap"): a bit slower than RAM, takes 5-10 minutes for decompression first
# - from Disk, compressed ("compressed"): very slow but no need for decompression, useful for debugging
# Note that this will download the dataset automatically if not found
dataset = env.get_dataset(mode="memmap", scale="small")

# Auxillary tools for computing normalized scores or extracting deathlevels
env.get_normalized_score(score=1337.0)
env.get_current_depth()

# We also provide an example of a sequential replay buffer
buffer = SequentialBuffer(
  dataset=dataset,
  seq_len=YOUR_SEQ_LEN,
  batch_size=YOUR_BATCH_SIZE, # Each batch element is a different trajectory
  seed=YOUR_SEED,
  add_next_step=True # if you want (s, a, r, s') instead of (s, a, r)
)

# What's inside the batch?
# Note that the next batch will include the +1 element as expected
batch = buffer.sample()
print(
  batch["tty_chars"],  # [batch_size, seq_len + 1, 80, 24]
  batch["tty_colors"], # [batch_size, seq_len + 1, 80, 24]
  batch["tty_cursor"], # [batch_size, seq_len + 1, 2]
  batch["actions"],    # [batch_size, seq_len + 1]
  batch["rewards"],    # [batch_size, seq_len + 1]
  batch["dones"]       # [batch_size, seq_len + 1]
)

# In case you don't want to store the decompressed dataset beyond code execution
dataset.close()
````


## Baselines

We also provide a set of offline RL baselines for discrete control augmented with recurrence. Implementations are based on the Chaotic-Dwarven-GPT-5 architecture and kept as simple as possible, feel free to dive in both full training logs and algorithms: you can find many useful stuff like sequential replay buffer or bias-corrected vectorized evaluation.

| Algorithm                                                                                                                       | Variants Implemented                               | Wandb Report |
|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------| ----------- |
| ✅ [Behavioral Cloning <br>(BC)](https://www.semanticscholar.org/paper/Cognitive-models-from-subcognitive-skills-Michie-Bain/d40aff59c9b0785e0d75765b0040430ffc377f2d)                                                                                                   | [`bc_chaotic_lstm.py`](algorithms/small_scale/bc_chaotic_lstm.py) |  [`Katakomba-All`](https://wandb.ai/tlab/NetHack/reports/-Offline-BC-Katakomba-All--Vmlldzo0NjA1OTI3)
| ✅ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)                      | [`cql_chaotic_lstm.py`](algorithms/small_scale/cql_chaotic_lstm.py)                      | [`Katakomba-All`](https://wandb.ai/tlab/NetHack/reports/-Offline-CQL-Katakomba-All--Vmlldzo0NjEwOTU0)
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)               | [`awac_chaotic_lstm.py`](algorithms/small_scale/awac_chaotic_lstm.py)                    | [`Katakomba-All`](https://wandb.ai/tlab/NetHack/reports/-Offline-AWAC-Katakomba-All--Vmlldzo0NjEwNzQx)
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)                         | [`iql_chaotic_lstm.py`](algorithms/small_scale/iql_chaotic_lstm.py)                      | [`Katakomba-All`](https://wandb.ai/tlab/NetHack/reports/-Offline-IQL-Katakomba-All--Vmlldzo0NjEwNzQ4)
| ✅ [An Optimistic Perspective on Offline Reinforcement Learning <br>(REM)](https://arxiv.org/abs/1907.04543)                     | [`rem_chaotic_lstm.py`](algorithms/small_scale/rem_chaotic_lstm.py)                      | [`Katakomba-All`](https://wandb.ai/tlab/NetHack/reports/-Offline-REM-Katakomba-All--Vmlldzo0NjEwOTYw)

## Datasets

In our benchmark, we treat every character configuration as a separate game to be solved -- different configurations may require highly varied forms of gameplay in the early game. To this end, we repacked the original large-scale AutoAscend (this symbolic agent is essentialy an early-game contender) dataset into 38 smaller datasets. This decomposition should allow practitioners to download less data and be more focused on specifics. 

Additionally, as benchmarking new algorithms on all of the datasets could be computationally expensive for many practitioners, we separate the benchmark into three categories, where ```roles > races > alignments``` as by wisdom of the NetHack community. 

We host all of the datasets on the [HuggingFace](https://huggingface.co/datasets/Howuhh/katakomba/tree/main/data), you can download them from there directly. But as we described above, our wrappers will take care of it automatically similar to the D4RL benchmark. The script for repacking the large-scale dataset can be found [here](scripts/generate_small_dataset.py).

### Tasks

| **Tasks**                        | **# Transitions** | **Median Turns** | **Median Score** | **Median Deathlvl** | **Size (GB)** | **Compressed Size (GB)** |
|---------------------------------------|-------------------------|-----------------------|-----------------------|--------------------------|--------------------|-------------------------------|
| **Base (Role-Centric)**          | -                       | -                     | -                     | -                        | -                  | -                             |
| ```arc-hum-neu```               | 24527163                | 32858.0               | 4802.5                | 2.0                      | 94.5               | 1.3                           |
| ```bar-hum-neu```               | 26266771                | 35716.0               | 11964.0               | 4.0                      | 101.1              | 1.7                           |
| ```cav-hum-neu```               | 21674680                | 30361.0               | 8152.0                | 4.0                      | 83.5               | 1.3                           |
| ```hea-hum-neu```               | 14473997                | 18051.0               | 2043.0                | 1.0                      | 55.7               | 0.8                           |
| ```kni-hum-law```               | 22287283                | 28246.0               | 6305.0                | 3.0                      | 85.8               | 1.5                           |
| ```mon-hum-neu```               | 33741542                | 42400.0               | 11356.0               | 4.0                      | 129.9              | 2.1                           |
| ```pri-hum-neu```               | 18376473                | 26796.5               | 5366.5                | 2.0                      | 70.8               | 1.1                           |
| ```ran-hum-neu```               | 17625493                | 25354.0               | 6168.0                | 2.0                      | 67.9               | 1.0                           |
| ```rog-hum-cha```               | 14284927                | 19334.0               | 3005.5                | 1.0                      | 55.0               | 0.8                           |
| ```sam-hum-law```               | 22422537                | 32951.0               | 7850.0                | 4.0                      | 86.3               | 1.3                           |
| ```tou-hum-neu```               | 13376498                | 17955.5               | 2554.5                | 1.0                      | 51.5               | 0.8                           |
| ```val-hum-neu```               | 27784788                | 35250.0               | 11402.5               | 4.0                      | 107.0              | 1.8                           |
| ```wiz-hum-neu```               | 14343449                | 19808.5               | 3132.5                | 1.0                      | 55.2               | 0.8                           |
| **Extended (Race-Centric)**      | -                       | -                     | -                     | -                        | -                  | -                             |
| ```pri-elf-cha```               | 18796560                | 26909.5               | 4718.5                | 2.0                      | 72.4               | 1.1                           |
| ```ran-elf-cha```               | 18238686                | 26607.0               | 7583.0                | 4.0                      | 70.2               | 1.1                           |
| ```wiz-elf-cha```               | 15277820                | 19512.0               | 2988.5                | 1.0                      | 58.8               | 0.9                           |
| ```arc-dwa-law```               | 25100788                | 34669.0               | 4026.0                | 1.0                      | 96.7               | 1.5                           |
| ```cav-dwa-law```               | 22871890                | 32261.0               | 7158.0                | 3.0                      | 88.1               | 1.5                           |
| ```val-dwa-law```               | 32787658                | 33973.0               | 8652.5                | 3.0                      | 126.6              | 2.5                           |
| ```arc-gno-neu```               | 24144048                | 34432.0               | 4077.5                | 1.0                      | 93.0               | 1.4                           |
| ```cav-gno-neu```               | 21624779                | 29860.0               | 6446.0                | 3.0                      | 83.3               | 1.4                           |
| ```hea-gno-neu```               | 14884704                | 18518.0               | 1980.5                | 1.0                      | 57.3               | 0.9                           |
| ```ran-gno-neu```               | 17571659                | 25970.0               | 5326.0                | 2.0                      | 67.7               | 1.1                           |
| ```wiz-gno-neu```               | 14193637                | 19206.0               | 2736.0                | 1.0                      | 54.7               | 0.9                           |
| ```bar-orc-cha```               | 27826356                | 39291.0               | 10499.0               | 4.0                      | 107.2              | 1.8                           |
| ```ran-orc-cha```               | 18127448                | 26707.0               | 5460.0                | 2.0                      | 69.8               | 1.1                           |
| ```rog-orc-cha```               | 16674806                | 22351.0               | 3103.0                | 1.0                      | 64.2               | 1.0                           |
| ```wiz-orc-cha```               | 15994150                | 22570.5               | 3241.5                | 1.0                      | 61.6               | 1.0                           |
| **Complete (Alignment-Centric)** | -                       | -                     | -                     | -                        | -                  | -                             |
| ```arc-hum-law```               | 23422383                | 31446.0               | 4188.0                | 1.0                      | 90.2               | 1.3                           |
| ```cav-hum-law```               | 22328494                | 31039.0               | 8174.0                | 4.0                      | 86.0               | 1.3                           |
| ```mon-hum-law```               | 30782317                | 39647.0               | 10855.0               | 4.0                      | 118.5              | 1.9                           |
| ```pri-hum-law```               | 18298816                | 27192.0               | 4833.0                | 1.0                      | 70.5               | 1.1                           |
| ```val-hum-law```               | 30171035                | 34570.5               | 9707.0                | 4.0                      | 116.2              | 2.1                           |
| ```bar-hum-cha```               | 25362111                | 35925.0               | 12574.0               | 5.0                      | 97.7               | 1.6                           |
| ```mon-hum-cha```               | 33662420                | 41730.5               | 11418.0               | 4.0                      | 129.6              | 2.1                           |
| ```pri-hum-cha```               | 18667816                | 28204.5               | 5847.0                | 2.0                      | 71.9               | 1.1                           |
| ```ran-hum-cha```               | 16999630                | 24698.5               | 6236.0                | 2.0                      | 65.6               | 1.0                           |
| ```wiz-hum-cha```               | 14635591                | 20257.0               | 3294.0                | 1.0                      | 56.4               | 0.9                           |

## Citing Katakamoba
```bibtex
@misc{kurenkov2023katakomba,
      title={Katakomba: Tools and Benchmarks for Data-Driven NetHack}, 
      author={Vladislav Kurenkov and Alexander Nikulin and Denis Tarasov and Sergey Kolesnikov},
      year={2023},
      eprint={2306.08772},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
