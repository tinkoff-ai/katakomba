from torch.utils.data import DataLoader

from d5rl.tasks import make_task_builder
from d5rl.utils.roles import Alignment, Race, Role, Sex


"""
Task Builder.

For each task (now it's just NetHackScore-v0-tty-bot-v0), make_task_builder
will output both environment builder and dataset builder, which could
further be used for precise evaluation and the dataset you use for 
your training purposes.
"""
env_builder, dataset_builder = make_task_builder("NetHackScore-v0-tty-bot-v0")


"""
Environment Builder.

- This class allows you to specify exactly which character traits combination you evaluate against. 
- You can specify all that's typically allowed by NetHack game: [roles, races, alignments, sex].
- Note that not all of the combinations are allowed by the NetHack, we filter out them for you.
- If you do not provide a specification, we assume that you evaluate against all possbile settings.
- You can also specify the seeds which are used for evaluation
  - This way you can make sure that you evaluate against the same dungeons
  - Not specifying anything would result in random dungeons at each evaluation
    (This is what should be done for reporting scores the ultimate true score)
"""
env_builder = (
    env_builder.roles([Role.MONK])
    .races([Race.HUMAN])
    .alignments([Alignment.NEUTRAL])
    .sex([Sex.MALE])
    .eval_seeds([1, 2, 3])
)


"""
An evaluation example.

- character is a short description that stands for 'role-race-alignment-sex'
  - Note that some characters do not possess sex.
  - Note that these are short labels (see utils/roles.py) for actual values.
- env is a typical gym env (with a reseed flag exception in reset)
- seed is the evaluation seed
  - In case none were provided, it will be None.
  - Note that you should specify the seed yourself.
"""
for character, env, seed in env_builder.evaluate():
    """
    In case you specified a certain set of seeds.
    Make sure you pass reseed=False in order to get the same dungeon for the same seed.
    """
    env.seed(seed, reseed=False)
    # your_super_eval_function(your_super_agent, env)


"""
Dataset Builder.

- This class allows you to specify which games will be in your training dataset.
- You can specify all that's typically allowed by NetHack game: [roles, races, alignments, sex].
- You can also specify concrete game_ids (but you better know what you're up to).
- You can also specify NetHack game versions to filter for (but you really better know what you're up to)
    - By default, we rely only on NetHack 3.6.6 trajectories.
- Note that not all of the combinations are allowed by the NetHack, we filter out them for you.
- If you do not provide a specification, we assume that you want the whole dataset for game_version=3.6.6.
- (!) You need to call build to get a dataset
    - batch_size is well, you know
    - seq_len is well, you also know
        - (!) sequences move by the seq_len, e.g., seq_len=4
            1st batch sequence timesteps = [1, 2, 3, 4]
            2nd batch sequence timesteps = [5, 6, 7, 9]
"""
dataset = (
    dataset_builder.roles([Role.MONK])
    .races([Race.HUMAN])
    .alignments([Alignment.NEUTRAL])
    .sex([Sex.MALE])
    .build(batch_size=4, seq_len=100)
)


"""
PyTorch DataLoader.

- As the dataset is already batched, we need to disable automatic batching.
"""
loader = DataLoader(
    dataset=dataset,
    # Disable automatic batching
    batch_sampler=None,
    batch_size=None,
)


"""
An iterator example. This will run indefinitely.
"""
for batch in loader:
    states, actions, rewards, dones, next_states = batch

    # [4, 100, 24, 80, 3]
    print(states.size())
    # [4, 100]
    print(actions.size())
    # [4, 100]
    print(rewards.size())
    # [4, 100]
    print(dones.size())
    # [4, 100, 24, 80, 3]
    print(next_states.size())
