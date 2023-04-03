"""
Key differneces or uncertanties to the original implementation:
 1. Dones are not used for masking out the rnn_state
 2. (?) Actions are argmaxed, not sampled (not sure yet how it's done in the original implementation)
"""
import pyrallis
from dataclasses import dataclass, asdict

import time
import random
import wandb
import sys
import os
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange
from collections import defaultdict
import numpy as np

from typing import Optional, Tuple
from katakomba.datasets.sa_chaotic_autoascend import SAChaoticAutoAscendTTYDataset
from katakomba.tasks import make_task_builder
from katakomba.utils.roles import Alignment, Race, Role, Sex
from katakomba.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from katakomba.utils.render import SCREEN_SHAPE

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


class timeit:
    def __enter__(self):
        self.start_gpu = torch.cuda.Event(enable_timing=True)
        self.end_gpu = torch.cuda.Event(enable_timing=True)
        self.start_cpu = time.time()
        self.start_gpu.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_gpu.record()
        torch.cuda.synchronize()
        self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        self.elapsed_time_cpu = time.time() - self.start_cpu


@dataclass
class TrainConfig:
    env: str = "NetHackScore-v0-ttyimg-bot-v0"
    data_path: str = "data/nle_data"
    db_path: str = "ttyrecs.db"
    # Wandb logging
    project: str = "NeuralNetHack"
    group: str = "ChaoticDwarfen-BC"
    name: str = "ChaoticDwarfen-BC"
    version: str = "v0"
    # Model
    rnn_hidden_dim: int = 512
    rnn_layers: int = 1
    use_prev_action: bool = True
    # Training
    update_steps: int = 180_000
    batch_size: int = 256
    seq_len: int = 32
    n_workers: int = 16
    learning_rate: float = 0.0001
    clip_grad_norm: Optional[float] = 4.0
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes: int = 50
    eval_processes: int = 8
    eval_seed: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-{self.env}-{self.version}"
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(
                self.checkpoints_path, self.group, self.name
            )


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class BC(nn.Module):
    def __init__(self, action_dim: int, rnn_hidden_dim: int = 512, rnn_layers: int = 1, use_prev_action: bool = True):
        super().__init__()
        # Action dimensions and prev actions
        self.num_actions = action_dim
        self.use_prev_action = use_prev_action
        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        # Encoders
        self.topline_encoder = TopLineEncoder()
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        screen_shape = (SCREEN_SHAPE[1], SCREEN_SHAPE[2])
        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )
        # Policy
        self.rnn = nn.LSTM(self.h_dim, rnn_hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Linear(rnn_hidden_dim, self.num_actions)

    def forward(self, inputs, state=None):
        B, T, C, H, W = inputs["screen_image"].shape
        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        encoded_state = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .float(memory_format=torch.contiguous_format)
                .view(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            encoded_state.append(
                torch.nn.functional.one_hot(
                    inputs["prev_actions"], self.num_actions
                ).view(T * B, -1)
            )

        encoded_state = torch.cat(encoded_state, dim=1)
        out, new_state = self.rnn(encoded_state.view(B, T, -1), state)
        logits = self.head(out)

        return logits, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        assert obs["tty_chars"].ndim == 3, "obs should be batched and without seq_len dim"
        inputs = {
            "tty_chars": torch.tensor(obs["tty_chars"][:, np.newaxis], device=device),
            "tty_colors": torch.tensor(obs["tty_colors"][:, np.newaxis], device=device),
            "screen_image": torch.tensor(obs["screen_image"][:, np.newaxis], device=device),
            "prev_actions": torch.tensor(obs["prev_actions"][:, np.newaxis], dtype=torch.long, device=device),
        }
        logits, new_state = self(inputs, state)
        actions = torch.argmax(logits.squeeze(1), dim=-1)
        return actions.cpu().numpy(), new_state


@torch.no_grad()
def evaluate_character(vec_env, actor, num_episodes, device="cpu", seed=None):
    # set seed for reproducibility (reseed=False by default)
    vec_env.seed(seed)
    # all is work is needed to mitigate bias for shorter episodes during vectorized evaluation, for more see:
    # https://github.com/DLR-RM/stable-baselines3/issues/402
    n_envs = vec_env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(num_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = vec_env.reset()
    observations["prev_actions"] = np.zeros(n_envs, dtype=float)

    states = None
    pbar = tqdm(total=num_episodes)
    while (episode_counts < episode_count_targets).any():
        actions, states = actor.vec_act(observations, states, device=device)

        observations, rewards, dones, infos = vec_env.step(actions)
        observations["prev_actions"] = actions

        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    pbar.update(1)

                    current_rewards[i] = 0
                    current_lengths[i] = 0

    pbar.close()
    result = {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_min": np.min(episode_rewards),
        "reward_max": np.max(episode_rewards),
        # "raw_episode_rewards": episode_rewards,
        # "raw_episode_lengths": episode_lengths
    }
    return result


def evaluate_all_characters(env_builder, actor, num_episodes, num_processes=8, device="cpu", seed=None):
    actor.eval()

    eval_stats = {}
    for character, vec_env in env_builder.vectorized_evaluate(num_processes):
        print(f"Evaluating {character}:")
        eval_stats[character] = evaluate_character(vec_env, actor, num_episodes, device, seed)

    actor.train()
    return eval_stats


@pyrallis.wrap()
def train(config: TrainConfig):
    print(f"Device: {DEVICE}")
    saved_config = asdict(config)
    saved_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")
    wandb.init(
        config=saved_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.train_seed)

    env_builder, dataset_builder = make_task_builder(
        config.env, data_path=config.data_path, db_path=config.db_path
    )
    env_builder = (
        env_builder.roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
    )

    dataset_builder = dataset_builder.roles([Role.MONK]).races([Race.HUMAN])
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_workers=config.n_workers,
        auto_ascend_cls=SAChaoticAutoAscendTTYDataset,
    )

    actor = BC(
        action_dim=env_builder.get_action_dim(),
        use_prev_action=config.use_prev_action,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_layers=config.rnn_layers,
    ).to(DEVICE)
    optim = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    print("Number of parameters:", sum(p.numel() for p in actor.parameters()))

    loader = DataLoader(
        dataset=dataset,
        # Disable automatic batching
        batch_sampler=None,
        batch_size=None,
        pin_memory=True,
    )
    scaler = torch.cuda.amp.GradScaler()

    prev_actions = torch.zeros((config.batch_size, 1), dtype=torch.long, device=DEVICE)
    rnn_state = None

    loader_iter = iter(loader)
    for step in trange(config.update_steps, desc="Training"):
        with timeit() as timer:
            tty_chars, tty_colors, tty_cursor, screen_image, actions = (
                a.to(DEVICE) for a in next(loader_iter)
            )
            actions = actions.long()

        wandb.log(
            {
                "times/batch_loading_cpu": timer.elapsed_time_cpu,
                "times/batch_loading_gpu": timer.elapsed_time_gpu,
            },
            step=step,
        )

        with timeit() as timer:
            with torch.cuda.amp.autocast():
                logits, rnn_state = actor(
                    inputs={
                        "tty_chars": tty_chars,
                        "tty_colors": tty_colors,
                        "screen_image": screen_image,
                        "prev_actions": torch.cat(
                            [prev_actions.long(), actions[:, :-1]], dim=1
                        ),
                    },
                    state=rnn_state,
                )
                rnn_state = [a.detach() for a in rnn_state]

                dist = Categorical(logits=logits)
                loss = -dist.log_prob(actions).mean()
                # update prev_actions for next iteration
                prev_actions = actions[:, -1].unsqueeze(-1)

        wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        with timeit() as timer:
            scaler.scale(loss).backward()

            if config.clip_grad_norm is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), config.clip_grad_norm
                )

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        wandb.log({"times/backward_pass": timer.elapsed_time_gpu}, step=step)

        wandb.log(
            {
                "loss": loss.detach().item(),
                "transitions": config.batch_size * config.seq_len * step,
            },
            step=step,
        )

        if (step + 1) % config.eval_every == 0:
            eval_stats = evaluate_all_characters(
                env_builder, actor, config.eval_episodes, config.eval_processes, device=DEVICE, seed=config.eval_seed
            )
            wandb.log(
                dict(
                    eval_stats,
                    **{"transitions": config.batch_size * config.seq_len * step},
                ),
                step=step,
            )

            if config.checkpoints_path is not None:
                torch.save(
                    actor.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
