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
import torch.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm, trange
from collections import defaultdict
from torch.distributions import Categorical
import numpy as np

from typing import Optional, Tuple
from d5rl.datasets.sa_chaotic_autoascend import SAChaoticAutoAscendTTYDataset
from d5rl.tasks import make_task_builder
from d5rl.utils.roles import Alignment, Race, Role, Sex
from d5rl.nn.chaotic_dwarf import TopLineEncoder, BottomLinesEncoder, ScreenEncoder
from d5rl.utils.render import SCREEN_SHAPE

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    group: str = "DummyBC"
    name: str = "DummyBC"
    version: str = "v0"
    # Training
    update_steps: int = 180_000
    batch_size: int = 256
    seq_len: int = 32
    n_workers: int = 16
    learning_rate: float = 0.0001
    clip_grad_norm: Optional[float] = 4.0
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes_per_seed: int = 1
    eval_seeds: int = 50
    train_seed: int = 42
    use_prev_action: bool = True

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


class ChaoticDwarvenGPT5(nn.Module):
    def __init__(self, action_dim: int, use_prev_action: bool = True):
        super(ChaoticDwarvenGPT5, self).__init__()

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

        self.hidden_dim = 512

        # Policy
        self.rnn = nn.LSTM(self.h_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.num_actions)

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.zeros(
                self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=DEVICE
            )
            for _ in range(2)
        )

    def forward(self, inputs, rnn_state):
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
        core_output, rnn_state = self.rnn(encoded_state.view(B, T, -1), rnn_state)
        policy_logits = self.head(core_output)

        return policy_logits, rnn_state

    @torch.no_grad()
    def act(self, obs, rnn_state):
        inputs = {
            "tty_chars": torch.tensor(
                obs["tty_chars"][np.newaxis, np.newaxis, ...], device=DEVICE
            ),
            "tty_colors": torch.tensor(
                obs["tty_colors"][np.newaxis, np.newaxis, ...], device=DEVICE
            ),
            "screen_image": torch.tensor(
                obs["screen_image"][np.newaxis, np.newaxis, ...], device=DEVICE
            ),
            "prev_actions": torch.tensor(
                np.array([obs["prev_actions"]]).reshape(1, 1),
                dtype=torch.long,
                device=DEVICE,
            ),
        }
        logits, new_state = self(inputs, rnn_state)
        # action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        return torch.argmax(logits).cpu().item(), new_state


@torch.no_grad()
def evaluate(
    env_builder, actor: ChaoticDwarvenGPT5, episodes_per_seed: int, device="cpu"
):
    actor.eval()
    eval_stats = defaultdict(dict)

    for (character, env, seed) in tqdm(env_builder.evaluate()):
        episodes_rewards = []
        for _ in trange(episodes_per_seed, desc="One seed evaluation", leave=False):
            env.seed(seed, reseed=False)

            obs, done, episode_reward = env.reset(), False, 0.0

            rnn_state = actor.initial_state(batch_size=1)
            obs["prev_actions"] = 0

            while not done:
                action, rnn_state = actor.act(obs, rnn_state)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                obs["prev_actions"] = action

            episodes_rewards.append(episode_reward)

        eval_stats[character][seed] = np.mean(episodes_rewards)

    # for each character also log mean across all seeds
    for character in eval_stats.keys():
        eval_stats[character]["mean_return"] = np.mean(
            list(eval_stats[character].values())
        )

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
        .eval_seeds(list(range(config.eval_seeds)))
    )

    dataset_builder = dataset_builder.roles([Role.MONK]).races([Race.HUMAN])
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_workers=config.n_workers,
        auto_ascend_cls=SAChaoticAutoAscendTTYDataset,
    )

    actor = ChaoticDwarvenGPT5(
        action_dim=env_builder.get_action_dim(), use_prev_action=config.use_prev_action
    ).to(DEVICE)
    optim = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)
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
    rnn_state = actor.initial_state(config.batch_size)

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
                    rnn_state=rnn_state,
                )
                rnn_state = [a.detach() for a in rnn_state]

                dist = Categorical(logits=logits)
                loss = -dist.log_prob(actions).mean()

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
            eval_stats = evaluate(
                env_builder, actor, config.eval_episodes_per_seed, device=DEVICE
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
