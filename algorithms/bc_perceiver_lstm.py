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
from torch.distributions import Categorical
import numpy as np

from typing import Optional
from d5rl.datasets.sa_autoascend import SAAutoAscendTTYDataset
from d5rl.tasks import make_task_builder
from d5rl.utils.roles import Alignment, Race, Role, Sex
from d5rl.nn.perceiver.perceiver import Perceiver

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# TODO:
#   1. implement filtering of params for the weight decay groups
#   2. oncycle rl scheduler
#   3. ...
#   4. label smoothing for cross entropy loss
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
    env: str = "NetHackScore-v0-tty-bot-v0"
    data_path: str = "data/nle_data"
    db_path: str = "ttyrecs.db"
    # Wandb logging
    project: str = "NeuralNetHack"
    group: str = "DummyBC"
    name: str = "DummyBC"
    version: str = "v0"
    # Model
    resnet_type: str = "ResNet20"
    lstm_layers: int = 1
    hidden_dim: int = 1024
    width_k: int = 1
    chrono_init_tmax: Optional[int] = None
    # Training
    update_steps: int = 180_000
    batch_size: int = 256
    seq_len: int = 32
    n_workers: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 10_000
    eval_episodes_per_seed: int = 1
    eval_seeds: int = 50
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.group}-{self.env}-{self.version}"
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group, self.name)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def chrono_init(lstm: nn.LSTM, t_max=100):
    """
        Reference: https://arxiv.org/pdf/1804.11188.pdf
    """
    for name, p in lstm.named_parameters():
        if "bias_ih" in name:
            torch.nn.init.zeros_(p)
            p.requires_grad_(False)
        elif "bias_hh" in name:
            hidden_size = p.nelement() // 4
            torch.nn.init.zeros_(p)
            # init forget gate
            p.data[hidden_size: 2 * hidden_size].data.copy_(
                torch.Tensor(hidden_size).uniform_(1.0, t_max - 1).log()
            )
            # init input gate
            p.data[:hidden_size] = -p.data[hidden_size: 2 * hidden_size]


class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, lstm_layers, width_k, resnet_type, chrono_init_tmax=None):
        super().__init__()
        resnet = getattr(sys.modules[__name__], resnet_type)
        self.state_encoder = resnet(img_channels=2, out_dim=hidden_dim, k=width_k)
        self.norm = nn.LayerNorm(hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        if chrono_init_tmax is not None:
            chrono_init(self.rnn, t_max=chrono_init_tmax)

        # TODO: ortho/chrono init for the lstm
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, state=None):
        # [batch_size, seq_len, ...]
        batch_size, seq_len, *_ = obs.shape

        out = self.state_encoder(obs.flatten(0, 1)).view(batch_size, seq_len, -1)
        out, new_state = self.rnn(self.norm(out), state)
        logits = self.head(out)

        return logits, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        assert obs.ndim == 3, "act only for single obs"
        obs = torch.tensor(obs, device=device, dtype=torch.float32).permute(2, 0, 1)
        logits, new_state = self(obs[None, None, ...], state)
        return torch.argmax(logits).cpu().item(), new_state


@torch.no_grad()
def evaluate(env_builder, actor, episodes_per_seed, device="cpu"):
    actor.eval()
    eval_stats = defaultdict(dict)

    for (character, env, seed) in tqdm(env_builder.evaluate()):
        episodes_rewards = []
        for _ in trange(episodes_per_seed, desc="One seed evaluation", leave=False):
            env.seed(seed, reseed=False)

            obs, done, episode_reward = env.reset(), False, 0.0
            rnn_state = None

            while not done:
                action, rnn_state = actor.act(obs[..., :2], rnn_state, device=device)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            episodes_rewards.append(episode_reward)

        eval_stats[character][seed] = np.mean(episodes_rewards)

    # for each character also log mean across all seeds
    for character in eval_stats.keys():
        eval_stats[character]["mean_return"] = np.mean(list(eval_stats[character].values()))

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
        save_code=True
    )
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    set_seed(config.train_seed)
    env_builder, dataset_builder = make_task_builder(
        config.env,
        data_path=config.data_path,
        db_path=config.db_path
    )
    env_builder = (
        env_builder.roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
        .eval_seeds(list(range(config.eval_seeds)))
    )
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        n_workers=config.n_workers,
        auto_ascend_cls=SAAutoAscendTTYDataset
    )
    actor = Actor(
        resnet_type=config.resnet_type,
        action_dim=env_builder.get_action_dim(),
        hidden_dim=config.hidden_dim,
        lstm_layers=config.lstm_layers,
        width_k=config.width_k,
        chrono_init_tmax=config.chrono_init_tmax
    ).to(DEVICE)
    print("Number of parameters:",  sum(p.numel() for p in actor.parameters()))
    # ONLY FOR MLC/TRS
    # actor = torch.compile(actor, mode="reduce-overhead")
    optim = torch.optim.AdamW(
        (p for p in actor.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    loader = DataLoader(
        dataset=dataset,
        # Disable automatic batching
        batch_sampler=None,
        batch_size=None,
        pin_memory=True
    )
    scaler = torch.cuda.amp.GradScaler()

    rnn_state = None
    loader_iter = iter(loader)
    for step in trange(config.update_steps, desc="Training"):
        with timeit() as timer:
            tty_chars, tty_colors, tty_cursor, actions = next(loader_iter)

        wandb.log({
            "times/batch_loading_cpu": timer.elapsed_time_cpu,
            "times/batch_loading_gpu": timer.elapsed_time_gpu
        }, step=step)

        with timeit() as timer:
            with torch.cuda.amp.autocast():
                states = torch.stack([tty_chars, tty_colors], axis=-1)
                logits, rnn_state = actor(
                    states.permute(0, 1, 4, 2, 3).to(DEVICE).to(torch.float32),
                    state=rnn_state
                )
                rnn_state = [a.detach() for a in rnn_state]

                dist = Categorical(logits=logits)
                loss = -dist.log_prob(actions.to(DEVICE)).mean()

        wandb.log({"times/forward_pass": timer.elapsed_time_gpu}, step=step)

        with timeit() as timer:
            scaler.scale(loss).backward()
            # loss.backward()
            if config.clip_grad_norm is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(actor.parameters(), config.clip_grad_norm)
            # optim.step()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        wandb.log({"times/backward_pass": timer.elapsed_time_gpu}, step=step)

        wandb.log({
            "loss": loss.detach().item(),
            "transitions": config.batch_size * config.seq_len * step
        }, step=step)

        if (step + 1) % config.eval_every == 0:
            eval_stats = evaluate(env_builder, actor, config.eval_episodes_per_seed, device=DEVICE)
            wandb.log(
                dict(eval_stats, **{"transitions": config.batch_size * config.seq_len * step}), step=step)

            if config.checkpoints_path is not None:
                torch.save(
                    actor.state_dict(),
                    os.path.join(config.checkpoints_path, f"{step}.pt"),
                )


if __name__ == "__main__":
    train()
