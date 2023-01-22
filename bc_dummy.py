from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
from collections import defaultdict
import os
from pathlib import Path
import random
import uuid

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from d5rl.tasks import make_task_builder, NetHackEnvBuilder
from d5rl.utils.roles import Role, Alignment, Race, Sex
from torch.utils.data import DataLoader

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # NetHack
    env        : str                 = "NetHackScore-v0-tty-bot-v0"
    character  : str                 = "mon-hum-neutral-male"
    eval_seeds: Optional[Tuple[int]] = (228, 1337, 1307, 2, 10000)

    # Training
    device          : str           = "cpu"
    seed            : int           = 0
    eval_freq       : int           = int(1000)
    n_episodes      : int           = 10
    max_timesteps   : int           = int(1e6)
    checkpoints_path: Optional[str] = None
    load_model      : str           = ""
    batch_size      : int           = 512

    # Wandb logging
    project: str = "NeuralNetHack"
    group  : str = "DummyBC"
    name   : str = "DummyBC"
    version: str = "v0"

    def __post_init__(self):
        self.group = f"{self.env}-{self.name}-{self.version}"
        self.name  = f"{self.group}-{str(uuid.uuid4())[:8]}"

        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config  = config,
        project = config["project"],
        group   = config["group"],
        name    = config["name"],
        id      = str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env_builder: NetHackEnvBuilder,
    actor      : nn.Module,
    device     : str,
    n_episodes : int,
) -> Dict[str, Dict[int, float]]:
    actor.eval()
    eval_stats = defaultdict(dict)

    for (character, env, seed) in env_builder.evaluate():
        episode_rewards = []
        for episode_ind in range(n_episodes):
            env.seed(seed, reseed=False)
            state, done = env.reset(), False
            episode_reward = 0.0
            while not done:
                action = actor.act(state, device)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
        eval_stats[character][seed] = np.mean(episode_rewards)


    actor.train()

    return eval_stats


class Actor(nn.Module):
    def __init__(self, action_dim: int):
        super(Actor, self).__init__()

        self.chars_encoder = nn.Sequential(
            nn.Linear(24 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.colors_encoder = nn.Sequential(
            nn.Linear(24 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.cursor_encoder = nn.Sequential(
            nn.Linear(24 * 80, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(256*3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        state      = state.view(batch_size, -1, 3) / 255.0

        chars_encoded  = self.chars_encoder(state[:, :, 0])
        colors_encoded = self.colors_encoder(state[:, :, 1])
        cursor_encoded = self.cursor_encoder(state[:, :, 2])

        return self.head(torch.concat([chars_encoded, colors_encoded, cursor_encoded], dim=-1))

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state  = torch.tensor(np.expand_dims(state, axis=0), device=device, dtype=torch.float32)
        logits = self(state)
        return torch.argmax(logits).cpu().item()


class BC:  # noqa
    def __init__(
        self,
        actor          : nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        device         : str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, done, next_state = batch

        # Compute actor loss
        pi = self.actor(state.squeeze())
        actor_loss = F.cross_entropy(pi, action.view(-1,))
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    # NetHack builders
    env_builder, dataset_builder = make_task_builder(config.env)
    env_builder = (
        env_builder
        .roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
        .eval_seeds(list(config.eval_seeds))
    )
    dataset = (
        dataset_builder
        .roles([Role.MONK])
        .races([Race.HUMAN])
        .alignments([Alignment.NEUTRAL])
        .sex([Sex.MALE])
        .build(batch_size=config.batch_size, seq_len=1, n_prefetched_batches=100)
    )
    loader = DataLoader(
        dataset       = dataset,
        # Disable automatic batching
        batch_sampler = None,
        batch_size    = None
    )

    # Get number of actions for the task of interest
    action_dim = env_builder.get_action_dim()

    # Save stuff
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds for training
    set_seed(config.seed)

    actor = Actor(action_dim).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "device": config.device,
    }

    # print("---------------------------------------")
    # print(f"Training BC, Env: {config.env}, Seed: {seed}")
    # print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t, batch in enumerate(loader):
        batch    = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        # Log train
        wandb.log(log_dict, step=trainer.total_it)

        # Evaluate episode
        if (t) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            eval_stats = eval_actor(
                env_builder = env_builder,
                actor       = actor,
                device      = config.device,
                n_episodes  = config.n_episodes,
            )

            print(eval_stats)
            wandb.log(eval_stats, step=trainer.total_it)
            # print("---------------------------------------")
            # print(
            #     f"Evaluation over {config.n_episodes} episodes: "
            #     f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            # )
            # print("---------------------------------------")
            # torch.save(
            #     trainer.state_dict(),
            #     os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
            # )
            # wandb.log(
            #     {"d4rl_normalized_score": normalized_eval_score},
            #     step=trainer.total_it,
            # )


if __name__ == "__main__":
    train()