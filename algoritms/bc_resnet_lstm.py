import pyrallis
from dataclasses import dataclass

import sys
import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.distributions import Categorical
import numpy as np

from typing import Optional, Tuple
from d5rl.tasks import NetHackEnvBuilder, make_task_builder
from d5rl.utils.roles import Alignment, Race, Role, Sex
from d5rl.nn.resnet import ResNet11, ResNet20, ResNet38, ResNet56, ResNet110

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    env: str = "NetHackScore-v0-tty-bot-v0"
    # Wandb logging
    project: str = "NeuralNetHack"
    group: str = "DummyBC"
    name: str = "DummyBC"
    version: str = "v0"
    # Model
    resnet_type: str = "ResNet11"
    lstm_layers: int = 2
    hidden_dim: int = 512
    width_k: int = 1
    # Training
    update_steps: int = 100_000
    batch_size: int = 256
    seq_len: int = 64
    learning_rate: float = 3e-4
    clip_grad: Optional[float] = None
    checkpoints_path: Optional[str] = None
    eval_every: int = 1000
    eval_episodes: int = 10
    eval_seeds: Tuple[int] = (228, 1337, 1307, 2, 10000)
    train_seed: int = 42

    def __post_init__(self):
        self.group = f"{self.env}-{self.name}-{self.version}"
        self.name = f"{self.group}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, lstm_layers, width_k, resnet_type):
        super().__init__()
        resnet = getattr(sys.modules[__name__], resnet_type)
        self.state_encoder = resnet(img_channels=3, out_dim=hidden_dim, k=width_k)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, state=None):
        # [batch_size, seq_len, ...]
        batch_size, seq_len, *_ = obs.shape

        out = self.state_encoder(obs.flatten(0, 1)).view(batch_size, seq_len, -1)
        out, new_state = self.rnn(out, state)
        logits = self.head(out)

        return logits, new_state

    @torch.no_grad()
    def act(self, obs, state=None, device="cpu"):
        assert obs.ndim == 3, "act only for single obs"
        obs = torch.tensor(state[None, None, ...], device=device, dtype=torch.float32)
        logits, new_state = self(obs, state)
        return torch.argmax(logits).cpu().item(), new_state


@pyrallis.wrap()
def train(config: TrainConfig):
    env_builder, dataset_builder = make_task_builder(config.env)

    env_builder = env_builder.eval_seeds(config.eval_seeds)
    dataset = dataset_builder.build(
        batch_size=config.batch_size,
        seq_len=config.seq_len
    )
    actor = Actor(
        resnet_type=config.resnet_type,
        action_dim=env_builder.get_action_dim(),
        hidden_dim=config.hidden_dim,
        lstm_layers=config.lstm_layers,
        width_k=config.width_k
    )
    optim = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)

    loader = DataLoader(
        dataset=dataset,
        # Disable automatic batching
        batch_sampler=None,
        batch_size=None,
    )

    rnn_state = None
    for idx, batch in enumerate(loader):
        if idx >= config.update_steps:
            break

        states, actions, *_ = batch
        logits, rnn_state = actor(
            states.permute(0, 1, 4, 2, 3).to(DEVICE).to(torch.float32),
            state=rnn_state
        )
        rnn_state = [a.detach() for a in rnn_state]

        dist = Categorical(logits=logits)
        loss = -dist.log_prob(actions.to(DEVICE)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm(actor.parameters(), config.clip_grad)
        optim.step()

        print(loss)


if __name__ == "__main__":
    train()
